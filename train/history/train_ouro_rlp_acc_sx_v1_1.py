import os
import json
import torch
from torchdata.stateful_dataloader import StatefulDataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
from modeling_ouro import OuroForCausalLM, OuroConfig
from accelerate import Accelerator
from accelerate.utils import set_seed
from accelerate import DistributedDataParallelKwargs
import hydra
from omegaconf import DictConfig, OmegaConf
import glob
import logging
import time
from utils import logprobs_from_logits,Tracking,append_to_dict,aggregate_metrics
from train.rlp_trainer import RLPTrainer
from utils.dataset_pt import OmniMathDataset,OpenThoughtsDataset,DataCollatorWithPadding
logger = logging.getLogger(__name__)


# ==========================================
# Main Execution
# ==========================================
@hydra.main(version_base=None, config_path="configs", config_name="ouro_rlp_acc_omnimath")
def main(config: DictConfig):
    # --- Config --- #
    CONFIG = OmegaConf.to_container(config, resolve=True)  # 转为普通字典
    if os.environ.get("RANK","-1")=="0":
        logger.info(f"RANK: {os.environ.get('RANK', '-1')}")
        logger.info(f"DEBUG_RANK: {os.environ.get('DEBUG_RANK', '-1')}")
        logger.info(f"DEBUG_MODE: {os.environ.get('DEBUG_MODE', '0')}")
        logger.info(json.dumps(CONFIG, indent=4))
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
    accelerator = Accelerator(
        gradient_accumulation_steps=CONFIG["gradient_accumulation_steps"],
        mixed_precision="bf16",
        kwargs_handlers=[ddp_kwargs],
    )
    accelerator.init_trackers(project_name="ouro-rlp-acc-sx-v1", config=CONFIG, init_kwargs={"name":CONFIG["experiment_name"] })
    set_seed(CONFIG["seed"] + accelerator.process_index)#XXX:check一下

    if accelerator.is_main_process:
        logging.info(f"Distributed training on {accelerator.num_processes} GPUs.")
        if not os.path.exists(CONFIG["output_dir"]):
            os.makedirs(CONFIG["output_dir"])

    tokenizer = AutoTokenizer.from_pretrained(CONFIG["model_path"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = (
            tokenizer.eos_token
        )  # FIXME:可能会存在最后的eos_token被mask，这里的eos_token是<|endoftext|>
        if CONFIG.get("apply_chat", False): # XXX:根据后续Ouro仓库的回复修改
            tokenizer.eos_token = "<|im_end|>" # THREEGOLD:如果按照chatml的template,eos_token应该设置为<|im_end|>
            
    config = OuroConfig.from_pretrained(CONFIG["model_path"])
    config.total_ut_steps = 4

    student = OuroForCausalLM.from_pretrained(
        CONFIG["model_path"],
        config=config,
        low_cpu_mem_usage=True,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2"
    )
    student.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    ) 

    teacher = OuroForCausalLM.from_pretrained(
        CONFIG["model_path"],
        config=config,
        low_cpu_mem_usage=True,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2"
    )
    teacher.eval() #FIXME:这里的eval需要吗?
    for p in teacher.parameters(): #FIXME:需不需要设置requires_grad=False?
        p.requires_grad = False
    # --- Data Loading & Splitting ---
    if CONFIG["dataset_name"] == "OpenThoughts":
        if os.path.isdir(CONFIG["train_data_path"]):    
            train_data_files = glob.glob(os.path.join(CONFIG["train_data_path"], "*.parquet"))
        else:
            train_data_files = [CONFIG["train_data_path"]]
        if os.path.isdir(CONFIG["val_data_path"]):
            val_data_files = glob.glob(os.path.join(CONFIG["val_data_path"], "*.parquet"))
        else:
            val_data_files = [CONFIG["val_data_path"]]
        train_dataset = OpenThoughtsDataset(train_data_files, tokenizer, apply_chat=CONFIG["apply_chat"],truncation=CONFIG["truncation"])
        val_dataset = OpenThoughtsDataset(val_data_files, tokenizer, apply_chat=CONFIG["apply_chat"],truncation=CONFIG["truncation"])
    collate_fn = DataCollatorWithPadding(tokenizer, max_length=CONFIG["max_length"], truncation=CONFIG["truncation"])
    train_dataloader = StatefulDataLoader(
        train_dataset, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=CONFIG.get("num_workers",4),drop_last=True, collate_fn=collate_fn
    )
    val_dataloader = StatefulDataLoader(
        val_dataset, batch_size=CONFIG["val_batch_size"], shuffle=False, num_workers=CONFIG.get("num_workers",4),drop_last=True, collate_fn=collate_fn
    )
    # [修改] 优化所有参数 (不再过滤 requires_grad，因为默认都是 True)
    optimizer = AdamW(
        student.parameters(),
        lr=CONFIG["lr"],
        weight_decay=CONFIG["weight_decay"],
    )
    num_training_steps = (len(train_dataloader) * CONFIG["num_epochs"]) // (CONFIG["gradient_accumulation_steps"]*accelerator.num_processes)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.03 * num_training_steps),
        num_training_steps=num_training_steps,
    )
    trainer = RLPTrainer(student, teacher, tokenizer, CONFIG, accelerator, train_dataloader, val_dataloader, optimizer, lr_scheduler)

    (
        trainer.student,
        trainer.teacher,
        trainer.optimizer,
        trainer.train_dataloader,
        trainer.val_dataloader,
        trainer.lr_scheduler,
    ) = accelerator.prepare(
        trainer.student,
        trainer.teacher,
        trainer.optimizer,
        trainer.train_dataloader,
        trainer.val_dataloader,
        trainer.lr_scheduler,
    )

    # --- Training Loop ---
    global_step = -1
    for epoch in range(CONFIG["num_epochs"]):
        trainer.student.eval()
        val_metrics = trainer.validate()
        trainer.student.train()
        for step, batch in enumerate(trainer.train_dataloader):
            metrics = {}
            with accelerator.accumulate(trainer.student):
                loss, train_metrics = trainer.compute_loss(
                    batch
                )
                accelerator.backward(loss)
                append_to_dict(metrics, train_metrics)
                if os.environ.get("RANK","-1") in os.environ.get(
                    "DEBUG_RANK", ""
                ) and "3" in os.environ.get("DEBUG_MODE", "0"):
                    breakpoint()
                if accelerator.sync_gradients:
                    time_start = time.time()
                    accelerator.clip_grad_norm_(trainer.student.parameters(), 1.0)
                    trainer.optimizer.step()
                    lr_scheduler.step()
                    trainer.optimizer.zero_grad()
                    trainer.update_ema()
                    global_step += 1
                    metrics.update({
                        "time/update": torch.tensor(time.time() - time_start).to(accelerator.device),
                    })
                    time_start = time.time()
                    reduced_metrics = aggregate_metrics(metrics) #平均值
                    # for key, val in reduced_metrics.items():
                    #     if isinstance(val, torch.Tensor):
                    #         reduced_metrics[key] = accelerator.reduce(val,reduction="mean") # 归并所有卡上的值
                    if global_step % CONFIG["log_interval"] == 0:
                        if accelerator.is_main_process:
                            total_loss = reduced_metrics["actor/total_loss"]
                            pg_loss = reduced_metrics["actor/pg_loss"]
                            model_loss = reduced_metrics["actor/model_loss"]
                            expected_step = reduced_metrics["student/expected_step"]
                            avg_teacher_step = reduced_metrics["teacher/avg_teacher_step"]
                            final_reward = reduced_metrics["group_student/final_reward"]
                            logger.info(
                                f'Ep {epoch}|St {global_step}|L:{total_loss:.4f}|PG:{pg_loss:.4f}|M:{model_loss:.4f}|Stp:{expected_step:.2f}|Ref:{avg_teacher_step:.2f}|Rwd:{final_reward:.4f}'
                            )
                            accelerator.log(reduced_metrics,global_step)
                        accelerator.wait_for_everyone()
                        
            # --- Validation Loop ---
            if (
                global_step % CONFIG["val_check_interval"] == 0
                and global_step >=0
            ):
                if accelerator.is_main_process:
                    logger.info(f"Running Validation at step {global_step}...")
                val_metrics = trainer.validate()
                val_metrics = accelerator.reduce(val_metrics,reduction="sum")
                trainer.student.train()
                if accelerator.is_main_process:
                    logger.info(f" >>> VAL | Peak Acc: {val_metrics['peak_correct_num'].item() / (val_metrics['pred_num'].item()+1e-6):.4f} | Adapt Acc: {val_metrics['adaptive_correct_num'].item() / (val_metrics['pred_num'].item()+1e-6):.4f} | Avg Steps: {val_metrics['exit_steps_num'].item() / (val_metrics['pred_num'].item()+1e-6):.2f}")
                    accelerator.log(val_metrics,global_step)
        if accelerator.is_main_process:
            save_path = os.path.join(CONFIG["output_dir"], f"epoch_{epoch}")
            accelerator.unwrap_model(trainer.student).save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)
    accelerator.wait_for_everyone()
    accelerator.end_training()



if __name__ == "__main__":
    main()
