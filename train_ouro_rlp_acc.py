import os
import json
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
from modeling_ouro import OuroForCausalLM, OuroConfig
from accelerate import Accelerator
from accelerate.utils import set_seed
from accelerate import DistributedDataParallelKwargs

# ==========================================
# Configuration (全量微调版)
# ==========================================
CONFIG = {
    "model_path": "/home/gtang/pretrain_model/Ouro-1.4B", 
    "data_path": "/home/gtang/self-learn/datasets/Omni-MATH/test.jsonl", 
    "output_dir": "/home/gtang/self-learn/ouro_rlp_checkpoints_bf16",
    "max_length": 2048,
    "batch_size": 4, 
    "gradient_accumulation_steps": 2,
    "lr": 1e-6,              # 全量微调保持低学习率，非常安全
    "weight_decay": 0.01,
    "num_epochs": 1,
    "ema_decay": 0.9995,
    "seed": 42,
    "warmup_ratio": 0.03,
    
    # --- RLP & Ouro Specifics ---
    "time_penalty": 0.05,    # 思考成本
    "kl_coef": 0.05,         # [关键] KL惩罚，防止骨干跑偏
    "entropy_coef": 0.01,    # 熵正则
    "reward_scale": 1.0,     # GRPO下无需过大
    
    # [调整] 权重平衡
    "pg_coef": 0.1,          # 降低 PG 权重，防止噪声干扰骨干学习
    "model_coef": 1.0,       # 保持 Model Loss 为主导，确保语言能力
}

# ==========================================
# Dataset
# ==========================================
class OmniMathDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length):
        self.data = []
        if os.path.exists(data_path):
            with open(data_path, "r") as f:
                for line in f:
                    try:
                        item = json.loads(line)
                        text = f"Problem: {item['problem']}\nSolution: {item['solution']}" + tokenizer.eos_token
                        self.data.append(text)
                    except:
                        continue
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data[idx]
        enc = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        labels = enc["input_ids"].squeeze(0).clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": labels
        }

# ==========================================
# Trainer
# ==========================================
class RLPTrainer:
    def __init__(self, student, teacher, tokenizer, config, accelerator):
        self.student = student
        self.teacher = teacher
        self.tokenizer = tokenizer
        self.config = config
        self.accelerator = accelerator 
        
        # [修改] 优化所有参数 (不再过滤 requires_grad，因为默认都是 True)
        self.optimizer = AdamW(
            self.student.parameters(), 
            lr=config["lr"], 
            weight_decay=config["weight_decay"]
        )
        
        # 初始化 Teacher EMA
        self.update_ema(decay=0.0)

    def update_ema(self, decay=None):
        """Update Teacher model with EMA of Student weights"""
        actual_decay = self.config["ema_decay"] if decay is None else decay
        student_model = self.accelerator.unwrap_model(self.student)
        teacher_model = self.accelerator.unwrap_model(self.teacher)
        
        # 全量 EMA 更新
        with torch.no_grad():
            for p_s, p_t in zip(student_model.parameters(), teacher_model.parameters()):
                p_t.data.mul_(actual_decay).add_(p_s.data, alpha=1 - actual_decay)

    def compute_loss(self, batch):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        
        shift_labels = labels[..., 1:].clone()
        active_mask = (shift_labels != -100).float()
        safe_shift_labels = shift_labels.clone()
        safe_shift_labels[shift_labels == -100] = 0

        # =================================================================
        # PHASE 1: Rollout & Evaluation (Inference Mode)
        # =================================================================
        self.student.eval()
        self.teacher.eval()
        
        with torch.no_grad():
            with self.accelerator.autocast():
                # 1.1 Reference (Teacher) LogProbs - 基准线
                teacher_outputs = self.teacher(input_ids, attention_mask=attention_mask, use_cache=False)
                teacher_logits = teacher_outputs.logits[..., :-1, :].contiguous()
                ref_logprobs = -F.cross_entropy(
                    teacher_logits.transpose(1, 2), safe_shift_labels, reduction='none'
                ) # (B, S-1)

                # 1.2 Student Rollout (所有 Steps)
                outputs_inf = self.student(input_ids, attention_mask=attention_mask, use_cache=False)
                raw_student = self.accelerator.unwrap_model(self.student)
                
                step_logprobs_list = []
                for hidden in outputs_inf.hidden_states_list:
                    logits = raw_student.lm_head(hidden)
                    shift_logits = logits[..., :-1, :].contiguous()
                    step_lp = -F.cross_entropy(
                        shift_logits.transpose(1, 2), safe_shift_labels, reduction='none'
                    )
                    step_logprobs_list.append(step_lp)
                
                # (Batch, Seq, Steps)
                student_logprobs_det = torch.stack(step_logprobs_list, dim=-1)

                # =================================================================
                # REWARD CALCULATION (GRPO + KL)
                # =================================================================
                
                # A. Info Gain (Task Reward)
                info_gain = student_logprobs_det - ref_logprobs.unsqueeze(-1)
                
                # B. KL Penalty (Log Ratio Approx)
                # P_student / P_ref
                kl_penalty = student_logprobs_det - ref_logprobs.unsqueeze(-1)
                
                # C. Time Penalty
                step_indices = torch.arange(student_logprobs_det.shape[-1], device=input_ids.device)
                
                # D. Total Reward
                # Reward = InfoGain - beta * KL - TimeCost
                rewards = info_gain - self.config["kl_coef"] * kl_penalty - self.config["time_penalty"] * step_indices
                rewards = rewards * self.config["reward_scale"]

                # E. GRPO Normalization (Normalize over Steps dimension)
                # 关键：这解决了 Reward 为负的问题
                r_mean = rewards.mean(dim=-1, keepdim=True)
                r_std = rewards.std(dim=-1, keepdim=True)
                advantages = (rewards - r_mean) / (r_std + 1e-8)
                
                # Debug Info
                expected_step_debug = 0 # Placeholder

        # =================================================================
        # PHASE 2: Training (Gradient Mode)
        # =================================================================
        self.student.train() # 全量微调
        
        with self.accelerator.autocast():
            # 2.1 Re-run Student (Grad enabled)
            outputs_train = self.student(input_ids, attention_mask=attention_mask, use_cache=False)
            
            # 2.2 Gate Policy PDF
            gate_logits = torch.cat(outputs_train.gate_list, dim=-1)
            gate_probs = torch.sigmoid(gate_logits)
            
            pdf_list = []
            remaining = torch.ones_like(gate_probs[..., 0])
            for i in range(gate_probs.shape[-1]):
                p = gate_probs[..., i]
                if i == gate_probs.shape[-1] - 1:
                    pdf_list.append(remaining)
                else:
                    pdf_list.append(p * remaining)
                    remaining = remaining * (1 - p)
            exit_pdf = torch.stack(pdf_list, dim=-1)
            
            # 统计步数
            step_vals = torch.arange(exit_pdf.shape[-1], device=exit_pdf.device).float()
            expected_step = (exit_pdf.detach() * step_vals).sum(dim=-1).mean()

            # 2.3 Policy Gradient Loss
            # 注意：全量微调时，这个 Loss 会回传到 Backbone。
            # 系数 pg_coef 设小一点(0.1)，让它主要影响 Gate，轻微调整 Backbone
            pg_loss_unreduced = -(exit_pdf * advantages.detach()).sum(dim=-1)
            pg_loss = (pg_loss_unreduced * active_mask).sum() / (active_mask.sum() + 1e-6)

            # 2.4 Entropy Loss
            entropy = -(exit_pdf * torch.log(exit_pdf + 1e-10)).sum(dim=-1)
            entropy_loss = -self.config["entropy_coef"] * (entropy * active_mask).sum() / (active_mask.sum() + 1e-6)

            # 2.5 Model Loss (Weighted NTP)
            # [关键] 这是维持语言能力和改进 Backbone 的主力 Loss
            raw_student = self.accelerator.unwrap_model(self.student)
            step_logprobs_train_list = []
            for hidden in outputs_train.hidden_states_list:
                logits = raw_student.lm_head(hidden)
                shift_logits = logits[..., :-1, :].contiguous()
                step_lp = -F.cross_entropy(shift_logits.transpose(1, 2), safe_shift_labels, reduction='none')
                step_logprobs_train_list.append(step_lp)
            step_logprobs_train = torch.stack(step_logprobs_train_list, dim=-1)

            # 我们希望模型在 Gate 选择概率高的 Step 上预测得更准
            model_loss_weighted = -(step_logprobs_train * exit_pdf.detach()).sum(dim=-1)
            model_loss = (model_loss_weighted * active_mask).sum() / (active_mask.sum() + 1e-6)

            # 2.6 Total Loss
            total_loss = self.config["pg_coef"] * pg_loss + \
                         self.config["model_coef"] * model_loss + \
                         entropy_loss

        return total_loss, pg_loss, model_loss, expected_step, rewards.mean()

# ==========================================
# Main Execution
# ==========================================
def main():
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
    accelerator = Accelerator(
        gradient_accumulation_steps=CONFIG["gradient_accumulation_steps"],
        mixed_precision="bf16",
        kwargs_handlers=[ddp_kwargs]
    )
    set_seed(CONFIG["seed"] + accelerator.process_index)

    if accelerator.is_main_process:
        print(f"Distributed training on {accelerator.num_processes} GPUs (A100 BF16 Mode).")

    # Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(CONFIG["model_path"])
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    
    # Load Config & Model
    config = OuroConfig.from_pretrained(CONFIG["model_path"])
    config.total_ut_steps = 4 
    
    student = OuroForCausalLM.from_pretrained(
        CONFIG["model_path"], config=config, 
        low_cpu_mem_usage=True, torch_dtype=torch.bfloat16
    )
    
    # [修改] 开启 Gradient Checkpointing 以节省显存 (全量微调必备)
    student.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    
    # [修改] 不再冻结参数，允许全量微调
    print("Full Parameter Finetuning Enabled (Backbone + Gate + Head)")
    
    # Load Teacher (Fully Frozen)
    teacher = OuroForCausalLM.from_pretrained(
        CONFIG["model_path"], config=config,
        low_cpu_mem_usage=True, torch_dtype=torch.bfloat16
    )
    teacher.eval()
    for p in teacher.parameters(): p.requires_grad = False
    
    # Data & Loader
    dataset = OmniMathDataset(CONFIG["data_path"], tokenizer, CONFIG["max_length"])
    dataloader = DataLoader(
        dataset, batch_size=CONFIG["batch_size"], shuffle=True,
        num_workers=4, pin_memory=True
    )
    
    # Initialize Trainer
    trainer = RLPTrainer(student, teacher, tokenizer, CONFIG, accelerator)
    
    # Scheduler
    num_training_steps = (len(dataloader) * CONFIG["num_epochs"]) // CONFIG["gradient_accumulation_steps"]
    num_warmup_steps = int(CONFIG["warmup_ratio"] * num_training_steps)
    
    lr_scheduler = get_cosine_schedule_with_warmup(
        trainer.optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps
    )

    # Accelerate Prepare
    trainer.student, trainer.optimizer, dataloader, lr_scheduler = accelerator.prepare(
        trainer.student, trainer.optimizer, dataloader, lr_scheduler
    )
    trainer.teacher.to(accelerator.device)
    
    if accelerator.is_main_process:
        print(f"Starting RLP Training (Full Finetune)... (Total Steps: {num_training_steps})")
    
    # Training Loop
    global_step = 0
    for epoch in range(CONFIG["num_epochs"]):
        for step, batch in enumerate(dataloader):
            with accelerator.accumulate(trainer.student):
                loss, pg, model_loss, avg_step, reward_val = trainer.compute_loss(batch)
                accelerator.backward(loss)
                
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(trainer.student.parameters(), 1.0)
                    trainer.optimizer.step()
                    lr_scheduler.step()
                    trainer.optimizer.zero_grad()
                    trainer.update_ema() 
                    global_step += 1
            
            if step % 10 == 0 and accelerator.is_main_process:
                print(f"Epoch {epoch} | Step {step} | Loss: {loss.item():.4f} (PG: {pg.item():.4f}, Model: {model_loss.item():.4f}) | Avg Step: {avg_step.item():.2f} | Rwd: {reward_val.item():.4f}")
        
        if accelerator.is_main_process:
            save_path = os.path.join(CONFIG["output_dir"], f"epoch_{epoch}")
            accelerator.unwrap_model(trainer.student).save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)
            print(f"Model saved to {save_path}")

if __name__ == "__main__":
    main()