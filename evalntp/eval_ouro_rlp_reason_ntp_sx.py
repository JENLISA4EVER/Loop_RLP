import torch
import json
import os
import random
import re
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer
from modeling_ouro import OuroForCausalLM, OuroConfig
import hydra
from omegaconf import DictConfig, OmegaConf
import logging
from torch.utils.data import DataLoader
from utils.dataset_pt import OmniMathGenerationDataset, DataCollatorWithPadding
from accelerate import Accelerator
from ouro_cache_fix import UniversalTransformerCache
logger = logging.getLogger(__name__)

# ==========================================
# 1. 核心工具函数
# ==========================================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def calculate_entropy(logits):
    """计算 Logits 的熵: H(p) = - sum(p * log(p))"""
    probs = F.softmax(logits, dim=-1)
    log_probs = F.log_softmax(logits, dim=-1)
    entropy = -torch.sum(probs * log_probs, dim=-1)
    return entropy

def extract_boxed_content(text):
    """从生成的文本中提取最后一个 \boxed{} 中的内容 (对标 RPT)"""
    matches = re.findall(r"\\boxed\{(.*?)\}", text)
    if matches:
        return matches[-1].strip()
    return ""

def load_model(path, steps):
    """严格按照配置加载 Ouro 模型"""
    
    cache = UniversalTransformerCache()
    config = OuroConfig.from_pretrained(path)
    config.total_ut_steps = steps
    # 从配置中同步 early_exit_threshold 等参数
    model = OuroForCausalLM.from_pretrained(
        path, 
        config=config,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        attn_implementation="flash_attention_2"
    )
    model.eval()
    return model

# ==========================================
# 2. 推理 Mode Prompt 设计 (对标 RPT v0)
# ==========================================
# 模板说明：融合了 RPT 的步进指令与 Ouro 的隐式推理特性，要求模型直接输出 Boxed 答案
REASONING_PROMPT_TEMPLATE = (
    "Predict the next token of the context and wrap it in \\boxed{{}}.\n\n"
    "Context: 1 + 1 =\nAnswer: \\boxed{{ 2}}\n\n"
    "Context: The capital of France is\nAnswer: \\boxed{{ Paris}}\n\n"
    "Context: {context}\nAnswer: \\boxed{{"
)

# ==========================================
# 3. 评测主逻辑
# ==========================================

@hydra.main(version_base=None, config_path="configs", config_name="ouro_ntp_omnimath_eval")
def main(config: DictConfig):
    CONFIG = OmegaConf.to_container(config, resolve=True)
    accelerator = Accelerator()
    set_seed(CONFIG.get("seed", 42))
    
    if accelerator.is_main_process:
        logger.info(f"Full Config: {json.dumps(CONFIG, indent=4)}")

    tokenizer = AutoTokenizer.from_pretrained(CONFIG["base_model_path"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        if CONFIG.get("apply_chat", False):
            tokenizer.eos_token = "<|im_end|>"

    # --- Step 1: 加载模型与数据 ---
    # 严格按照总步数配置加载
    total_steps = CONFIG.get("total_ut_steps", 4)
    # base_model = load_model(CONFIG["base_model_path"], total_steps)
    rlp_model = load_model(CONFIG["rlp_model_path"], total_steps)
    
    EACH_CATEGORY_MAX_NUM = CONFIG.get("each_category_max_num", 20)
    
    # --- Step 1.1 限制在每个类别最多300个样本 ---
    collator = DataCollatorWithPadding(tokenizer=tokenizer, max_length=CONFIG["max_seq_len"], truncation="right",max_length_method="group_max",padding_side="left")
    easy_dataset = OmniMathGenerationDataset(
        data_source=CONFIG["easy_data_path"], 
        tokenizer=tokenizer, max_length=CONFIG["max_seq_len"], 
        apply_chat=True, 
        prompt_key="problem", 
        response_key="response",
        max_num=EACH_CATEGORY_MAX_NUM,
        truncation="no"
    )
    easy_dataloader = DataLoader(easy_dataset, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=4, collate_fn=collator)
    medium_dataset = OmniMathGenerationDataset(
        data_source=CONFIG["medium_data_path"], 
        tokenizer=tokenizer, max_length=CONFIG["max_seq_len"], 
        apply_chat=True, 
        prompt_key="problem", 
        response_key="response",
        max_num=EACH_CATEGORY_MAX_NUM,
        truncation="no"
    )
    medium_dataloader = DataLoader(medium_dataset, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=4, collate_fn=collator)
    hard_dataset = OmniMathGenerationDataset(
        data_source=CONFIG["hard_data_path"], 
        tokenizer=tokenizer, max_length=CONFIG["max_seq_len"], 
        apply_chat=True, 
        prompt_key="problem", 
        response_key="response",
        max_num=EACH_CATEGORY_MAX_NUM,
        truncation="no"
    )
    hard_dataloader = DataLoader(hard_dataset, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=4, collate_fn=collator)
    # --- Step 2: 初始化指标统计器 ---
    cats = ["Easy", "Medium", "Hard"]
    def init_metrics():
        return {cat: {
            "reason_peak_correct": torch.tensor(0).to(accelerator.device),
            "reason_adapt_correct": torch.tensor(0).to(accelerator.device),
            "reason_total": torch.tensor(0.0).to(accelerator.device),
        } for cat in cats}

    base_results = init_metrics()
    rlp_results = init_metrics()

    # 准备 Accelerator
    easy_dataloader, medium_dataloader, hard_dataloader, rlp_model = accelerator.prepare(easy_dataloader, medium_dataloader, hard_dataloader, rlp_model)
    # --- Step 3: 循环测评 ---
    cat2dataloader = {"Easy": easy_dataloader, "Medium": medium_dataloader, "Hard": hard_dataloader}
    # --- Step 3.1: Easy Evaluation ---
    for cat in cats:
        if accelerator.is_main_process:
            bar = tqdm(total=len(cat2dataloader[cat]), desc=f"{cat} Evaluation")
        for batch in cat2dataloader[cat]:
            attention_mask = batch["attention_mask"]
            input_ids = batch["input_ids"]
            labels = batch["labels"]
            with accelerator.autocast():
                def run_reasoning(model, tokenizer, input_ids, labels, attention_mask, mode="peak", config=None):
                    with torch.no_grad():
                        # 【注意】确保 config 被正确传入
                        threshold = 1.0 if mode == "peak" else config["exit_threshold"]
                        gen_out = model.generate(
                            input_ids,
                            max_new_tokens=16, 
                            exit_threshold=threshold,
                            attention_mask=attention_mask,
                            use_cache=True,
                            do_sample=False,
                        )
                    batch_target_str = tokenizer.batch_decode(labels, skip_special_tokens=True)
                    batch_gen_text = tokenizer.batch_decode(gen_out[:, input_ids.shape[1]:], skip_special_tokens=False)
                    predictions = [gen_text.split("}")[0].strip() for gen_text in batch_gen_text]
                    is_correct = [1 if prediction == target_str.strip() else 0 for prediction, target_str in zip(predictions, batch_target_str)]
                    return torch.tensor(sum(is_correct))


            # 【修复】传入 prefix_ids 而非 input_ids，并传入 config=CONFIG
            rlp_results[cat]["reason_peak_correct"] += run_reasoning(
                model=rlp_model, 
                tokenizer=tokenizer, 
                input_ids=input_ids, 
                labels=labels,
                attention_mask=attention_mask,
                mode="peak",
                config=CONFIG
            )
            rlp_results[cat]["reason_adapt_correct"] += run_reasoning(
                model=rlp_model, 
                tokenizer=tokenizer, 
                input_ids=input_ids, 
                labels=labels,
                attention_mask=attention_mask,
                mode="adapt",
                config=CONFIG
            )
            rlp_results[cat]["reason_total"] += len(batch["input_ids"])
            if accelerator.is_main_process:
                bar.update(1)
        if accelerator.is_main_process:
            bar.close()

    # --- Step 4: 归约与报告 ---
    accelerator.wait_for_everyone()
    # --- Step 4.1: Evaluation ---
    rlp_results = accelerator.reduce(rlp_results, reduction="sum")
    base_results = accelerator.reduce(base_results, reduction="sum")
    def finalize_metrics(m_dict):
        final = {}
        for cat in cats:
            reason_total = m_dict[cat]["reason_total"].item() + 1e-9
            # Reasoning 指标 (由于是采样，此处假设采样总数即为进入该逻辑的次数)
            reason_peak_acc = m_dict[cat]["reason_peak_correct"].item() / reason_total
            reason_adapt_acc = m_dict[cat]["reason_adapt_correct"].item() / reason_total
            # 在实际科研中，建议额外维护一个采样计数器
            final[cat] = {
                "reason_peak": f"{reason_peak_acc:.2%}",   # 现在可以显示百分比了
                "reason_adapt": f"{reason_adapt_acc:.2%}", # 现在可以显示百分比了
                "reason_count": f"{int(reason_total)}",    # 记录总采样数方便 Check
            }
        return final
    if accelerator.is_main_process:
        report_base = finalize_metrics(base_results)
        report_rlp = finalize_metrics(rlp_results)
        
        print("\n" + "="*110)
        print(f"{'CATEGORY':<10} | {'MODE':<10} | {'BASE(Peak)':<12} | {'BASE(Adp)':<12} | {'RLP(Peak)':<12} | {'RLP(Adp)':<12}")
        print("-" * 110)
        for cat in cats:
            print(f"{'':<10} | {'Reason Acc':<10} | {report_base[cat]['reason_peak']:<12} | {report_base[cat]['reason_adapt']:<12} | {report_rlp[cat]['reason_peak']:<12} | {report_rlp[cat]['reason_adapt']:<12}")
            print(f"{'':<10} | {'Reason Count':<10} | {report_base[cat]['reason_count']:<12} | {'-':<12} | {'-':<12} | {'-':<12}")
            print("-" * 110)

        # 保存结果
        with open(os.path.join(CONFIG["output_dir"], "eval_summary.json"), "w") as f:
            json.dump({"base": report_base, "rlp": report_rlp}, f, indent=4)

if __name__ == "__main__":
    main()