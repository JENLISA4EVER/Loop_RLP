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
from utils.dataset_pt import OmniMathDataset
from accelerate import Accelerator

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
    base_model = load_model(CONFIG["base_model_path"], total_steps)
    rlp_model = load_model(CONFIG["rlp_model_path"], total_steps)
    
    dataset = OmniMathDataset(
        data_source=CONFIG["raw_data_path"], 
        tokenizer=tokenizer, max_length=CONFIG["max_seq_len"], 
        apply_chat=True, 
        prompt_key="problem", 
        response_key="solution",
        truncation="right"
    )
    
    dataloader = DataLoader(dataset, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=4)

    # --- Step 2: 初始化指标统计器 ---
    cats = ["Easy", "Medium", "Hard"]
    def init_metrics():
        return {cat: {
            "ntp_peak_correct": torch.tensor(0).to(accelerator.device),
            "ntp_adapt_correct": torch.tensor(0).to(accelerator.device),
            "reason_peak_correct": torch.tensor(0).to(accelerator.device),
            "reason_adapt_correct": torch.tensor(0).to(accelerator.device),
            "ntp_steps": torch.tensor(0.0).to(accelerator.device),
            "reason_total": torch.tensor(0.0).to(accelerator.device),
            "total_tokens": torch.tensor(0).to(accelerator.device)
        } for cat in cats}

    base_results = init_metrics()
    rlp_results = init_metrics()

    # 准备 Accelerator
    dataloader, base_model, rlp_model = accelerator.prepare(dataloader, base_model, rlp_model)

    if accelerator.is_main_process:
        bar = tqdm(total=len(dataloader), desc="Full Evaluation")

    # --- Step 3: 循环测评 ---
    for batch in dataloader:
        input_ids = batch["input_ids"]
        labels = batch["labels"]
        shift_labels = labels[:, 1:]
        shift_loss_mask = batch["loss_mask"][:, :-1]
        
        with torch.no_grad():
            with accelerator.autocast():
                # 3.1 使用 Base 模型在 Step 0 计算熵分类 (对标 RPT 逻辑)
                # 使用 exit_at_step=0 确保是直觉熵
                base_out_s0 = base_model(input_ids, exit_at_step=0)
                entropy = calculate_entropy(base_out_s0.logits[:, :-1, :])
                
                entropy_mask = torch.zeros_like(entropy, dtype=torch.long)
                # 分类阈值: 0.5, 1.0, 1.5 [cite: 611]
                thresholds = sorted(CONFIG["entropy_thresholds"].items(), key=lambda x: x[1])
                for idx, (cat, thresh) in enumerate(thresholds):
                    entropy_mask[entropy > thresh] = idx + 1
                
                cat_map = {i+1: name for i, name in enumerate([t[0] for t in thresholds])}
                eval_mask = shift_loss_mask * entropy_mask

                # 3.2 批量测评函数 (Standard NTP Mode)
                def eval_ntp_mode(model, results_dict):
                    # Peak Mode (T=Max)
                    out_peak = model(input_ids, exit_at_step=total_steps-1)
                    preds_peak = torch.argmax(out_peak.logits[:, :-1, :], dim=-1)
                    
                    # Adaptive Mode
                    out_adapt = model(input_ids, exit_threshold=CONFIG["exit_threshold"])
                    preds_adapt = torch.argmax(out_adapt.logits[:, :-1, :], dim=-1)
                    exit_steps = out_adapt.exit_steps[:, :-1] + 1
                    
                    for eid, ename in cat_map.items():
                        m = (eval_mask == eid)
                        results_dict[ename]["ntp_peak_correct"] += (preds_peak[m] == shift_labels[m]).sum()
                        results_dict[ename]["ntp_adapt_correct"] += (preds_adapt[m] == shift_labels[m]).sum()
                        results_dict[ename]["ntp_steps"] += exit_steps[m].sum().float()
                        results_dict[ename]["total_tokens"] += m.sum()

                eval_ntp_mode(base_model, base_results)
                eval_ntp_mode(rlp_model, rlp_results)

                # 3.3 采样评测 (Next-Token Reasoning Mode)
                # 注意：Reasoning 模式涉及 generate，速度极慢，通常采样评测
                def run_reasoning(model, tokenizer, input_ids, target_str, mode="peak", config=None):
                    # 【修复】确保传入的是 1D 序列，并转为 list 或 cpu tensor
                    if torch.is_tensor(input_ids):
                        input_ids = input_ids.tolist()
                        
                    # 注意：这里 context 不要带聊天模板，直接用原始文本
                    context_text = tokenizer.decode(input_ids, skip_special_tokens=True).strip()
                    
                    # 构造 Prompt
                    full_prompt = REASONING_PROMPT_TEMPLATE.format(context=context_text)
                    inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)
                    
                    with torch.no_grad():
                        # 【注意】确保 config 被正确传入
                        threshold = 1.0 if mode == "peak" else config["exit_threshold"]
                        gen_out = accelerator.unwrap_model(model).generate(
                            **inputs,
                            max_new_tokens=16, 
                            exit_threshold=threshold,
                            do_sample=False,
                            eos_token_id=tokenizer.encode("}", add_special_tokens=False)[-1],
                            pad_token_id=tokenizer.pad_token_id
                        )
                    
                    gen_text = tokenizer.decode(gen_out[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True)
                    prediction = gen_text.split("}")[0].strip()
                    is_correct = 1 if prediction == target_str.strip() else 0
                    
                    if accelerator.is_main_process and getattr(main, "debug_count", 0) < 5:
                        print(f"\n[DEBUG] Target: '{target_str.strip()}' | Pred: '{prediction}'")
                        print(f"[DEBUG] Full Output: '{gen_text}'")
                        main.debug_count = getattr(main, "debug_count", 0) + 1
                        
                    return is_correct
                
                if CONFIG.get("eval_reasoning", True):
                    for b_idx in range(input_ids.shape[0]):
                        for eid, ename in cat_map.items():
                            valid_indices = (eval_mask[b_idx] == eid).nonzero(as_tuple=True)[0]
                            if valid_indices.numel() == 0: continue
                            
                            t_pos = valid_indices[0].item()
                            base_results[ename]["reason_total"] += 1
                            rlp_results[ename]["reason_total"] += 1

                            # 提取当前样本的 1D 前缀
                            prefix_ids = input_ids[b_idx, :t_pos+1]
                            target_str = tokenizer.decode([shift_labels[b_idx, t_pos]]).strip()

                            # 【修复】传入 prefix_ids 而非 input_ids，并传入 config=CONFIG
                            base_results[ename]["reason_peak_correct"] += run_reasoning(
                                model=base_model, 
                                tokenizer=tokenizer, 
                                input_ids=prefix_ids, # 改为当前样本前缀
                                target_str=target_str,
                                mode="peak",
                                config=CONFIG # 必须传入配置
                            )
                            base_results[ename]["reason_adapt_correct"] += run_reasoning(
                                model=base_model, 
                                tokenizer=tokenizer, 
                                input_ids=prefix_ids, 
                                target_str=target_str,
                                mode="adapt",
                                config=CONFIG
                            )
                            rlp_results[ename]["reason_peak_correct"] += run_reasoning(
                                model=rlp_model, 
                                tokenizer=tokenizer, 
                                input_ids=prefix_ids, 
                                target_str=target_str,
                                mode="peak",
                                config=CONFIG
                            )
                            rlp_results[ename]["reason_adapt_correct"] += run_reasoning(
                                model=rlp_model, 
                                tokenizer=tokenizer, 
                                input_ids=prefix_ids, 
                                target_str=target_str,
                                mode="adapt",
                                config=CONFIG
                            )

        if accelerator.is_main_process:
            bar.update(1)

    # --- Step 4: 归约与报告 ---
    accelerator.wait_for_everyone()
    
    def finalize_metrics(m_dict):
        final = {}
        for cat in cats:
            ntp_total = m_dict[cat]["total_tokens"].item() + 1e-9
            reason_total = m_dict[cat]["reason_total"].item() + 1e-9
            # NTP 指标
            ntp_peak_acc = m_dict[cat]["ntp_peak_correct"].item() / ntp_total
            ntp_adapt_acc = m_dict[cat]["ntp_adapt_correct"].item() / ntp_total
            avg_steps = m_dict[cat]["ntp_steps"].item() / ntp_total
            # Reasoning 指标 (由于是采样，此处假设采样总数即为进入该逻辑的次数)
            reason_peak_acc = m_dict[cat]["reason_peak_correct"].item() / reason_total
            reason_adapt_acc = m_dict[cat]["reason_adapt_correct"].item() / reason_total
            # 在实际科研中，建议额外维护一个采样计数器
            final[cat] = {
                "ntp_peak": f"{ntp_peak_acc:.2%}",
                "ntp_adapt": f"{ntp_adapt_acc:.2%}",
                "ntp_steps": f"{avg_steps:.2f}",
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
            print(f"{cat:<10} | {'NTP Acc':<10} | {report_base[cat]['ntp_peak']:<12} | {report_base[cat]['ntp_adapt']:<12} | {report_rlp[cat]['ntp_peak']:<12} | {report_rlp[cat]['ntp_adapt']:<12}")
            print(f"{'':<10} | {'Avg Steps':<10} | {'-':<12} | {report_base[cat]['ntp_steps']:<12} | {'-':<12} | {report_rlp[cat]['ntp_steps']:<12}")
            print(f"{'':<10} | {'Reason Acc':<10} | {report_base[cat]['reason_peak']:<12} | {report_base[cat]['reason_adapt']:<12} | {report_rlp[cat]['reason_peak']:<12} | {report_rlp[cat]['reason_adapt']:<12}")
            print(f"{'':<10} | {'Reason Count':<10} | {report_base[cat]['reason_count']:<12} | {'-':<12} | {'-':<12} | {'-':<12}")
            print("-" * 110)

        # 保存结果
        with open(os.path.join(CONFIG["output_dir"], "eval_summary.json"), "w") as f:
            json.dump({"base": report_base, "rlp": report_rlp}, f, indent=4)

if __name__ == "__main__":
    main()