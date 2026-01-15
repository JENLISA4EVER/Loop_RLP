import torch
import json
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import random
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer
from ouro.modeling_ouro import OuroForCausalLM, OuroConfig
import time
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



# ==========================================
# 1. 全局配置 (Configuration)
# ==========================================
CONFIG = {
    # 原始未训练的 Base 模型 (用于计算熵 + 作为基准)
    "base_model_path": "/share/home/sxjiang/model/Ouro-1.4B",
    
    # RLP 训练后的 Checkpoint
    "rlp_model_path": "/share/home/sxjiang/myproject/self-learn/ouro_rlp_checkpoints_bf16/epoch_2",
    
    # 原始数据路径 (jsonl 格式: {"problem":..., "solution":...})
    "raw_data_path": "/share/home/sxjiang/myproject/self-learn/datasets/Omni-MATH/rlp_test_set.jsonl",
    
    "device": "cuda",
    "max_seq_len": 2048,
    "total_ut_steps": 4,   # T_max
    "exit_threshold": 0.5, # 自适应退出阈值
    
    # RPT 论文定义的难度阈值 (Entropy)
    # Easy: 0.5 ~ 1.0 (论文中通常忽略 <0.5 的极简单 token)
    # Medium: 1.0 ~ 1.5
    # Hard: > 1.5
    "entropy_thresholds": {
        "Easy":   (0.5, 1.0),
        "Medium": (1.0, 1.5),
        "Hard":   (1.5, 10.0)#FIXME:根据rpt来说，存在大于10.0的token，需要考虑
    },
    
    "samples_per_category": 300, # 每个难度级别采集多少个样本进行精测
}

# ==========================================
# 2. 核心工具函数
# ==========================================

def calculate_entropy(logits):
    """计算 Logits 的熵: H(p) = - sum(p * log(p))"""
    probs = F.softmax(logits, dim=-1)
    log_probs = F.log_softmax(logits, dim=-1)
    entropy = -torch.sum(probs * log_probs, dim=-1)
    return entropy

def calculate_exit_step(gate_list, threshold=0.9):
    """根据 Gate 输出计算退出步数 (Strictly aligned with Ouro)"""
    gate_logits = torch.cat(gate_list, dim=-1)
    gate_probs = torch.sigmoid(gate_logits)
    
    pdf_list = []
    remaining_prob = torch.ones_like(gate_probs[..., 0])
    T_max = gate_probs.shape[-1]
    
    for i in range(T_max):
        lambda_i = gate_probs[..., i]
        if i == T_max - 1:
            p_i = remaining_prob
        else:
            p_i = lambda_i * remaining_prob
            remaining_prob = remaining_prob * (1.0 - lambda_i)
        pdf_list.append(p_i)
    
    exit_pdf = torch.stack(pdf_list, dim=-1) # (B, S, T)
    exit_cdf = torch.cumsum(exit_pdf, dim=-1)
    
    threshold_mask = exit_cdf >= threshold
    exit_indices = torch.argmax(threshold_mask.float(), dim=-1)
    
    never_exceeded = ~threshold_mask.any(dim=-1)
    exit_indices[never_exceeded] = T_max - 1
    
    return exit_indices

def load_model(path, steps, device):
    print(f"\n[System] Loading model from {path} (Steps={steps})...")
    config = OuroConfig.from_pretrained(path)
    config.total_ut_steps = steps
    
    model = OuroForCausalLM.from_pretrained(
        path, 
        config=config,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True
    ).to(device)
    model.eval()
    return model

# ==========================================
# 3. 数据准备阶段 (Data Preparation)
# ==========================================

def prepare_rpt_dataset(base_model, tokenizer):
    """
    [Debug Version] 带详细日志的数据准备函数
    """
    print("\n" + "="*50)
    print("[Phase 1] Scanning dataset for Entropy-based Filtering (DEBUG MODE)...")
    print("="*50)
    
    # 临时将 Base Model 设为 T=1 用于快速计算熵
    base_model.model.total_ut_steps = 1
    
    categorized_samples = {k: [] for k in CONFIG["entropy_thresholds"].keys()}
    
    # 检查文件路径
    if not os.path.exists(CONFIG["raw_data_path"]):
        print(f"❌ Error: File not found at {CONFIG['raw_data_path']}")
        return categorized_samples

    with open(CONFIG["raw_data_path"], "r") as f:
        lines = f.readlines()
        
    print(f"Loaded {len(lines)} lines from raw data.")
    random.shuffle(lines)
    
    pbar = tqdm(total=CONFIG["samples_per_category"] * 3)
    
    # 计数器
    total_processed = 0
    errors_count = 0
    skipped_len = 0
    skipped_threshold = 0
    
    with torch.no_grad():
        for i, line in enumerate(lines):
            # 检查是否所有桶都满了
            if all(len(v) >= CONFIG["samples_per_category"] for v in categorized_samples.values()):
                print("All buckets filled. Stopping scan.")
                break
            
            # 限制只扫描前 100 条用于 Debug，防止刷屏
            if total_processed > 100 and sum(len(v) for v in categorized_samples.values()) == 0:
                print("⚠️ Scanned 100 items but found 0 valid samples. Stopping to prevent infinite loop.")
                break
                
            try:
                item = json.loads(line)
                
                # [Check 1] 检查 Key 是否存在
                if 'problem' not in item or 'solution' not in item:
                    print(f"⚠️ Line {i}: Missing 'problem' or 'solution' keys. Keys found: {list(item.keys())}")
                    errors_count += 1
                    continue

                full_text = f"Problem: {item['problem']}\nSolution: {item['solution']}"
                input_ids = tokenizer.encode(full_text, return_tensors="pt").to(CONFIG["device"])
                
                # [Check 2] 长度过滤
                if input_ids.shape[1] > CONFIG["max_seq_len"]:
                    input_ids = input_ids[:, :CONFIG["max_seq_len"]]
                if input_ids.shape[1] < 50: 
                    skipped_len += 1
                    continue

                # Forward pass (T=1)
                outputs = base_model(input_ids, exit_at_step=0, use_cache=False)
                logits = outputs.logits 
                
                # 计算 Entropy
                shift_logits = logits[0, :-1, :]
                entropies = calculate_entropy(shift_logits)
                
                # 随机采样几个点
                valid_indices = torch.arange(entropies.shape[0])
                # 确保不越界
                if len(valid_indices) < 3: 
                    continue
                    
                # random_indices = valid_indices[torch.randperm(len(valid_indices))[:5]] # 多采几个点试试
                
                found_match = False
                for idx in valid_indices:
                    h = entropies[idx].item()
                    
                    # [Debug Print] 打印前几个熵值，看看范围是多少
                    if total_processed < 5:
                        print(f"Sample {i} | Token Idx {idx} | Entropy: {h:.4f}")

                    category = None
                    for cat, (low, high) in CONFIG["entropy_thresholds"].items():
                        if low <= h < high:
                            category = cat
                            break
                    
                    if category and len(categorized_samples[category]) < CONFIG["samples_per_category"]:
                        target_id = input_ids[0, idx+1].item()
                        context = input_ids[:, :idx+1]
                        
                        categorized_samples[category].append({
                            "input_ids": context,
                            "target_id": target_id,
                            "entropy": h
                        })
                        pbar.update(1)
                        found_match = True
                
                if not found_match:
                    skipped_threshold += 1
                
                total_processed += 1

            except Exception as e:
                print(f"❌ Error processing line {i}: {str(e)}")
                errors_count += 1
                import traceback
                traceback.print_exc()
                continue
                
    pbar.close()
    
    print("\n" + "="*30)
    print(f"Scan Summary:")
    print(f"  Processed: {total_processed}")
    print(f"  Errors: {errors_count} (JSON/Key errors)")
    print(f"  Skipped (Too Short): {skipped_len}")
    print(f"  Skipped (No Threshold Match): {skipped_threshold} (Entropy too low/high?)")
    print("="*30)
    
    # 统计信息
    print("\n[Data Stats]")
    for cat, samples in categorized_samples.items():
        avg_ent = np.mean([s['entropy'] for s in samples]) if samples else 0.0
        print(f"  - {cat:<6}: {len(samples)} samples (Avg Entropy: {avg_ent:.2f})")
    
    return categorized_samples

# ==========================================
# 4. 评测执行阶段
# ==========================================
def evaluate_samples(model, samples, mode="peak"):
    correct = 0
    total = 0
    total_steps = 0
    
    model.model.total_ut_steps = CONFIG["total_ut_steps"]
    
    with torch.no_grad():
        for sample in samples:
            input_ids = sample["input_ids"].to(CONFIG["device"])
            target_id = sample["target_id"]
            
            # Forward with use_cache=False for safety
            base_out, hidden_list, gate_list, add_noise_list = model.model(input_ids, use_cache=False)
            
            if mode == "peak":
                step_idx = CONFIG["total_ut_steps"] - 1
            elif mode == "adaptive":
                last_token_gates = [g[:, -1:, :] for g in gate_list]
                exit_indices = calculate_exit_step(last_token_gates, threshold=CONFIG["exit_threshold"])
                step_idx = exit_indices.item()
            
            selected_hidden = hidden_list[step_idx][:, -1:, :]
            logits = model.lm_head(selected_hidden)
            pred = torch.argmax(logits[0, 0, :]).item()
            
            if pred == target_id:
                correct += 1
            total += 1
            total_steps += (step_idx + 1)
            
    return correct / total, total_steps / total

# ==========================================
# 5. 主流程 (Main) - Modified Table
# ==========================================
def main():
    set_seed(42)
    tokenizer = AutoTokenizer.from_pretrained(CONFIG["base_model_path"])
    
    # --- Step 1: Base Model & Data ---
    base_model = load_model(CONFIG["base_model_path"], CONFIG["total_ut_steps"], CONFIG["device"])
    categorized_data = prepare_rpt_dataset(base_model, tokenizer)
    
    # --- Step 2: Evaluate Base Model ---
    print("\n" + "="*50)
    print("[Phase 2] Evaluating Base Model (Untrained)...")
    print("="*50)
    
    results_base = {}
    for cat, samples in categorized_data.items():
        if not samples: continue
        # [修改] 同时测试 Base 的 Peak 和 Adaptive
        acc_peak, _ = evaluate_samples(base_model, samples, mode="peak")
        acc_adapt, steps_adapt = evaluate_samples(base_model, samples, mode="adaptive")
        
        results_base[cat] = {
            "peak_acc": acc_peak,
            "adapt_acc": acc_adapt,
            "avg_steps": steps_adapt
        }
        print(f"  Category {cat:<6}: Peak={acc_peak:.2%} | Adapt={acc_adapt:.2%} (Step={steps_adapt:.2f})")
        
    del base_model
    torch.cuda.empty_cache()
    
    # --- Step 3: Evaluate RLP Model ---
    print("\n" + "="*50)
    print("[Phase 3] Evaluating RLP Model (Trained)...")
    print("="*50)
    
    results_final = {}
    if os.path.exists(CONFIG["rlp_model_path"]):
        rlp_model = load_model(CONFIG["rlp_model_path"], CONFIG["total_ut_steps"], CONFIG["device"])
        for cat, samples in categorized_data.items():
            if not samples: continue
            acc_peak, _ = evaluate_samples(rlp_model, samples, mode="peak")
            acc_adapt, steps_adapt = evaluate_samples(rlp_model, samples, mode="adaptive")
            results_final[cat] = {
                "peak_acc": acc_peak,
                "adapt_acc": acc_adapt,
                "avg_steps": steps_adapt
            }
            print(f"  Category {cat:<6}: Peak={acc_peak:.2%} | Adapt={acc_adapt:.2%} (Step={steps_adapt:.2f})")
    else:
        print("RLP Model not found, skipping Phase 3.")

    # --- Step 4: Comparative Report (New Format) ---
    print("\n" + "#"*100)
    print(f"{'FINAL REPORT (Full Comparison)':^100}")
    print("#"*100)
    
    # Header: 包含 Base(Adapt) 和 Step(B)
    header = f"{'Diff':<8} | {'Base(Peak)':<10} | {'Base(Adp)':<10} | {'RLP(Peak)':<10} | {'RLP(Adp)':<10} | {'Gain':<8} | {'Step(B)':<8} | {'Step(R)':<8}"
    print(header)
    print("-" * 100)
    
    for cat in ["Easy", "Medium", "Hard"]:
        if cat not in results_base or cat not in results_final: continue
        
        # Base Stats
        b_peak = results_base[cat]["peak_acc"]
        b_adapt = results_base[cat]["adapt_acc"]
        b_step = results_base[cat]["avg_steps"]
        
        # RLP Stats
        r_peak = results_final[cat]["peak_acc"]
        r_adapt = results_final[cat]["adapt_acc"]
        r_step = results_final[cat]["avg_steps"]
        
        gain = r_adapt - b_adapt
        
        row = f"{cat:<8} | {b_peak:.2%}     | {b_adapt:.2%}     | {r_peak:.2%}     | {r_adapt:.2%}     | {gain:+.2%}   | {b_step:.2f}     | {r_step:.2f}"
        print(row)
        
    print("#"*100)
    print("Metrics Key:")
    print(" - Base(Adp): Untrained model using its own gate (Natural behavior)")
    print(" - RLP(Adp) : Trained model using optimized gate (Learned behavior)")
    print(" - Step(B/R): Average steps taken by Base / RLP model")
    print("If Step(R) > Step(B) in Hard, the model learned to 'think longer' for hard problems.")

if __name__ == "__main__":
    main()