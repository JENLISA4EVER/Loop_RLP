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
import hydra
from omegaconf import DictConfig, OmegaConf
import logging
import wandb
from torch.utils.data import Dataset, DataLoader
from utils.dataset_pt import OmniMathDataset
from accelerate import Accelerator
from tqdm import tqdm
logger = logging.getLogger(__name__)
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
#参考configs/ouro_ntp_omnimath_eval.yaml
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

def load_model(path, steps):
    if os.environ.get("RANK", "-1") == "0":
        print(f"\n[System] Loading model from {path} (Steps={steps})...")
    config = OuroConfig.from_pretrained(path)
    config.total_ut_steps = steps
    
    model = OuroForCausalLM.from_pretrained(
        path, 
        config=config,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        attn_implementation="flash_attention_2"
    )
    model.eval()
    return model

# ==========================================
# 4. 评测执行阶段
# ==========================================
def evaluate_samples(model, samples, mode="peak",CONFIG=None):
    correct = 0
    total = 0
    total_steps = 0
    
    model.model.total_ut_steps = CONFIG["total_ut_steps"]
    
    with torch.no_grad():
        for sample in samples:
            input_ids = sample["input_ids"]
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
@hydra.main(version_base=None, config_path="configs", config_name="ouro_ntp_omnimath_eval")
def main(config: DictConfig):
    # --- Config --- #
    CONFIG = OmegaConf.to_container(config, resolve=True)  # 转为普通字典
    accelerator = Accelerator()
    if os.environ.get("RANK", "-1") == "0":
        logger.info(json.dumps(CONFIG, indent=4))
    
    set_seed(42)
    tokenizer = AutoTokenizer.from_pretrained(CONFIG["base_model_path"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = (
            tokenizer.eos_token
        )  # FIXME:可能会存在最后的eos_token被mask，这里的eos_token是<|endoftext|>
        if CONFIG.get("apply_chat", False): # XXX:根据后续Ouro仓库的回复修改
            tokenizer.eos_token = "<|im_end|>" # THREEGOLD:如果按照chatml的template,eos_token应该设置为<|im_end|>
    # --- Step 0: Resume --- #
    resume_flag = False
    if CONFIG.get("resume", False) and os.path.exists(os.path.join(CONFIG["output_dir"], "results_final.json")):
        logger.info(f"Already evaluated, skipping...")
        resume_flag = True
    if resume_flag:
        return

    
    
    # --- Step 1: Base Model & Data ---
    base_model = load_model(CONFIG["base_model_path"], CONFIG["total_ut_steps"])
    rlp_model = load_model(CONFIG["rlp_model_path"], CONFIG["total_ut_steps"])
    dataset = OmniMathDataset(data_source=CONFIG["raw_data_path"], tokenizer=tokenizer, max_length=CONFIG["max_seq_len"], apply_chat=True, prompt_key="problem", response_key="solution",truncation="right")
    dataloader = DataLoader(dataset, batch_size=CONFIG["batch_size"], shuffle=False,num_workers=8)
    # --- Step 2: Evaluate Base Model ---
    if os.environ.get("RANK", "-1") == "0":
        print("\n" + "="*50)
        print("[Phase 2] Evaluating Base Model (Untrained)...")
        print("="*50)
    results_base = {}
    results_final = {}
    for cat in CONFIG["entropy_thresholds"].keys():
        results_base[cat] = {
            "steps_num": torch.tensor(0).to(accelerator.device),
            "peak_correct_num": torch.tensor(0).to(accelerator.device),
            "adapt_correct_num": torch.tensor(0).to(accelerator.device),
            "pred_num": torch.tensor(0).to(accelerator.device),
        }
        results_final[cat] = {
            "steps_num": 0,
            "peak_correct_num": 0,
            "adapt_correct_num": 0,
            "pred_num": 0,
        }
    if "1" in os.environ.get("DEBUG_MODE", "0") and os.environ.get("RANK", "-1") == "0":
        breakpoint()
    dataloader,base_model,rlp_model = accelerator.prepare(dataloader,base_model,rlp_model)
    if accelerator.is_main_process:
        bar = tqdm(total=len(dataloader), desc="Evaluating Base Model")
    with torch.no_grad():
        with accelerator.autocast():
            for _, batch in enumerate(dataloader):
                input_ids = batch["input_ids"]
                labels = batch["labels"]
                shift_labels = labels[:, 1:]
                loss_mask = batch["loss_mask"]
                shift_loss_mask = loss_mask[:, :-1]
                # --- Step 2.1: Evaluate Base Model ---
                base_model_output = base_model(input_ids, use_cache=True,exit_threshold=CONFIG["exit_threshold"])
                
                # --- Step 2.2: Evaluate Entropy ---
                hidden_state = base_model_output.hidden_states_list[CONFIG.get("entropy_hidden_state_index",0)] #XXX:和train不一样
                logits = accelerator.unwrap_model(base_model).lm_head(hidden_state) # (B, S, V)
                shift_logits = logits[..., :-1, :] # (B, S-1, V)
                entropy = calculate_entropy(shift_logits) # (B, S-1)
                entropy_mask = torch.zeros_like(entropy) # (B, S-1)
                for idx, (cat, threshold) in enumerate(sorted(CONFIG["entropy_thresholds"].items(), key=lambda x: x[1])):#利用sorted函数对阈值进行排序，从低到高
                    entropy_mask[entropy>threshold] = idx+1
                # 0--> none(<easy_threshold)
                # 1--> easy(easy_threshold,medium_threshold)
                # 2--> medium(medium_threshold,hard_threshold)
                # 3--> hard(hard_threshold)
                entropy_category_map = {
                    1 : "Easy",
                    2 : "Medium",
                    3 : "Hard"
                }
                evaluate_mask = shift_loss_mask * entropy_mask # (B, S-1)
                #FIXME:把base model只用来计算entropy，不用来计算peak和adapt
                # --- Step 2.3: Evaluate Base Model Peak Mode ---
                if CONFIG.get("evaluate_base_model", False):
                    hidden_state = base_model_output.hidden_states_list[CONFIG.get("peak_hidden_state_index",-1)] 
                    logits = accelerator.unwrap_model(base_model).lm_head(hidden_state) # (B, S, V)
                    shift_logits = logits[...,  :-1, :] # (B, S-1, V)
                    peak_preds = torch.argmax(shift_logits, dim=-1) # (B, S-1),避免重复命名导致显存浪费
                    for entropy_id, entropy_cat in entropy_category_map.items():
                        results_base[entropy_cat]["peak_correct_num"] += ((peak_preds == shift_labels) * (evaluate_mask == entropy_id)).sum()
                        results_base[entropy_cat]["pred_num"] += (evaluate_mask == entropy_id).sum()
                    
                    # --- Step 2.4: Evaluate Base Model Adaptive Mode ---
                    logits = base_model_output.logits[...,  :-1, :] # (B, S-1, V) 这里已经是exit_threshold的logits了
                    adaptive_preds = torch.argmax(logits, dim=-1) # (B, S-1)
                    shift_exit_steps = base_model_output.exit_steps[..., :-1]
                    for entropy_id, entropy_cat in entropy_category_map.items():
                        results_base[entropy_cat]["adapt_correct_num"] += ((adaptive_preds == shift_labels) * (evaluate_mask == entropy_id)).sum()
                        results_base[entropy_cat]["steps_num"] += shift_exit_steps[evaluate_mask == entropy_id].sum()
                        # results_base[entropy_cat]["pred_num"] += (evaluate_mask == entropy_id).sum().item() #计算一次就行
                
                # --- Step 3: Evaluate RLP Model ---
                rlp_model_output = rlp_model(input_ids, use_cache=True,exit_threshold=CONFIG["exit_threshold"])
                # --- Step 3.1: Evaluate RLP Model Peak Mode ---
                hidden_state = rlp_model_output.hidden_states_list[CONFIG.get("peak_hidden_state_index",-1)] 
                logits = accelerator.unwrap_model(rlp_model).lm_head(hidden_state) # (B, S, V)
                shift_logits = logits[...,  :-1, :] # (B, S-1, V)
                peak_preds = torch.argmax(shift_logits, dim=-1) # (B, S-1),避免重复命名导致显存浪费
                for entropy_id, entropy_cat in entropy_category_map.items():
                    results_final[entropy_cat]["peak_correct_num"] += ((peak_preds == shift_labels) * (evaluate_mask == entropy_id)).sum()
                    results_final[entropy_cat]["pred_num"] += (evaluate_mask == entropy_id).sum()
                
                # --- Step 3.2: Evaluate RLP Model Adaptive Mode ---
                logits = rlp_model_output.logits[...,  :-1, :] # (B, S-1, V) 这里已经是exit_threshold的logits了
                adaptive_preds = torch.argmax(logits, dim=-1) # (B, S-1)
                shift_exit_steps = rlp_model_output.exit_steps[..., :-1]+1
                for entropy_id, entropy_cat in entropy_category_map.items():
                    results_final[entropy_cat]["adapt_correct_num"] += ((adaptive_preds == shift_labels) * (evaluate_mask == entropy_id)).sum()
                    results_final[entropy_cat]["steps_num"] += shift_exit_steps[evaluate_mask == entropy_id].sum()
                if accelerator.is_main_process:
                    bar.update(1)

    if "3" in os.environ.get("DEBUG_MODE", "0") and os.environ.get("RANK", "-1") in os.environ.get("DEBUG_RANK", ""):
        breakpoint()
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        bar.close()
    # --- Step 4: Calculate Accuracy ---
    # --- Step 4.1: DDP Gather Results ---   
    results_base = accelerator.reduce(results_base,reduction="sum")
    results_final = accelerator.reduce(results_final,reduction="sum")
    for cat in entropy_category_map.values():
        results_base[cat]["peak_acc"] = results_base[cat]["peak_correct_num"].item() / (results_base[cat]["pred_num"].item()+1e-6)
        results_base[cat]["adapt_acc"] = results_base[cat]["adapt_correct_num"].item() / (results_base[cat]["pred_num"].item()+1e-6)
        results_base[cat]["avg_steps"] = results_base[cat]["steps_num"].item() / (results_base[cat]["pred_num"].item()+1e-6)
        results_final[cat]["peak_acc"] = results_final[cat]["peak_correct_num"].item() / (results_final[cat]["pred_num"].item()+1e-6)
        results_final[cat]["adapt_acc"] = results_final[cat]["adapt_correct_num"].item() / (results_final[cat]["pred_num"].item()+1e-6)
        results_final[cat]["avg_steps"] = results_final[cat]["steps_num"].item() / (results_final[cat]["pred_num"].item()+1e-6)
    # --- Step 5: Comparative Report (New Format) ---
    if accelerator.is_main_process:
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
        result_dir = CONFIG["output_dir"]
        os.makedirs(result_dir, exist_ok=True)
        with open(os.path.join(result_dir, "results_base.json"), "w") as f:
            pop_keys = ["peak_correct_num", "adapt_correct_num", "pred_num", "steps_num"]
            print(results_base.keys())
            for cat in ["Easy", "Medium", "Hard"]:
                for key in pop_keys:
                    results_base[cat].pop(key)
            json.dump(results_base, f,indent=4)
        with open(os.path.join(result_dir, "results_final.json"), "w") as f:
            pop_keys = ["peak_correct_num", "adapt_correct_num", "pred_num", "steps_num"]
            for cat in ["Easy", "Medium", "Hard"]:
                for key in pop_keys:
                    results_final[cat].pop(key)
            json.dump(results_final, f,indent=4)

if __name__ == "__main__":
    main()