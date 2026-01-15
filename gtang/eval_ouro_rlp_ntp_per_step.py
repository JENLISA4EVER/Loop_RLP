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

    exit_pdf = torch.stack(pdf_list, dim=-1)  # (B, S, T)
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
# 5. 主流程 (Main)
# ==========================================
@hydra.main(version_base=None, config_path="configs", config_name="ouro_ntp_omnimath_eval")
def main(config: DictConfig):
    CONFIG = OmegaConf.to_container(config, resolve=True)
    accelerator = Accelerator()
    if accelerator.is_main_process:
        logger.info(json.dumps(CONFIG, indent=4))

    set_seed(42)

    tokenizer = AutoTokenizer.from_pretrained(CONFIG["base_model_path"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        if CONFIG.get("apply_chat", False):
            tokenizer.eos_token = "<|im_end|>"

    # --- Resume ---
    if CONFIG.get("resume", False) and os.path.exists(os.path.join(CONFIG["output_dir"], "results_final.json")):
        logger.info("Already evaluated, skipping...")
        return

    # --- Load models & data ---
    total_ut_steps = int(CONFIG["total_ut_steps"])
    base_model = load_model(CONFIG["base_model_path"], total_ut_steps)
    rlp_model = load_model(CONFIG["rlp_model_path"], total_ut_steps)

    dataset = OmniMathDataset(
        data_source=CONFIG["raw_data_path"],
        tokenizer=tokenizer,
        max_length=CONFIG["max_seq_len"],
        apply_chat=True,
        prompt_key="problem",
        response_key="solution",
        truncation="right"
    )
    dataloader = DataLoader(dataset, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=8)

    dataloader, base_model, rlp_model = accelerator.prepare(dataloader, base_model, rlp_model)

    if accelerator.is_main_process:
        print("\n" + "=" * 80)
        print("[Eval] NTP Evaluation with Per-Step Reporting")
        print("=" * 80)

    # --- Prepare result containers ---
    entropy_category_map = {1: "Easy", 2: "Medium", 3: "Hard"}
    cats = ["Easy", "Medium", "Hard"]

    # 主结果（你原来就有）
    results_base = {}
    results_final = {}

    # 新增：per-step 统计（base / rlp 各一份）
    # 结构：results_per_step_base[cat]["step_correct_num"][t], pred_num 与原一致
    results_per_step_base = {}
    results_per_step_final = {}

    for cat in cats:
        results_base[cat] = {
            "steps_num": torch.tensor(0, device=accelerator.device, dtype=torch.long),
            "peak_correct_num": torch.tensor(0, device=accelerator.device, dtype=torch.long),
            "adapt_correct_num": torch.tensor(0, device=accelerator.device, dtype=torch.long),
            "pred_num": torch.tensor(0, device=accelerator.device, dtype=torch.long),
        }
        results_final[cat] = {
            "steps_num": torch.tensor(0, device=accelerator.device, dtype=torch.long),
            "peak_correct_num": torch.tensor(0, device=accelerator.device, dtype=torch.long),
            "adapt_correct_num": torch.tensor(0, device=accelerator.device, dtype=torch.long),
            "pred_num": torch.tensor(0, device=accelerator.device, dtype=torch.long),
        }

        # per-step correct 统计：长度 = total_ut_steps
        results_per_step_base[cat] = {
            "pred_num": torch.tensor(0, device=accelerator.device, dtype=torch.long),
            "step_correct_num": torch.zeros(total_ut_steps, device=accelerator.device, dtype=torch.long),
        }
        results_per_step_final[cat] = {
            "pred_num": torch.tensor(0, device=accelerator.device, dtype=torch.long),
            "step_correct_num": torch.zeros(total_ut_steps, device=accelerator.device, dtype=torch.long),
        }

    # 用于打印时保存纯 python（防止 tensor 不能 json）
    results_base_py = {cat: {} for cat in cats}
    results_final_py = {cat: {} for cat in cats}
    per_step_base_py = {cat: {"per_step_acc": []} for cat in cats}
    per_step_final_py = {cat: {"per_step_acc": []} for cat in cats}

    if accelerator.is_main_process:
        bar = tqdm(total=len(dataloader), desc="Evaluating")

    # unwrap lm_head
    base_lm_head = accelerator.unwrap_model(base_model).lm_head
    rlp_lm_head = accelerator.unwrap_model(rlp_model).lm_head

    with torch.no_grad():
        with accelerator.autocast():
            for _, batch in enumerate(dataloader):
                input_ids = batch["input_ids"]
                labels = batch["labels"]
                shift_labels = labels[:, 1:]
                loss_mask = batch["loss_mask"]
                shift_loss_mask = loss_mask[:, :-1]

                # =========================
                # 1) Base forward（用于 entropy 分桶）
                # =========================
                base_model_output = base_model(input_ids, use_cache=True, exit_threshold=CONFIG["exit_threshold"])

                # --- Entropy bucket (用 base 的某个 hidden_state_index 算 entropy) ---
                hidden_state_for_entropy = base_model_output.hidden_states_list[CONFIG.get("entropy_hidden_state_index", 0)]
                logits_for_entropy = base_lm_head(hidden_state_for_entropy)  # (B, S, V)
                shift_logits_for_entropy = logits_for_entropy[..., :-1, :]   # (B, S-1, V)
                entropy = calculate_entropy(shift_logits_for_entropy)         # (B, S-1)

                entropy_mask = torch.zeros_like(entropy)  # (B, S-1)
                # threshold 从低到高排序：>threshold 的赋更大 idx
                for idx, (cat_name, threshold) in enumerate(sorted(CONFIG["entropy_thresholds"].items(), key=lambda x: x[1])):
                    entropy_mask[entropy > threshold] = idx + 1

                evaluate_mask = shift_loss_mask * entropy_mask  # (B, S-1)

                # =========================
                # 2) Base 模型评测（可选）
                # =========================
                if CONFIG.get("evaluate_base_model", False):
                    # Peak（你原来）
                    hidden_state_peak = base_model_output.hidden_states_list[CONFIG.get("peak_hidden_state_index", -1)]
                    logits_peak = base_lm_head(hidden_state_peak)
                    shift_logits_peak = logits_peak[..., :-1, :]
                    peak_preds = torch.argmax(shift_logits_peak, dim=-1)

                    # Adaptive（你原来）
                    logits_adapt = base_model_output.logits[..., :-1, :]  # 已是 exit_threshold 后 logits
                    adaptive_preds = torch.argmax(logits_adapt, dim=-1)
                    shift_exit_steps = base_model_output.exit_steps[..., :-1]  # base 这里你原来没 +1

                    for entropy_id, entropy_cat in entropy_category_map.items():
                        mask = (evaluate_mask == entropy_id)
                        results_base[entropy_cat]["peak_correct_num"] += ((peak_preds == shift_labels) & mask).sum()
                        results_base[entropy_cat]["adapt_correct_num"] += ((adaptive_preds == shift_labels) & mask).sum()
                        results_base[entropy_cat]["pred_num"] += mask.sum()
                        results_base[entropy_cat]["steps_num"] += shift_exit_steps[mask].sum()

                    # ================
                    # ✅ 新增：Base per-step（从同一次 forward 的 hidden_states_list[step]）
                    # ================
                    # 注意：per-step 的统计不依赖 exit_threshold；这是“中间表征质量”的测量
                    for step_idx in range(total_ut_steps):
                        hs = base_model_output.hidden_states_list[step_idx]
                        step_logits = base_lm_head(hs)[..., :-1, :]  # (B,S-1,V)
                        step_preds = torch.argmax(step_logits, dim=-1)

                        for entropy_id, entropy_cat in entropy_category_map.items():
                            mask = (evaluate_mask == entropy_id)
                            # pred_num 对每个 cat 只加一次（避免重复加）
                            # 这里 pred_num 用 results_per_step_base 维护
                            results_per_step_base[entropy_cat]["step_correct_num"][step_idx] += ((step_preds == shift_labels) & mask).sum()

                    for entropy_id, entropy_cat in entropy_category_map.items():
                        mask = (evaluate_mask == entropy_id)
                        results_per_step_base[entropy_cat]["pred_num"] += mask.sum()

                # =========================
                # 3) RLP 模型评测（总是评）
                # =========================
                rlp_model_output = rlp_model(input_ids, use_cache=True, exit_threshold=CONFIG["exit_threshold"])

                # Peak（你原来）
                hidden_state_peak = rlp_model_output.hidden_states_list[CONFIG.get("peak_hidden_state_index", -1)]
                logits_peak = rlp_lm_head(hidden_state_peak)
                shift_logits_peak = logits_peak[..., :-1, :]
                peak_preds = torch.argmax(shift_logits_peak, dim=-1)

                # Adaptive（你原来）
                logits_adapt = rlp_model_output.logits[..., :-1, :]
                adaptive_preds = torch.argmax(logits_adapt, dim=-1)
                shift_exit_steps = rlp_model_output.exit_steps[..., :-1] + 1  # 你原来给 RLP +1

                for entropy_id, entropy_cat in entropy_category_map.items():
                    mask = (evaluate_mask == entropy_id)
                    results_final[entropy_cat]["peak_correct_num"] += ((peak_preds == shift_labels) & mask).sum()
                    results_final[entropy_cat]["adapt_correct_num"] += ((adaptive_preds == shift_labels) & mask).sum()
                    results_final[entropy_cat]["pred_num"] += mask.sum()
                    results_final[entropy_cat]["steps_num"] += shift_exit_steps[mask].sum()

                # ================
                # ✅ 新增：RLP per-step（同一次 forward 的 hidden_states_list[step]）
                # ================
                for step_idx in range(total_ut_steps):
                    hs = rlp_model_output.hidden_states_list[step_idx]
                    step_logits = rlp_lm_head(hs)[..., :-1, :]  # (B,S-1,V)
                    step_preds = torch.argmax(step_logits, dim=-1)

                    for entropy_id, entropy_cat in entropy_category_map.items():
                        mask = (evaluate_mask == entropy_id)
                        results_per_step_final[entropy_cat]["step_correct_num"][step_idx] += ((step_preds == shift_labels) & mask).sum()

                for entropy_id, entropy_cat in entropy_category_map.items():
                    mask = (evaluate_mask == entropy_id)
                    results_per_step_final[entropy_cat]["pred_num"] += mask.sum()

                if accelerator.is_main_process:
                    bar.update(1)

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        bar.close()

    # =========================
    # 4) DDP Reduce
    # =========================
    results_base = accelerator.reduce(results_base, reduction="sum")
    results_final = accelerator.reduce(results_final, reduction="sum")
    results_per_step_base = accelerator.reduce(results_per_step_base, reduction="sum")
    results_per_step_final = accelerator.reduce(results_per_step_final, reduction="sum")

    # =========================
    # 5) 计算最终指标（python）
    # =========================
    for cat in cats:
        # Base（如果没 evaluate_base_model，就只输出 per-step=空）
        if CONFIG.get("evaluate_base_model", False):
            denom_b = results_base[cat]["pred_num"].item() + 1e-6
            results_base_py[cat]["peak_acc"] = results_base[cat]["peak_correct_num"].item() / denom_b
            results_base_py[cat]["adapt_acc"] = results_base[cat]["adapt_correct_num"].item() / denom_b
            results_base_py[cat]["avg_steps"] = results_base[cat]["steps_num"].item() / denom_b

            denom_pb = results_per_step_base[cat]["pred_num"].item() + 1e-6
            per_step_acc_b = (results_per_step_base[cat]["step_correct_num"].float() / denom_pb).tolist()
            per_step_base_py[cat]["per_step_acc"] = per_step_acc_b

        # RLP
        denom_r = results_final[cat]["pred_num"].item() + 1e-6
        results_final_py[cat]["peak_acc"] = results_final[cat]["peak_correct_num"].item() / denom_r
        results_final_py[cat]["adapt_acc"] = results_final[cat]["adapt_correct_num"].item() / denom_r
        results_final_py[cat]["avg_steps"] = results_final[cat]["steps_num"].item() / denom_r

        denom_pr = results_per_step_final[cat]["pred_num"].item() + 1e-6
        per_step_acc_r = (results_per_step_final[cat]["step_correct_num"].float() / denom_pr).tolist()
        per_step_final_py[cat]["per_step_acc"] = per_step_acc_r

    # =========================
    # 6) 打印报告（含 per-step）
    # =========================
    if accelerator.is_main_process:
        print("\n" + "#" * 110)
        print(f"{'FINAL REPORT (with Per-Step Acc)':^110}")
        print("#" * 110)

        header = f"{'Diff':<8} | {'Base(Peak)':<10} | {'Base(Adp)':<10} | {'RLP(Peak)':<10} | {'RLP(Adp)':<10} | {'Gain':<8} | {'Step(B)':<8} | {'Step(R)':<8}"
        print(header)
        print("-" * 110)

        for cat in cats:
            if CONFIG.get("evaluate_base_model", False):
                b_peak = results_base_py[cat]["peak_acc"]
                b_adapt = results_base_py[cat]["adapt_acc"]
                b_step = results_base_py[cat]["avg_steps"]
            else:
                b_peak, b_adapt, b_step = float("nan"), float("nan"), float("nan")

            r_peak = results_final_py[cat]["peak_acc"]
            r_adapt = results_final_py[cat]["adapt_acc"]
            r_step = results_final_py[cat]["avg_steps"]

            gain = r_adapt - (b_adapt if CONFIG.get("evaluate_base_model", False) else 0.0)

            row = f"{cat:<8} | {b_peak:.2%}     | {b_adapt:.2%}     | {r_peak:.2%}     | {r_adapt:.2%}     | {gain:+.2%}   | {b_step:.2f}     | {r_step:.2f}"
            print(row)

        print("-" * 110)
        print("[Per-Step Accuracy] 说明：固定同一个 checkpoint（total_ut_steps 不变），从 hidden_states_list[step] 分别计算 step=1..T 的 NTP Acc。")

        # Base per-step block
        if CONFIG.get("evaluate_base_model", False):
            print("\n" + "=" * 110)
            print(f"{'BASE MODEL Per-Step Acc':^110}")
            print("=" * 110)
            for cat in cats:
                per_step = per_step_base_py[cat]["per_step_acc"]
                # step 从 1 开始展示
                per_step_str = " | ".join([f"s{i+1}:{v*100:5.2f}" for i, v in enumerate(per_step)])
                print(f"{cat:<8} | {per_step_str}")

        # RLP per-step block
        print("\n" + "=" * 110)
        print(f"{'RLP MODEL Per-Step Acc':^110}")
        print("=" * 110)
        for cat in cats:
            per_step = per_step_final_py[cat]["per_step_acc"]
            per_step_str = " | ".join([f"s{i+1}:{v*100:5.2f}" for i, v in enumerate(per_step)])
            print(f"{cat:<8} | {per_step_str}")

        print("\n" + "#" * 110)
        print("Metrics Key:")
        print(" - Base(Adp): Base model using its own gate (Natural behavior)")
        print(" - RLP(Adp) : RLP model using optimized gate (Learned behavior)")
        print(" - Step(B/R): Average exit steps in Adaptive mode")
        print(" - Per-Step : Accuracy computed from hidden_states_list[step] (NOT by truncating total_ut_steps)")
        print("#" * 110)

        # =========================
        # 7) Save json（含 per-step）
        # =========================
        result_dir = CONFIG["output_dir"]
        os.makedirs(result_dir, exist_ok=True)

        # 清理 tensor 字段（避免保存无意义大对象）
        with open(os.path.join(result_dir, "results_base.json"), "w") as f:
            json.dump(results_base_py, f, indent=4)

        with open(os.path.join(result_dir, "results_final.json"), "w") as f:
            json.dump(results_final_py, f, indent=4)

        with open(os.path.join(result_dir, "results_base_per_step.json"), "w") as f:
            json.dump(per_step_base_py, f, indent=4)

        with open(os.path.join(result_dir, "results_final_per_step.json"), "w") as f:
            json.dump(per_step_final_py, f, indent=4)

if __name__ == "__main__":
    main()
