import os
import json
import copy
import torch
import math
import random
import hashlib
import numpy as np
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
    "model_path": "/share/home/sxjiang/model/Ouro-1.4B", 
    "data_path": "/share/home/sxjiang/myproject/self-learn/datasets/Omni-MATH/test.jsonl", 
    "output_dir": "/share/home/sxjiang/myproject/self-learn/ouro_rlp_checkpoints_bf16_test",
    "test_data_save_path": "/share/home/sxjiang/myproject/self-learn/datasets/Omni-MATH/rlp_test_set.jsonl",
    "max_length": 2048,
    "batch_size": 4, 
    "gradient_accumulation_steps": 4,
    "lr": 5e-6,              # 全量微调保持低学习率，非常安全
    "weight_decay": 0.01,
    "num_epochs": 3,
    "ema_decay": 0.995,
    "seed": 42,
    "warmup_ratio": 0.03,
    
    # --- RLP & Ouro Specifics ---
    "time_penalty_base": 0.02,    # base思考惩罚系数
    "time_penalty_scale": 0.5,     # γ,思考系数变化scale
    "kl_coef": 0.001,         # KL惩罚，防止骨干跑偏
    "entropy_coef": 0.01,    # 熵正则
    "reward_scale": 1.0,     
    
    "pg_coef": 1.0,          #  PG 权重
    "model_coef": 1.0,       #  Model Loss 
    "val_check_interval": 10, # 每多少步验证一次
    "num_val_samples": 200,   # 论文设定的验证集大小 [cite: 6941]
    "num_test_samples": 400,  # 预留测试集大小

    "num_rollouts": 8,         # G: 每个上下文采样几条 latent reasoning 轨迹
    "latent_noise_type": "gaussian",  # "gaussian" or "dropout"
    "latent_noise_std": 0.1,   # 高斯噪声 std
    "latent_dropout": 0.1,     # dropout 概率
    "gate_threshold": 0.5,     # calculate_gate_stats 用的阈值
}

print(json.dumps(CONFIG,indent=4))

def calculate_gate_stats(gate_list, threshold=0.5):
    """
    计算 Gate 的分布以及决定退出步数 (Strictly aligned with Ouro modeling_ouro.py)
    """
    # gate_list: list of (B, S, 1) tensors
    # Concatenate to (B, S, T_max)
    gate_logits = torch.cat(gate_list, dim=-1)
    # 1. Sigmoid: 得到每一步的条件退出概率 lambda_t
    gate_probs = torch.sigmoid(gate_logits) 
    
    # 2. 计算 PDF (Probability Density Function)
    # 逻辑源自 modeling_ouro.py 中 compute_expected_logits 的实现
    pdf_list = []
    # remaining 记录"存活至今"的概率，初始为 1.0
    remaining_prob = torch.ones_like(gate_probs[..., 0]) 
    
    T_max = gate_probs.shape[-1]
    
    for i in range(T_max):
        lambda_i = gate_probs[..., i]
        
        if i == T_max - 1:
            # [关键对齐] 最后一步强制吸收所有剩余概率，保证 sum(PDF) = 1
            # 对应 paper 公式 (3)
            p_i = remaining_prob
        else:
            # p_t = lambda_t * S_{t-1}
            p_i = lambda_i * remaining_prob
            # 更新剩余存活概率: S_t = S_{t-1} * (1 - lambda_t)
            remaining_prob = remaining_prob * (1.0 - lambda_i)
            
        pdf_list.append(p_i)
    
    exit_pdf = torch.stack(pdf_list, dim=-1) # (B, S, T)
    
    # 3. 计算 CDF (Cumulative Distribution Function)
    exit_cdf = torch.cumsum(exit_pdf, dim=-1)
    
    # 4. 确定退出步数 (Q-Exit 策略)
    # 逻辑源自 modeling_ouro.py lines 420-435
    
    # 找到所有 CDF >= threshold 的位置
    threshold_mask = exit_cdf >= threshold
    
    # argmax 返回第一个 True 的索引。
    # 隐患：如果 threshold 设置很高(如 1.0)，且 fp16 导致 cdf 最高只有 0.999，
    # argmax 会因为全 False 而返回 index 0 (第1步)。这是绝对错误的。
    t_ref_indices = torch.argmax(threshold_mask.float(), dim=-1)
    
    # [关键修正] 兜底逻辑：如果没有一步满足阈值，强制设为最后一步
    # 对应 modeling_ouro.py 中的 `never_exceeded` 处理
    never_exceeded = ~threshold_mask.any(dim=-1)
    last_step_idx = T_max - 1
    t_ref_indices[never_exceeded] = last_step_idx
    
    return exit_pdf, t_ref_indices



# ==========================================
# Dataset
# ==========================================
class OmniMathDataset(Dataset):
    def __init__(self, data_source, tokenizer, max_length):
        """
        data_source: 可以是文件路径(str) 或 已经加载好的 list
        """
        self.data = []
        if isinstance(data_source, str) and os.path.exists(data_source):
            with open(data_source, "r") as f:
                for line in f:
                    try:
                        item = json.loads(line)
                        # RLP将整个语料视为序列 [cite: 6863-6866]
                        text = f"Problem: {item['problem']}\nSolution: {item['solution']}" + tokenizer.eos_token
                        self.data.append(text)
                    except:
                        continue
        elif isinstance(data_source, list):
            self.data = data_source
            
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
        self.entropy_scale = math.log(self.student.config.vocab_size)

    def update_ema(self, decay=None):
        """Update Teacher model with EMA of Student weights"""
        actual_decay = self.config["ema_decay"] if decay is None else decay
        student_model = self.accelerator.unwrap_model(self.student)
        teacher_model = self.accelerator.unwrap_model(self.teacher)
        
        # 全量 EMA 更新
        with torch.no_grad():
            for p_s, p_t in zip(student_model.parameters(), teacher_model.parameters()):
                p_t.data.mul_(actual_decay).add_(p_s.data, alpha=1 - actual_decay)

    def compute_loss_old(self, batch):
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
                # -------------------------------------------------------------
                # 1. Teacher (Reference) Forward -> 确定动态基线 t_ref
                # -------------------------------------------------------------
                # Teacher 也要跑全量 Forward 以获取每一层的 Logits
                teacher_outputs = self.teacher(input_ids, attention_mask=attention_mask, use_cache=False)
                
                # 计算 Teacher 的 t_ref (它觉得自己该在哪一步停)
                # 使用 Ouro 默认的 0.5 或者 config 中的阈值
                _, t_ref_indices = calculate_gate_stats(teacher_outputs.gate_list)
                # t_ref_indices shape: (B, S) - 注意这里的 S 是 seq_len，我们需要对应到 shift_labels (S-1)
                t_ref_indices_shifted = t_ref_indices[..., :-1] # (B, S-1)
                
                # 获取 Teacher 所有层的 Logprobs
                raw_teacher = self.accelerator.unwrap_model(self.teacher)
                teacher_step_logprobs = []
                for hidden in teacher_outputs.hidden_states_list:
                    # lm_head 预测下一个 token
                    logits = raw_teacher.lm_head(hidden) 
                    shift_logits = logits[..., :-1, :].contiguous()
                    step_lp = -F.cross_entropy(
                        shift_logits.transpose(1, 2), safe_shift_labels, reduction='none'
                    )
                    teacher_step_logprobs.append(step_lp)
                teacher_logprobs_stack = torch.stack(teacher_step_logprobs, dim=-1) # (B, S-1, T_max)
                
                # 【关键】Gather Teacher 在 t_ref 这一步的 Logprobs 作为基线
                # gather index 需要 unsqueeze 变成 (B, S-1, 1)
                gather_idx = t_ref_indices_shifted.unsqueeze(-1)
                ref_dynamic_baseline = torch.gather(teacher_logprobs_stack, -1, gather_idx).squeeze(-1) # (B, S-1)

                # -------------------------------------------------------------
                # 2. Student Rollout -> 获取所有步的表现
                # -------------------------------------------------------------
                outputs_inf = self.student(input_ids, attention_mask=attention_mask, use_cache=False)
                raw_student = self.accelerator.unwrap_model(self.student)
                
                student_step_logprobs = []
                for hidden in outputs_inf.hidden_states_list:
                    logits = raw_student.lm_head(hidden)
                    shift_logits = logits[..., :-1, :].contiguous()
                    step_lp = -F.cross_entropy(
                        shift_logits.transpose(1, 2), safe_shift_labels, reduction='none'
                    )
                    student_step_logprobs.append(step_lp)
                
                student_logprobs_det = torch.stack(student_step_logprobs, dim=-1) # (B, S-1, T_max)

                # =================================================================
                # REWARD CALCULATION: Dynamic Ref + Relative Cost
                # =================================================================
                
                # A. 准确率增益 (相对于 Teacher 的动态基线)
                # 形状广播: (B, S-1, T) - (B, S-1, 1)
                accuracy_gain = student_logprobs_det - ref_dynamic_baseline.unsqueeze(-1)
                
                # B. 相对时间成本 (Relative Time Cost)
                # t: 当前步数索引 (0 to T-1)
                # t_ref: Teacher 的步数索引
                T_max = student_logprobs_det.shape[-1]
                device = input_ids.device
                step_indices = torch.arange(T_max, device=device).view(1, 1, -1) # (1, 1, T)
                
                # cost = lambda * (t - t_ref)
                # t_ref 需要广播到 (B, S-1, 1)
                t_diff = step_indices - t_ref_indices_shifted.unsqueeze(-1)
                time_cost = self.config["time_penalty_base"] * t_diff
                
                # C. KL Penalty (Optional, usually small)
                # 防止 Student 分布偏离 Teacher 当前步太远 (Local Constraint)
                # 这里简单用 log ratio 近似
                # kl_penalty = student_logprobs_det - teacher_logprobs_stack
                
                # D. Total Reward
                # rewards = accuracy_gain - time_cost - self.config["kl_coef"] * kl_penalty
                rewards = accuracy_gain - time_cost
                rewards = rewards * self.config["reward_scale"]

                # E. GRPO Advantage Calculation (Group Normalization over T dimension)
                # 将 T_max 个步数视为一个 Group
                r_mean = rewards.mean(dim=-1, keepdim=True)
                r_std = rewards.std(dim=-1, keepdim=True)
                advantages = (rewards - r_mean) / (r_std + 1e-8)
                
                # 用于日志记录
                avg_teacher_step = t_ref_indices_shifted.float().mean()

        # =================================================================
        # PHASE 2: Training (Gradient Mode)
        # =================================================================
        self.student.train() # 全量微调
        
        with self.accelerator.autocast():
            # 2.1 Re-run Student
            outputs_train = self.student(input_ids, attention_mask=attention_mask, use_cache=False)
            
            # 2.2 Re-calculate PDF (Training graph connected)
            exit_pdf_train, _ = calculate_gate_stats(outputs_train.gate_list)
            # 注意：gate_list 长度是 S，我们需要取对应的 S-1 部分与 labels 对齐
            exit_pdf_shifted = exit_pdf_train[..., :-1, :] # (B, S-1, T)
            
            # 统计学生步数
            step_vals = torch.arange(exit_pdf_shifted.shape[-1], device=device).float()
            expected_step = (exit_pdf_shifted.detach() * step_vals).sum(dim=-1).mean()

            # 2.3 Policy Gradient Loss (Optimize Gate)
            # Gate 应该倾向于选择 advantage 高的步数
            pg_loss_unreduced = -(exit_pdf_shifted * advantages.detach()).sum(dim=-1)
            pg_loss = (pg_loss_unreduced * active_mask).sum() / (active_mask.sum() + 1e-6)

            # 2.4 Entropy Loss
            entropy = -(exit_pdf_shifted * torch.log(exit_pdf_shifted + 1e-10)).sum(dim=-1)
            entropy_loss = -self.config["entropy_coef"] * (entropy * active_mask).sum() / (active_mask.sum() + 1e-6)

            # 2.5 Model Loss (Backbone Optimization)
            # 使用 ReLU(advantages) 进行加权，只学习正向收益
            # 这里的改进：不仅仅是用 exit_pdf 加权，还要考虑 advantage
            # 我们希望 backbone 在 advantage 大的层表现更好。
            # 如果只用 exit_pdf 加权，那是标准 Ouro 训练。
            # RLP 改进：
            # loss = - log_prob * (ReLU(Adv) * weighting)
            # 这里 weighting 可以是 1，也可以是 exit_pdf_shifted.detach()
            # 建议保留 exit_pdf 权重，保证模型主要优化 Gate 选择的层
            
            raw_student = self.accelerator.unwrap_model(self.student)
            
            model_loss_list = []
            kl_loss_list = []
            
            # Re-compute student logprobs with grad
            for t_idx, hidden in enumerate(outputs_train.hidden_states_list):
                logits = raw_student.lm_head(hidden)
                shift_logits = logits[..., :-1, :].contiguous()
                
                # --- Model Loss (Cross Entropy) ---
                # PPO-style: Maximize log_prob(gold)
                step_lp = -F.cross_entropy(shift_logits.transpose(1, 2), safe_shift_labels, reduction='none')
                
                # Weighting: Gate Prob * ReLU(Advantage)
                # Note: exit_pdf_shifted is detached to block grad to gate from backbone loss
                weight = exit_pdf_shifted[..., t_idx].detach() * (1.0 + torch.relu(advantages[..., t_idx].detach()))
                step_model_loss = -step_lp * weight
                model_loss_list.append(step_model_loss)
                
                # --- [New] Schulman K3 KL Loss ---
                # kl = ref_logprob - logprob
                # 我们已经有 ref_logprob (teacher_logprobs_stack) 和 logprob (step_lp)
                # 注意: step_lp 是 log_prob(gold), 而不是完整的分布 KL
                # 在 Token-level PPO 中，通常近似为对 target token 的 KL
                # ref_logprob: teacher_logprobs_stack[..., t_idx] (No Grad)
                # logprob: step_lp (With Grad)
                
                ref_lp = teacher_logprobs_stack[..., t_idx].to(step_lp.dtype)
                
                # Calculate KL: log(P_ref) - log(P_stu)
                # kl_diff = ref_lp - step_lp 
                
                # K3 Approx: ratio - log(ratio) - 1
                # ratio = P_stu / P_ref = exp(log_stu - log_ref) = exp(-kl_diff)
                # kld = exp(-kl_diff) - (-kl_diff) - 1
                #     = exp(step_lp - ref_lp) - (step_lp - ref_lp) - 1
                
                log_ratio = ref_lp - step_lp
                ratio = torch.exp(log_ratio)
                k3_kld = (ratio - log_ratio - 1).contiguous()
                
                # Clamp as per image
                k3_kld_clamped = torch.clamp(k3_kld, min=-10, max=10)
                
                # KL Loss is also weighted by the gate probability (we only care about active steps)
                # Use standard gate prob weighting
                kl_loss_list.append(k3_kld_clamped * exit_pdf_shifted[..., t_idx].detach())

            # Aggregate Model Loss
            model_loss_tensor = torch.stack(model_loss_list, dim=-1).sum(dim=-1)
            model_loss = (model_loss_tensor * active_mask).sum() / (active_mask.sum() + 1e-6)
            
            # Aggregate KL Loss
            kl_loss_tensor = torch.stack(kl_loss_list, dim=-1).sum(dim=-1)
            kl_loss = (kl_loss_tensor * active_mask).sum() / (active_mask.sum() + 1e-6)

            # 4. Total Loss
            # Total = PG + Model + Entropy + KL
            total_loss = self.config["pg_coef"] * pg_loss + \
                         self.config["model_coef"] * model_loss + \
                         entropy_loss + \
                         self.config["kl_coef"] * kl_loss
                         

        return total_loss, pg_loss, model_loss, expected_step, rewards.mean(), avg_teacher_step
        
    def compute_loss(self, batch):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        device = input_ids.device
        
        # ===== 预处理 labels =====
        shift_labels = labels[..., 1:].clone()       # (B, S-1)
        active_mask = (shift_labels != -100).float()
        safe_shift_labels = shift_labels.clone()
        safe_shift_labels[shift_labels == -100] = 0

        # =================================================================
        # PHASE 1: Teacher & Student (eval) -> per-step reward (不依赖 rollout)
        # =================================================================
        self.student.eval()
        self.teacher.eval()
        
        with torch.no_grad():
            with self.accelerator.autocast():
                # ---------------------- 1. Teacher 动态 baseline ----------------------
                teacher_outputs = self.teacher(
                    input_ids, 
                    attention_mask=attention_mask, 
                    use_cache=False
                )
                
                # --- [新增] RPT Token Selection Strategy ---
                # 使用 Teacher 最后一层的输出计算由数据本质决定的“难度(熵)”
                # 为什么用最后一层？因为这是模型最终的确定性判断，最能代表该 Token 是否“难”。
                raw_teacher = self.accelerator.unwrap_model(self.teacher)
                last_hidden = teacher_outputs.hidden_states_list[-1] 
                last_logits = raw_teacher.lm_head(last_hidden) # (B, S, V)
                
                # 计算熵: H(p) = -sum(p * log(p))
                # 只取 shift 后的部分 (B, S-1, V)
                shift_teacher_logits = last_logits[..., :-1, :].contiguous()
                probs = F.softmax(shift_teacher_logits, dim=-1)
                log_probs = F.log_softmax(shift_teacher_logits, dim=-1)
                entropy = -(probs * log_probs).sum(dim=-1) # (B, S-1)
                
                # [RPT 核心对齐]: 只选择 Top-K% 高熵 Token (例如 Top 20%)
                # 论文引用: "RPT applies reinforcement only to tokens pre-selected... via entropy filtering"
                # 计算每条数据的 80% 分位点 (即只保留 >80% 的部分)
                rpt_ratio = 0.2 # 训练 Top 20%
                quantile_threshold = torch.quantile(entropy, 1 - rpt_ratio, dim=1, keepdim=True)
                rpt_mask = (entropy >= quantile_threshold).float()
                
                # 更新 active_mask：既要是有效 Token，又要是高熵 Token
                active_mask = active_mask * rpt_mask
                # -------------------------------------------------------------

                # gate -> exit pdf & teacher 的 t_ref
                _, t_ref_indices = calculate_gate_stats(
                    teacher_outputs.gate_list, 
                    threshold=self.config.get("gate_threshold", 0.5)
                )  # (B, S)
                t_ref_indices_shifted = t_ref_indices[..., :-1]  # 对齐 S-1
                
                teacher_step_logprobs = []
                teacher_shift_logits_list = []
                
                for hidden in teacher_outputs.hidden_states_list:
                    logits = raw_teacher.lm_head(hidden)  # (B, S, V)
                    shift_logits = logits[..., :-1, :].contiguous()  # (B, S-1, V)
                    teacher_shift_logits_list.append(shift_logits)    # 保存起来

                    # log_prob(gold)
                    step_lp = -F.cross_entropy(
                        shift_logits.transpose(1, 2),  # (B, V, S-1)
                        safe_shift_labels,
                        reduction="none"
                    )  # (B, S-1)
                    teacher_step_logprobs.append(step_lp)
                
                # 堆成 (B, S-1, T_max)
                teacher_logprobs_stack = torch.stack(
                    teacher_step_logprobs, dim=-1
                )

                
                # 把 teacher 在 t_ref 这一步的 logprob 当作 baseline
                gather_idx = t_ref_indices_shifted.unsqueeze(-1)  # (B, S-1, 1)
                ref_dynamic_baseline = torch.gather(
                    teacher_logprobs_stack, -1, gather_idx
                ).squeeze(-1)  # (B, S-1)

                # 1) 把 logits 堆成一个 4D 张量，方便 gather
                teacher_shift_logits_stack = torch.stack(teacher_shift_logits_list, dim=-2)
                # 形状: (B, S-1, T_max, V)

                # 2) 在 teacher 参考步数 t_ref 上 gather logits
                gather_idx = t_ref_indices_shifted.unsqueeze(-1).unsqueeze(-1)  # (B, S-1, 1, 1)
                gather_idx = gather_idx.expand(-1, -1, 1, teacher_shift_logits_stack.size(-1))
                selected_logits = torch.gather(
                    teacher_shift_logits_stack, -2, gather_idx
                ).squeeze(-2)                           # (B, S-1, V)

                # 3) softmax → 概率分布
                probs = selected_logits.float().softmax(dim=-1)   # (B, S-1, V)

                # 4) 熵 H_i
                token_entropy = -(probs * probs.log()).sum(dim=-1)    # (B, S-1)

                # 5) 映射到 [0, 1] 难度 d_i
                difficulty = (token_entropy / self.entropy_scale).clamp(0.0, 1.0)  # (B, S-1)

                lambda_base  = self.config.get("time_penalty_base", 0.02)
                lambda_scale = self.config.get("time_penalty_scale", 1.0)

                # 简单 token (difficulty 小) → λ_i 更大；困难 token → λ_i 更小
                lambda_i = lambda_base * (1.0 + lambda_scale * (1.0 - difficulty))   # (B, S-1)

                # 扩展到 step 维度，方便和 t_diff 相乘
                lambda_i_expanded = lambda_i.unsqueeze(-1)    # (B, S-1, 1)

                # ---------------------- 2. Student per-step logprob ----------------------
                outputs_inf = self.student(
                    input_ids, 
                    attention_mask=attention_mask, 
                    use_cache=False
                )
                raw_student_eval = self.accelerator.unwrap_model(self.student)
                
                student_step_logprobs = []
                for hidden in outputs_inf.hidden_states_list:
                    logits = raw_student_eval.lm_head(hidden)  # (B, S, V)
                    shift_logits = logits[..., :-1, :].contiguous()
                    step_lp = -F.cross_entropy(
                        shift_logits.transpose(1, 2),
                        safe_shift_labels,
                        reduction="none"
                    )  # (B, S-1)
                    student_step_logprobs.append(step_lp)
                
                # (B, S-1, T_max)
                student_logprobs_det = torch.stack(
                    student_step_logprobs, dim=-1
                )

                # =================================================================
                # REWARD: per-step reward[b,s,t] = accuracy_gain - time_cost
                # =================================================================
                T_max = student_logprobs_det.shape[-1]
                step_indices = torch.arange(T_max, device=device).view(1, 1, -1).float()  # (1,1,T)

                accuracy_gain = student_logprobs_det - ref_dynamic_baseline.unsqueeze(-1)  # (B,S-1,T)

                t_diff = step_indices - t_ref_indices_shifted.unsqueeze(-1).float()        # (B,S-1,T)

                # 使用难度感知的 λ_i
                time_cost = lambda_i_expanded * t_diff   # (B,S-1,T)

                rewards_stepwise = (accuracy_gain - time_cost) * self.config["reward_scale"]
                
                # 这个是 backbone 用的 token-level advantage（在 T 上标准化）
                r_mean_token = rewards_stepwise.mean(dim=-1, keepdim=True)
                r_std_token = rewards_stepwise.std(dim=-1, keepdim=True)
                token_advantages = (rewards_stepwise - r_mean_token) / (r_std_token + 1e-8)
                
                # teacher 的平均 t_ref（日志用）
                avg_teacher_step = t_ref_indices_shifted.float().mean()

        # =================================================================
        # PHASE 2: Training (带梯度) —— latent rollouts + GRPO + backbone
        # =================================================================
        self.student.train()
        
        with self.accelerator.autocast():
            # ---------------------- 1. student 再前向（train 模式） ----------------------
            outputs_train = self.student(
                input_ids, 
                attention_mask=attention_mask, 
                use_cache=False
            )
            raw_student = self.accelerator.unwrap_model(self.student)
            
            # 1.1 用“确定性 gate”算一个 exit_pdf_train：用于 entropy + backbone 权重 + 日志
            exit_pdf_train, _ = calculate_gate_stats(
                outputs_train.gate_list, 
                threshold=self.config.get("gate_threshold", 0.5)
            )  # (B, S, T_max)
            exit_pdf_shifted = exit_pdf_train[..., :-1, :]  # 对齐 S-1
            
            step_vals = torch.arange(
                exit_pdf_shifted.shape[-1], device=device
            ).float()  # (T,)
            expected_step = (exit_pdf_shifted.detach() * step_vals).sum(dim=-1).mean()

            # ---------------------- 2. latent rollouts: G 条轨迹 ----------------------
            G = self.config.get("num_rollouts", 4)
            noise_type = self.config.get("latent_noise_type", "gaussian")
            noise_std = self.config.get("latent_noise_std", 0.1)
            dropout_p = self.config.get("latent_dropout", 0.1)

            group_rewards = []
            group_log_probs = []
            
            # 直接从 train 的 hidden_states_list 里取 latent 状态
            hidden_list_train = outputs_train.hidden_states_list  # 长度 T_max, 每个 (B, S, H)
            
            for g in range(G):
                noisy_gate_list = []
                for hidden in hidden_list_train:
                    # 只对有 label 的位置 (S-1) 计算 gate
                    latent = hidden[..., :-1, :]  # (B, S-1, H)
                    
                    if noise_type == "gaussian":
                        noise = torch.randn_like(latent) * noise_std
                        latent_noisy = latent + noise
                    elif noise_type == "dropout":
                        latent_noisy = F.dropout(latent, p=dropout_p, training=True)
                    else:
                        latent_noisy = latent  # no noise

                    gate_logits = raw_student.model.early_exit_gate(latent_noisy)  # (B, S-1, 1)
                    noisy_gate_list.append(gate_logits)
                
                # 对这一条 rollout 的 gate 计算 exit 分布
                exit_pdf_g, _ = calculate_gate_stats(
                    noisy_gate_list, 
                    threshold=self.config.get("gate_threshold", 0.5)
                )  # (B, S-1, T_max)
                
                # 从 exit_pdf_g 采样 exit step：t^{(g)}_{b,s}
                T = exit_pdf_g.shape[-1]
                probs_flat = exit_pdf_g.view(-1, T)  # (B*(S-1), T)
                step_idx_flat = torch.multinomial(probs_flat, num_samples=1).squeeze(-1)
                step_idx = step_idx_flat.view_as(shift_labels)  # (B, S-1)
                
                # 对应的 log π_g
                chosen_prob = torch.gather(
                    exit_pdf_g, -1, step_idx.unsqueeze(-1)
                ).squeeze(-1)  # (B, S-1)
                log_pi = torch.log(chosen_prob + 1e-8)
                
                # 用 Phase 1 的 per-step reward 表，取出对应 step 的 reward
                reward_g = torch.gather(
                    rewards_stepwise, -1, step_idx.unsqueeze(-1)
                ).squeeze(-1)  # (B, S-1)
                
                group_rewards.append(reward_g)
                group_log_probs.append(log_pi)
            
            # 堆叠成 (G, B, S-1)
            rewards_group = torch.stack(group_rewards, dim=0)
            logp_group = torch.stack(group_log_probs, dim=0)
            
            # 在 G 维度上做 GRPO 标准化
            r_mean = rewards_group.mean(dim=0, keepdim=True)  # (1, B, S-1)
            r_std = rewards_group.std(dim=0, keepdim=True)
            adv_group = (rewards_group - r_mean) / (r_std + 1e-8)  # (G, B, S-1)
            
            # Policy Gradient loss: sum over G，再按 (B,S-1) 加权平均
            pg_loss_unreduced = -(adv_group.detach() * logp_group).sum(dim=0)  # (B, S-1)
            pg_loss = (pg_loss_unreduced * active_mask).sum() / (active_mask.sum() + 1e-6)

            # ---------------------- 3. gate 的 entropy 正则（用确定性 exit_pdf_train） ----------------------
            entropy = -(exit_pdf_shifted * torch.log(exit_pdf_shifted + 1e-10)).sum(dim=-1)
            entropy_loss = -self.config["entropy_coef"] * (entropy * active_mask).sum() / (active_mask.sum() + 1e-6)

            # ---------------------- 4. backbone + KL（基本保留原逻辑） ----------------------
            model_loss_list = []
            kl_loss_list = []
            
            for t_idx, hidden in enumerate(outputs_train.hidden_states_list):
                logits = raw_student.lm_head(hidden)    # (B, S, V)
                shift_logits = logits[..., :-1, :].contiguous()
                
                # log_prob(gold)
                step_lp = -F.cross_entropy(
                    shift_logits.transpose(1, 2),
                    safe_shift_labels,
                    reduction="none"
                )  # (B, S-1)
                
                # 用 per-step 的 token_advantages 做 backbone 的额外加权
                weight = exit_pdf_shifted[..., t_idx].detach() * (
                    1.0 + torch.relu(token_advantages[..., t_idx].detach())
                )
                step_model_loss = -step_lp * weight
                model_loss_list.append(step_model_loss)
                
                # KL (Schulman K3) 部分保持原来的近似：teacher vs student logprob
                ref_lp = teacher_logprobs_stack[..., t_idx].to(step_lp.dtype)
                log_ratio = ref_lp - step_lp
                ratio = torch.exp(log_ratio)
                k3_kld = ratio - log_ratio - 1
                k3_kld_clamped = torch.clamp(k3_kld, min=-10, max=10)
                kl_loss_list.append(k3_kld_clamped * exit_pdf_shifted[..., t_idx].detach())

            model_loss_tensor = torch.stack(model_loss_list, dim=-1).sum(dim=-1)
            model_loss = (model_loss_tensor * active_mask).sum() / (active_mask.sum() + 1e-6)
            
            kl_loss_tensor = torch.stack(kl_loss_list, dim=-1).sum(dim=-1)
            kl_loss = (kl_loss_tensor * active_mask).sum() / (active_mask.sum() + 1e-6)

            # ---------------------- 5. 总 loss ----------------------
            total_loss = (
                self.config["pg_coef"] * pg_loss
                + self.config["model_coef"] * model_loss
                + entropy_loss
                + self.config["kl_coef"] * kl_loss
            )

        # 日志里可以把 reward 平均一下
        avg_reward = rewards_group.mean()

        return total_loss, pg_loss, model_loss, expected_step, avg_reward, avg_teacher_step

# ==========================================
# Main Execution
# ==========================================
def main():
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
    accelerator = Accelerator(gradient_accumulation_steps=CONFIG["gradient_accumulation_steps"], mixed_precision="bf16", kwargs_handlers=[ddp_kwargs])
    set_seed(CONFIG["seed"] + accelerator.process_index)

    if accelerator.is_main_process:
        print(f"Distributed training on {accelerator.num_processes} GPUs.")
        if not os.path.exists(CONFIG["output_dir"]):
            os.makedirs(CONFIG["output_dir"])

    tokenizer = AutoTokenizer.from_pretrained(CONFIG["model_path"])
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    
    config = OuroConfig.from_pretrained(CONFIG["model_path"])
    config.total_ut_steps = 4 
    
    student = OuroForCausalLM.from_pretrained(CONFIG["model_path"], config=config, low_cpu_mem_usage=True, torch_dtype=torch.bfloat16)
    student.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    
    teacher = OuroForCausalLM.from_pretrained(CONFIG["model_path"], config=config, low_cpu_mem_usage=True, torch_dtype=torch.bfloat16)
    teacher.eval()
    for p in teacher.parameters(): p.requires_grad = False
    
    # --- Data Loading & Splitting ---
    # 读取原始数据
    raw_data = []
    with open(CONFIG["data_path"], "r") as f:
        for line in f:
            try:
                raw_data.append(json.loads(line))
            except: pass
            
    total_len = len(raw_data)
    if accelerator.is_main_process:
        print(f"Total samples loaded: {total_len}")

    # [关键修改 1] 确定性随机打乱 (Deterministic Shuffle)
    # 必须确保所有 GPU 进程使用完全相同的 Seed 进行打乱
    # 否则不同卡上的训练数据会错位，甚至导致测试集泄露
    rng = random.Random(CONFIG["seed"]) # 使用 Config 中的固定 Seed (42)
    rng.shuffle(raw_data) # 原地打乱

    # 执行划分
    test_size = CONFIG["num_test_samples"]
    val_size = CONFIG["num_val_samples"]
    train_size = total_len - test_size - val_size
    
    # 切片 (Slicing) - 物理隔离
    # Train: [0 : train_size]
    # Val:   [train_size : train_size + val_size]
    # Test:  [train_size + val_size : ]
    train_raw = raw_data[:train_size]
    val_raw = raw_data[train_size:train_size+val_size]
    test_raw = raw_data[train_size+val_size:]
    
    # [关键修改 2] 数据指纹校验 (Data Fingerprint Check)
    # 计算第一条数据的 Hash，确保所有进程看到的数据顺序是一致的
    def get_data_fingerprint(data_list):
        if not data_list: return "empty"
        # 取第一条和最后一条的 content 做 hash
        sample_str = str(data_list[0]) + str(data_list[-1])
        return hashlib.md5(sample_str.encode()).hexdigest()[:8]

    train_fp = get_data_fingerprint(train_raw)
    test_fp = get_data_fingerprint(test_raw)
    
    # 打印指纹，人工检查是否一致
    print(f"[Process {accelerator.process_index}] Split Fingerprint | Train: {train_fp} | Test: {test_fp}")
    
    # 只有主进程负责保存测试集
    if accelerator.is_main_process:
        print(f"Data Split Stats: Train={len(train_raw)}, Val={len(val_raw)}, Test={len(test_raw)}")
        
        # 检查是否覆盖，防止误操作
        if os.path.exists(CONFIG["test_data_save_path"]):
            print(f"[Warning] Overwriting existing test set at {CONFIG['test_data_save_path']}")
            
        with open(CONFIG["test_data_save_path"], "w") as f:
            for item in test_raw:
                f.write(json.dumps(item) + "\n")
        print(f"✅ Test set saved to {CONFIG['test_data_save_path']} (Fingerprint: {test_fp})")
        print("IMPORTANT: Please ensure the Evaluation Script uses THIS specific file.")

    # 同步所有进程，确保主进程保存完文件，且大家数据一致
    accelerator.wait_for_everyone()

    # 构造 text list
    def fmt_data(items):
        return [f"Problem: {x['problem']}\nSolution: {x['solution']}" + tokenizer.eos_token for x in items]

    train_dataset = OmniMathDataset(fmt_data(train_raw), tokenizer, CONFIG["max_length"])
    val_dataset = OmniMathDataset(fmt_data(val_raw), tokenizer, CONFIG["max_length"])
    
    train_dataloader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], shuffle=True, num_workers=4)
    val_dataloader = DataLoader(val_dataset, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=4)

    trainer = RLPTrainer(student, teacher, tokenizer, CONFIG, accelerator)
    
    num_training_steps = (len(train_dataloader) * CONFIG["num_epochs"]) // CONFIG["gradient_accumulation_steps"]
    lr_scheduler = get_cosine_schedule_with_warmup(trainer.optimizer, num_warmup_steps=int(0.03*num_training_steps), num_training_steps=num_training_steps)

    trainer.student, trainer.optimizer, train_dataloader, val_dataloader, lr_scheduler = accelerator.prepare(
        trainer.student, trainer.optimizer, train_dataloader, val_dataloader, lr_scheduler
    )
    trainer.teacher.to(accelerator.device)
    
    # --- Training Loop ---
    global_step = 0
    for epoch in range(CONFIG["num_epochs"]):
        trainer.student.train()
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(trainer.student):
                loss, pg, m_loss, avg_step, avg_rwd, ref_step = trainer.compute_loss(batch)
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(trainer.student.parameters(), 1.0)
                    trainer.optimizer.step()
                    lr_scheduler.step()
                    trainer.optimizer.zero_grad()
                    trainer.update_ema()
                    global_step += 1
            
            if step % 10 == 0 and accelerator.is_main_process:
                print(f"Ep {epoch}|St {step}|L:{loss.item():.4f}|PG:{pg.item():.4f}|M:{m_loss.item():.4f}|Stp:{avg_step.item():.2f}|Ref:{ref_step.item():.2f}|Rwd:{avg_rwd.item():.4f}")

            # --- Validation Loop ---
            if global_step % CONFIG["val_check_interval"] == 0 and global_step > 0 and accelerator.sync_gradients:
                if accelerator.is_main_process:
                    print(f"Running Validation at step {global_step}...")
                trainer.student.eval()
                val_rewards = []
                val_steps = []
                with torch.no_grad():
                    for val_batch in val_dataloader:
                        # 只需要前向计算 Reward 和 Step
                        _, _, _, v_step, v_rwd, _ = trainer.compute_loss(val_batch)
                        val_rewards.append(v_rwd.item())
                        val_steps.append(v_step.item())
                
                trainer.student.train()
                if accelerator.is_main_process:
                    print(f" >>> VAL | Avg Reward: {np.mean(val_rewards):.4f} | Avg Step: {np.mean(val_steps):.2f}")

        if accelerator.is_main_process:
            save_path = os.path.join(CONFIG["output_dir"], f"epoch_{epoch}")
            accelerator.unwrap_model(trainer.student).save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)

if __name__ == "__main__":
    main()