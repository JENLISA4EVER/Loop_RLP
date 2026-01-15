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
import hydra
from omegaconf import DictConfig, OmegaConf
import logging

logger = logging.getLogger(__name__)
def get_global_entropy_top_mask(entropy, loss_mask, top_ratio=0.2):
    """
    Select the top `top_ratio` high-entropy tokens among all response tokens in a batch.
    ref: https://github.com/Shenzhi-Wang/Beyond-the-80-20-Rule-RLVR/blob/main/verl/trainer/ppo/core_algos.py#L53
    Args:
        entropy: [B, S] tensor of token entropies.
        loss_mask: [B, S] tensor (1 = response token, 0 = non-response).
        top_ratio: fraction of response tokens to keep (e.g. 0.2 = top 20%).
        
    Returns:
        entropy_top_mask: [B, S] binary mask (1 = selected top entropy token)
    """
    
    # Flatten
    flat_entropy = entropy.flatten()
    flat_mask = loss_mask.flatten().bool()
    
    # Filter response token
    response_entropy = flat_entropy[flat_mask]
    if response_entropy.numel() == 0:
        return torch.zeros_like(entropy, dtype=torch.long)

    # Top-k selection
    top_k = max(1, int(len(response_entropy) * top_ratio + 0.9999)) # ceil
    _, topk_idx = torch.topk(response_entropy, k=top_k)
    
    # Map back to original flat indices
    response_positions = flat_mask.nonzero(as_tuple=False).squeeze(1)
    top_positions = response_positions[topk_idx]
    
    # Build mask
    flat_out = torch.zeros_like(flat_entropy, dtype=torch.long)
    flat_out[top_positions] = 1
    
    return flat_out.view_as(entropy)
def get_local_entropy_top_mask(entropy, loss_mask, top_ratio=0.2):
    """
    Select the top `top_ratio` high-entropy tokens among all response tokens in a batch.
    ref: https://github.com/Shenzhi-Wang/Beyond-the-80-20-Rule-RLVR/blob/main/verl/trainer/ppo/core_algos.py#L53
    Args:
        entropy: [B, S-1] tensor of token entropies.
        loss_mask: [B, S-1] tensor (1 = response token, 0 = non-response). shift_labels之后的mask
        top_ratio: fraction of response tokens to keep (e.g. 0.2 = top 20%).
        
    Returns:
        entropy_top_mask: [B, S-1] binary mask (1 = selected top entropy token)
    """
    # 1. 获取有效长度的mask
    valid_entropy_mask = loss_mask.bool()#FIXME:loss_mask是shift_labels之后的mask，还是用attention_mask比较好？
    
    # 2. 过滤有效长度的entropy,每条数据计算一个阈值
    row_thresholds = []
    for i in range(entropy.size(0)):
        row_valid_entropy = entropy[i][valid_entropy_mask[i]]
        if row_valid_entropy.numel() == 0:
            row_thresholds.append(torch.tensor(0.0,device=entropy.device))
        else:
            row_threshold = torch.quantile(row_valid_entropy, 1 - top_ratio)
            row_thresholds.append(row_threshold)
    row_thresholds = torch.stack(row_thresholds, dim=0) #(B,)
    # 3. 计算mask，大于阈值为1，否则为0
    entropy_top_mask = entropy > row_thresholds.unsqueeze(1) #(B, S-1)
    return entropy_top_mask
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

    exit_pdf = torch.stack(pdf_list, dim=-1)  # (B, S, T)

    # 3. 计算 CDF (Cumulative Distribution Function)
    exit_cdf = torch.cumsum(exit_pdf, dim=-1)  # FIXME:sum(exit_cdf)

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
    def __init__(
        self, data_source, tokenizer, max_length, apply_chat=False, prompt_key="problem",
        response_key="solution",truncation="left"
    ):
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
                        self.data.append(item)
                    except:
                        continue
        elif isinstance(data_source, list):
            self.data = data_source

        self.tokenizer = tokenizer
        self.max_length = max_length
        self.apply_chat = apply_chat
        self.prompt_key = prompt_key
        self.response_key = response_key
        self.truncation = truncation

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        #reference: verl/sft_dataset.py
        if os.environ.get("RANK","0") in os.environ.get(
            "DEBUG_RANK", "-1"
        ) and "2" in os.environ.get("DEBUG_MODE", "0"):
            breakpoint()  # FIXME:默认right padding
        # 1. 分别根据prompt_key和response_key获取 prompt 和 response
        item = self.data[idx]
        prompt = item[self.prompt_key]
        response = item[self.response_key]

        # 2. string
        if self.apply_chat:
            prompt_chat = [{"role": "user", "content": prompt}]
            prompt_str = self.tokenizer.apply_chat_template(
                prompt_chat,
                add_generation_prompt=True,
                tokenize=False,
            )
        else:
            # 和之前没chat_template一样，只是分开编码prompt和response
            prompt_str = f"Problem: {prompt}\nSolution: "    
        response_str = response + self.tokenizer.eos_token #FIXME:ouro的tokenizer.eos_token和config.eos_token不一样
        
        # 3. tokenize
        prompt_ids_output = self.tokenizer(
            prompt_str,
            return_tensors="pt",
            add_special_tokens=False,
        )
        prompt_ids = prompt_ids_output["input_ids"][0]
        prompt_attention_mask = prompt_ids_output["attention_mask"][0]
        response_ids_output = self.tokenizer(
            response_str,
            add_special_tokens=False,
            return_tensors="pt",
        )
        response_ids = response_ids_output["input_ids"][0]
        response_attention_mask = response_ids_output["attention_mask"][0]
        
        prompt_length = prompt_ids.shape[0]
        response_length = response_ids.shape[0]
        
        input_ids = torch.cat((prompt_ids, response_ids), dim=-1)
        attention_mask = torch.cat((prompt_attention_mask, response_attention_mask), dim=-1)
        
        # 4. padding to max_length
        sequence_length = prompt_length + response_length
        if sequence_length < self.max_length:
            padded_input_ids = torch.ones(size=(self.max_length - sequence_length,),
                                          dtype=input_ids.dtype) * self.tokenizer.pad_token_id
            padded_attention_mask = torch.zeros(size=(self.max_length - sequence_length,), dtype=attention_mask.dtype)
            input_ids = torch.cat((input_ids, padded_input_ids), dim=-1)
            attention_mask = torch.cat((attention_mask, padded_attention_mask), dim=-1)
        elif sequence_length > self.max_length:
            assert self.truncation == "right", "truncation must be right"
            if self.truncation == "right":
                input_ids = input_ids[:self.max_length]
                attention_mask = attention_mask[:self.max_length]
            elif self.truncation == "left":
                input_ids = input_ids[-self.max_length:]
                attention_mask = attention_mask[-self.max_length:]
            else:
                raise ValueError(f"Invalid truncation: {self.truncation}")
        # 5. compute position_ids
        def compute_position_id_with_mask(mask):
            return torch.clip(torch.cumsum(mask, dim=-1) - 1, min=0, max=None)
        
        position_ids = compute_position_id_with_mask(attention_mask)

        # 6. loss mak
        loss_mask = attention_mask.clone()
        if prompt_length > 1:
            # mask out prompt loss.
            loss_mask[:min(prompt_length, loss_mask.size(0)) - 1] = 0 #XXX:这里也要注意shift_labels
        # mask out the last token in response
        loss_mask[min(prompt_length + response_length, loss_mask.size(0)) - 1] = 0 #FIXME:不确定

        labels = input_ids.clone()
        labels[labels == self.tokenizer.pad_token_id] = -100 #FIXME:有了loss_mask，这里还需要吗？
                
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "position_ids": position_ids,
            "loss_mask": loss_mask,
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
            weight_decay=config["weight_decay"],
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

    def compute_loss(self, batch):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        if "labels" in batch:
            labels = batch["labels"]
        else:
            labels = input_ids.clone()
        loss_mask = batch["loss_mask"] 
        position_ids = batch["position_ids"]
        device = input_ids.device

        # ===== 预处理 labels =====
        shift_labels = labels[..., 1:].clone()  # (B, S-1)
        shift_loss_mask = loss_mask[..., :-1].clone() #FIXME:shift_loss_mask
        active_mask = (shift_labels != -100).float() #FIXME:用loss_mask代替
        safe_shift_labels = shift_labels.clone() #FIXME:是否需要safe_shift_labels,应该不用，因为shift_labels不管怎么样都会计算logprob，利用loss_mask mask
        safe_shift_labels[shift_labels == -100] = 0

        # =================================================================
        # PHASE 1: Teacher & Student (eval) -> per-step reward (不依赖 rollout)
        # =================================================================
        self.student.eval()
        self.teacher.eval()

        with torch.no_grad():
            with self.accelerator.autocast():
                if os.environ.get("RANK","0") in os.environ.get(
                    "DEBUG_RANK", "-1"
                ) and "3" in os.environ.get("DEBUG_MODE", "0"):
                    breakpoint()  # FIXME:默认right padding
                # ---------------------- 1. Teacher 动态 baseline ----------------------
                # THREEGOLDCHANGE:传入threshold
                teacher_outputs = self.teacher(
                    input_ids, attention_mask=attention_mask, use_cache=False, threshold=self.config.get("gate_threshold", 0.5)
                )

                # --- [新增] RPT Token Selection Strategy ---
                # 使用 Teacher 最后一层的输出计算由数据本质决定的“难度(熵)”
                # 为什么用最后一层？因为这是模型最终的确定性判断，最能代表该 Token 是否“难”。
                raw_teacher = self.accelerator.unwrap_model(self.teacher)
                last_hidden = teacher_outputs.hidden_states_list[
                    -1
                ]  # FIXME:(bz,sq,sq,ut)为什么取最后step
                last_logits = raw_teacher.lm_head(last_hidden)  # (B, S, V)

                # 计算熵: H(p) = -sum(p * log(p))
                # 只取 shift 后的部分 (B, S-1, V)
                shift_teacher_logits = last_logits[..., :-1, :].contiguous()
                probs = F.softmax(shift_teacher_logits, dim=-1)
                log_probs = F.log_softmax(shift_teacher_logits, dim=-1)
                entropy = -(probs * log_probs).sum(dim=-1)  # (B, S-1)

                # [RPT 核心对齐]: 只选择 Top-K% 高熵 Token (例如 Top 20%)
                # 论文引用: "RPT applies reinforcement only to tokens pre-selected... via entropy filtering"
                # 计算每条数据的 80% 分位点 (即只保留 >80% 的部分) #FIXME:要取有效长度的前20%吗，attention_mask是有效长度
                rpt_ratio = self.config.get("rpt_ratio", 0.2)  # 训练 Top 20%
                # THREEGOLD:修改entropy计算方式，只取有效长度内的entropy
                rpt_mask = get_local_entropy_top_mask(entropy, shift_loss_mask, top_ratio=rpt_ratio)
                
                # 更新 active_mask/shift_loss_mask：既要是有效/计算loss的 Token，又要是高熵 Token
                active_mask = active_mask * rpt_mask
                shift_loss_mask = shift_loss_mask * rpt_mask
                # -------------------------------------------------------------

                # gate -> exit pdf & teacher 的 t_ref,用于计算效率
                _, t_ref_indices = calculate_gate_stats(
                    teacher_outputs.gate_list,
                    threshold=self.config.get("gate_threshold", 0.5), 
                )  # (B, S)
                t_ref_indices_shifted = t_ref_indices[..., :-1]  # 对齐 S-1

                teacher_step_logprobs = [] #存储label在每一步对应的logprob
                teacher_shift_logits_list = [] #存储VOCAB_SIZE在每一步对应的logits

                for hidden in teacher_outputs.hidden_states_list:
                    logits = raw_teacher.lm_head(hidden)  # (B, S, V)
                    shift_logits = logits[..., :-1, :].contiguous()  # (B, S-1, V)
                    teacher_shift_logits_list.append(shift_logits)  # 保存起来

                    # log_prob(gold)
                    step_lp = -F.cross_entropy(
                        shift_logits.transpose(1, 2),  # (B, V, S-1)
                        # safe_shift_labels,
                        shift_labels,
                        reduction="none",
                    )  # (B, S-1)
                    teacher_step_logprobs.append(step_lp) # 获得teacher在每个step golden label的logprob

                # 堆成 (B, S-1, T_max)
                teacher_logprobs_stack = torch.stack(teacher_step_logprobs, dim=-1) #计算KL散度

                # 把 teacher 在 t_ref 这一步的 logprob 当作 baseline
                gather_idx = t_ref_indices_shifted.unsqueeze(-1)  # (B, S-1, 1)
                ref_dynamic_baseline = torch.gather(
                    teacher_logprobs_stack, -1, gather_idx
                ).squeeze(
                    -1
                )  # (B, S-1)#FIXME:等价于 -F.cross_entropy(teacher_outputs.logits[...,:-1,:].contiguous().transpose(1, 2), shift_labels,   reduction="none", )

                ref_dynamic_baseline2 = -F.cross_entropy(
                    teacher_outputs.logits[...,:-1,:].contiguous().transpose(1, 2), 
                    shift_labels,
                    reduction="none", ) # (B, S-1)
                
                # 1) 把 logits 堆成一个 4D 张量，方便 gather
                teacher_shift_logits_stack = torch.stack(
                    teacher_shift_logits_list, dim=-2
                )
                # 形状: (B, S-1, T_max, V)

                # 2) 在 teacher 参考步数 t_ref 上 gather logits
                gather_idx = t_ref_indices_shifted.unsqueeze(-1).unsqueeze(
                    -1
                )  # (B, S-1, 1, 1)
                gather_idx = gather_idx.expand(
                    -1, -1, 1, teacher_shift_logits_stack.size(-1)
                )
                selected_logits = torch.gather(
                    teacher_shift_logits_stack, -2, gather_idx
                ).squeeze(
                    -2
                )  # (B, S-1, V)#FIXME:等价于teacher_output.logits[..., :-1, :]

                # THREEGOLDCHANGE:直接利用相同的threshold计算teacher_output.logits[..., :-1, :]
                selected_logits2 = teacher_outputs.logits[..., :-1, :]   
                
                # 3) softmax → 概率分布
                probs = selected_logits.float().softmax(dim=-1)  # (B, S-1, V)

                # 4) 熵 H_i
                token_entropy = -(probs * probs.log()).sum(dim=-1)  # (B, S-1) #和之前entropy不一样的，这里是ref_indices对于步数的熵

                # 5) 映射到 [0, 1] 难度 d_i
                difficulty = (token_entropy / self.entropy_scale).clamp(
                    0.0, 1.0
                )  # (B, S-1)

                lambda_base = self.config.get("time_penalty_base", 0.02)
                lambda_scale = self.config.get("time_penalty_scale", 1.0)

                # 简单 token (difficulty 小) → λ_i 更大；困难 token → λ_i 更小
                lambda_i = lambda_base * (
                    1.0 + lambda_scale * (1.0 - difficulty)
                )  # (B, S-1)

                # 扩展到 step 维度，方便和 t_diff 相乘
                lambda_i_expanded = lambda_i.unsqueeze(-1)  # (B, S-1, 1)

                # ---------------------- 2. Student per-step logprob ----------------------
                outputs_inf = self.student(
                    input_ids, attention_mask=attention_mask, use_cache=False, threshold=self.config.get("gate_threshold", 0.5)
                )
                raw_student_eval = self.accelerator.unwrap_model(self.student)

                student_step_logprobs = []
                for hidden in outputs_inf.hidden_states_list:
                    logits = raw_student_eval.lm_head(hidden)  # (B, S, V)
                    shift_logits = logits[..., :-1, :].contiguous()
                    step_lp = -F.cross_entropy(
                        shift_logits.transpose(1, 2),
                        # safe_shift_labels, #THREEGOLD
                        shift_labels,
                        reduction="none",
                    )  # (B, S-1)
                    student_step_logprobs.append(step_lp)

                # (B, S-1, T_max)
                student_logprobs_det = torch.stack(student_step_logprobs, dim=-1)

                # =================================================================
                # REWARD: per-step reward[b,s,t] = accuracy_gain - time_cost
                # =================================================================
                T_max = student_logprobs_det.shape[-1]
                step_indices = (
                    torch.arange(T_max, device=device).view(1, 1, -1).float()
                )  # (1,1,T)

                accuracy_gain = student_logprobs_det - ref_dynamic_baseline.unsqueeze(
                    -1
                )  # (B,S-1,T) 

                t_diff = (
                    step_indices - t_ref_indices_shifted.unsqueeze(-1).float()
                )  # (B,S-1,T) #步数的差值

                # 使用难度感知的 λ_i
                time_cost = lambda_i_expanded * t_diff  # (B,S-1,T)

                rewards_stepwise = (accuracy_gain - time_cost) * self.config[
                    "reward_scale"
                ]

                # 这个是 backbone 用的 token-level advantage（在 T 上标准化）
                r_mean_token = rewards_stepwise.mean(dim=-1, keepdim=True)
                r_std_token = rewards_stepwise.std(dim=-1, keepdim=True)
                token_advantages = (rewards_stepwise - r_mean_token) / (
                    r_std_token + 1e-8
                )

                # teacher 的平均 t_ref（日志用）
                avg_teacher_step = t_ref_indices_shifted.float().mean()
        #XXX:显存的释放？
        # =================================================================
        # PHASE 2: Training (带梯度) —— latent rollouts + GRPO + backbone
        # =================================================================
        self.student.train()

        with self.accelerator.autocast():
            # ---------------------- 1. student 再前向（train 模式） ----------------------
            outputs_train = self.student(
                input_ids, attention_mask=attention_mask, use_cache=False
            )
            raw_student = self.accelerator.unwrap_model(self.student)

            # 1.1 用“确定性 gate”算一个 exit_pdf_train：用于 entropy + backbone 权重 + 日志
            exit_pdf_train, _ = calculate_gate_stats(
                outputs_train.gate_list,
                threshold=self.config.get("gate_threshold", 0.5),
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
            hidden_list_train = (
                outputs_train.hidden_states_list
            )  # 长度 T_max, 每个 (B, S, H)

            for g in range(G):
                noisy_gate_list = []
                for hidden in hidden_list_train: #按step循环
                    # 只对有 label 的位置 (S-1) 计算 gate
                    latent = hidden[..., :-1, :]  # (B, S-1, H)

                    if noise_type == "gaussian":
                        noise = torch.randn_like(latent) * noise_std
                        latent_noisy = latent + noise
                    elif noise_type == "dropout":
                        latent_noisy = F.dropout(latent, p=dropout_p, training=True)
                    else:
                        latent_noisy = latent  # no noise

                    gate_logits = raw_student.model.early_exit_gate(
                        latent_noisy
                    )  # (B, S-1, 1)
                    noisy_gate_list.append(gate_logits)

                # 对这一条 rollout 的 gate 计算 exit 分布(退出步的概率分布，t_max个概率)
                exit_pdf_g, _ = calculate_gate_stats(
                    noisy_gate_list, threshold=self.config.get("gate_threshold", 0.5)
                )  # (B, S-1, T_max)

                # 从 exit_pdf_g 采样 exit step：t^{(g)}_{b,s}
                T = exit_pdf_g.shape[-1]
                probs_flat = exit_pdf_g.view(-1, T)  # (B*(S-1), T)
                step_idx_flat = torch.multinomial(probs_flat, num_samples=1).squeeze(-1)#FIXME:为什么是采样而不是和之前一样cdf>threshold？
                step_idx = step_idx_flat.view_as(shift_labels)  # (B, S-1)

                # 对应的 log π_g
                chosen_prob = torch.gather(
                    exit_pdf_g, -1, step_idx.unsqueeze(-1)
                ).squeeze(
                    -1
                )  # (B, S-1) #XXX:为什么是用这个作为log_pi?查看是否存在梯度被截断?(和CISPO的实现类似)
                log_pi = torch.log(chosen_prob + 1e-8)

                # 用 Phase 1 的 per-step reward 表，取出对应 step 的 reward
                reward_g = torch.gather(
                    rewards_stepwise, -1, step_idx.unsqueeze(-1)
                ).squeeze(
                    -1
                )  # (B, S-1)

                group_rewards.append(reward_g)
                group_log_probs.append(log_pi)

            # 堆叠成 (G, B, S-1)
            rewards_group = torch.stack(group_rewards, dim=0)
            logp_group = torch.stack(group_log_probs, dim=0)

            # 在 G 维度上做 GRPO 标准化 #XXX:group_normalization还是batch_normalization?/Step-Advantage
            r_mean = rewards_group.mean(dim=0, keepdim=True)  # (1, B, S-1)
            r_std = rewards_group.std(dim=0, keepdim=True)
            adv_group = (rewards_group - r_mean) / (r_std + 1e-8)  # (G, B, S-1)

            # Policy Gradient loss: sum over G，再按 (B,S-1) 加权平均
            pg_loss_unreduced = -(adv_group.detach() * logp_group).sum(
                dim=0
            )  # (B, S-1) #XXX:如果没有 log π_old,会有梯度吗?
            pg_loss = (pg_loss_unreduced * shift_loss_mask).sum() / (
                shift_loss_mask.sum() + 1e-6
            ) #FIXME:是否按照Dr.GRPO/DAPO修改loss平均的方式，不用因为这里每个rollout的长度都是一样的，不会存在不同rollout长度不同导致loss权重                                                                     

            # ---------------------- 3. gate 的 entropy 正则（用确定性 exit_pdf_train） ----------------------
            entropy = -(exit_pdf_shifted * torch.log(exit_pdf_shifted + 1e-10)).sum(
                dim=-1
            )
            entropy_loss = (
                -self.config["entropy_coef"]
                * (entropy * shift_loss_mask).sum()
                / (shift_loss_mask.sum() + 1e-6)
            )

            # ---------------------- 4. backbone + KL（基本保留原逻辑） ----------------------
            model_loss_list = []
            kl_loss_list = []

            for t_idx, hidden in enumerate(outputs_train.hidden_states_list):
                logits = raw_student.lm_head(hidden)  # (B, S, V)
                shift_logits = logits[..., :-1, :].contiguous()

                # log_prob(gold)
                step_lp = -F.cross_entropy(
                    shift_logits.transpose(1, 2), safe_shift_labels, reduction="none"
                )  # (B, S-1)

                # 用 per-step 的 token_advantages 做 backbone 的额外加权
                weight = exit_pdf_shifted[..., t_idx].detach() * (
                    1.0 + torch.relu(token_advantages[..., t_idx].detach())
                )#XXX:为什么要额外加权，和ouro原论文保持一致？
                step_model_loss = -step_lp * weight
                model_loss_list.append(step_model_loss)

                # KL (Schulman K3) 部分保持原来的近似：teacher vs student logprob
                ref_lp = teacher_logprobs_stack[..., t_idx].to(step_lp.dtype)
                log_ratio = ref_lp - step_lp
                # THREEGOLDCHANGE:follow https://github.com/volcengine/verl/blob/1c99f4727ed184937e87c5b363ae69c0e79b8049/verl/trainer/ppo/core_algos.py#L1464 and https://github.com/volcengine/verl/issues/891
                if self.config.get("kl_loss_clamp", False):
                    log_ratio = torch.clamp(log_ratio, min=-20, max=20) 
                ratio = torch.exp(log_ratio)
                k3_kld = (ratio - log_ratio - 1).contiguous()
                k3_kld_clamped = torch.clamp(k3_kld, min=-10, max=10)
                kl_loss_list.append(
                    k3_kld_clamped * exit_pdf_shifted[..., t_idx].detach()
                )

            model_loss_tensor = torch.stack(model_loss_list, dim=-1).sum(dim=-1) #在T_max上求和
            model_loss = (model_loss_tensor * shift_loss_mask).sum() / (
                shift_loss_mask.sum() + 1e-6
            )

            kl_loss_tensor = torch.stack(kl_loss_list, dim=-1).sum(dim=-1) #在T_max上求和
            kl_loss = (kl_loss_tensor * shift_loss_mask).sum() / (shift_loss_mask.sum() + 1e-6)

            # ---------------------- 5. 总 loss ----------------------
            total_loss = (
                self.config["pg_coef"] * pg_loss
                + self.config["model_coef"] * model_loss
                + entropy_loss
                + self.config["kl_coef"] * kl_loss
            )

        # 日志里可以把 reward 平均一下
        avg_reward = rewards_group.mean()

        return (
            total_loss,
            pg_loss,
            model_loss,
            expected_step,
            avg_reward,
            avg_teacher_step,
        )


# ==========================================
# Main Execution
# ==========================================
@hydra.main(version_base=None, config_path="configs", config_name="ouro_rlp_acc_omnimath")
def main(config: DictConfig):
    # --- Config --- #
    CONFIG = OmegaConf.to_container(config, resolve=True)  # 转为普通字典
    logger.info(json.dumps(CONFIG, indent=4))
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
    accelerator = Accelerator(
        gradient_accumulation_steps=CONFIG["gradient_accumulation_steps"],
        mixed_precision="bf16",
        kwargs_handlers=[ddp_kwargs],
    )
    if str(accelerator.process_index) in os.environ.get(
        "DEBUG_RANK", "-1"
    ) and "1" in os.environ.get("DEBUG_MODE", "0"):
        breakpoint()
    set_seed(CONFIG["seed"] + accelerator.process_index)

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
        torch_dtype=torch.bfloat16,
    )
    student.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )  # FIXME:accerlate应该有这个设置来着

    teacher = OuroForCausalLM.from_pretrained(
        CONFIG["model_path"],
        config=config,
        low_cpu_mem_usage=True,
        torch_dtype=torch.bfloat16,
    )
    teacher.eval() #FIXME:这里的eval需要吗?
    for p in teacher.parameters(): #FIXME:需不需要设置requires_grad=False?
        p.requires_grad = False

    # --- Data Loading & Splitting ---
    # 读取原始数据
    raw_data = []
    with open(CONFIG["data_path"], "r") as f:
        for line in f:
            try:
                raw_data.append(json.loads(line))
            except:
                pass

    total_len = len(raw_data)
    if accelerator.is_main_process:
        print(f"Total samples loaded: {total_len}")

    # [关键修改 1] 确定性随机打乱 (Deterministic Shuffle)
    # 必须确保所有 GPU 进程使用完全相同的 Seed 进行打乱
    # 否则不同卡上的训练数据会错位，甚至导致测试集泄露
    rng = random.Random(CONFIG["seed"])  # 使用 Config 中的固定 Seed (42)
    rng.shuffle(raw_data)  # 原地打乱 #FIXME:应该也可以放在dataloader里面shuffle吧

    # 执行划分
    test_size = CONFIG["num_test_samples"]  # CHECK:看OpenThought是否需要修改
    val_size = CONFIG["num_val_samples"]
    train_size = total_len - test_size - val_size

    # 切片 (Slicing) - 物理隔离
    # Train: [0 : train_size]
    # Val:   [train_size : train_size + val_size]
    # Test:  [train_size + val_size : ]
    train_raw = raw_data[:train_size]
    val_raw = raw_data[train_size : train_size + val_size]
    test_raw = raw_data[train_size + val_size :]

    # [关键修改 2] 数据指纹校验 (Data Fingerprint Check)
    # 计算第一条数据的 Hash，确保所有进程看到的数据顺序是一致的
    def get_data_fingerprint(data_list):  # ASK:不确定是否必须
        if not data_list:
            return "empty"
        # 取第一条和最后一条的 content 做 hash
        sample_str = str(data_list[0]) + str(data_list[-1])
        return hashlib.md5(sample_str.encode()).hexdigest()[:8]

    train_fp = get_data_fingerprint(train_raw)
    test_fp = get_data_fingerprint(test_raw)

    # 打印指纹，人工检查是否一致
    print(
        f"[Process {accelerator.process_index}] Split Fingerprint | Train: {train_fp} | Test: {test_fp}"
    )

    # 只有主进程负责保存测试集
    if accelerator.is_main_process:
        print(
            f"Data Split Stats: Train={len(train_raw)}, Val={len(val_raw)}, Test={len(test_raw)}"
        )

        # 检查是否覆盖，防止误操作
        if os.path.exists(CONFIG["test_data_save_path"]):
            print(
                f"[Warning] Overwriting existing test set at {CONFIG['test_data_save_path']}"
            )

        with open(CONFIG["test_data_save_path"], "w") as f:
            for item in test_raw:
                f.write(json.dumps(item) + "\n")
        print(
            f"✅ Test set saved to {CONFIG['test_data_save_path']} (Fingerprint: {test_fp})"
        )
        print("IMPORTANT: Please ensure the Evaluation Script uses THIS specific file.")

    # 同步所有进程，确保主进程保存完文件，且大家数据一致
    accelerator.wait_for_everyone()

    if str(accelerator.process_index) in os.environ.get(
        "DEBUG_RANK", "-1"
    ) and "1" in os.environ.get("DEBUG_MODE", "0"):
        breakpoint()

    train_dataset = OmniMathDataset(train_raw, tokenizer, CONFIG["max_length"],apply_chat=CONFIG["apply_chat"])
    val_dataset = OmniMathDataset(val_raw, tokenizer, CONFIG["max_length"],apply_chat=CONFIG["apply_chat"])

    train_dataloader = DataLoader(
        train_dataset, batch_size=CONFIG["batch_size"], shuffle=True, num_workers=CONFIG.get("num_workers",4)
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=CONFIG.get("num_workers",4)
    )

    trainer = RLPTrainer(student, teacher, tokenizer, CONFIG, accelerator)

    num_training_steps = (len(train_dataloader) * CONFIG["num_epochs"]) // CONFIG[
        "gradient_accumulation_steps"
    ]
    lr_scheduler = get_cosine_schedule_with_warmup(
        trainer.optimizer,
        num_warmup_steps=int(0.03 * num_training_steps),
        num_training_steps=num_training_steps,
    )

    (
        trainer.student,
        trainer.optimizer,
        train_dataloader,
        val_dataloader,
        lr_scheduler,
    ) = accelerator.prepare(
        trainer.student,
        trainer.optimizer,
        train_dataloader,
        val_dataloader,
        lr_scheduler,
    )
    trainer.teacher.to(accelerator.device)
    
    # --- Resume Training ---
    if CONFIG.get("resume_from_checkpoint", False):
        trainer.load_checkpoint(CONFIG["resume_from_checkpoint"])
    # --- Training Loop ---
    global_step = 0
    for epoch in range(CONFIG["num_epochs"]):
        trainer.student.train()
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(trainer.student):
                if os.environ.get("RANK","0") in os.environ.get(
                    "DEBUG_RANK", "-1"
                ) and "1" in os.environ.get("DEBUG_MODE", "0"):
                    breakpoint()  # FIXME:默认right padding
                loss, pg, m_loss, avg_step, avg_rwd, ref_step = trainer.compute_loss(
                    batch
                )
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(trainer.student.parameters(), 1.0)
                    trainer.optimizer.step()
                    lr_scheduler.step()
                    trainer.optimizer.zero_grad()
                    trainer.update_ema()
                    global_step += 1

            if step % 10 == 0 and accelerator.is_main_process:
                print(
                    f"Ep {epoch}|St {step}|L:{loss.item():.4f}|PG:{pg.item():.4f}|M:{m_loss.item():.4f}|Stp:{avg_step.item():.2f}|Ref:{ref_step.item():.2f}|Rwd:{avg_rwd.item():.4f}"
                )

            # --- Validation Loop ---
            if (
                global_step % CONFIG["val_check_interval"] == 0
                and global_step > 0
                and accelerator.sync_gradients
            ):
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
                    print(
                        f" >>> VAL | Avg Reward: {np.mean(val_rewards):.4f} | Avg Step: {np.mean(val_steps):.2f}"
                    )

        if accelerator.is_main_process:
            save_path = os.path.join(CONFIG["output_dir"], f"epoch_{epoch}")
            accelerator.unwrap_model(trainer.student).save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)


if __name__ == "__main__":
    main()
