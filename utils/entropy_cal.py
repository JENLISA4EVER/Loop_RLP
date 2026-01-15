import torch
import torch.nn.functional as F
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
def get_local_entropy_top_mask(entropy, loss_mask, top_ratio=0.2,top_k=16, mode="threshold"):
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
    if entropy.dtype != torch.float32:
        entropy = entropy.float()
    # 2. 过滤有效长度的entropy,每条数据计算一个阈值
    if mode == "threshold":
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
    elif mode == "topk":
        
        entropy_top_mask = torch.zeros_like(entropy, dtype=torch.long)
        for i in range(entropy.size(0)):
            row_valid_entropy = entropy[i][valid_entropy_mask[i]]
            if row_valid_entropy.numel() == 0:
                continue
            top_k = min(top_k, round(row_valid_entropy.numel() * top_ratio))
            topk_indices = torch.topk(row_valid_entropy, k=top_k)[1]
            entropy_top_mask[i][topk_indices] = 1
    else:
        raise ValueError(f"Invalid mode: {mode}")
    return entropy_top_mask

def calculate_entropy(logits):
    """计算 Logits 的熵: H(p) = - sum(p * log(p))"""
    probs = F.softmax(logits, dim=-1)
    log_probs = F.log_softmax(logits, dim=-1)
    entropy = -torch.sum(probs * log_probs, dim=-1)
    return entropy
def entropy_from_logits(logits):
    """Calculate entropy from logits."""
    '''
    ref:verl
    减少额外的log_softmax计算
    '''
    pd = F.softmax(logits, dim=-1)
    entropy = torch.logsumexp(logits, dim=-1) - torch.sum(pd * logits, dim=-1)
    return entropy