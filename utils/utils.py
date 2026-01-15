import torch
import wandb
import torch.nn.functional as F
import numpy as np
try:
    from flash_attn.ops.triton.cross_entropy import cross_entropy_loss
    FLAH_ATTN_CROSS_ENTROPY_LOSS_AVAILABLE = True
except ImportError:
    FLAH_ATTN_CROSS_ENTROPY_LOSS_AVAILABLE = False
def logprobs_from_logits_flash_attn(logits, labels):
    output = cross_entropy_loss(logits, labels)
    assert isinstance(
        output, tuple), "please make sure flash-attn>=2.4.3 where cross_entropy_loss returns Tuple[losses, z_losses]."
    return -output[0]
def logprobs_from_logits_v2(logits: torch.FloatTensor, labels):
    """
    A memory efficient implementation of logprobs_from_logits
    """
    if logits.dtype in [torch.float32, torch.float64]:
        logits_labels = torch.gather(logits, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
        # loop to reduce peak mem consumption
        logsumexp_values = torch.stack([torch.logsumexp(l, dim=-1) for l in logits])
        logprobs_labels = logits_labels - logsumexp_values  # log_softmax(x_i) = x_i - logsumexp(x)
    else:
        # logsumexp approach is unstable with bfloat16, fall back to slightly less efficent approach
        logprobs_labels = []
        for row_logits, row_labels in zip(logits, labels):  # loop to reduce peak mem consumption
            row_logprobs = F.log_softmax(row_logits, dim=-1)
            row_logprobs_labels = row_logprobs.gather(dim=-1, index=row_labels.unsqueeze(-1)).squeeze(-1)
            logprobs_labels.append(row_logprobs_labels)
        logprobs_labels = torch.stack(logprobs_labels)
    return logprobs_labels

def add_noise_function(
    x: torch.Tensor,
    std: float = 0.1,
    mask: torch.Tensor | None = None,
):
    """
    Add noise to vector x.

    Args:
    x (torch.Tensor): Input vector (can be of any shape).
    std (float): Standard deviation or amplitude of the noise.
    mask (torch.Tensor): (optional) Apply noise only to elements where mask==1.

    Returns:
    torch.Tensor: Noisy vector
    """
    noise = torch.randn_like(x) * std

    if mask is not None:
        noise = noise * mask.to(noise.dtype).to(x.device)
    x_noisy = x + noise
    del noise
    return x_noisy, None #FIXME:暂时不返回noise 

def logprobs_from_logits(logits, labels):
    """
    See: https://github.com/pytorch/pytorch/issues/563#issuecomment-330103591
    """
    if FLAH_ATTN_CROSS_ENTROPY_LOSS_AVAILABLE:
        batch_dim = logits.shape[:-1]
        last_dim = logits.shape[-1]
        logits = logits.reshape(-1, last_dim)
        labels = labels.reshape(-1)
        output = logprobs_from_logits_flash_attn(logits, labels)
        output = output.view(*batch_dim)
    else:
        output = logprobs_from_logits_v2(logits, labels)
    return output
def chunk_logprob_from_hidden_state(hidden_state,model,labels,chunk_size=1024):
    '''
    hidden_state: (B, S-1, D) shift_hidden_state
    model: model(raw_model)
    labels: (B, S-1) shift_labels
    chunk_size: chunk size
    '''
    logprobs = []
    for i in range(0, hidden_state.size(1) , chunk_size):
        chunk_hidden_state = hidden_state[:,i:i+chunk_size,:]
        chunk_logits = model.lm_head(chunk_hidden_state)
        chunk_labels = labels[:,i:i+chunk_size]
        chunk_logprobs = logprobs_from_logits(chunk_logits, chunk_labels)
        logprobs.append(chunk_logprobs)
        del chunk_logits
        
    return torch.cat(logprobs, dim=1)
    
    

class Tracking:
    def __init__(self, config):
        wandb.init(project=config["project"], name=config["experiment_name"], config=config)
    
    def log(self,data,step):
        wandb.log(data,step=step)
        
def append_to_dict(data, new_data):
    """Append values from new_data to lists in data.

    For each key in new_data, this function appends the corresponding value to a list
    stored under the same key in data. If the key doesn't exist in data, a new list is created.

    Args:
        data (Dict): The target dictionary containing lists as values.
        new_data (Dict): The source dictionary with values to append.

    Returns:
        None: The function modifies data in-place.
    """
    for key, val in new_data.items():
        if key not in data:
            data[key] = []
        data[key].append(val)


def aggregate_metrics(metrics):
    """
    Reduces a dictionary of metric lists by computing the mean, max, or min of each list.
    The reduce operation is determined by the key name:
    - If the key contains "max", np.max is used
    - If the key contains "min", np.min is used
    - Otherwise, np.mean is used

    Args:
        metrics: A dictionary mapping metric names to lists of metric values.

    Returns:
        A dictionary with the same keys but with each list replaced by its reduced value.

    Example:
        >>> metrics = {
        ...     "loss": [1.0, 2.0, 3.0],
        ...     "accuracy": [0.8, 0.9, 0.7],
        ...     "max_reward": [5.0, 8.0, 6.0],
        ...     "min_error": [0.1, 0.05, 0.2]
        ... }
        >>> aggregate_metrics(metrics)
        {"loss": 2.0, "accuracy": 0.8, "max_reward": 8.0, "min_error": 0.05}
    """
    for key, val in metrics.items():
        if isinstance(val, np.ndarray):
            if "max" in key:
                metrics[key] = np.max(val)
            elif "min" in key:
                metrics[key] = np.min(val)
            else:
                metrics[key] = np.mean(val)
        elif isinstance(val, torch.Tensor):
            if "max" in key:
                metrics[key] = val.max()
            elif "min" in key:
                metrics[key] = val.min()
            else:
                metrics[key] = val.mean()
        elif isinstance(val, list):
            if isinstance(val[0], np.ndarray):
                metrics[key] = np.mean(val)
            elif isinstance(val[0], torch.Tensor):
                metrics[key] = torch.stack(val).mean()
            else:
                metrics[key] = np.mean(val)
    return metrics