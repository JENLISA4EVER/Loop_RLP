import json
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import random
import torch
from transformers import AutoTokenizer
from modeling_ouro import OuroForCausalLM, OuroConfig
from utils.dataset_pt import OmniMathDataset, DataCollatorWithPadding
from utils.entropy_cal import entropy_from_logits,get_local_entropy_top_mask,calculate_entropy
from torch.utils.data import Dataset, DataLoader
from accelerate import Accelerator
from tqdm import tqdm
import argparse

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="/share/home/sxjiang/myproject/self-learn/datasets/processed/omni_math/train.jsonl")
    parser.add_argument("--ouro_model_path", type=str, default="/share/home/sxjiang/model/Ouro-1.4B")
    parser.add_argument("--output_dir", type=str, default="/share/home/sxjiang/myproject/self-learn/datasets/processed/omni_math/entropy_split_topk_16_len_4096")
    parser.add_argument("--entropy_threshold", type=float, default=0.2)
    parser.add_argument("--apply_chat", type=int, default=1)
    parser.add_argument("--chunk", type=int, default=0)
    parser.add_argument("--max_length", type=int, default=4096)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--exit_threshold", type=float, default=0.5)
    parser.add_argument("--entropy_hidden_state_index", type=int, default=-1)
    parser.add_argument("--entropy_mode", type=str, default="topk")
    return parser.parse_args()
 

if __name__ == "__main__":
    set_seed(42)
    args = get_args()
    ouro_model = OuroForCausalLM.from_pretrained(args.ouro_model_path,dtype=torch.bfloat16,attn_implementation="flash_attention_2")
    tokenizer = AutoTokenizer.from_pretrained(args.ouro_model_path)
    accelerator = Accelerator()
    apply_chat = True if args.apply_chat == 1 else False
    dataset = OmniMathDataset(data_source=args.data_path, tokenizer=tokenizer, max_length=args.max_length, apply_chat=apply_chat, truncation="no")
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, max_length=args.max_length, truncation="right")
    tokenizer.pad_token_id = tokenizer.eos_token_id #这里是<|endoftext|>
    if apply_chat:
        tokenizer.eos_token = "<|im_end|>"
    else:
        tokenizer.eos_token = "<|endoftext|>"

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=data_collator)
    
    ouro_model, dataloader = accelerator.prepare(ouro_model, dataloader)
    new_dataset = []
    if accelerator.is_main_process:
        bar = tqdm(total=len(dataloader), desc="Processing Data")
    with torch.no_grad():
        with accelerator.autocast():
            for batch in dataloader:
                if accelerator.is_main_process:
                    bar.update(1)
                input_ids = batch["input_ids"]
                attention_mask = batch["attention_mask"]
                loss_mask = batch["loss_mask"]
                shift_loss_mask = loss_mask[..., :-1]
                outputs = ouro_model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True,exit_threshold=args.exit_threshold)
                entropy_hidden_state = outputs.hidden_states_list[args.entropy_hidden_state_index]
                logits =  accelerator.unwrap_model(ouro_model).lm_head(entropy_hidden_state)
                shift_logits = logits[..., :-1, :]
                entropy = entropy_from_logits(shift_logits) #(B, S-1) 
                entropy_top_mask = get_local_entropy_top_mask(entropy, shift_loss_mask, top_ratio=args.entropy_threshold, mode=args.entropy_mode)
                new_loss_mask = shift_loss_mask * entropy_top_mask
                # for idx, input_id in enumerate(input_ids):
                #     full_prefix = tokenizer.decode(input_id)
                #     predict_idx_list = torch.where(new_loss_mask[idx]).tolist()
                #     for predict_idx in predict_idx_list:
                #         new_dataset.append({
                #             "prefix": tokenizer.decode(input_id[:predict_idx]),
                #             "response": tokenizer.decode(input_id[predict_idx]),
                #         })
                # 1. 预先计算所有需要提取的索引
                # 使用 nonzero 获取所有 mask 为 1 的坐标 (batch_idx, seq_idx)
                indices = torch.nonzero(new_loss_mask)

                # 2. 避免在循环中频繁切片，改用批量操作
                for b_idx, s_idx in indices:
                    # 依然需要解码，但我们尽量减少 decode 的次数
                    # 技巧：如果 response 只是单个 token，可以直接用 tokenizer.convert_ids_to_tokens
                    prefix_ids = input_ids[b_idx, :s_idx]
                    target_id = input_ids[b_idx, s_idx]
                    
                    new_dataset.append({
                        "prefix": tokenizer.decode(prefix_ids, skip_special_tokens=False),
                        "response": tokenizer.decode(target_id)
                    })
    if accelerator.is_main_process:
        bar.close()
        os.makedirs(args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()
    with open(os.path.join(args.output_dir, f"train_{accelerator.process_index}.jsonl"), "w", encoding="utf-8") as f:
        for data in new_dataset:
            f.write(json.dumps(data) + "\n")
            
    accelerator.end_training()