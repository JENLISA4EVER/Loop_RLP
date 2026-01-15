import glob
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
import logging
from logging import getLogger
logger = getLogger(__name__)
logger.setLevel(logging.INFO)
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
    parser.add_argument("--output_dir", type=str, default="/share/home/sxjiang/myproject/self-learn/datasets/processed/omni_math/entropy_split_topk_difficulty_len_2048_debug")
    parser.add_argument("--entropy_threshold", type=float, default=0.2)
    parser.add_argument("--apply_chat", type=int, default=1)
    parser.add_argument("--chunk", type=int, default=0)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--exit_threshold", type=float, default=0.5)
    parser.add_argument("--entropy_hidden_state_index", type=int, default=0)
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
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, max_length=args.max_length, truncation="right", max_length_method="group_max")
    tokenizer.pad_token_id = tokenizer.eos_token_id #这里是<|endoftext|>
    entropy_thresholds = {
        "Easy": 0.5,
        "Medium": 1.0,
        "Hard": 1.5
    }
    cat_map = {
        1: "Easy",
        2: "Medium",
        3: "Hard"
    }
    REASONING_PROMPT_TEMPLATE = (
        "Predict the next token of the context and wrap it in \\boxed{{}}.\n\n"
        "Context: '1 + 1 ='\\nAnswer: \\boxed{{ 2}}\n\n"
        "Context: 'The capital of France is'\\nAnswer: \\boxed{{ Paris}}\n\n"
        "Context: '{context}'\\nAnswer: \\boxed{{"
    )
    
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
                entropy_mask = torch.zeros_like(entropy, dtype=torch.long)
                thresholds = sorted(entropy_thresholds.items(), key=lambda x: x[1])
                for idx, (cat, thresh) in enumerate(thresholds):
                    entropy_mask[entropy > thresh] = idx + 1
                new_loss_mask = shift_loss_mask * entropy_mask
                indices = torch.nonzero(new_loss_mask)

                # 2. 避免在循环中频繁切片，改用批量操作
                for b_idx, s_idx in indices:
                    # 依然需要解码，但我们尽量减少 decode 的次数
                    # 技巧：如果 response 只是单个 token，可以直接用 tokenizer.convert_ids_to_tokens
                    prefix_ids = input_ids[b_idx, :s_idx]
                    target_id = input_ids[b_idx, s_idx]
                    #XXX:Problem:+Solution:
                    new_problem = tokenizer.decode(prefix_ids, skip_special_tokens=True).split("user\n")[1].split("\nassistant")[0]
                    new_solution = tokenizer.decode(prefix_ids, skip_special_tokens=True).split("\nassistant")[1]
                    new_problem = REASONING_PROMPT_TEMPLATE.format(context=f"Problem:{new_problem}\nSolution:{new_solution}")
                    if accelerator.is_main_process:
                        logger.info(f"origin_prefix: {tokenizer.decode(prefix_ids, skip_special_tokens=True)}")
                        logger.info(f"new_problem: {new_problem}")
                    new_dataset.append({
                        "problem": new_problem,
                        "response": tokenizer.decode(target_id),
                        "difficulty": cat_map[entropy_mask[b_idx, s_idx].item()]
                    })
                if os.environ.get("DEBUG_MODE") == "1":
                    break
                    
    if accelerator.is_main_process:
        bar.close()
        os.makedirs(args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()
    with open(os.path.join(args.output_dir, f"test_ntp_difficulty_split_{accelerator.process_index}.jsonl"), "w", encoding="utf-8") as f:
        for data in new_dataset:
            f.write(json.dumps(data) + "\n")
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        process_files = glob.glob(os.path.join(args.output_dir, "test_ntp_difficulty_split_*.jsonl"))
        assert len(process_files) == accelerator.num_processes
        easy_dataset = []
        medium_dataset = []
        hard_dataset = []
        for file in process_files:
            if file.endswith(".jsonl"):
                with open(os.path.join(args.output_dir, file), "r", encoding="utf-8") as f:
                    for line in f:
                        data = json.loads(line)
                        if data["difficulty"] == "Easy":
                            easy_dataset.append(data)
                        elif data["difficulty"] == "Medium":
                            medium_dataset.append(data)
                        elif data["difficulty"] == "Hard":
                            hard_dataset.append(data)
        split_dir = os.path.join(args.output_dir, "split")
        os.makedirs(split_dir, exist_ok=True)
        with open(os.path.join(split_dir, f"test_ntp_difficulty_split_easy.jsonl"), "w", encoding="utf-8") as f:
            for data in easy_dataset:
                f.write(json.dumps(data) + "\n")
        with open(os.path.join(split_dir, f"test_ntp_difficulty_split_medium.jsonl"), "w", encoding="utf-8") as f:
            for data in medium_dataset:
                f.write(json.dumps(data) + "\n")
        with open(os.path.join(split_dir, f"test_ntp_difficulty_split_hard.jsonl"), "w", encoding="utf-8") as f:
            for data in hard_dataset:
                f.write(json.dumps(data) + "\n")
    accelerator.end_training()