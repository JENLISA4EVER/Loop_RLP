from datasets import load_dataset
import os
import json
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import random
if __name__ == "__main__":
    data_dir = "/share/home/sxjiang/myproject/self-learn/datasets/OpenThoughts3-1.2M/"
    data_files = []
    for root,_,files in os.walk(data_dir):
        for file in files:
            if file.endswith(".jsonl"):
                data_files.append(os.path.join(root, file))
    ds1 = load_dataset(
        "json",
        data_files=data_files,   # 8 个 JSON
        split="train",
        streaming=True
    )
    # ds2 = OpenThoughtsDataset(data_files, apply_chat=False, truncation="left")
    data_files = []
    data_dir = "/share/home/sxjiang/myproject/self-learn/datasets/OpenThoughts3-1.2M/"
    for root,_,files in os.walk(data_dir):
        for file in files:
            if file.endswith(".parquet"):
                data_files.append(os.path.join(root, file))
    ds = load_dataset(
        "parquet",
        data_files=data_files,   # 8 个 JSON
        split="train",
        streaming=True
    )
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("/share/home/sxjiang/model/Ouro-1.4B")
    domain_count = {
        "ma"
    }
    
    def process_item_chat(item):
        conversations = item["conversations"]
        new_conversations = []
        for conversation in conversations:
            role_keys = ["from","role"]
            user_values = ["human","user"]
            assistant_values = ["gpt","assistant","model"]
            new_conversation = {}
            for key in role_keys:
                if key in conversation:
                    if conversation[key] in user_values:
                        new_conversation["role"] = "user"
                    elif conversation[key] in assistant_values:
                        new_conversation["role"] = "assistant"
                    else:
                        new_conversation["role"] = "system"
            content_keys = ["value","content"]
            for key in content_keys:
                if key in conversation:
                    new_conversation["content"] = conversation[key]
            new_conversations.append(new_conversation)
        item["conversations"] = new_conversations
        return item
    def get_length(item,tokenizer):
        conversations = item["conversations"]
        prompt_ids = tokenizer.apply_chat_template(
            conversations,
            add_generation_prompt=False,
            tokenize=True,
        )
        return len(prompt_ids)
    import hashlib
    def hash_id(s) -> str:
        return hashlib.sha256(s.encode("utf-8")).hexdigest()
    all_count = 0
    full_think_count = 0
    full_think_box_count = 0
    full_think_no_box_count = 0
    MAX_SEQ_LEN = 20480
    TARGET_LIMITS = {
        'math': 300000,    # Math 限制 30万
        'code': 200000,    # Code 限制 20万
        'science': 200000, # Science 限制 20万 (或者尽可能多)
    }
    TEMP_DIR = {
        "math": "/share/home/sxjiang/myproject/self-learn/temp_parts/math",
        "code": "/share/home/sxjiang/myproject/self-learn/temp_parts/code",
        "science": "/share/home/sxjiang/myproject/self-learn/temp_parts/science",
    }
    EXISTING_FILES = {
        "math": [],
        "code": [],
        "science": [],
    }
    for domain in ["math", "code", "science"]:
        os.makedirs(TEMP_DIR[domain], exist_ok=True)
        EXISTING_FILES[domain] = os.listdir(TEMP_DIR[domain])
    def normalize_domain(raw_domain):
        if not isinstance(raw_domain, str):
            return None 

        d = raw_domain.lower().strip()

        # 1. Math
        if any(x in d for x in ['math', 'algebra', 'geometry', 'calculus', 'statistics', 'puzzle', 'logic']):
            return 'math'
        # 2. Code
        if any(x in d for x in ['code', 'python', 'java', 'programming', 'script', 'software']):
            return 'code'
        # 3. Science
        if any(x in d for x in ['science', 'physics', 'chemistry', 'biology', 'medical', 'medicine']):
            return 'science'

        return None    
    def process_item(item):
        data_id = hash_id(item["conversations"][0]["value"]) #利用question的hash_id作为temp_file的name
        domain = normalize_domain(item["domain"])
        if domain is None:
            return True
        item = process_item_chat(item)
        length = get_length(item,tokenizer)
        if length < MAX_SEQ_LEN:
            file_base_name = data_id+"-"+str(random.randint(1,500)).zfill(3)
            with open(os.path.join(TEMP_DIR[domain], file_base_name+".json"), "w", encoding="utf-8") as f:
                item["id"] = file_base_name
                json.dump(item, f)
    with ThreadPoolExecutor(max_workers=28) as executor:
        futures = []
        for item in tqdm(ds):
            if "</think>" in item["conversations"][1]["value"]:
                full_think_count += 1
                if "\\boxed{" in item["conversations"][1]["value"]:
                    full_think_box_count += 1
                else:
                    full_think_no_box_count += 1
            all_count += 1
            futures.append(executor.submit(process_item, item))
        from concurrent.futures import as_completed
        for future in tqdm(as_completed(futures), total=len(futures)):
            try:
                future.result()
            except Exception as e:
                print(f"Error: {e}")
                from traceback import format_exc
                print(format_exc())
                raise e
    print(f"all_count: {all_count}")
    print(f"full_think_count: {full_think_count}")
    print(f"full_think_box_count: {full_think_box_count}")
    print(f"full_think_no_box_count: {full_think_no_box_count}")
    pass