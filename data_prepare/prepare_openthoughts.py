import os
import glob
import json
import gc
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
import pandas as pd
import pyarrow.parquet as pq
from transformers import AutoTokenizer
from tqdm import tqdm
from collections import defaultdict

# ================= 配置区 =================
# 原始数据路径
DATA_DIR = "/home/gtang/self-learn/datasets/OpenThoughts3-1.2M/data"

# 输出目录配置
OUTPUT_DIRS = {
    'Math': "/home/gtang/self-learn/datasets/OpenThoughts3-1.2M/processed_4k_math",
    'Code': "/home/gtang/self-learn/datasets/OpenThoughts3-1.2M/processed_4k_code",
    'Science': "/home/gtang/self-learn/datasets/OpenThoughts3-1.2M/processed_4k_science"
}

# [修复] 数量配额限制 (根据 5:3:2 比例)
# 设置为 float('inf') 则代表不限制，全量提取
TARGET_LIMITS = {
    'Math': 300000,    # Math 限制 30万
    'Code': 200000,    # Code 限制 20万
    'Science': 200000, # Science 限制 20万 (或者尽可能多)
}

# 临时文件目录
TEMP_DIR = "/home/gtang/self-learn/datasets/OpenThoughts3-1.2M/temp_parts"
MODEL_PATH = "/home/gtang/pretrain_model/Ouro-1.4B"

# 过滤参数
MAX_SEQ_LEN = 20480

# [性能参数]
NUM_WORKERS = 8          # 进程数
BATCH_SIZE = 100         # 单次读取行数
LINES_PER_FILE = 50000   # 切分大小：每个 jsonl 文件存 5万行

# =========================================

def normalize_domain(raw_domain):
    if not isinstance(raw_domain, str):
        return None 
    
    d = raw_domain.lower().strip()
    
    # 1. Math
    if any(x in d for x in ['math', 'algebra', 'geometry', 'calculus', 'statistics', 'puzzle', 'logic']):
        return 'Math'
    # 2. Code
    if any(x in d for x in ['code', 'python', 'java', 'programming', 'script', 'software']):
        return 'Code'
    # 3. Science
    if any(x in d for x in ['science', 'physics', 'chemistry', 'biology', 'medical', 'medicine']):
        return 'Science'
    
    return None

def format_conversation(conversations):
    text = ""
    for turn in conversations:
        role_raw = turn.get('from', '') or turn.get('role', '')
        content = turn.get('value', '') or turn.get('content', '')
        if content is None: content = ""
        
        if role_raw in ['human', 'user']:
            role_tag = 'user'
        elif role_raw in ['gpt', 'assistant', 'model']:
            role_tag = 'assistant'
        elif role_raw in ['system']:
            role_tag = 'system'
        else:
            continue
        text += f"<|im_start|>{role_tag}\n{content}<|im_end|>\n"
    return text + "<|im_end|>"

def process_single_file(file_path, output_dir, model_path, max_len):
    # 关闭 Tokenizer 并行
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    except Exception as e:
        print(f"Tokenizer load failed: {e}")
        return None
    
    file_name = os.path.basename(file_path)
    
    # 准备临时文件句柄
    temp_handles = {}
    temp_paths = {}
    
    for cat in ['Math', 'Code', 'Science']:
        path = os.path.join(output_dir, f"{cat}_{file_name}.jsonl")
        temp_paths[cat] = path
        temp_handles[cat] = open(path, 'w', encoding='utf-8')

    local_stats = defaultdict(lambda: {'count': 0, 'discarded': 0})
    
    try:
        parquet_file = pq.ParquetFile(file_path)
        num_rows = parquet_file.metadata.num_rows
        total_batches = (num_rows + BATCH_SIZE - 1) // BATCH_SIZE
        
        iterator = parquet_file.iter_batches(batch_size=BATCH_SIZE)
        
        for batch in tqdm(iterator, total=total_batches, desc=f"Proc {file_name[:10]}...", leave=False):
            df_batch = batch.to_pandas()
            
            for _, row in df_batch.iterrows():
                # 1. 分类
                raw_domain = row['domain'] if 'domain' in row else 'unknown'
                category = normalize_domain(raw_domain)
                if category is None: continue
                
                # 2. 提取
                convs = row.get('conversations')
                if convs is None or len(convs) == 0: continue
                    
                full_text = format_conversation(convs)
                if not full_text.strip(): continue

                # 3. 过滤
                token_ids = tokenizer.encode(full_text, add_special_tokens=False)
                seq_len = len(token_ids)
                
                if seq_len > max_len:
                    local_stats[category]['discarded'] += 1
                    continue
                
                # 4. 写入临时文件 (不在这里做数量截断，为了并行效率，统一在 Merge 阶段截断)
                record = {
                    "text": full_text,
                    "length": seq_len,
                    "source": "OpenThoughts"
                }
                temp_handles[category].write(json.dumps(record, ensure_ascii=False) + "\n")
                local_stats[category]['count'] += 1
            
            del df_batch
            gc.collect()
            
    except Exception as e:
        print(f"Error processing {file_name}: {e}")
    finally:
        for f in temp_handles.values():
            f.close()
            
    return temp_paths, dict(local_stats)

def merge_and_split(category, temp_files, output_dir, lines_per_file, limit):
    """
    合并阶段：读取临时文件，写入最终文件夹，并支持切分和数量限制。
    """
    print(f"Merging {category} into {output_dir} (Limit: {limit})...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    file_idx = 0
    line_count = 0
    current_out_path = os.path.join(output_dir, f"train_part_{file_idx}.jsonl")
    f_out = open(current_out_path, 'w', encoding='utf-8')
    
    total_saved = 0
    
    # 遍历所有临时文件
    for t_file in tqdm(temp_files, desc=f"Merging {category}"):
        # [关键修复] 如果已经达到总限额，直接跳出循环，不再读取剩余临时文件
        if total_saved >= limit:
            break
            
        if not os.path.exists(t_file): continue
            
        with open(t_file, 'r', encoding='utf-8') as f_in:
            for line in f_in:
                # [关键修复] 双重检查：每行写入前都检查是否超限
                if total_saved >= limit:
                    break
                
                f_out.write(line)
                line_count += 1
                total_saved += 1
                
                # 切分文件
                if line_count >= lines_per_file:
                    f_out.close()
                    file_idx += 1
                    line_count = 0
                    current_out_path = os.path.join(output_dir, f"train_part_{file_idx}.jsonl")
                    f_out = open(current_out_path, 'w', encoding='utf-8')
        
        # 处理完一个临时文件，删除它
        os.remove(t_file)
        
    f_out.close()
    
    # 清理：如果最后一个文件是空的（或者因超限提前退出），删掉它
    if line_count == 0 and os.path.exists(current_out_path):
        os.remove(current_out_path)
    # 如果完全没写数据，且产生了第一个空文件
    if total_saved == 0 and os.path.exists(os.path.join(output_dir, "train_part_0.jsonl")):
         os.remove(os.path.join(output_dir, "train_part_0.jsonl"))
        
    return total_saved

def main():
    if not os.path.exists(TEMP_DIR):
        os.makedirs(TEMP_DIR)
    
    for d in OUTPUT_DIRS.values():
        if not os.path.exists(d):
            os.makedirs(d)
        
    parquet_files = glob.glob(os.path.join(DATA_DIR, "*.parquet"))
    print(f"Found {len(parquet_files)} parquet files.")
    
    # 1. 并行处理 (Map Phase)
    category_temp_files = defaultdict(list)
    global_stats = defaultdict(lambda: {'count': 0, 'discarded': 0})
    
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = {
            executor.submit(process_single_file, f, TEMP_DIR, MODEL_PATH, MAX_SEQ_LEN): f 
            for f in parquet_files
        }
        
        for future in tqdm(as_completed(futures), total=len(parquet_files), desc="Map Phase"):
            try:
                result = future.result()
                if result is None: continue
                
                temp_paths_dict, stats = result
                
                for cat, path in temp_paths_dict.items():
                    if os.path.exists(path) and os.path.getsize(path) > 0:
                        category_temp_files[cat].append(path)
                    elif os.path.exists(path):
                        os.remove(path)
                
                for cat, s in stats.items():
                    global_stats[cat]['count'] += s['count']
                    global_stats[cat]['discarded'] += s['discarded']
                    
            except Exception as e:
                print(f"Worker exception: {e}")
                continue

    print("\nMap Phase complete. Starting Reduce (Merge & Limit)...")
    
    # 2. 合并、切分与限额 (Reduce Phase)
    final_counts = {}
    
    for cat in ['Math', 'Code', 'Science']:
        temp_list = category_temp_files[cat]
        out_dir = OUTPUT_DIRS[cat]
        limit = TARGET_LIMITS.get(cat, float('inf'))
        
        if not temp_list:
            final_counts[cat] = 0
            continue
            
        saved_count = merge_and_split(cat, temp_list, out_dir, LINES_PER_FILE, limit)
        final_counts[cat] = saved_count
        
        # 如果因为限额提前结束，剩余的临时文件也要删掉，不然会占磁盘
        for remaining_file in temp_list:
            if os.path.exists(remaining_file):
                os.remove(remaining_file)

    try:
        os.rmdir(TEMP_DIR)
    except:
        pass

    # 3. 最终报告
    print("\n" + "="*65)
    print(f"{'FINAL DATASET REPORT':^65}")
    print("="*65)
    print(f"{'Category':<10} | {'Limit':<10} | {'Saved':<10} | {'Discarded(>4k)':<15}")
    print("-" * 65)
    
    for cat in ['Math', 'Code', 'Science']:
        limit = TARGET_LIMITS.get(cat, 'Inf')
        saved = final_counts.get(cat, 0)
        discarded = global_stats[cat]['discarded']
        print(f"{cat:<10} | {str(limit):<10} | {saved:<10} | {discarded:<15}")
        
    print("-" * 65)
    print("Output directories:")
    for cat, d in OUTPUT_DIRS.items():
        print(f" - {cat}: {d}")

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)
    main()