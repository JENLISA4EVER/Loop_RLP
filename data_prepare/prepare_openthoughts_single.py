import os
import glob
import json
import gc
import pandas as pd
import pyarrow.parquet as pq
from transformers import AutoTokenizer
from tqdm import tqdm
from collections import defaultdict

# ================= 配置区 =================
# 输入数据路径 (Parquet文件夹)
DATA_DIR = "/share/home/sxjiang/myproject/self-learn/datasets/OpenThoughts3-1.2M/data"
# 输出文件路径
OUTPUT_FILE = "/share/home/sxjiang/myproject/self-learn/datasets/OpenThoughts3-1.2M/processed_ouro_train_4k.jsonl"
# 模型路径 (用于计算Token长度)
MODEL_PATH = "/share/home/sxjiang/model/Ouro-1.4B"

# 最大长度 (超过此长度直接丢弃)
MAX_SEQ_LEN = 4096 

# 采样配额 (根据您的训练配比需求)
TARGET_LIMITS = {
    'Math': 300000,   
    'Code': 200000,
    'Science': 200000,
    'General': 100000
}
# =========================================

def normalize_domain(raw_domain):
    """
    将数据集中的 domain (如 'code', 'math') 映射到四大类
    """
    if not isinstance(raw_domain, str):
        return 'General'
    
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
    
    # 4. 默认归为 General
    return 'General'

def format_conversation(conversations):
    """
    将 OpenThoughts 的 conversations 列表转换为 Ouro/Qwen 格式的纯文本。
    关键修复：处理 'from': 'human' 的情况。
    """
    text = ""
    
    for turn in conversations:
        # 兼容处理：获取角色和内容
        # 数据集里主要是 'from' 和 'value'
        role_raw = turn.get('from', '') or turn.get('role', '')
        content = turn.get('value', '') or turn.get('content', '')
        
        if content is None: 
            content = ""
        
        # --- 角色映射 (关键修复) ---
        # OpenThoughts 可能使用 'human' 代表用户
        if role_raw in ['human', 'user']:
            role_tag = 'user'
        elif role_raw in ['gpt', 'assistant', 'model']:
            role_tag = 'assistant'
        elif role_raw in ['system']:
            role_tag = 'system'
        else:
            # 遇到未知角色，跳过或默认处理
            continue
            
        # 拼接 ChatML 格式 (或您模型支持的其他格式)
        text += f"<|im_start|>{role_tag}\n{content}<|im_end|>\n"
            
    return text + "<|im_end|>"

def main():
    print(f"Loading tokenizer from {MODEL_PATH}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    
    parquet_files = glob.glob(os.path.join(DATA_DIR, "*.parquet"))
    print(f"Found {len(parquet_files)} parquet files. Processing...")
    
    # 统计计数器
    stats = defaultdict(lambda: {'count': 0, 'total_len': 0, 'discarded': 0})
    current_counts = defaultdict(int)
    
    # 打开输出文件
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f_out:
        for p_idx, p_file in enumerate(parquet_files):
            file_name = os.path.basename(p_file)
            print(f"[{p_idx+1}/{len(parquet_files)}] Streaming {file_name}...")
            
            try:
                # 使用 pyarrow 流式读取，避免 OOM
                parquet_file = pq.ParquetFile(p_file)
                
                # 每次只读 2000 行，显存占用极小
                for batch in parquet_file.iter_batches(batch_size=1000):
                    df_batch = batch.to_pandas()
                    
                    # 遍历当前 Batch
                    for _, row in df_batch.iterrows():
                        
                        # 1. 处理 Domain
                        raw_domain = row['domain'] if 'domain' in row else 'unknown'
                        category = normalize_domain(raw_domain)
                        
                        # 2. 检查配额是否已满
                        if current_counts[category] >= TARGET_LIMITS.get(category, float('inf')):
                            continue
                        
                        # 3. 提取对话
                        convs = row.get('conversations')
                        # 判空检查 (必须做，否则报错)
                        if convs is None or len(convs) == 0:
                            continue
                        
                        # 4. 格式化文本
                        full_text = format_conversation(convs)
                        
                        # 如果格式化后是空的（例如没匹配到角色），跳过
                        if not full_text.strip():
                            continue

                        # 5. 长度过滤
                        token_ids = tokenizer.encode(full_text, add_special_tokens=False)
                        seq_len = len(token_ids)
                        
                        if seq_len > MAX_SEQ_LEN:
                            stats[category]['discarded'] += 1
                            continue
                        
                        # 6. 写入 JSONL
                        record = {
                            "text": full_text,
                            "category": category,
                            "original_domain": str(raw_domain), # 确保转为字符串
                            "length": seq_len
                        }
                        f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
                        
                        # 更新统计
                        stats[category]['count'] += 1
                        stats[category]['total_len'] += seq_len
                        current_counts[category] += 1
                    
                    # [内存管理] 处理完一个 Batch 后手动清理
                    del df_batch
                
                # 处理完一个文件后，强制垃圾回收
                gc.collect()

            except Exception as e:
                print(f"Error reading {file_name}: {e}")
                continue

    # ================= 打印最终统计报表 =================
    print("\n" + "="*80)
    print(f"{'DATASET STATISTICS REPORT':^80}")
    print("="*80)
    print(f"{'Category':<15} | {'Count':<10} | {'Avg Len':<10} | {'Discarded(>4k)':<15}")
    print("-" * 80)
    
    for cat in ['Math', 'Code', 'Science', 'General']:
        s = stats[cat]
        c = s['count']
        avg = s['total_len'] / c if c > 0 else 0
        print(f"{cat:<15} | {c:<10} | {avg:<10.0f} | {s['discarded']:<15}")
        
    print("-" * 80)
    print(f"Output saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()