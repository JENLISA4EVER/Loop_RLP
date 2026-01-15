import os
import glob
import pandas as pd
import numpy as np
from transformers import AutoTokenizer
import pyarrow.parquet as pq

# 配置
DATA_DIR = "/home/gtang/self-learn/datasets/OpenThoughts3-1.2M/data"
MODEL_PATH = "/home/gtang/pretrain_model/Ouro-1.4B"

def format_conversation(conversations):
    text = ""
    for turn in conversations:
        role = turn.get('from', '') or turn.get('role', '')
        content = turn.get('value', '') or turn.get('content', '')
        if role in ['human', 'user']:
            role_tag = 'user'
        elif role in ['gpt', 'assistant', 'model']:
            role_tag = 'assistant'
        else:
            continue
        text += f"<|im_start|>{role_tag}\n{content}<|im_end|>\n"
    return text + "<|im_end|>"

def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    # 只读第一个文件做采样分析
    p_file = glob.glob(os.path.join(DATA_DIR, "*.parquet"))[0]
    print(f"Analyzing lengths in: {os.path.basename(p_file)}...")

    lengths = []
    # 读 2000 条看看分布
    parquet_file = pq.ParquetFile(p_file)
    for batch in parquet_file.iter_batches(batch_size=2000):
        df = batch.to_pandas()
        for _, row in df.iterrows():
            convs = row.get('conversations')
                # 判空检查 (必须做，否则报错)
            if convs is None or len(convs) == 0:
                continue
            full_text = format_conversation(row['conversations'])
            lengths.append(len(tokenizer.encode(full_text, add_special_tokens=False)))
        break # 只看第一批 2000 条足够了

    lengths = np.array(lengths)
    print("\n=== Length Distribution Report ===")
    print(f"Total Samples: {len(lengths)}")
    print(f"Min: {np.min(lengths)}")
    print(f"Avg: {np.mean(lengths):.1f}")
    print(f"Max: {np.max(lengths)}")
    print("-" * 30)
    print(f"P50 (Median): {np.percentile(lengths, 50):.0f}")
    print(f"P80: {np.percentile(lengths, 80):.0f}")
    print(f"P90: {np.percentile(lengths, 90):.0f}")
    print(f"P95: {np.percentile(lengths, 95):.0f}")
    print(f"P99: {np.percentile(lengths, 99):.0f}")
    print("-" * 30)
    print(f"Count < 4096:  {np.sum(lengths < 4096)} ({np.mean(lengths < 4096)*100:.1f}%)")
    print(f"Count < 8192:  {np.sum(lengths < 8192)} ({np.mean(lengths < 8192)*100:.1f}%)")
    print(f"Count < 16384: {np.sum(lengths < 16384)} ({np.mean(lengths < 16384)*100:.1f}%)")

if __name__ == "__main__":
    main()