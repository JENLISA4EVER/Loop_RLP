import json
import pandas as pd
import ast
import os
from tqdm import tqdm
from datasets import load_dataset
import pyarrow as pa
import pyarrow.parquet as pq
import glob
import time
import orjson  # pip install orjson
from concurrent.futures import ThreadPoolExecutor
def write_to_parquet(json_files, parquet_dir,parquet_index):
    schema = pa.schema([
        ("difficulty", pa.string()),
        ("source", pa.string()),
        ("domain", pa.string()),
        ("conversations", pa.list_(pa.struct([
            ("role", pa.string()),
            ("content", pa.string()),
        ]))),
        ("id", pa.string()),
    ])

    os.makedirs(parquet_dir, exist_ok=True)
    if os.path.exists(os.path.join(parquet_dir, f"{parquet_index}.parquet")):
        try:
            data = pd.read_parquet(os.path.join(parquet_dir, f"{parquet_index}.parquet"))
            print(f"{len(json_files)} json files, {len(data)} rows in {parquet_index}.parquet")
            if abs(len(data) - len(json_files)) <= 5:
                print(f"json_files: {len(json_files)}")
                print(f"Skipping {parquet_index}.parquet, {len(data)} rows")
                return
        except Exception as e:
            print(f"Error reading {parquet_index}.parquet: {e}")
            print(f"Deleting {parquet_index}.parquet")
            os.remove(os.path.join(parquet_dir, f"{parquet_index}.parquet"))
            print(f"Creating {parquet_index}.parquet, {len(json_files)} rows")
    else:
        print(f"Creating {parquet_index}.parquet, {len(json_files)} rows")
    index = 0
    max_rows_per_parquet = 20000
    BATCH_SIZE = 2000  # 可以从 1w 起试，内存够就再加

    def new_writer(idx):
        path = os.path.join(parquet_dir, f"{idx}.parquet")
        return pq.ParquetWriter(
            path,
            schema,
        )

    writer = new_writer(parquet_index)
    buffer = []
    rows_in_current_file = 0

    bar = tqdm(total=len(json_files), desc=f"Merging {parquet_dir} {parquet_index}.parquet", mininterval=0.5)
    for path in json_files:
        try:
            with open(path, "r", encoding="utf-8") as f:
                obj = json.load(f)
                for key,value in obj.items():
                    if isinstance(value,int):
                        obj[key] = str(value)
                buffer.append(obj)
        except Exception as e:
            print(f"Error loading {path}: {e}")
            continue
        bar.update(1)

        if len(buffer) >= BATCH_SIZE:
            table = pa.Table.from_pylist(buffer, schema=schema)
            writer.write_table(table)
            rows_in_current_file += len(buffer)
            buffer.clear()

            # ✅ flush 后再判断是否需要轮转
            if rows_in_current_file >= max_rows_per_parquet:
                index += 1
                print(f"Flushed {parquet_index}.parquet,{rows_in_current_file} rows")
                break

    bar.close()

    if buffer:
        table = pa.Table.from_pylist(buffer, schema=schema)
        writer.write_table(table)

    writer.close()
    buffer.clear()


def merge_json_to_parquet(json_files, parquet_dir):
    schema = pa.schema([
        ("difficulty", pa.string()),
        ("source", pa.string()),
        ("domain", pa.string()),
        ("conversations", pa.list_(pa.struct([
            ("role", pa.string()),
            ("content", pa.string()),
        ]))),
        ("id", pa.string()),
    ])

    os.makedirs(parquet_dir, exist_ok=True)

    index = 0
    max_rows_per_parquet = 20000
    BATCH_SIZE = 2000  # 可以从 1w 起试，内存够就再加
    json_files_chunks = [json_files[i:i+max_rows_per_parquet] for i in range(0, len(json_files), max_rows_per_parquet)]
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = []
        print(f"Merging {parquet_dir} {len(json_files_chunks)} chunks")
        for idx, json_files_chunk in enumerate(json_files_chunks):
            futures.append(executor.submit(write_to_parquet, json_files_chunk, parquet_dir, idx))
        for future in futures:
            future.result()

def merge_json_to_parquet_old(json_files, parquet_dir):
    schema = pa.schema([
        ("difficulty", pa.string()),   # null 会自动兼容
        ("source", pa.string()),
        ("domain", pa.string()),
        (
            "conversations",
            pa.list_(
                pa.struct([
                        ("role", pa.string()),
                        ("content", pa.string()),
                    ])
                ),
            ),
        ("id", pa.string()),
    ])
    index = 0
    max_size_per_parquet = 30000
    os.makedirs(parquet_dir, exist_ok=True)
    print(f"Merging {parquet_dir}")
    writer = pq.ParquetWriter(os.path.join(parquet_dir, f"{index}.parquet"), schema)
    buffer = []
    BATCH_SIZE = 10000  # 推荐 1k~10k
    bar = tqdm(total=len(json_files), desc=f"Merging {parquet_dir}")
    t1 = time.time()
    for i, path in enumerate(json_files):
        try:
            with open(path, "r", encoding="utf-8") as f:
                obj = json.load(f)
                buffer.append(obj)
        except Exception as e:
            print(f"Error loading {path}: {e}")
            continue
        bar.update(1)
        if len(buffer) >= BATCH_SIZE:
            table = pa.Table.from_pylist(buffer, schema=schema)
            writer.write_table(table)
            buffer.clear()
            t2 = time.time()
            print(f"Time taken: {t2 - t1} seconds, batch size: {BATCH_SIZE}")
            t1 = time.time()
        if i % max_size_per_parquet == 0:
            writer.close()
            index += 1
            writer = pq.ParquetWriter(os.path.join(parquet_dir, f"{index}.parquet"), schema)
    bar.close()
    # 写剩余
    if buffer:
        table = pa.Table.from_pylist(buffer, schema=schema)
        writer.write_table(table)

    writer.close()

if __name__ == "__main__":
    TEMP_DIR = {
        "math": "/share/home/sxjiang/myproject/self-learn/temp_parts/math",
        "code": "/share/home/sxjiang/myproject/self-learn/temp_parts/code",
        "science": "/share/home/sxjiang/myproject/self-learn/temp_parts/science",
    }
    TARGET_LIMITS = {
        'math': 300000,    # Math 限制 30万
        'code': 200000,    # Code 限制 20万
        'science': 200000, # Science 限制 20万 (或者尽可能多)
    }
    # 读取每个领域的数据并保存为parquet文件
    output_dir = "/share/home/sxjiang/myproject/self-learn/processed_new"
    os.makedirs(output_dir, exist_ok=True)
    error_count = 0
    for domain in TEMP_DIR.keys():
        domain_dir = TEMP_DIR[domain]
        domain_json_files = glob.glob(os.path.join(domain_dir, "*.json"))
        domain_json_files.sort()
        domain_json_files = domain_json_files[:TARGET_LIMITS[domain]]
        print(len(domain_json_files))
        merge_json_to_parquet(domain_json_files, os.path.join(output_dir, domain))
                    
    # 合并数据:按照math:code:science=3:2:1的比例
    math_files =  glob.glob(os.path.join(TEMP_DIR["math"], "*.json"))
    code_files = glob.glob(os.path.join(TEMP_DIR["code"], "*.json"))
    science_files = glob.glob(os.path.join(TEMP_DIR["science"], "*.json"))
    merged_files = []
    math_files.sort()
    code_files.sort()
    science_files.sort()
    math_per_batch = 5
    code_per_batch = 3
    science_per_batch = 2
    math_count = 0
    code_count = 0  
    science_count = 0
    math_index = 0
    code_index = 0
    science_index = 0
    break_type = ""
    while math_count < TARGET_LIMITS['math'] and code_count < TARGET_LIMITS['code'] and science_count < TARGET_LIMITS['science']:
        for i in range(math_per_batch):
            if math_index < len(math_files):
                merged_files.append(math_files[math_index])
                math_count += 1
                math_index += 1
            if math_count >= TARGET_LIMITS['math']:
                break_type = "math"
                break
        for i in range(code_per_batch):
            if code_index < len(code_files):
                merged_files.append(code_files[code_index])
                code_count += 1
                code_index += 1
            if code_count >= TARGET_LIMITS['code']:
                break_type = "code"
                break
        for i in range(science_per_batch):
            if science_index < len(science_files):
                merged_files.append(science_files[science_index])
                science_count += 1
                science_index += 1
            if science_count >= TARGET_LIMITS['science']:
                break_type = "science"
                break
    print(f"break_type: {break_type}")
    print(f"最后组成:math:{math_count}, code:{code_count}, science:{science_count}")
    # for i in range(math_index, TARGET_LIMITS['math']):
    #     merged_files.append(math_files[i])
    # for i in range(code_index, TARGET_LIMITS['code']):
    #     merged_files.append(code_files[i])
    # for i in range(science_index, TARGET_LIMITS['science']):
    #     merged_files.append(science_files[i])
    # 先严格按照比例合并
    print(f"Merging {len(merged_files)} files")
    merge_json_to_parquet(merged_files, os.path.join(output_dir, "merged_5_3_2_train"))