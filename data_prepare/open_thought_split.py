import json
import random
import os


def set_seed(seed):
    random.seed(seed)
 
if __name__ == "__main__":
    set_seed(42)
    print("Hello, World!")
    omni_math_data_path = "/share/home/sxjiang/myproject/self-learn/datasets/Omni-MATH/test.jsonl"
    output_dir = "/share/home/sxjiang/myproject/self-learn/datasets/processed/omni_math"
    os.makedirs(output_dir, exist_ok=True)
    omni_math_data = []
    val_data_len = 200 # 和RPT保持一致
    with open(omni_math_data_path, "r") as f:
        for line in f:
            omni_math_data.append(json.loads(line))
    print(f"Original data length: {len(omni_math_data)}")
    random.shuffle(omni_math_data)
    train_data = omni_math_data[:-val_data_len]
    val_data = omni_math_data[-val_data_len:]
    print(f"Train data length: {len(train_data)}")
    print(f"Val data length: {len(val_data)}")
    with open(os.path.join(output_dir, "train.jsonl"), "w") as f:
        for data in train_data:
            f.write(json.dumps(data) + "\n")
    with open(os.path.join(output_dir, "val.jsonl"), "w") as f:
        for data in val_data:
            f.write(json.dumps(data) + "\n")