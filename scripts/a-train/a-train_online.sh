#!/bin/bash

source ~/.bashrc
source ~/miniconda3/bin/activate
conda activate ouro3
cd /share/home/sxjiang/myproject/self-learn/ #NEED_CHANGE
LOG_DIR=/share/home/sxjiang/myproject/self-learn/logs/train/ #NEED_CHANGE
TIME=$(date +%Y%m%d-%H-%M-%S)
export WANDB_MODE=offline
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 #NEED_CHANGE
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export TORCH_NCCL_AVOID_RECORD_STREAMS="1"
export PYTHONPATH=/share/home/sxjiang/myproject/self-learn:$PYTHONPATH
RUN_NAME=ouro_rlp_checkpoints_bf16_bz8_no_kl_debug_v2_test_env
OUTPUT_DIR=/share/home/sxjiang/myproject/self-learn/checkpoints/ #NEED_CHANGE
mkdir -p $LOG_DIR/$RUN_NAME
accelerate launch train_ouro_rlp_acc_sx_v2.py\
    --config-path /share/home/sxjiang/myproject/self-learn/configs/train\
    --config-name ouro_rlp_acc_omnimathntp\
    num_rollouts=8\
    output_dir=${OUTPUT_DIR}/${RUN_NAME} \
    experiment_name=${RUN_NAME} \
    gradient_accumulation_steps=1 \
    val_check_interval=10\
    save_interval=20\
    online_entropy_threshold_mask=true\
    resume_from_checkpoint=false\
    kl_coef=0.0001 \
    truncation=right\
    num_epochs=3\
    disable_cast=true\
    dataset_name=OmniMathDataset\
    train_data_path=/share/home/sxjiang/myproject/self-learn/datasets/processed/omni_math/train.jsonl \
    val_data_path=/share/home/sxjiang/myproject/self-learn/datasets/processed/omni_math/val.jsonl \
    batch_size=1 \
    max_length=4096 \
    max_length_method=group_max\
    num_workers=4 | tee $LOG_DIR/$RUN_NAME/${TIME}.log
