#!/bin/bash

# 1. 环境设置
source ~/.bashrc
source ~/miniconda3/bin/activate
conda activate ouro
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTHONPATH=/share/home/sxjiang/myproject/self-learn:$PYTHONPATH
TIME=$(date +%Y-%m-%d-%H-%M-%S)
CONFIG_FILE="/share/home/sxjiang/myproject/self-learn/configs/lm-eval/default.yaml"
# 2. 输出设置
logs_dir="/share/home/sxjiang/myproject/self-learn/logs/lm-evaluation-harness/mmlu_pro" #NEED_CHANGE
OUTPUT_DIR="/share/home/sxjiang/myproject/self-learn/results/mmlu_pro/fp16_flash_attn_2/" #NEED_CHANGE
export OUTPUT_EXIT_STEPS_DIR=$OUTPUT_DIR

# 3. 实验配置 
# 3.1 通用设置
TASKS="mmlu_pro" #NEED_CHANGE
BATCH_SIZE=4 #NEED_CHANGE
NUM_FEWSHOT=5 #NEED_CHANGE
mkdir -p $logs_dir/$MODEL_NAME
#备注:最终的数据文件在 OUTPUT_DIR 目录下,文件夹名为配置的 EXPERIMENT_NAME
# 3.2 分别进行实验

# (1) 自适应退出:ouro-1.4b-peak
MODEL_NAME="ouro-1.4b-peak" #NEED_CHANGE
MODEL_PATH="/share/home/sxjiang/model/Ouro-1.4B" #NEED_CHANGE
EXIT_THRESHOLD=1.0 #NEED_CHANGE 
EXIT_STEP=None #NEED_CHANGE
mkdir -p $logs_dir/$MODEL_NAME
#备注:最终的数据文件在 OUTPUT_DIR 目录下,文件夹名为配置的 EXPERIMENT_NAME
accelerate launch -m lm_eval\
    --model hf-ouro \
    --tasks $TASKS \
    --batch_size $BATCH_SIZE\
    --output_path $OUTPUT_DIR \
    --num_fewshot $NUM_FEWSHOT \
    --model_args pretrained=$MODEL_PATH,dtype=float16,early_exit_threshold=$EXIT_THRESHOLD,early_exit_step=$EXIT_STEP,model_name=$MODEL_NAME,attn_implementation=flash_attention_2,mixed_precision_dtype=float16\
    --config $CONFIG_FILE\
| tee $logs_dir/$MODEL_NAME/$TIME.log


# (2) 自适应退出+RLP:ouro-1.4b-rlp-peak
MODEL_NAME="ouro-1.4b-rlp-peak" #NEED_CHANGE
MODEL_PATH="/share/home/sxjiang/myproject/self-learn/checkpoints/1.4b_bf16_bz8_kl0001/epoch_2" #NEED_CHANGE
EXIT_THRESHOLD=1.0 #NEED_CHANGE 
EXIT_STEP=None #NEED_CHANGE
mkdir -p $logs_dir/$MODEL_NAME
#备注:最终的数据文件在 OUTPUT_DIR/MODEL_NAME目录下
accelerate launch -m lm_eval\
    --model hf-ouro \
    --tasks $TASKS \
    --batch_size $BATCH_SIZE\
    --output_path $OUTPUT_DIR \
    --num_fewshot $NUM_FEWSHOT \
    --model_args pretrained=$MODEL_PATH,dtype=float16,early_exit_threshold=$EXIT_THRESHOLD,early_exit_step=$EXIT_STEP,model_name=$MODEL_NAME,attn_implementation=flash_attention_2,mixed_precision_dtype=float16\
    --config $CONFIG_FILE\
| tee $logs_dir/$MODEL_NAME/$TIME.log

# (3) peak退出:ouro-1.4b-adaptive-exit-0.7
MODEL_NAME="ouro-1.4b-adaptive-exit-0.7" #NEED_CHANGE
MODEL_PATH="/share/home/sxjiang/model/Ouro-1.4B" #NEED_CHANGE
EXIT_THRESHOLD=0.7 #NEED_CHANGE 
EXIT_STEP=None #NEED_CHANGE
mkdir -p $logs_dir/$MODEL_NAME
#备注:最终的数据文件在 OUTPUT_DIR/MODEL_NAME目录下
accelerate launch -m lm_eval\
    --model hf-ouro \
    --tasks $TASKS \
    --batch_size $BATCH_SIZE\
    --output_path $OUTPUT_DIR \
    --num_fewshot $NUM_FEWSHOT \
    --model_args pretrained=$MODEL_PATH,dtype=float16,early_exit_threshold=$EXIT_THRESHOLD,early_exit_step=$EXIT_STEP,model_name=$MODEL_NAME,attn_implementation=flash_attention_2,mixed_precision_dtype=float16\
    --config $CONFIG_FILE\
| tee $logs_dir/$MODEL_NAME/$TIME.log


# (4) peak退出+RLP:ouro-1.4b-rlp-adaptive-exit-0.7
MODEL_NAME="ouro-1.4b-rlp-adaptive-exit-0.7" #NEED_CHANGE
MODEL_PATH="/share/home/sxjiang/myproject/self-learn/checkpoints/1.4b_bf16_bz8_kl0001/epoch_2" #NEED_CHANGE
EXIT_THRESHOLD=0.7 #NEED_CHANGE 
EXIT_STEP=None #NEED_CHANGE
mkdir -p $logs_dir/$MODEL_NAME
#备注:最终的数据文件在 OUTPUT_DIR/MODEL_NAME目录下
accelerate launch -m lm_eval\
    --model hf-ouro \
    --tasks $TASKS \
    --batch_size $BATCH_SIZE\
    --output_path $OUTPUT_DIR \
    --num_fewshot $NUM_FEWSHOT \
    --model_args pretrained=$MODEL_PATH,dtype=float16,early_exit_threshold=$EXIT_THRESHOLD,early_exit_step=$EXIT_STEP,model_name=$MODEL_NAME,attn_implementation=flash_attention_2,mixed_precision_dtype=float16\
    --config $CONFIG_FILE\
| tee $logs_dir/$MODEL_NAME/$TIME.log