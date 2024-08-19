#!/bin/bash
export HF_ENDPOINT="https://hf-mirror.com"
export JT_SYNC=1
export trace_py_var=3
MODEL_NAME="stabilityai/stable-diffusion-2-1"
BASE_INSTANCE_DIR="/media/php/code/Jittor/generater/data_B"
RESOLUTION=512
GRADIENT_ACCUMULATION_STEPS=1
LEARNING_RATE=1e-4
LR_SCHEDULER="constant"
LR_WARMUP_STEPS=0
SEED=0
GPU_COUNT=1
MAX_NUM=28

TRAIN_BATCH_SIZE=$1
MAX_TRAIN_STEPS=$2
CHECKPOINTING_STEPS=$MAX_TRAIN_STEPS
RANK=$3
GPU_VISIBLE=$4
METHOD=$5
PROMPT_CLASS_NAME=1
STYLE_PATH="checkpoint/style_${METHOD}_${TRAIN_BATCH_SIZE}_${MAX_TRAIN_STEPS}_${RANK}"
OUTPUT_DIR_PREFIX="$STYLE_PATH/style_"
OUTPUT_IMG="results/${METHOD}_${TRAIN_BATCH_SIZE}_${MAX_TRAIN_STEPS}"
echo $OUTPUT_DIR_PREFIX

for ((folder_number = 0; folder_number < $MAX_NUM; folder_number+=$GPU_COUNT)); do
    for ((gpu_id = 0; gpu_id < GPU_COUNT; gpu_id++)); do
        current_folder_number=$((folder_number + gpu_id))
        if [$current_folder_number -gt $MAX_NUM]; then
            break
        fi
        INSTANCE_DIR="${BASE_INSTANCE_DIR}/$(printf "%02d" $current_folder_number)/images"
        OUTPUT_DIR="${OUTPUT_DIR_PREFIX}$(printf "%02d" $current_folder_number)"
        CUDA_VISIBLE_DEVICES=$GPU_VISIBLE
        PROMPT=$(printf "style_%02d" $current_folder_number)
        echo "$gpu_id $folder_number $current_folder_number $INSTANCE_DIR $OUTPUT_DIR $PROMPT"
 
        COMMAND="CUDA_VISIBLE_DEVICES=$GPU_VISIBLE python train.py \
            --pretrained_model_name_or_path=$MODEL_NAME \
            --instance_data_dir=$INSTANCE_DIR \
            --output_dir=$OUTPUT_DIR \
            --prompt_class_name=$PROMPT_CLASS_NAME
            --instance_prompt=$PROMPT \
            --resolution=$RESOLUTION \
            --train_batch_size=$TRAIN_BATCH_SIZE \
            --gradient_accumulation_steps=$GRADIENT_ACCUMULATION_STEPS \
            --learning_rate=$LEARNING_RATE \
            --lr_scheduler=$LR_SCHEDULER \
            --lr_warmup_steps=$LR_WARMUP_STEPS \
            --max_train_steps=$MAX_TRAIN_STEPS \
            --seed=$SEED \
            --rank=$RANK"

        eval $COMMAND 
        # sleep 3
    done
    wait
done
