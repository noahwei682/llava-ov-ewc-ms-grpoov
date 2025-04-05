#!/bin/bash

# Environment variables
export NCCL_IB_DISABLE=1
export NCCL_DEBUG=INFO
export ACCELERATE_CPU_AFFINITY=1 
export NPROC_PER_NODE=4
export NODES=1 
export NODE_RANK=0 

# Model and checkpoint settings
LLM_VERSION="lmms-lab/llava-onevision-qwen2-7b-si"
LLM_VERSION_CLEAN="${LLM_VERSION//\//_}"
VISION_MODEL_VERSION="google/siglip-so400m-patch14-384"
VISION_MODEL_VERSION_CLEAN="${VISION_MODEL_VERSION//\//_}"

# Training settings
RUN_NAME="llava-onevision-${VISION_MODEL_VERSION_CLEAN}-${LLM_VERSION_CLEAN}-grpo-r1-onevision"
OUTPUT_DIR="/blob/weiwei/llava_checkpoint/${RUN_NAME}"
PREV_STAGE_CHECKPOINT="lmms-lab/llava-onevision-qwen2-7b-si"

# Run the training script
ACCELERATE_CPU_AFFINITY=1 torchrun --nproc_per_node $NPROC_PER_NODE --nnodes $NODES --node_rank $NODE_RANK \
    llava/train/train_grpo.py \
    --lora_enable \
    --deepspeed scripts/zero3.json \
    --model_name_or_path $PREV_STAGE_CHECKPOINT \
    --version "qwen_1_5" \
    --data_path "/home/aiscuser/lmms-eval/llava-ov-ewc-ms/msdata/onevision_r1/VizWiz_MathV360K/onevision_r1_VizWiz_MathV360K.json" \
    --image_folder "/home/aiscuser/lmms-eval/llava-ov-ewc-ms/msdata/onevision_r1/images/VizWiz_MathV360K/" \
    --mm_vision_tower_lr=2e-6 \
    --vision_tower $VISION_MODEL_VERSION \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end \
    --mm_use_im_patch_token \
    --group_by_modality_length \
    --image_aspect_ratio anyres_max_9 \
    --image_grid_pinpoints "(1x1),...,(6x6)" \
    --mm_patch_merge_type spatial_unpad \
    --bf16 \
    --run_name $RUN_NAME \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 300 \
    --save_total_limit 1 \
    --learning_rate 1e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 \
    --model_max_length 32768 \
    --gradient_checkpointing \
    --dataloader_num_workers 4 \
    --lazy_preprocess \
    --report_to wandb \
    --torch_compile \
    --torch_compile_backend "inductor" \
    --dataloader_drop_last \
    --frames_upbound 32 \
    --use_ewc false \
    --ewc_lambda 0.1 \
    --grpo_num_generations 2 \
    --grpo_max_prompt_length 256 \
    --grpo_max_completion_length 300 \
    --grpo_learning_rate 5e-6 \
    --grpo_weight_decay 0.1 \
    --grpo_warmup_ratio 0.1 \
    --grpo_lr_scheduler_type "cosine" \
    --grpo_max_grad_norm 0.1 \
    --grpo_report_to "tensorboard" \
    --grpo_ddp_find_unused_parameters 
