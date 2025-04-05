# export OMP_NUM_THREADS=8
# export NCCL_IB_DISABLE=0
# export NCCL_IB_GID_INDEX=3
# export NCCL_SOCKET_IFNAME=br-intranet
# export NCCL_DEBUG=INFO

# export ACCELERATE_CPU_AFFINITY=1 
# export NPROC_PER_NODE=8
# export NODES=1 
# export NODE_RANK=0 
# export MASTER_ADDR=172.17.100.112 
# export MASTER_PORT=23456 
# export RUN_NAME=llava-onevision-google-siglip-so400m-patch14-384-lmms-lab-llava-onevision-qwen2-7b-si-ewc-stage-lambda-1
# export OUTPUT_DIR=/blob/weiwei/llava_checkpoint/$RUN_NAME
# export RUN_NAME="llava-onevision-google-siglip-so400m-patch14-384-lmms-lab-llava-onevision-qwen2-7b-si-ewc-stage-lambda-1"
# export OUTPUT_DIR="/blob/weiwei/llava_checkpoint/$RUN_NAME"
# export PREV_STAGE_CHECKPOINT="lmms-lab/llava-onevision-qwen2-7b-si"
# export VISION_MODEL_VERSION="google/siglip-so400m-patch14-384"

# - export OMP_NUM_THREADS=7
export NCCL_IB_DISABLE=1
# - export NCCL_IB_GID_INDEX=3
# - export NCCL_SOCKET_IFNAME="eth0"
export NCCL_DEBUG=INFO
export ACCELERATE_CPU_AFFINITY=1 
export NPROC_PER_NODE=4
export NODES=1 
export NODE_RANK=0 
# export WORLD_SIZE=4
# - export MASTER_ADDR="10.36.38.89"
# - export MASTER_PORT="23456"

# export RUN_NAME="llava-onevision-google-siglip-so400m-patch14-384-lmms-lab-llava-onevision-qwen2-7b-si-ewc-lambda01-amazon-multisource-all"
# export OUTPUT_DIR="/blob/weiwei/llava_checkpoint/llava-onevision-google-siglip-so400m-patch14-384-lmms-lab-llava-onevision-qwen2-7b-si-ewc-lambda01-amazon-multisource-all"
# export PREV_STAGE_CHECKPOINT="lmms-lab/llava-onevision-qwen2-7b-si"

# LLM_VERSION="lmms-lab/llava-onevision-qwen2-7b-si" 
# for 7b model we recommend bs=1, accum=2, 16 nodes, 128 gpus, lr=1e-5, warmup=0.03
# for 72b model we recommend bs=1, accum=1, 32 nodes, 256 gpus, lr=1e-5, warmup=0.03
LLM_VERSION_CLEAN="${LLM_VERSION//\//_}"
VISION_MODEL_VERSION="google/siglip-so400m-patch14-384"
VISION_MODEL_VERSION_CLEAN="${VISION_MODEL_VERSION//\//_}"
# RANK=${RANK:-0}
# PORT=${PORT:-12345} 
# NUM_GPUS=${NUM_GPUS:5}
# NNODES=${NNODES:1}

############### Pretrain ################

BASE_RUN_NAME="llavanext-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-mlp2x_gelu-pretrain_blip558k_plain"
echo "BASE_RUN_NAME: ${BASE_RUN_NAME}"

############### Finetune ################

# Stage 2
PROMPT_VERSION="qwen_1_5"
# RUN_NAME="llava-onevision-${VISION_MODEL_VERSION_CLEAN}-${LLM_VERSION_CLEAN}-ewc-stage-lambda-1" 
PREV_STAGE_CHECKPOINT="lmms-lab/llava-onevision-qwen2-7b-si" # replace it with your last checkpoint training from single image collection
echo "PREV_STAGE_CHECKPOINT: ${PREV_STAGE_CHECKPOINT}"
echo "MID_RUN_NAME: ${RUN_NAME}"

# ACCELERATE_CPU_AFFINITY=1 torchrun --nproc_per_node 8 --nnodes 1 --node_rank 0 --master_addr 172.17.100.112 --master_port 23456 \
ACCELERATE_CPU_AFFINITY=1 torchrun --nproc_per_node $NPROC_PER_NODE --nnodes $NODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT \
    llava/train/train_mem.py \
    --deepspeed scripts/zero3.json \
    --lora_enable True \
    --model_name_or_path $PREV_STAGE_CHECKPOINT \
    --version $PROMPT_VERSION \
    --data_path "./msdata/split/split_small/{${TRAIN_JSON}}.json" \
    --image_folder ./msdata/images/images/ \
    --mm_vision_tower_lr=2e-6 \
    --vision_tower $VISION_MODEL_VERSION \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --group_by_modality_length True \
    --image_aspect_ratio anyres_max_9 \
    --image_grid_pinpoints  "(1x1),...,(6x6)" \
    --mm_patch_merge_type spatial_unpad \
    --bf16 True \
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
    --tf32 True \
    --model_max_length 32768 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb \
    --torch_compile True \
    --torch_compile_backend "inductor" \
    --dataloader_drop_last True \
    --frames_upbound 32 \
    --use_ewc true \
    --ewc_lambda 0.1
exit 0;


    # --mm_tunable_parts="mm_vision_tower,mm_mlp_adapter,mm_language_model" \

    # --data_path "/home/aiscuser/dataset/Amazon/{meta_All_Beauty_exist_gene_ITdata,meta_Appliances_exist_gene_ITdata,meta_Arts_Crafts_and_Sewing_exist_gene_ITdata,meta_Beauty_and_Personal_Care_exist_gene_ITdata,meta_Cell_Phones_and_Accessories_exist_gene_ITdata,meta_Digital_Music_exist_gene_ITdata,meta_Electronics_exist_gene_ITdata,meta_Gift_Cards_exist_gene_ITdata,meta_Grocery_and_Gourmet_Food_exist_gene_ITdata,meta_Health_and_Household_exist_gene_ITdata,meta_Industrial_and_Scientific_exist_gene_ITdata,meta_Magazine_Subscriptions_exist_gene_ITdata,meta_Movies_and_TV_exist_gene_ITdata,meta_Patio_Lawn_and_Garden_exist_gene_ITdata,meta_Sports_and_Outdoors_exist_gene_ITdata,meta_Subscription_Boxes_exist_gene_ITdata,meta_Toys_and_Games_exist_gene_ITdata}.json" \
    # --data_path "/home/aiscuser/dataset/Amazon/{meta_Gift_Cards_exist_gene_ITdata}.json" \


# You can delete the sdpa attn_implementation if you want to use flash attn
#     --video_folder /mnt/bn/vl-research/data/llava_video \

    # --data_path ./download_data/mydatasets/llava_onevision/llava_onevision_FigureQA_MathV360K.json \
    # --image_folder ./download_data/mydatasets/llava_onevision/images/FigureQA_MathV360K \
