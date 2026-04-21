#!/usr/bin/env bash

# Specify resource requirements
#SBATCH --time=2:00:00               # Job time limit (2 hours, change as necessary)
#SBATCH --partition=gpu40G           # Partition for GPU (adjust based on your cluster's partition)
#SBATCH --nodes=1                     # Number of nodes
#SBATCH --ntasks-per-node=1           # Number of tasks per node
#SBATCH --cpus-per-task=8             # Number of CPU cores per task
#SBATCH --mem=64G                     # Memory allocation for the job
#SBATCH --gres=gpu:8                  # 8 GPUs per node (adjust based on how many GPUs you need)
#SBATCH --output=./slurms_logs/R-%j-%x.out

# Configure environment variables
source /home/users/tle/miniforge3/etc/profile.d/conda.sh  # Activate conda
conda init
conda activate lwdetr                       # Activate the environment

# Set the number of GPUs and processes per node
export NPROC_PER_NODE=8  
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
# export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256
model_name = 'lwdetr_tiny_coco_quant'
# Run the distributed training job with `torch.distributed.launch`
srun python -u -m torch.distributed.launch \
    --nproc_per_node=4 \
    --use_env \
    main.py \
    --lr 1e-4 \
    --lr_encoder 1.5e-4 \
    --batch_size 2 \
    --weight_decay 1e-4 \
    --epochs 20 \
    --lr_drop 10 \
    --lr_vit_layer_decay 0.8 \
    --lr_component_decay 0.7 \
    --encoder vit_tiny \
    --vit_encoder_num_layers 6 \
    --window_block_indexes 0 2 4 \
    --out_feature_indexes 1 3 5 \
    --dec_layers 3 \
    --group_detr 13 \
    --two_stage \
    --projector_scale P4 \
    --hidden_dim 256 \
    --sa_nheads 8 \
    --ca_nheads 16 \
    --dec_n_points 2 \
    --bbox_reparam \
    --lite_refpoint_refine \
    --num_queries 100 \
    --ia_bce_loss \
    --cls_loss_coef 1 \
    --num_select 100 \
    --dataset_file coco \
    --coco_path "../../../../../data/dataset/coco" \
    --square_resize_div_64 \
    --pretrained_encoder pretrain_weights/caev2_tiny_300e_objects365.pth \
    --pretrain_weights pretrain_weights/LWDETR_tiny_30e_objects365.pth \
    --resume pretrain_weights/LWDETR_tiny_60e_coco.pth \
    --qat \
    --pretrain_keys_modify_to_load transformer.enc_out_class_embed.0.weight transformer.enc_out_class_embed.1.weight transformer.enc_out_class_embed.2.weight transformer.enc_out_class_embed.3.weight transformer.enc_out_class_embed.4.weight transformer.enc_out_class_embed.5.weight transformer.enc_out_class_embed.6.weight transformer.enc_out_class_embed.7.weight transformer.enc_out_class_embed.8.weight transformer.enc_out_class_embed.9.weight transformer.enc_out_class_embed.10.weight transformer.enc_out_class_embed.11.weight transformer.enc_out_class_embed.12.weight transformer.enc_out_class_embed.0.bias transformer.enc_out_class_embed.1.bias transformer.enc_out_class_embed.2.bias transformer.enc_out_class_embed.3.bias transformer.enc_out_class_embed.4.bias transformer.enc_out_class_embed.5.bias transformer.enc_out_class_embed.6.bias transformer.enc_out_class_embed.7.bias transformer.enc_out_class_embed.8.bias transformer.enc_out_class_embed.9.bias transformer.enc_out_class_embed.10.bias transformer.enc_out_class_embed.11.bias transformer.enc_out_class_embed.12.bias class_embed.weight class_embed.bias \
    --output_dir output/$model_name