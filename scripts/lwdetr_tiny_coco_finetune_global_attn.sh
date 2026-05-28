#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: bash $0 /path/to/coco [checkpoint.pth] [output_dir]" >&2
    exit 1
fi

coco_path="$1"
checkpoint="${2:-pretrain_weights/LWDETR_tiny_60e_coco.pth}"
output_dir="${3:-output/lwdetr_tiny_coco_global_attn_ft}"

nproc_per_node="${NPROC_PER_NODE:-4}"
batch_size="${BATCH_SIZE:-1}"
epochs="${EPOCHS:-12}"
lr_drop="${LR_DROP:-10}"
lr="${LR:-1e-5}"
lr_encoder="${LR_ENCODER:-1.5e-5}"

python -u -m torch.distributed.launch \
    --nproc_per_node="${nproc_per_node}" \
    --use_env \
    main.py \
    --lr "${lr}" \
    --lr_encoder "${lr_encoder}" \
    --batch_size "${batch_size}" \
    --weight_decay 1e-4 \
    --epochs "${epochs}" \
    --lr_drop "${lr_drop}" \
    --lr_vit_layer_decay 0.8 \
    --lr_component_decay 0.7 \
    --encoder vit_tiny \
    --vit_encoder_num_layers 6 \
    --window_block_indexes -1 \
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
    --coco_path "${coco_path}" \
    --square_resize_div_64 \
    --use_ema \
    --pretrain_weights "${checkpoint}" \
    --output_dir "${output_dir}"
