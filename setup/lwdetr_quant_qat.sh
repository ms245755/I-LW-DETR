#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash setup/lwdetr_quant_qat.sh tiny /path/to/coco /path/to/checkpoint.pth
#   BACKBONE_QUANT_COMPONENTS="vit" ENCODER_QUANT_BLOCKS="0 3 5" \
#     ENCODER_QUANT_COMPONENTS="linear softmax layernorm gelu" \
#     PROJECTOR_QUANT_COMPONENTS="conv2d layernorm" \
#     bash setup/lwdetr_quant_qat.sh small /path/to/coco /path/to/checkpoint.pth

model_size="${1:-tiny}"
coco_path="${2:?Usage: bash setup/lwdetr_quant_qat.sh <tiny|small|medium> <coco_path> <checkpoint>}"
checkpoint="${3:?Usage: bash setup/lwdetr_quant_qat.sh <tiny|small|medium> <coco_path> <checkpoint>}"

backbone_quant_components="${BACKBONE_QUANT_COMPONENTS:-vit projector}"
encoder_quant_blocks="${ENCODER_QUANT_BLOCKS:-all}"
encoder_quant_components="${ENCODER_QUANT_COMPONENTS:-linear conv2d matmul softmax gelu layernorm}"
projector_quant_components="${PROJECTOR_QUANT_COMPONENTS:-conv2d gelu layernorm}"
act_observer="${ACT_OBSERVER:-ema_minmax}"
act_percentile="${ACT_PERCENTILE:-99.9}"

read -r -a backbone_quant_component_args <<< "$backbone_quant_components"
component_suffix="$(printf '%s\n' "${backbone_quant_component_args[@]}" | paste -sd '_' -)"

pretrain_keys_modify_to_load=(
  transformer.enc_out_class_embed.0.weight
  transformer.enc_out_class_embed.1.weight
  transformer.enc_out_class_embed.2.weight
  transformer.enc_out_class_embed.3.weight
  transformer.enc_out_class_embed.4.weight
  transformer.enc_out_class_embed.5.weight
  transformer.enc_out_class_embed.6.weight
  transformer.enc_out_class_embed.7.weight
  transformer.enc_out_class_embed.8.weight
  transformer.enc_out_class_embed.9.weight
  transformer.enc_out_class_embed.10.weight
  transformer.enc_out_class_embed.11.weight
  transformer.enc_out_class_embed.12.weight
  transformer.enc_out_class_embed.0.bias
  transformer.enc_out_class_embed.1.bias
  transformer.enc_out_class_embed.2.bias
  transformer.enc_out_class_embed.3.bias
  transformer.enc_out_class_embed.4.bias
  transformer.enc_out_class_embed.5.bias
  transformer.enc_out_class_embed.6.bias
  transformer.enc_out_class_embed.7.bias
  transformer.enc_out_class_embed.8.bias
  transformer.enc_out_class_embed.9.bias
  transformer.enc_out_class_embed.10.bias
  transformer.enc_out_class_embed.11.bias
  transformer.enc_out_class_embed.12.bias
  class_embed.weight
  class_embed.bias
)

case "$model_size" in
  tiny)
    model_name="lwdetr_tiny_coco"
    encoder="vit_tiny"
    vit_encoder_num_layers=6
    window_block_indexes="0 2 4"
    out_feature_indexes="1 3 5"
    batch_size="${BATCH_SIZE:-1}"
    epochs="${EPOCHS:-2}"
    lr_drop="${LR_DROP:-60}"
    lr_vit_layer_decay="${LR_VIT_LAYER_DECAY:-0.8}"
    lr_component_decay="${LR_COMPONENT_DECAY:-0.7}"
    num_queries=100
    num_select=100
    nproc_per_node="${NPROC_PER_NODE:-4}"
    pretrained_encoder="${PRETRAINED_ENCODER:-pretrain_weights/caev2_tiny_300e_objects365.pth}"
    pretrain_weights="${PRETRAIN_WEIGHTS:-pretrain_weights/LWDETR_tiny_30e_objects365.pth}"
    ;;
  small)
    model_name="lwdetr_small_coco"
    encoder="vit_tiny"
    vit_encoder_num_layers=10
    window_block_indexes="0 1 3 6 7 9"
    out_feature_indexes="2 4 5 9"
    batch_size="${BATCH_SIZE:-4}"
    epochs="${EPOCHS:-2}"
    lr_drop="${LR_DROP:-60}"
    lr_vit_layer_decay="${LR_VIT_LAYER_DECAY:-0.8}"
    lr_component_decay="${LR_COMPONENT_DECAY:-0.7}"
    num_queries=300
    num_select=300
    nproc_per_node="${NPROC_PER_NODE:-8}"
    pretrained_encoder="${PRETRAINED_ENCODER:-pretrain_weights/caev2_tiny_300e_objects365.pth}"
    pretrain_weights="${PRETRAIN_WEIGHTS:-pretrain_weights/LWDETR_small_60e_coco.pth}"
    ;;
  medium)
    model_name="lwdetr_medium_coco"
    encoder="vit_small"
    vit_encoder_num_layers=10
    window_block_indexes="0 1 3 6 7 9"
    out_feature_indexes="2 4 5 9"
    batch_size="${BATCH_SIZE:-2}"
    epochs="${EPOCHS:-2}"
    lr_drop="${LR_DROP:-60}"
    lr_vit_layer_decay="${LR_VIT_LAYER_DECAY:-0.7}"
    lr_component_decay="${LR_COMPONENT_DECAY:-0.7}"
    num_queries=300
    num_select=300
    nproc_per_node="${NPROC_PER_NODE:-8}"
    pretrained_encoder="${PRETRAINED_ENCODER:-pretrain_weights/caev2_small_300e_objects365.pth}"
    pretrain_weights="${PRETRAIN_WEIGHTS:-pretrain_weights/LWDETR_medium_60e_coco.pth}"
    ;;
  *)
    echo "Unsupported model size: $model_size" >&2
    exit 2
    ;;
esac

lr="${LR:-1e-4}"
lr_encoder="${LR_ENCODER:-1.5e-4}"
weight_decay="${WEIGHT_DECAY:-1e-4}"

quant_block_args=()
if [[ "$encoder_quant_blocks" != "all" ]]; then
  read -r -a quant_block_values <<< "$encoder_quant_blocks"
  quant_block_args=(--encoder_quant_blocks "${quant_block_values[@]}")
fi

read -r -a encoder_quant_component_args <<< "$encoder_quant_components"
read -r -a projector_quant_component_args <<< "$projector_quant_components"
read -r -a window_block_args <<< "$window_block_indexes"
read -r -a out_feature_args <<< "$out_feature_indexes"

python -u -m torch.distributed.launch \
    --nproc_per_node "$nproc_per_node" \
    --use_env \
    main.py \
    --lr "$lr" \
    --lr_encoder "$lr_encoder" \
    --batch_size "$batch_size" \
    --weight_decay "$weight_decay" \
    --epochs "$epochs" \
    --lr_drop "$lr_drop" \
    --lr_vit_layer_decay "$lr_vit_layer_decay" \
    --lr_component_decay "$lr_component_decay" \
    --encoder "$encoder" \
    --vit_encoder_num_layers "$vit_encoder_num_layers" \
    --window_block_indexes "${window_block_args[@]}" \
    --out_feature_indexes "${out_feature_args[@]}" \
    --backbone_quant_components "${backbone_quant_component_args[@]}" \
    "${quant_block_args[@]}" \
    --encoder_quant_components "${encoder_quant_component_args[@]}" \
    --projector_quant_components "${projector_quant_component_args[@]}" \
    --act_observer "$act_observer" \
    --act_percentile "$act_percentile" \
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
    --num_queries "$num_queries" \
    --ia_bce_loss \
    --cls_loss_coef 1 \
    --num_select "$num_select" \
    --dataset_file coco \
    --coco_path "$coco_path" \
    --square_resize_div_64 \
    --pretrained_encoder "$pretrained_encoder" \
    --pretrain_weights "$pretrain_weights" \
    --resume "$checkpoint" \
    --qat \
    --pretrain_keys_modify_to_load "${pretrain_keys_modify_to_load[@]}" \
    --output_dir "output/${model_name}_quant_${component_suffix}_${model_size}"
