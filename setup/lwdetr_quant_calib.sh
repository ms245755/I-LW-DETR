#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash setup/lwdetr_quant_calib.sh tiny /path/to/coco /path/to/checkpoint.pth
#   BACKBONE_QUANT_COMPONENTS="vit" ENCODER_QUANT_BLOCKS="0 3 5" \
#     ENCODER_QUANT_COMPONENTS="linear softmax layernorm gelu" \
#     PROJECTOR_QUANT_COMPONENTS="conv2d layernorm" \
#     bash setup/lwdetr_quant_calib.sh small /path/to/coco /path/to/checkpoint.pth

model_size="${1:-tiny}"
coco_path="${2:?Usage: bash setup/lwdetr_quant_calib.sh <tiny|small|medium> <coco_path> <checkpoint>}"
checkpoint="${3:?Usage: bash setup/lwdetr_quant_calib.sh <tiny|small|medium> <coco_path> <checkpoint>}"

backbone_quant_components="${BACKBONE_QUANT_COMPONENTS:-vit projector}"
encoder_quant_blocks="${ENCODER_QUANT_BLOCKS:-all}"
encoder_quant_components="${ENCODER_QUANT_COMPONENTS:-linear conv2d matmul softmax gelu layernorm}"
projector_quant_components="${PROJECTOR_QUANT_COMPONENTS:-conv2d gelu layernorm}"
act_observer="${ACT_OBSERVER:-percentile}"
act_percentile="${ACT_PERCENTILE:-99}"

read -r -a backbone_quant_component_args <<< "$backbone_quant_components"
component_suffix="$(printf '%s\n' "${backbone_quant_component_args[@]}" | paste -sd '_' -)"

case "$model_size" in
  tiny)
    model_name="lwdetr_tiny_coco"
    encoder="vit_tiny"
    vit_encoder_num_layers=6
    window_block_indexes="0 2 4"
    out_feature_indexes="1 3 5"
    batch_size=4
    num_select=100
    ;;
  small)
    model_name="lwdetr_small_coco"
    encoder="vit_tiny"
    vit_encoder_num_layers=10
    window_block_indexes="0 1 3 6 7 9"
    out_feature_indexes="2 4 5 9"
    batch_size=4
    num_select=300
    ;;
  medium)
    model_name="lwdetr_medium_coco"
    encoder="vit_small"
    vit_encoder_num_layers=10
    window_block_indexes="0 1 3 6 7 9"
    out_feature_indexes="2 4 5 9"
    batch_size=2
    num_select=300
    ;;
  *)
    echo "Unsupported model size: $model_size" >&2
    exit 2
    ;;
esac

quant_block_args=()
if [[ "$encoder_quant_blocks" != "all" ]]; then
  read -r -a quant_block_args <<< "$encoder_quant_blocks"
  quant_block_args=(--encoder_quant_blocks "${quant_block_args[@]}")
fi

read -r -a encoder_quant_component_args <<< "$encoder_quant_components"
read -r -a projector_quant_component_args <<< "$projector_quant_components"
read -r -a window_block_args <<< "$window_block_indexes"
read -r -a out_feature_args <<< "$out_feature_indexes"

python main.py \
    --batch_size "$batch_size" \
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
    --num_queries 100 \
    --num_select "$num_select" \
    --dataset_file coco \
    --coco_path "$coco_path" \
    --square_resize_div_64 \
    --calib --resume "$checkpoint" \
    --output_dir "output/${model_name}_quant_${component_suffix}_${model_size}"
