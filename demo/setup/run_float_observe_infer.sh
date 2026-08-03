#!/usr/bin/env bash
set -euo pipefail

usage="Usage: bash demo/setup/run_float_observe_infer.sh <tiny|small|medium> <checkpoint> <image> <output_dir> [observed_root]"

model_size="${1:?$usage}"
checkpoint="${2:?$usage}"
image_path="${3:?$usage}"
output_dir="${4:?$usage}"
observed_root="${5:-observed_tensor}"

case "$model_size" in
  tiny)
    encoder="vit_tiny"
    vit_encoder_num_layers=6
    window_block_indexes=(0 2 4)
    out_feature_indexes=(1 3 5)
    num_select=100
    num_queries=(--num_queries 100)
    ;;
  small)
    encoder="vit_tiny"
    vit_encoder_num_layers=10
    window_block_indexes=(0 1 3 6 7 9)
    out_feature_indexes=(2 4 5 9)
    num_select=300
    num_queries=()
    ;;
  medium)
    encoder="vit_small"
    vit_encoder_num_layers=10
    window_block_indexes=(0 1 3 6 7 9)
    out_feature_indexes=(2 4 5 9)
    num_select=300
    num_queries=()
    ;;
  *)
    echo "Unsupported model size: $model_size. Use tiny, small, or medium." >&2
    exit 2
    ;;
esac

export OBSERVED_TENSOR_ROOT="$observed_root"

python demo/demo.py \
  --encoder "$encoder" \
  --vit_encoder_num_layers "$vit_encoder_num_layers" \
  --window_block_indexes "${window_block_indexes[@]}" \
  --out_feature_indexes "${out_feature_indexes[@]}" \
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
  --num_select "$num_select" \
  "${num_queries[@]}" \
  --weights "$checkpoint" \
  --input "$image_path" \
  --output_dir "$output_dir" \
  --observe_stat \
  --mode float \
  --confidence_threshold "${CONFIDENCE_THRESHOLD:-0.5}" \
  --device "${DEVICE:-cuda}"
