#!/usr/bin/env bash
set -euo pipefail

usage="Usage: bash demo/setup/run_quant_observe_infer.sh <tiny|small|medium> <checkpoint> <image> <output_dir> [observed_root]"

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

read -r -a quant_model_component_args <<< "${QUANT_MODEL_COMPONENTS:-backbone transformer}"
read -r -a backbone_quant_component_args <<< "${BACKBONE_QUANT_COMPONENTS:-vit projector}"
read -r -a encoder_quant_component_args <<< "${ENCODER_QUANT_COMPONENTS:-linear conv2d matmul softmax gelu layernorm}"
read -r -a projector_quant_component_args <<< "${PROJECTOR_QUANT_COMPONENTS:-conv2d gelu silu layernorm}"
read -r -a transformer_quant_component_args <<< "${TRANSFORMER_QUANT_COMPONENTS:-linear matmul softmax layernorm deformable_attention}"

easy_quantize_args=()
for component in "${quant_model_component_args[@]}"; do
  if [[ "$component" == "transformer" ]]; then
    easy_quantize_args=(--easy_quantize)
    break
  fi
done

decoder_two_stage_float_args=()
if [[ "${DECODER_TWO_STAGE_FLOAT:-false}" == "1" || "${DECODER_TWO_STAGE_FLOAT:-false}" == "true" || "${DECODER_TWO_STAGE_FLOAT:-false}" == "yes" ]]; then
  decoder_two_stage_float_args=(--decoder_two_stage_float)
fi

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
  "${easy_quantize_args[@]}" \
  "${decoder_two_stage_float_args[@]}" \
  --mode quant \
  --quant_model_components "${quant_model_component_args[@]}" \
  --backbone_quant_components "${backbone_quant_component_args[@]}" \
  --encoder_quant_components "${encoder_quant_component_args[@]}" \
  --projector_quant_components "${projector_quant_component_args[@]}" \
  --transformer_quant_components "${transformer_quant_component_args[@]}" \
  --act_observer "${ACT_OBSERVER:-mean}" \
  --act_percentile "${ACT_PERCENTILE:-99}" \
  --confidence_threshold "${CONFIDENCE_THRESHOLD:-0.5}" \
  --device "${DEVICE:-cuda}"
