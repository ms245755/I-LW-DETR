#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/../.." && pwd)"
# shellcheck source=../../setup/common.sh
source "$repo_root/setup/common.sh"

usage="Usage: bash demo/setup/run_one_image_quant_infer.sh <vit_only|vit_projector|vit_projector_transformer_float_head|fully_quantized> <tiny|small|medium> <checkpoint> <image> <output_dir> [observed_root]"

if [[ $# -lt 5 || $# -gt 6 ]]; then
  echo "$usage" >&2
  exit 2
fi

scenario="$1"
export MODEL_SIZE="$2"
checkpoint="$(absolute_existing_path "$3")"
image_path="$(absolute_existing_path "$4")"
output_dir="$5"
observed_root="${6:-observed_tensor}"

configure_model_size
configure_scenario
prepare_common_args

decoder_cross_attn_float="${DECODER_CROSS_ATTN_FLOAT:-false}"
decoder_cross_attn_value_per_channel="${DECODER_CROSS_ATTN_VALUE_PER_CHANNEL:-false}"
decoder_norm_output_qact="${DECODER_NORM_OUTPUT_QACT:-false}"

decoder_cross_attn_float_args=()
if [[ "$decoder_cross_attn_float" == "1" || "$decoder_cross_attn_float" == "true" || "$decoder_cross_attn_float" == "yes" ]]; then
  decoder_cross_attn_float_args=(--decoder_cross_attn_float)
fi

decoder_cross_attn_value_per_channel_args=()
if [[ "$decoder_cross_attn_value_per_channel" == "1" || "$decoder_cross_attn_value_per_channel" == "true" || "$decoder_cross_attn_value_per_channel" == "yes" ]]; then
  decoder_cross_attn_value_per_channel_args=(--decoder_cross_attn_value_per_channel)
fi

decoder_norm_output_qact_args=()
if [[ "$decoder_norm_output_qact" == "1" || "$decoder_norm_output_qact" == "true" || "$decoder_norm_output_qact" == "yes" ]]; then
  decoder_norm_output_qact_args=(--decoder_norm_output_qact)
fi

observe_args=()
if [[ "${OBSERVE_STAT:-0}" == "1" || "${OBSERVE_STAT:-false}" == "true" || "${OBSERVE_STAT:-false}" == "yes" ]]; then
  observe_args=(--observe_stat)
fi

export OBSERVED_TENSOR_ROOT="$observed_root"

cd "$repo_root"

python demo/demo.py \
  --encoder "$encoder" \
  --vit_encoder_num_layers "$vit_encoder_num_layers" \
  --window_block_indexes "${window_block_args[@]}" \
  --out_feature_indexes "${out_feature_args[@]}" \
  --quant_model_components "${quant_model_component_args[@]}" \
  --backbone_quant_components "${backbone_quant_component_args[@]}" \
  "${quant_block_args[@]}" \
  --encoder_quant_components "${encoder_quant_component_args[@]}" \
  --projector_quant_components "${projector_quant_component_args[@]}" \
  --transformer_quant_components "${transformer_quant_component_args[@]}" \
  --lwdetr_quant_components "${lwdetr_quant_component_args[@]}" \
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
  --num_select "$num_select" \
  --dataset_file coco \
  --easy_quantize \
  "${decoder_cross_attn_float_args[@]}" \
  "${decoder_cross_attn_value_per_channel_args[@]}" \
  "${decoder_norm_output_qact_args[@]}" \
  --weights "$checkpoint" \
  --input "$image_path" \
  --output_dir "$output_dir" \
  "${observe_args[@]}" \
  --mode quant \
  --act_observer "${ACT_OBSERVER:-mean}" \
  --act_percentile "${ACT_PERCENTILE:-99}" \
  --confidence_threshold "${CONFIDENCE_THRESHOLD:-0.5}" \
  --device "${DEVICE:-cuda}"
