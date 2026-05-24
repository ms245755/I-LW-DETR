#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/../.." && pwd)"
# shellcheck source=../../setup/common.sh
source "$repo_root/setup/common.sh"

usage="Usage: bash demo/setup/run_one_image_observe_compare.sh <vit_only|vit_projector|vit_projector_transformer_float_head|fully_quantized> <tiny|small|medium> <float_checkpoint> <quant_checkpoint> <image> [run_dir]"

if [[ $# -lt 5 || $# -gt 6 ]]; then
  echo "$usage" >&2
  exit 2
fi

scenario="$1"
export MODEL_SIZE="$2"
float_checkpoint="$(absolute_existing_path "$3")"
quant_checkpoint="$(absolute_existing_path "$4")"
image_path="$(absolute_existing_path "$5")"
run_dir="${6:-${OBSERVATION_ROOT:-observation}/${MODEL_SIZE}_${scenario}}"
output_root="$run_dir/observe_output"
observed_root="${OBSERVED_ROOT:-$run_dir/observed_tensor}"

configure_model_size
configure_scenario
prepare_common_args

case "$scenario" in
  vit_only)
    stats_components=(vit projector)
    ;;
  vit_projector)
    stats_components=(vit projector)
    ;;
  vit_projector_transformer_float_head)
    stats_components=(vit projector transformer lwdetr)
    ;;
  fully_quantized)
    stats_components=(vit projector transformer lwdetr)
    ;;
esac

decoder_cross_attn_float="${DECODER_CROSS_ATTN_FLOAT:-false}"
decoder_cross_attn_value_per_channel="${DECODER_CROSS_ATTN_VALUE_PER_CHANNEL:-false}"
decoder_two_stage_float="${DECODER_TWO_STAGE_FLOAT:-false}"
decoder_norm_output_qact="${DECODER_NORM_OUTPUT_QACT:-false}"

decoder_cross_attn_float_args=()
if [[ "$decoder_cross_attn_float" == "1" || "$decoder_cross_attn_float" == "true" || "$decoder_cross_attn_float" == "yes" ]]; then
  decoder_cross_attn_float_args=(--decoder_cross_attn_float)
fi

decoder_cross_attn_value_per_channel_args=()
if [[ "$decoder_cross_attn_value_per_channel" == "1" || "$decoder_cross_attn_value_per_channel" == "true" || "$decoder_cross_attn_value_per_channel" == "yes" ]]; then
  decoder_cross_attn_value_per_channel_args=(--decoder_cross_attn_value_per_channel)
fi

decoder_two_stage_float_args=()
if [[ "$decoder_two_stage_float" == "1" || "$decoder_two_stage_float" == "true" || "$decoder_two_stage_float" == "yes" ]]; then
  decoder_two_stage_float_args=(--decoder_two_stage_float)
fi

decoder_norm_output_qact_args=()
if [[ "$decoder_norm_output_qact" == "1" || "$decoder_norm_output_qact" == "true" || "$decoder_norm_output_qact" == "yes" ]]; then
  decoder_norm_output_qact_args=(--decoder_norm_output_qact)
fi

common_args=(
  --encoder "$encoder"
  --vit_encoder_num_layers "$vit_encoder_num_layers"
  --window_block_indexes "${window_block_args[@]}"
  --out_feature_indexes "${out_feature_args[@]}"
  --dec_layers 3
  --group_detr 13
  --two_stage
  --projector_scale P4
  --hidden_dim 256
  --sa_nheads 8
  --ca_nheads 16
  --dec_n_points 2
  --bbox_reparam
  --lite_refpoint_refine
  --num_queries "$num_queries"
  --num_select "$num_select"
  --dataset_file coco
  --easy_quantize
  "${decoder_cross_attn_float_args[@]}"
  "${decoder_cross_attn_value_per_channel_args[@]}"
  "${decoder_two_stage_float_args[@]}"
  "${decoder_norm_output_qact_args[@]}"
  --input "$image_path"
  --confidence_threshold "${CONFIDENCE_THRESHOLD:-0.5}"
  --device "${DEVICE:-cuda}"
  --observe_stat
)

if [[ "${CLEAR_OBSERVED_TENSOR:-1}" == "1" ]]; then
  rm -rf "$observed_root"
  rm -rf observed_tensor
fi
mkdir -p "$output_root" "$observed_root"
export OBSERVED_TENSOR_ROOT="$observed_root"

cd "$repo_root"

echo "[1/4] Running float inference and saving observed tensors..."
python demo/demo.py \
  "${common_args[@]}" \
  --weights "$float_checkpoint" \
  --mode float \
  --output_dir "$output_root/float"

echo "[2/4] Running quant inference and saving observed tensors..."
python demo/demo.py \
  "${common_args[@]}" \
  --weights "$quant_checkpoint" \
  --mode quant \
  --output_dir "$output_root/quant" \
  --quant_model_components "${quant_model_component_args[@]}" \
  --backbone_quant_components "${backbone_quant_component_args[@]}" \
  "${quant_block_args[@]}" \
  --encoder_quant_components "${encoder_quant_component_args[@]}" \
  --projector_quant_components "${projector_quant_component_args[@]}" \
  --transformer_quant_components "${transformer_quant_component_args[@]}" \
  "${decoder_quant_block_args[@]}" \
  "${decoder_quant_norm_args[@]}" \
  "${decoder_self_attn_quant_component_args[@]}" \
  "${decoder_cross_attn_quant_component_args[@]}" \
  "${decoder_ffn_quant_component_args[@]}" \
  "${decoder_refpoint_quant_component_args[@]}" \
  "${decoder_two_stage_quant_component_args[@]}" \
  "${decoder_output_norm_quant_component_args[@]}" \
  --lwdetr_quant_components "${lwdetr_quant_component_args[@]}" \
  --act_observer "${ACT_OBSERVER:-mean}" \
  --act_percentile "${ACT_PERCENTILE:-99}"

echo "[3/4] Generating tensor statistics..."
python demo/generate_tensor_statistics.py \
  --components "${stats_components[@]}" \
  --observed-root "$observed_root" \
  --skip-missing

echo "[4/4] Generating summary plots..."
python demo/read_comparision_result.py \
  --components "${stats_components[@]}" \
  --observed-root "$observed_root" \
  --skip-missing

echo "Done. Visualizations: $output_root"
echo "Done. Observed tensors: $observed_root"
echo "Done. Tensor comparisons: $observed_root/comparison"
echo "Done. Metric plots: $observed_root/plots"
