#!/usr/bin/env bash
set -euo pipefail

# Run float and quant demo inference with --observe_stat, then generate
# tensor statistics and summary plots.
#
# Usage:
#   bash demo/run_observe_compare.sh <vit|vit_projector|transformer|full> <tiny|small|medium> <float_checkpoint> <quant_checkpoint> <image> [run_dir]
#
# Examples:
#   bash demo/run_observe_compare.sh vit tiny pretrain_weights/lwdetr_tiny.pth output/lwdetr_tiny_quant/checkpoint.pth demo/catdog.jpg
#   bash demo/run_observe_compare.sh vit_projector tiny pretrain_weights/lwdetr_tiny.pth output/lwdetr_tiny_quant/checkpoint.pth demo/catdog.jpg
#   bash demo/run_observe_compare.sh full tiny pretrain_weights/lwdetr_tiny.pth output/lwdetr_tiny_quant/checkpoint.pth demo/catdog.jpg

usage="Usage: bash demo/run_observe_compare.sh <vit|vit_projector|transformer|full> <tiny|small|medium> <float_checkpoint> <quant_checkpoint> <image> [run_dir]"

quant_scope="${1:?$usage}"
model_size="${2:?$usage}"
float_checkpoint="${3:?$usage}"
quant_checkpoint="${4:?$usage}"
image_path="${5:?$usage}"
run_root="${6:-${OBSERVATION_ROOT:-observation}/${model_size}_${quant_scope}}"
output_root="$run_root/observe_output"
observed_root="${OBSERVED_ROOT:-$run_root/observed_tensor}"

if [[ ! -f "$float_checkpoint" ]]; then
  echo "Float checkpoint not found: $float_checkpoint" >&2
  exit 2
fi
if [[ ! -f "$quant_checkpoint" ]]; then
  echo "Quant checkpoint not found: $quant_checkpoint" >&2
  exit 2
fi
if [[ ! -f "$image_path" ]]; then
  echo "Image not found: $image_path" >&2
  exit 2
fi

case "$quant_scope" in
  vit)
    backbone_quant_components="${BACKBONE_QUANT_COMPONENTS:-vit}"
    quant_model_components="${QUANT_MODEL_COMPONENTS:-backbone}"
    lwdetr_quant_components="${LWDETR_QUANT_COMPONENTS:-none}"
    stats_components=(vit projector)
    ;;
  vit_projector)
    backbone_quant_components="${BACKBONE_QUANT_COMPONENTS:-vit projector}"
    quant_model_components="${QUANT_MODEL_COMPONENTS:-backbone}"
    lwdetr_quant_components="${LWDETR_QUANT_COMPONENTS:-none}"
    stats_components=(vit projector)
    ;;
  transformer)
    backbone_quant_components="${BACKBONE_QUANT_COMPONENTS:-vit projector}"
    quant_model_components="${QUANT_MODEL_COMPONENTS:-backbone transformer}"
    lwdetr_quant_components="${LWDETR_QUANT_COMPONENTS:-none}"
    stats_components=(transformer lwdetr)
    ;;
  full)
    backbone_quant_components="${BACKBONE_QUANT_COMPONENTS:-vit projector}"
    quant_model_components="${QUANT_MODEL_COMPONENTS:-backbone transformer}"
    lwdetr_quant_components="${LWDETR_QUANT_COMPONENTS:-linear}"
    stats_components=(vit projector transformer lwdetr)
    ;;
  *)
    echo "Unsupported quant scope: $quant_scope. Use vit, vit_projector, transformer, or full." >&2
    exit 2
    ;;
esac

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

read -r -a backbone_quant_component_args <<< "$backbone_quant_components"
read -r -a quant_model_component_args <<< "$quant_model_components"
read -r -a encoder_quant_component_args <<< "${ENCODER_QUANT_COMPONENTS:-linear conv2d matmul softmax gelu layernorm}"
read -r -a projector_quant_component_args <<< "${PROJECTOR_QUANT_COMPONENTS:-conv2d gelu silu layernorm}"
read -r -a transformer_quant_component_args <<< "${TRANSFORMER_QUANT_COMPONENTS:-self_attn cross_attn ffn norm refpoint two_stage decoder_output_norm}"
read -r -a lwdetr_quant_component_args <<< "$lwdetr_quant_components"

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

quant_block_args=()
if [[ "${ENCODER_QUANT_BLOCKS:-all}" != "all" ]]; then
  read -r -a encoder_quant_block_args <<< "${ENCODER_QUANT_BLOCKS}"
  quant_block_args=(--encoder_quant_blocks "${encoder_quant_block_args[@]}")
fi

common_args=(
  --encoder "$encoder"
  --vit_encoder_num_layers "$vit_encoder_num_layers"
  --window_block_indexes "${window_block_indexes[@]}"
  --out_feature_indexes "${out_feature_indexes[@]}"
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
  --num_select "$num_select"
  "${num_queries[@]}"
  --input "$image_path"
  --confidence_threshold "${CONFIDENCE_THRESHOLD:-0.5}"
  --device "${DEVICE:-cuda}"
  --observe_stat
  "${easy_quantize_args[@]}"
  "${decoder_two_stage_float_args[@]}"
)

mkdir -p "$output_root"

# Tensor save paths are hard-coded to observed_tensor/... in the model code.
# save_tensor_from_pathname rewrites that prefix to OBSERVED_TENSOR_ROOT.
if [[ "${CLEAR_OBSERVED_TENSOR:-1}" == "1" ]]; then
  rm -rf "$observed_root"
  rm -rf observed_tensor
fi
mkdir -p "$observed_root"
export OBSERVED_TENSOR_ROOT="$observed_root"

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
  --lwdetr_quant_components "${lwdetr_quant_component_args[@]}" \
  --act_observer "${ACT_OBSERVER:-mean}" \
  --act_percentile "${ACT_PERCENTILE:-99}"

echo "[3/4] Generating tensor statistics..."
python demo/generate_tensor_statistics.py \
  --components "${stats_components[@]}" \
  --observed-root "$observed_root"

echo "[4/4] Generating summary plots..."
python demo/read_comparision_result.py \
  --components "${stats_components[@]}" \
  --observed-root "$observed_root"

echo "Done. Visualizations: $output_root"
echo "Done. Observed tensors: $observed_root"
echo "Done. Tensor comparisons: $observed_root/comparison"
echo "Done. Metric plots: $observed_root/plots"
