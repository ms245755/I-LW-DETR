#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/../.." && pwd)"

usage() {
  cat >&2 <<'EOF'
Usage:
  bash setup/export/run.sh <tiny|small|medium|large|xlarge> <coco_path> <checkpoint> [export_model args]

Examples:
  bash setup/export/run.sh tiny ../../data/COCO pretrain_weights/coco/tiny/LWDETR_tiny_60e_coco.pth
  bash setup/export/run.sh small ../../data/COCO /path/to/checkpoint.pth --simplify
  bash setup/export/run.sh medium ../../data/COCO /path/to/checkpoint.pth --simplify --trt

Optional environment variables:
  DATASET_FILE=coco|custom_coco   Dataset metadata type. Default: coco
  OUTPUT_DIR=<path>               Export output directory. Default: output/lwdetr_<size>_<dataset>
  DEVICE=<cuda|cpu|index>         Device id passed to export. Default: cuda
  EASY_QUANTIZE=true              Export a checkpoint trained with --easy_quantize
  QUANT=true                      Export a checkpoint using quantized modules
  QUANT_MODEL_COMPONENTS          Override top-level quantized components
  BACKBONE_QUANT_COMPONENTS       Override quantized backbone components
  ENCODER_QUANT_BLOCKS            ViT block indexes to quantize, or all
  DECODER_QUANT_BLOCKS            Decoder layer indexes to quantize, or all
  DECODER_QUANT_NORMS             Decoder norms to quantize, or all
  ENCODER_QUANT_COMPONENTS        Override quantized ViT ops
  PROJECTOR_QUANT_COMPONENTS      Override quantized projector ops
  TRANSFORMER_QUANT_COMPONENTS    Override quantized transformer blocks
  DECODER_SELF_ATTN_QUANT_COMPONENTS
  DECODER_CROSS_ATTN_QUANT_COMPONENTS
  DECODER_FFN_QUANT_COMPONENTS
  DECODER_REFPOINT_QUANT_COMPONENTS
  DECODER_TWO_STAGE_QUANT_COMPONENTS
  DECODER_OUTPUT_NORM_QUANT_COMPONENTS
  LWDETR_QUANT_COMPONENTS         Override quantized prediction-head ops
  TRANSFORMER_EXP_MODE=float|ivit Default: ivit
  LWDETR_EXP_MODE=float|ivit      Default: ivit
  DECODER_CROSS_ATTN_FLOAT=true   Match checkpoints trained with this fallback
  DECODER_CROSS_ATTN_VALUE_PER_CHANNEL=true
  DECODER_TWO_STAGE_FLOAT=true
  DECODER_NORM_OUTPUT_QACT=true
  DECODER_REFPOINT_HEAD_FLOAT=true
  AIDGE_EXPORT=true               Avoid selected ONNX patterns unsupported by Aidge CPU
  DRY_RUN=true                    Print the generated command and exit

Output:
  <OUTPUT_DIR>/inference_model.onnx
  <OUTPUT_DIR>/inference_model.sim.onnx when --simplify is passed
  <OUTPUT_DIR>/inference_model[.sim].engine when --trt is passed
EOF
}

if [[ $# -lt 3 ]]; then
  usage
  exit 2
fi

model_size="$1"
coco_path="$2"
checkpoint="$3"
shift 3

dataset_file="${DATASET_FILE:-coco}"
device="${DEVICE:-cuda}"
easy_quantize="${EASY_QUANTIZE:-false}"
quant="${QUANT:-false}"
aidge_export="${AIDGE_EXPORT:-false}"

case "$dataset_file" in
  coco|custom_coco) ;;
  *)
    echo "Unsupported DATASET_FILE: $dataset_file" >&2
    usage
    exit 2
    ;;
esac

case "$easy_quantize" in
  true|false) ;;
  *)
    echo "EASY_QUANTIZE must be true or false, got: $easy_quantize" >&2
    usage
    exit 2
    ;;
esac

case "$quant" in
  true|false) ;;
  *)
    echo "QUANT must be true or false, got: $quant" >&2
    usage
    exit 2
    ;;
esac

case "$aidge_export" in
  true|false) ;;
  *)
    echo "AIDGE_EXPORT must be true or false, got: $aidge_export" >&2
    usage
    exit 2
    ;;
esac

case "$model_size" in
  tiny)
    encoder="vit_tiny"
    vit_encoder_num_layers=6
    window_block_indexes=(0 2 4)
    out_feature_indexes=(1 3 5)
    projector_scale=(P4)
    hidden_dim=256
    sa_nheads=8
    ca_nheads=16
    dec_n_points=2
    num_queries_args=(--num_queries 100)
    num_select=100
    ;;
  small)
    encoder="vit_tiny"
    vit_encoder_num_layers=10
    window_block_indexes=(0 1 3 6 7 9)
    out_feature_indexes=(2 4 5 9)
    projector_scale=(P4)
    hidden_dim=256
    sa_nheads=8
    ca_nheads=16
    dec_n_points=2
    num_queries_args=()
    num_select=300
    ;;
  medium)
    encoder="vit_small"
    vit_encoder_num_layers=10
    window_block_indexes=(0 1 3 6 7 9)
    out_feature_indexes=(2 4 5 9)
    projector_scale=(P4)
    hidden_dim=256
    sa_nheads=8
    ca_nheads=16
    dec_n_points=2
    num_queries_args=()
    num_select=300
    ;;
  large)
    encoder="vit_small"
    vit_encoder_num_layers=10
    window_block_indexes=(0 1 3 6 7 9)
    out_feature_indexes=(2 4 5 9)
    projector_scale=(P3 P5)
    hidden_dim=384
    sa_nheads=12
    ca_nheads=24
    dec_n_points=4
    num_queries_args=()
    num_select=300
    ;;
  xlarge)
    encoder="vit_base"
    vit_encoder_num_layers=10
    window_block_indexes=(0 1 3 6 7 9)
    out_feature_indexes=(2 4 5 9)
    projector_scale=(P3 P5)
    hidden_dim=384
    sa_nheads=12
    ca_nheads=24
    dec_n_points=4
    num_queries_args=()
    num_select=300
    ;;
  *)
    echo "Unsupported model size: $model_size" >&2
    usage
    exit 2
    ;;
esac

if [[ ! -d "$coco_path" ]]; then
  echo "COCO path does not exist or is not a directory: $coco_path" >&2
  exit 2
fi

if [[ ! -f "$checkpoint" ]]; then
  echo "Checkpoint does not exist or is not a file: $checkpoint" >&2
  exit 2
fi

model_name="lwdetr_${model_size}_${dataset_file}"
output_dir="${OUTPUT_DIR:-output/$model_name}"

split_args() {
  local value="$1"
  local -n out_ref="$2"
  read -r -a out_ref <<< "$value"
}

bool_flag_args() {
  local value="$1"
  local flag="$2"
  local -n out_ref="$3"
  case "$value" in
    true) out_ref=("$flag") ;;
    false) out_ref=() ;;
    *)
      echo "${flag#--} must be true or false, got: $value" >&2
      usage
      exit 2
      ;;
  esac
}

quant_args=()
if [[ "$quant" == "true" ]]; then
  quant_model_components="${QUANT_MODEL_COMPONENTS:-backbone transformer}"
  backbone_quant_components="${BACKBONE_QUANT_COMPONENTS:-vit projector}"
  transformer_quant_components="${TRANSFORMER_QUANT_COMPONENTS:-self_attn cross_attn ffn norm refpoint two_stage decoder_output_norm}"
  lwdetr_quant_components="${LWDETR_QUANT_COMPONENTS:-linear}"
  encoder_quant_blocks="${ENCODER_QUANT_BLOCKS:-all}"
  decoder_quant_blocks="${DECODER_QUANT_BLOCKS:-all}"
  decoder_quant_norms="${DECODER_QUANT_NORMS:-all}"
  encoder_quant_components="${ENCODER_QUANT_COMPONENTS:-linear conv2d matmul softmax gelu layernorm}"
  projector_quant_components="${PROJECTOR_QUANT_COMPONENTS:-conv2d gelu silu layernorm}"
  decoder_self_attn_quant_components="${DECODER_SELF_ATTN_QUANT_COMPONENTS:-all}"
  decoder_cross_attn_quant_components="${DECODER_CROSS_ATTN_QUANT_COMPONENTS:-all}"
  decoder_ffn_quant_components="${DECODER_FFN_QUANT_COMPONENTS:-all}"
  decoder_refpoint_quant_components="${DECODER_REFPOINT_QUANT_COMPONENTS:-all}"
  decoder_two_stage_quant_components="${DECODER_TWO_STAGE_QUANT_COMPONENTS:-all}"
  decoder_output_norm_quant_components="${DECODER_OUTPUT_NORM_QUANT_COMPONENTS:-all}"
  transformer_exp_mode="${TRANSFORMER_EXP_MODE:-ivit}"
  lwdetr_exp_mode="${LWDETR_EXP_MODE:-ivit}"

  split_args "$quant_model_components" quant_model_component_args
  split_args "$backbone_quant_components" backbone_quant_component_args
  split_args "$encoder_quant_components" encoder_quant_component_args
  split_args "$projector_quant_components" projector_quant_component_args
  split_args "$transformer_quant_components" transformer_quant_component_args
  split_args "$lwdetr_quant_components" lwdetr_quant_component_args

  quant_args=(
    --quant
    --quant_model_components "${quant_model_component_args[@]}"
    --backbone_quant_components "${backbone_quant_component_args[@]}"
    --encoder_quant_components "${encoder_quant_component_args[@]}"
    --projector_quant_components "${projector_quant_component_args[@]}"
    --transformer_quant_components "${transformer_quant_component_args[@]}"
    --lwdetr_quant_components "${lwdetr_quant_component_args[@]}"
    --transformer_exp_mode "$transformer_exp_mode"
    --lwdetr_exp_mode "$lwdetr_exp_mode"
  )

  if [[ "$encoder_quant_blocks" != "all" ]]; then
    split_args "$encoder_quant_blocks" encoder_quant_block_args
    quant_args+=(--encoder_quant_blocks "${encoder_quant_block_args[@]}")
  fi
  if [[ "$decoder_quant_blocks" != "all" ]]; then
    split_args "$decoder_quant_blocks" decoder_quant_block_args
    quant_args+=(--decoder_quant_blocks "${decoder_quant_block_args[@]}")
  fi
  if [[ "$decoder_quant_norms" != "all" ]]; then
    split_args "$decoder_quant_norms" decoder_quant_norm_arg_values
    quant_args+=(--decoder_quant_norms "${decoder_quant_norm_arg_values[@]}")
  fi
  if [[ "$decoder_self_attn_quant_components" != "all" ]]; then
    split_args "$decoder_self_attn_quant_components" decoder_self_attn_quant_component_args
    quant_args+=(--decoder_self_attn_quant_components "${decoder_self_attn_quant_component_args[@]}")
  fi
  if [[ "$decoder_cross_attn_quant_components" != "all" ]]; then
    split_args "$decoder_cross_attn_quant_components" decoder_cross_attn_quant_component_args
    quant_args+=(--decoder_cross_attn_quant_components "${decoder_cross_attn_quant_component_args[@]}")
  fi
  if [[ "$decoder_ffn_quant_components" != "all" ]]; then
    split_args "$decoder_ffn_quant_components" decoder_ffn_quant_component_args
    quant_args+=(--decoder_ffn_quant_components "${decoder_ffn_quant_component_args[@]}")
  fi
  if [[ "$decoder_refpoint_quant_components" != "all" ]]; then
    split_args "$decoder_refpoint_quant_components" decoder_refpoint_quant_component_args
    quant_args+=(--decoder_refpoint_quant_components "${decoder_refpoint_quant_component_args[@]}")
  fi
  if [[ "$decoder_two_stage_quant_components" != "all" ]]; then
    split_args "$decoder_two_stage_quant_components" decoder_two_stage_quant_component_args
    quant_args+=(--decoder_two_stage_quant_components "${decoder_two_stage_quant_component_args[@]}")
  fi
  if [[ "$decoder_output_norm_quant_components" != "all" ]]; then
    split_args "$decoder_output_norm_quant_components" decoder_output_norm_quant_component_args
    quant_args+=(--decoder_output_norm_quant_components "${decoder_output_norm_quant_component_args[@]}")
  fi

  decoder_cross_attn_float_args=()
  decoder_cross_attn_value_per_channel_args=()
  decoder_two_stage_float_args=()
  decoder_norm_output_qact_args=()
  decoder_refpoint_head_float_args=()
  bool_flag_args "${DECODER_CROSS_ATTN_FLOAT:-false}" --decoder_cross_attn_float decoder_cross_attn_float_args
  bool_flag_args "${DECODER_CROSS_ATTN_VALUE_PER_CHANNEL:-false}" --decoder_cross_attn_value_per_channel decoder_cross_attn_value_per_channel_args
  bool_flag_args "${DECODER_TWO_STAGE_FLOAT:-false}" --decoder_two_stage_float decoder_two_stage_float_args
  bool_flag_args "${DECODER_NORM_OUTPUT_QACT:-false}" --decoder_norm_output_qact decoder_norm_output_qact_args
  bool_flag_args "${DECODER_REFPOINT_HEAD_FLOAT:-false}" --decoder_refpoint_head_float decoder_refpoint_head_float_args
  quant_args+=(
    "${decoder_cross_attn_float_args[@]}"
    "${decoder_cross_attn_value_per_channel_args[@]}"
    "${decoder_two_stage_float_args[@]}"
    "${decoder_norm_output_qact_args[@]}"
    "${decoder_refpoint_head_float_args[@]}"
  )
fi

easy_quantize_args=()
if [[ "$easy_quantize" == "true" ]]; then
  easy_quantize_args=(--easy_quantize)
fi

aidge_export_args=()
if [[ "$aidge_export" == "true" ]]; then
  aidge_export_args=(--aidge_export)
fi

cmd=(
  python main.py
  --batch_size 1
  --encoder "$encoder"
  --vit_encoder_num_layers "$vit_encoder_num_layers"
  --window_block_indexes "${window_block_indexes[@]}"
  --out_feature_indexes "${out_feature_indexes[@]}"
  --dec_layers 3
  --group_detr 13
  --two_stage
  --projector_scale "${projector_scale[@]}"
  --hidden_dim "$hidden_dim"
  --sa_nheads "$sa_nheads"
  --ca_nheads "$ca_nheads"
  --dec_n_points "$dec_n_points"
  --bbox_reparam
  --lite_refpoint_refine
  "${easy_quantize_args[@]}"
  "${quant_args[@]}"
  "${num_queries_args[@]}"
  --num_select "$num_select"
  --dataset_file "$dataset_file"
  --coco_path "$coco_path"
  --square_resize_div_64
  --use_ema
  --eval
  --resume "$checkpoint"
  --output_dir "$output_dir"
  --device "$device"
  export_model
  "${aidge_export_args[@]}"
  "$@"
)

cd "$repo_root"
printf 'Export command:'
printf ' %q' "${cmd[@]}"
printf '\n'
if [[ "${DRY_RUN:-false}" == "true" ]]; then
  exit 0
fi
exec "${cmd[@]}"
