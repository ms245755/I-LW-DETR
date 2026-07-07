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
