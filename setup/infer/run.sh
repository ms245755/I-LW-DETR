#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../common.sh
source "$script_dir/../common.sh"

usage() {
  cat >&2 <<'EOF'
Usage: bash setup/infer/run.sh <tiny|small|medium> <qr|qat> <checkpoint-or-dir> <image-path>

Runs PyTorch inference for one image.

Arguments:
  qr                EasyQuant/quantization-ready float checkpoint
  qat               QAT-trained quantized checkpoint
  checkpoint-or-dir A checkpoint file, or a directory containing the default
                    checkpoint name for the selected size and variant
  image-path        Path to one JPG, PNG, BMP, WebP, or TIFF image

Optional environment variables:
  DEVICE                       Default: cuda
  OUTPUT_DIR                   Default: output/infer/<size>_<variant>
  CONFIDENCE_THRESHOLD         Default: 0.5

Examples:
  bash setup/infer/run.sh tiny qr pretrain_weights/coco/tiny/checkpoint_easy_quant.pth demo/images/catdog.jpg
  bash setup/infer/run.sh tiny qat pretrain_weights/coco/tiny/lwdetr_tiny_coco_qat.pth demo/images/catdog.jpg
EOF
}

if [[ $# -ne 4 ]]; then
  usage
  exit 2
fi

export MODEL_SIZE="$1"
variant="$2"
checkpoint_root="$(absolute_existing_path "$3")"
image_path="$(absolute_existing_path "$4")"

if [[ ! -f "$image_path" ]]; then
  echo "Image path is not a file: $image_path" >&2
  exit 2
fi

case "${image_path,,}" in
  *.jpg|*.jpeg|*.png|*.bmp|*.webp|*.tif|*.tiff) ;;
  *)
    echo "Unsupported image format: $image_path" >&2
    exit 2
    ;;
esac

case "$variant" in
  qr|easy_quant)
    variant="qr"
    mode="float"
    ;;
  qat)
    mode="quant"
    ;;
  *)
    echo "Unsupported variant: $variant. Use qr or qat." >&2
    exit 2
    ;;
esac

configure_model_size
prepare_common_args
build_architecture_model_args

if [[ -f "$checkpoint_root" ]]; then
  checkpoint="$checkpoint_root"
else
  case "$variant:$model_size" in
    qr:tiny) checkpoint_name="checkpoint_easy_quant.pth" ;;
    qr:small) checkpoint_name="checkpoint_easy_quant_small.pth" ;;
    qr:medium) checkpoint_name="checkpoint_easy_quant_medium.pth" ;;
    qat:tiny) checkpoint_name="lwdetr_tiny_coco_qat.pth" ;;
    qat:small) checkpoint_name="lwdetr_small_coco_qat.pth" ;;
    qat:medium) checkpoint_name="lwdetr_medium_coco_qat.pth" ;;
  esac
  checkpoint="$(resolve_path "$checkpoint_root" "$checkpoint_name" "$variant checkpoint")"
fi

device="${DEVICE:-cuda}"
confidence_threshold="${CONFIDENCE_THRESHOLD:-0.5}"
output_dir="${OUTPUT_DIR:-output/infer/${model_size}_${variant}}"

model_args=(
  "${architecture_model_args[@]}"
  --dataset_file "$dataset_file"
  --easy_quantize
)

mkdir -p "$output_dir"
filename="$(basename "$image_path")"
image_name="${filename%.*}"
image_output_dir="$output_dir/$image_name"

cd "$repo_root"
echo "Model size: $model_size"
echo "Variant: $variant"
echo "Checkpoint: $checkpoint"
echo "Image: $image_path"
echo "Output: $image_output_dir"

python -u demo/demo.py \
  "${model_args[@]}" \
  --weights "$checkpoint" \
  --input "$image_path" \
  --output_dir "$image_output_dir" \
  --device "$device" \
  --mode "$mode" \
  --save_predictions \
  --confidence_threshold "$confidence_threshold"

echo "Inference complete: $image_output_dir"
