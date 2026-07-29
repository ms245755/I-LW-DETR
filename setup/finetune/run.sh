#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../common.sh
source "$script_dir/../common.sh"

usage() {
  cat >&2 <<'EOF'
Usage:
  bash setup/finetune/run.sh <tiny|small|medium> <data_path> [checkpoint.pth]

Environment variables:
  DATASET_FILE                   coco or custom_coco. Default: custom_coco
  OUTPUT_DIR                    Default: output/lwdetr_<size>_<dataset>_finetune
  EPOCHS                        Default: 60 for easy-quant finetune
  LR_DROP                       Default: 80% of EPOCHS
  LR_SCHEDULER                  step or delayed_cosine. Default: delayed_cosine for easy-quant finetune
  LR_DECAY_START_EPOCH          Delayed cosine start epoch. Default: 10 for easy-quant finetune
  LR_MIN_FACTOR                 Final LR ratio for cosine. Default: 0.01
  LR                            Pretrained detector LR. Default: 5e-5 for easy-quant finetune
  LR_ENCODER                    Encoder LR. Default: 1e-5 for easy-quant finetune
  LR_CLASS_HEAD                 New classification-head LR. Default: 1e-4
  CLASS_HEAD_WARMUP_EPOCHS      Head-only training epochs. Default: 0 for easy-quant finetune
  EASY_QUANTIZE                 true to use quantization-friendly float ops. Default: true
  RESUME                        true to continue from a training checkpoint
  BATCH_SIZE                    Per-process batch size. Default: size-specific
  NPROC_PER_NODE                GPU process count. Default: size-specific
  MASTER_PORT                   Default: 29500
  DEVICE                        Default: cuda
  KEEP_CLASS_HEAD               true to load existing classification heads. Default: true for easy-quant finetune, else false
  DRY_RUN                       true to print the command without running it

Examples:
  bash setup/finetune/run.sh tiny /path/to/custom-coco

  EPOCHS=40 CLASS_HEAD_WARMUP_EPOCHS=5 \
  bash setup/finetune/run.sh small /path/to/custom-coco /path/to/checkpoint.pth

  RESUME=true OUTPUT_DIR=output/my_run \
  bash setup/finetune/run.sh medium /path/to/custom-coco
EOF
}

if [[ $# -lt 2 || $# -gt 3 ]]; then
  usage
  exit 2
fi

model_size="$1"
data_path="$(absolute_existing_path "$2")"
checkpoint="${3:-}"

case "$model_size" in
  tiny|small|medium) ;;
  *)
    echo "Unsupported model size: $model_size" >&2
    exit 2
    ;;
esac

dataset_file="${DATASET_FILE:-custom_coco}"
case "$dataset_file" in
  coco|custom_coco) ;;
  *)
    echo "Unsupported DATASET_FILE: $dataset_file" >&2
    exit 2
    ;;
esac

for required_path in \
  "$data_path/train2017" \
  "$data_path/val2017" \
  "$data_path/annotations/instances_train2017.json" \
  "$data_path/annotations/instances_val2017.json"; do
  if [[ ! -e "$required_path" ]]; then
    echo "Required dataset path does not exist: $required_path" >&2
    exit 2
  fi
done

easy_quantize="${EASY_QUANTIZE:-true}"
case "$easy_quantize" in
  true|false) ;;
  *)
    echo "EASY_QUANTIZE must be true or false, got: $easy_quantize" >&2
    exit 2
    ;;
esac

if [[ "$easy_quantize" == "true" && -z "${BATCH_SIZE:-}" ]]; then
  export BATCH_SIZE=1
fi

requested_epochs="${EPOCHS:-$([[ "$easy_quantize" == "true" ]] && echo 60 || echo 40)}"
requested_lr_drop="${LR_DROP:-}"

export MODEL_SIZE="$model_size"
export DATASET_FILE="$dataset_file"
configure_model_size

epochs="$requested_epochs"
lr_drop="${requested_lr_drop:-$((epochs * 4 / 5))}"
lr_scheduler="${LR_SCHEDULER:-$([[ "$easy_quantize" == "true" ]] && echo delayed_cosine || echo step)}"
lr_decay_start_epoch="${LR_DECAY_START_EPOCH:-$([[ "$easy_quantize" == "true" ]] && echo 10 || echo $((epochs / 4)))}"
lr_min_factor="${LR_MIN_FACTOR:-0.01}"
lr="${LR:-$([[ "$easy_quantize" == "true" ]] && echo 5e-5 || echo 1e-5)}"
lr_encoder="${LR_ENCODER:-$([[ "$easy_quantize" == "true" ]] && echo 1e-5 || echo 1.5e-5)}"
lr_class_head="${LR_CLASS_HEAD:-1e-4}"
class_head_warmup_epochs="${CLASS_HEAD_WARMUP_EPOCHS:-$([[ "$easy_quantize" == "true" ]] && echo 0 || echo 3)}"
keep_class_head="${KEEP_CLASS_HEAD:-$([[ "$easy_quantize" == "true" ]] && echo true || echo false)}"
resume_training="${RESUME:-false}"
master_port="${MASTER_PORT:-29500}"
device="${DEVICE:-cuda}"
output_dir="${OUTPUT_DIR:-output/lwdetr_${model_size}_${dataset_file}_finetune}"

case "$resume_training" in
  true|false) ;;
  *)
    echo "RESUME must be true or false, got: $resume_training" >&2
    exit 2
    ;;
esac

case "$keep_class_head" in
  true|false) ;;
  *)
    echo "KEEP_CLASS_HEAD must be true or false, got: $keep_class_head" >&2
    exit 2
    ;;
esac

if [[ "$resume_training" == "true" ]]; then
  if [[ -z "$checkpoint" ]]; then
    checkpoint="$output_dir/checkpoint.pth"
  fi
else
  if [[ -z "$checkpoint" ]]; then
    checkpoint="pretrain_weights/$model_size/$resume_name"
  fi
fi
checkpoint="$(absolute_existing_path "$checkpoint")"

prepare_common_args
build_architecture_model_args

class_head_args=()
if [[ "$keep_class_head" != "true" ]]; then
  class_head_args=(
    --pretrain_exclude_keys
    'class_embed.*'
    'transformer.enc_out_class_embed.*'
  )
fi

easy_quantize_args=()
if [[ "$easy_quantize" == "true" ]]; then
  easy_quantize_args=(--easy_quantize)
fi

checkpoint_args=()
if [[ "$resume_training" == "true" ]]; then
  checkpoint_args=(
    --resume "$checkpoint"
    --resume_optimizer
  )
else
  checkpoint_args=(
    --pretrain_weights "$checkpoint"
    "${class_head_args[@]}"
  )
fi

train_command=(
  python -u -m torch.distributed.launch
  --nproc_per_node "$nproc_per_node"
  --master_port "$master_port"
  --use_env
  main.py
  --lr "$lr"
  --lr_encoder "$lr_encoder"
  --lr_class_head "$lr_class_head"
  --class_head_warmup_epochs "$class_head_warmup_epochs"
  --batch_size "$batch_size"
  --weight_decay 1e-4
  --epochs "$epochs"
  --lr_drop "$lr_drop"
  --lr_scheduler "$lr_scheduler"
  --lr_decay_start_epoch "$lr_decay_start_epoch"
  --lr_min_factor "$lr_min_factor"
  --lr_vit_layer_decay "$lr_vit_layer_decay"
  --lr_component_decay "$lr_component_decay"
  "${architecture_model_args[@]}"
  "${easy_quantize_args[@]}"
  --ia_bce_loss
  --cls_loss_coef 1
  --dataset_file "$dataset_file"
  --coco_path "$data_path"
  --square_resize_div_64
  --use_ema
  "${checkpoint_args[@]}"
  --output_dir "$output_dir"
  --device "$device"
)

echo "model_size=$model_size"
echo "dataset_file=$dataset_file"
echo "data_path=$data_path"
echo "checkpoint=$checkpoint"
echo "output_dir=$output_dir"
echo "nproc_per_node=$nproc_per_node"
echo "batch_size=$batch_size"
echo "epochs=$epochs"
echo "lr_drop=$lr_drop"
echo "lr_scheduler=$lr_scheduler"
echo "lr_decay_start_epoch=$lr_decay_start_epoch"
echo "lr_min_factor=$lr_min_factor"
echo "lr=$lr"
echo "lr_encoder=$lr_encoder"
echo "lr_class_head=$lr_class_head"
echo "class_head_warmup_epochs=$class_head_warmup_epochs"
echo "easy_quantize=$easy_quantize"
echo "resume=$resume_training"

cd "$repo_root"
mkdir -p "$output_dir"

if [[ "${DRY_RUN:-false}" == "true" ]]; then
  printf 'Command:'
  printf ' %q' "${train_command[@]}"
  printf '\n'
  exit 0
fi

"${train_command[@]}"
