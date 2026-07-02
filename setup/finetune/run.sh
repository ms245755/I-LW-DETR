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
  EPOCHS                        Default: 40
  LR_DROP                       Default: 80% of EPOCHS
  LR_SCHEDULER                  step or delayed_cosine. Default: step
  LR_DECAY_START_EPOCH          Delayed cosine start epoch. Default: 25% of EPOCHS
  LR_MIN_FACTOR                 Final LR ratio for cosine. Default: 0.01
  LR                            Pretrained detector LR. Default: 1e-5
  LR_ENCODER                    Encoder LR. Default: 1.5e-5
  LR_CLASS_HEAD                 New classification-head LR. Default: 1e-4
  CLASS_HEAD_WARMUP_EPOCHS      Head-only training epochs. Default: 3
  EASY_QUANTIZE                 true to use quantization-friendly float ops
  RESUME                        true to continue from a training checkpoint
  BATCH_SIZE                    Per-process batch size. Default: size-specific
  NPROC_PER_NODE                GPU process count. Default: size-specific
  MASTER_PORT                   Default: 29500
  DEVICE                        Default: cuda
  KEEP_CLASS_HEAD               true to load existing classification heads
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

requested_epochs="${EPOCHS:-40}"
requested_lr_drop="${LR_DROP:-}"

export MODEL_SIZE="$model_size"
export DATASET_FILE="$dataset_file"
configure_model_size

epochs="$requested_epochs"
lr_drop="${requested_lr_drop:-$((epochs * 4 / 5))}"
lr_scheduler="${LR_SCHEDULER:-step}"
lr_decay_start_epoch="${LR_DECAY_START_EPOCH:-$((epochs / 4))}"
lr_min_factor="${LR_MIN_FACTOR:-0.01}"
lr="${LR:-1e-5}"
lr_encoder="${LR_ENCODER:-1.5e-5}"
lr_class_head="${LR_CLASS_HEAD:-1e-4}"
class_head_warmup_epochs="${CLASS_HEAD_WARMUP_EPOCHS:-3}"
easy_quantize="${EASY_QUANTIZE:-false}"
resume_training="${RESUME:-false}"
master_port="${MASTER_PORT:-29500}"
device="${DEVICE:-cuda}"
output_dir="${OUTPUT_DIR:-output/lwdetr_${model_size}_${dataset_file}_finetune}"

case "$easy_quantize" in
  true|false) ;;
  *)
    echo "EASY_QUANTIZE must be true or false, got: $easy_quantize" >&2
    exit 2
    ;;
esac

case "$resume_training" in
  true|false) ;;
  *)
    echo "RESUME must be true or false, got: $resume_training" >&2
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

window_block_args=()
out_feature_args=()
split_args "$window_block_indexes" window_block_args
split_args "$out_feature_indexes" out_feature_args

class_head_args=()
if [[ "${KEEP_CLASS_HEAD:-false}" != "true" ]]; then
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
  "${easy_quantize_args[@]}"
  --num_queries "$num_queries"
  --num_select "$num_select"
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
