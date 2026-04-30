#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../common.sh
source "$script_dir/../common.sh"

if [[ $# -ne 2 ]]; then
  usage_common "setup/ptq/run.sh"
  echo "Set SCENARIO to one of: vit_only, vit_projector, vit_projector_transformer_float_head, fully_quantized" >&2
  exit 2
fi

coco_path="$(absolute_existing_path "$1")"
checkpoint_root="$(absolute_existing_path "$2")"
scenario="${SCENARIO:?SCENARIO must be set by a scenario wrapper}"

configure_model_size
configure_scenario
prepare_common_args
build_common_model_args

checkpoint="${RESUME:-$(resolve_path "$checkpoint_root" "$resume_name" "resume checkpoint")}"
act_observer="${ACT_OBSERVER:-mean}"
act_percentile="${ACT_PERCENTILE:-99}"
output_root="${OUTPUT_ROOT:-output}"
output_dir="${OUTPUT_DIR:-${output_root}/${model_name}_ptq_${scenario_label}_${model_size}}"
ptq_reconstruct="${PTQ_RECONSTRUCT:-false}"
read -r -a ptq_reconstruct_modules <<< "${PTQ_RECONSTRUCT_MODULES:-auto}"
ptq_reconstruct_batches="${PTQ_RECONSTRUCT_BATCHES:-16}"
ptq_reconstruct_lr="${PTQ_RECONSTRUCT_LR:-1e-5}"
ptq_reconstruct_weight_decay="${PTQ_RECONSTRUCT_WEIGHT_DECAY:-0.0}"
ptq_reconstruct_clip_max_norm="${PTQ_RECONSTRUCT_CLIP_MAX_NORM:-0.1}"
ptq_output_distill_batches="${PTQ_OUTPUT_DISTILL_BATCHES:-16}"
ptq_output_distill_lr="${PTQ_OUTPUT_DISTILL_LR:-1e-6}"
ptq_output_distill_logit_weight="${PTQ_OUTPUT_DISTILL_LOGIT_WEIGHT:-1.0}"
ptq_output_distill_box_weight="${PTQ_OUTPUT_DISTILL_BOX_WEIGHT:-5.0}"
ptq_output_distill_temperature="${PTQ_OUTPUT_DISTILL_TEMPERATURE:-2.0}"

ptq_reconstruct_args=()
if [[ "$ptq_reconstruct" == "1" || "$ptq_reconstruct" == "true" || "$ptq_reconstruct" == "yes" ]]; then
  ptq_reconstruct_args=(
    --ptq_reconstruct
    --ptq_reconstruct_modules "${ptq_reconstruct_modules[@]}"
    --ptq_reconstruct_batches "$ptq_reconstruct_batches"
    --ptq_reconstruct_lr "$ptq_reconstruct_lr"
    --ptq_reconstruct_weight_decay "$ptq_reconstruct_weight_decay"
    --ptq_reconstruct_clip_max_norm "$ptq_reconstruct_clip_max_norm"
    --ptq_output_distill_batches "$ptq_output_distill_batches"
    --ptq_output_distill_lr "$ptq_output_distill_lr"
    --ptq_output_distill_logit_weight "$ptq_output_distill_logit_weight"
    --ptq_output_distill_box_weight "$ptq_output_distill_box_weight"
    --ptq_output_distill_temperature "$ptq_output_distill_temperature"
  )
fi

cd "$repo_root"

python main.py \
  --batch_size "$batch_size" \
  "${common_model_args[@]}" \
  --act_observer "$act_observer" \
  --act_percentile "$act_percentile" \
  --calib \
  "${ptq_reconstruct_args[@]}" \
  --resume "$checkpoint" \
  --output_dir "$output_dir"
