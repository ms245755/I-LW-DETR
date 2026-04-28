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

cd "$repo_root"

python main.py \
  --batch_size "$batch_size" \
  "${common_model_args[@]}" \
  --act_observer "$act_observer" \
  --act_percentile "$act_percentile" \
  --calib \
  --resume "$checkpoint" \
  --output_dir "$output_dir"
