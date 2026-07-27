#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../common.sh
source "$script_dir/../common.sh"

if [[ $# -ne 2 ]]; then
  usage_common "setup/ptq/run.sh"
  echo "Use setup/ptq/fully_quantized.sh for PTQ." >&2
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
master_port="${MASTER_PORT:-29500}"
device="${DEVICE:-cuda}"
act_observer="${ACT_OBSERVER:-mean}"
act_percentile="${ACT_PERCENTILE:-99}"
calib_batches="${CALIB_BATCHES:-500}"
calib_eval="${CALIB_EVAL:-true}"
output_root="${OUTPUT_ROOT:-output}"
output_dir="${OUTPUT_DIR:-${output_root}/${model_name}_ptq_${scenario_label}_${model_size}}"
decoder_cross_attn_value_per_channel="${DECODER_CROSS_ATTN_VALUE_PER_CHANNEL:-false}"
decoder_norm_output_qact="${DECODER_NORM_OUTPUT_QACT:-false}"
softmax_calibrate_m="${SOFTMAX_CALIBRATE_M:-true}"
softmax_m_safe_guard="${SOFTMAX_M_SAFE_GUARD:-${SAFE_GUARD:-4}}"
softmax_m_calib_mode="${SOFTMAX_M_CALIB_MODE:-max}"
softmax_m_percentile="${SOFTMAX_M_PERCENTILE:-99.9}"
softmax_m_ema_momentum="${SOFTMAX_M_EMA_MOMENTUM:-0.99}"
observe_stat="${OBSERVE_STAT:-false}"

decoder_cross_attn_value_per_channel_args=()
if [[ "$decoder_cross_attn_value_per_channel" == "1" || "$decoder_cross_attn_value_per_channel" == "true" || "$decoder_cross_attn_value_per_channel" == "yes" ]]; then
  decoder_cross_attn_value_per_channel_args=(--decoder_cross_attn_value_per_channel)
fi

decoder_norm_output_qact_args=()
if [[ "$decoder_norm_output_qact" == "1" || "$decoder_norm_output_qact" == "true" || "$decoder_norm_output_qact" == "yes" ]]; then
  decoder_norm_output_qact_args=(--decoder_norm_output_qact)
fi

calib_eval_args=()
if [[ "$calib_eval" == "0" || "$calib_eval" == "false" || "$calib_eval" == "no" ]]; then
  calib_eval_args=(--no_calib_eval)
fi

softmax_calibrate_m_args=()
if [[ "$softmax_calibrate_m" == "0" || "$softmax_calibrate_m" == "false" || "$softmax_calibrate_m" == "no" ]]; then
  softmax_calibrate_m_args=(--no_softmax_calibrate_m)
fi

observe_stat_args=()
if [[ "$observe_stat" == "1" || "$observe_stat" == "true" || "$observe_stat" == "yes" ]]; then
  observe_stat_args=(--observe_stat)
fi

cd "$repo_root"

	python -u -m torch.distributed.launch \
	  --nproc_per_node "$nproc_per_node" \
	  --master_port "$master_port" \
	  --use_env \
  main.py \
  --device "$device" \
  --batch_size "$batch_size" \
  "${common_model_args[@]}" \
  --act_observer "$act_observer" \
  --act_percentile "$act_percentile" \
  --softmax_m_safe_guard "$softmax_m_safe_guard" \
  --softmax_m_calib_mode "$softmax_m_calib_mode" \
  --softmax_m_percentile "$softmax_m_percentile" \
  --softmax_m_ema_momentum "$softmax_m_ema_momentum" \
  "${softmax_calibrate_m_args[@]}" \
  --calib_batches "$calib_batches" \
  --quant \
  --calib \
  "${observe_stat_args[@]}" \
  "${calib_eval_args[@]}" \
  "${decoder_cross_attn_value_per_channel_args[@]}" \
  "${decoder_norm_output_qact_args[@]}" \
  --resume "$checkpoint" \
  --output_dir "$output_dir"
