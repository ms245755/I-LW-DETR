#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../common.sh
source "$script_dir/../common.sh"

if [[ $# -eq 3 ]]; then
  export MODEL_SIZE="$1"
  shift
fi

if [[ $# -ne 2 ]]; then
  usage_common "setup/qat/run.sh"
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

if [[ -f "$checkpoint_root" ]]; then
  checkpoint_dir="$(cd "$(dirname "$checkpoint_root")" && pwd)"
  resume="${RESUME:-$checkpoint_root}"
else
  checkpoint_dir="$checkpoint_root"
  resume="${RESUME:-$(resolve_path "$checkpoint_dir" "$resume_name" "resume checkpoint")}"
fi
skip_pretrain="${SKIP_PRETRAIN:-false}"
pretrain_args=()
if [[ "$skip_pretrain" == "1" || "$skip_pretrain" == "true" || "$skip_pretrain" == "yes" ]]; then
  echo "SKIP_PRETRAIN is set; QAT will initialize from --resume without loading pretrained encoder/pretrain weights."
else
  pretrained_encoder="${PRETRAINED_ENCODER:-$(resolve_path "$checkpoint_dir" "$pretrained_encoder_name" "pretrained encoder")}"
  pretrain_weights="${PRETRAIN_WEIGHTS:-$(resolve_path "$checkpoint_dir" "$pretrain_weights_name" "pretrain weights")}"
  pretrain_args=(
    --pretrained_encoder "$pretrained_encoder"
    --pretrain_weights "$pretrain_weights"
  )
fi

lr="${LR:-2e-5}"
lr_encoder="${LR_ENCODER:-2e-5}"
weight_decay="${WEIGHT_DECAY:-1e-4}"
master_port="${MASTER_PORT:-29500}"
act_observer="${ACT_OBSERVER:-ema_minmax}"
act_percentile="${ACT_PERCENTILE:-99.9}"
qat_freeze_observers="${QAT_FREEZE_OBSERVERS:-false}"
output_root="${OUTPUT_ROOT:-output}"
output_dir="${OUTPUT_DIR:-${output_root}/${model_name}_qat_${scenario_label}_${model_size}}"
resume_optimizer="${RESUME_OPTIMIZER:-false}"
decoder_cross_attn_float="${DECODER_CROSS_ATTN_FLOAT:-false}"
decoder_cross_attn_value_per_channel="${DECODER_CROSS_ATTN_VALUE_PER_CHANNEL:-false}"
decoder_two_stage_float="${DECODER_TWO_STAGE_FLOAT:-false}"
decoder_norm_output_qact="${DECODER_NORM_OUTPUT_QACT:-false}"
tensorboard="${TENSORBOARD:-false}"
tensorboard_dir="${TENSORBOARD_DIR:-}"

resume_optimizer_args=()
if [[ "$resume_optimizer" == "1" || "$resume_optimizer" == "true" || "$resume_optimizer" == "yes" ]]; then
  resume_optimizer_args=(--resume_optimizer)
fi

qat_freeze_observer_args=()
if [[ "$qat_freeze_observers" == "1" || "$qat_freeze_observers" == "true" || "$qat_freeze_observers" == "yes" ]]; then
  qat_freeze_observer_args=(--qat_freeze_observers)
fi

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

tensorboard_args=()
if [[ "$tensorboard" == "1" || "$tensorboard" == "true" || "$tensorboard" == "yes" ]]; then
  tensorboard_args=(--tensorboard)
  if [[ -n "$tensorboard_dir" ]]; then
    tensorboard_args+=(--tensorboard_dir "$tensorboard_dir")
  fi
fi

pretrain_keys_modify_to_load=(
  transformer.enc_out_class_embed.0.weight
  transformer.enc_out_class_embed.1.weight
  transformer.enc_out_class_embed.2.weight
  transformer.enc_out_class_embed.3.weight
  transformer.enc_out_class_embed.4.weight
  transformer.enc_out_class_embed.5.weight
  transformer.enc_out_class_embed.6.weight
  transformer.enc_out_class_embed.7.weight
  transformer.enc_out_class_embed.8.weight
  transformer.enc_out_class_embed.9.weight
  transformer.enc_out_class_embed.10.weight
  transformer.enc_out_class_embed.11.weight
  transformer.enc_out_class_embed.12.weight
  transformer.enc_out_class_embed.0.bias
  transformer.enc_out_class_embed.1.bias
  transformer.enc_out_class_embed.2.bias
  transformer.enc_out_class_embed.3.bias
  transformer.enc_out_class_embed.4.bias
  transformer.enc_out_class_embed.5.bias
  transformer.enc_out_class_embed.6.bias
  transformer.enc_out_class_embed.7.bias
  transformer.enc_out_class_embed.8.bias
  transformer.enc_out_class_embed.9.bias
  transformer.enc_out_class_embed.10.bias
  transformer.enc_out_class_embed.11.bias
  transformer.enc_out_class_embed.12.bias
  class_embed.weight
  class_embed.bias
)

cd "$repo_root"

python -u -m torch.distributed.launch \
  --nproc_per_node "$nproc_per_node" \
  --master_port "$master_port" \
  --use_env \
  main.py \
  --lr "$lr" \
  --lr_encoder "$lr_encoder" \
  --batch_size "$batch_size" \
  --weight_decay "$weight_decay" \
  --epochs "$epochs" \
  --lr_drop "$lr_drop" \
  --lr_vit_layer_decay "$lr_vit_layer_decay" \
  --lr_component_decay "$lr_component_decay" \
  "${common_model_args[@]}" \
  --act_observer "$act_observer" \
  --act_percentile "$act_percentile" \
  --ia_bce_loss \
  --cls_loss_coef 1 \
  "${pretrain_args[@]}" \
  --resume "$resume" \
  --qat \
  --find_unused_parameters \
  "${resume_optimizer_args[@]}" \
  "${qat_freeze_observer_args[@]}" \
  "${decoder_cross_attn_float_args[@]}" \
  "${decoder_cross_attn_value_per_channel_args[@]}" \
  "${decoder_two_stage_float_args[@]}" \
  "${decoder_norm_output_qact_args[@]}" \
  "${tensorboard_args[@]}" \
  --pretrain_keys_modify_to_load "${pretrain_keys_modify_to_load[@]}" \
  --output_dir "$output_dir"
