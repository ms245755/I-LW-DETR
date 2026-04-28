#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../common.sh
source "$script_dir/../common.sh"

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
pretrained_encoder="${PRETRAINED_ENCODER:-$(resolve_path "$checkpoint_dir" "$pretrained_encoder_name" "pretrained encoder")}"
pretrain_weights="${PRETRAIN_WEIGHTS:-$(resolve_path "$checkpoint_dir" "$pretrain_weights_name" "pretrain weights")}"

lr="${LR:-1e-4}"
lr_encoder="${LR_ENCODER:-1.5e-4}"
weight_decay="${WEIGHT_DECAY:-1e-4}"
act_observer="${ACT_OBSERVER:-ema_minmax}"
act_percentile="${ACT_PERCENTILE:-99.9}"
output_root="${OUTPUT_ROOT:-output}"
output_dir="${OUTPUT_DIR:-${output_root}/${model_name}_qat_${scenario_label}_${model_size}}"

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
  --pretrained_encoder "$pretrained_encoder" \
  --pretrain_weights "$pretrain_weights" \
  --resume "$resume" \
  --qat \
  --pretrain_keys_modify_to_load "${pretrain_keys_modify_to_load[@]}" \
  --output_dir "$output_dir"
