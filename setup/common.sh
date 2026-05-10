#!/usr/bin/env bash

setup_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$setup_dir/.." && pwd)"

usage_common() {
  local script_name="$1"
  cat >&2 <<EOF
Usage: bash $script_name <coco_path> <checkpoint_path_or_dir>

Optional environment variables:
  MODEL_SIZE=tiny|small|medium  Default: tiny
  OUTPUT_ROOT=output            Default output root
  ENCODER_QUANT_BLOCKS          ViT block indexes to quantize, or all
  ENCODER_QUANT_COMPONENTS      Ops quantized inside ViT blocks
  PROJECTOR_QUANT_COMPONENTS    Ops quantized inside the projector
  TRANSFORMER_QUANT_COMPONENTS  Ops quantized inside the transformer
  LWDETR_QUANT_COMPONENTS       Ops quantized inside prediction heads
  DECODER_TWO_STAGE_FLOAT       Keep two-stage proposal/top-k path float
EOF
}

configure_model_size() {
  model_size="${MODEL_SIZE:-tiny}"

  case "$model_size" in
    tiny)
      model_name="lwdetr_tiny_coco"
      encoder="vit_tiny"
      vit_encoder_num_layers=6
      window_block_indexes="0 2 4"
      out_feature_indexes="1 3 5"
      batch_size="${BATCH_SIZE:-4}"
      epochs="${EPOCHS:-30}"
      lr_drop="${LR_DROP:-25}"
      lr_vit_layer_decay="${LR_VIT_LAYER_DECAY:-0.8}"
      lr_component_decay="${LR_COMPONENT_DECAY:-0.7}"
      num_queries=100
      num_select=100
      nproc_per_node="${NPROC_PER_NODE:-4}"
      resume_name="LWDETR_tiny_60e_coco.pth"
      pretrained_encoder_name="caev2_tiny_300e_objects365.pth"
      pretrain_weights_name="LWDETR_tiny_30e_objects365.pth"
      ;;
    small)
      model_name="lwdetr_small_coco"
      encoder="vit_tiny"
      vit_encoder_num_layers=10
      window_block_indexes="0 1 3 6 7 9"
      out_feature_indexes="2 4 5 9"
      batch_size="${BATCH_SIZE:-4}"
      epochs="${EPOCHS:-2}"
      lr_drop="${LR_DROP:-60}"
      lr_vit_layer_decay="${LR_VIT_LAYER_DECAY:-0.8}"
      lr_component_decay="${LR_COMPONENT_DECAY:-0.7}"
      num_queries=300
      num_select=300
      nproc_per_node="${NPROC_PER_NODE:-8}"
      resume_name="LWDETR_small_60e_coco.pth"
      pretrained_encoder_name="caev2_tiny_300e_objects365.pth"
      pretrain_weights_name="LWDETR_small_60e_coco.pth"
      ;;
    medium)
      model_name="lwdetr_medium_coco"
      encoder="vit_small"
      vit_encoder_num_layers=10
      window_block_indexes="0 1 3 6 7 9"
      out_feature_indexes="2 4 5 9"
      batch_size="${BATCH_SIZE:-2}"
      epochs="${EPOCHS:-2}"
      lr_drop="${LR_DROP:-60}"
      lr_vit_layer_decay="${LR_VIT_LAYER_DECAY:-0.7}"
      lr_component_decay="${LR_COMPONENT_DECAY:-0.7}"
      num_queries=300
      num_select=300
      nproc_per_node="${NPROC_PER_NODE:-8}"
      resume_name="LWDETR_medium_60e_coco.pth"
      pretrained_encoder_name="caev2_small_300e_objects365.pth"
      pretrain_weights_name="LWDETR_medium_60e_coco.pth"
      ;;
    *)
      echo "Unsupported MODEL_SIZE: $model_size" >&2
      exit 2
      ;;
  esac
}

resolve_path() {
  local root="$1"
  local filename="$2"
  local label="$3"

  if [[ -f "$root" ]]; then
    printf '%s\n' "$root"
    return
  fi

  local path="$root/$filename"
  if [[ ! -f "$path" ]]; then
    echo "Could not find $label at: $path" >&2
    echo "Pass a file directly or set the matching override environment variable." >&2
    exit 2
  fi
  printf '%s\n' "$path"
}

absolute_existing_path() {
  local path="$1"
  if [[ -d "$path" ]]; then
    cd "$path" && pwd
  elif [[ -f "$path" ]]; then
    local dir
    dir="$(cd "$(dirname "$path")" && pwd)"
    printf '%s/%s\n' "$dir" "$(basename "$path")"
  else
    echo "Path does not exist: $path" >&2
    exit 2
  fi
}

configure_scenario() {
  case "$scenario" in
    vit_only)
      scenario_label="vit_only"
      quant_model_components="${QUANT_MODEL_COMPONENTS:-backbone}"
      backbone_quant_components="${BACKBONE_QUANT_COMPONENTS:-vit}"
      transformer_quant_components="${TRANSFORMER_QUANT_COMPONENTS:-none}"
      lwdetr_quant_components="${LWDETR_QUANT_COMPONENTS:-none}"
      ;;
    vit_projector)
      scenario_label="vit_projector"
      quant_model_components="${QUANT_MODEL_COMPONENTS:-backbone}"
      backbone_quant_components="${BACKBONE_QUANT_COMPONENTS:-vit projector}"
      transformer_quant_components="${TRANSFORMER_QUANT_COMPONENTS:-none}"
      lwdetr_quant_components="${LWDETR_QUANT_COMPONENTS:-none}"
      ;;
    vit_projector_transformer_float_head)
      scenario_label="vit_projector_transformer_float_head"
      quant_model_components="${QUANT_MODEL_COMPONENTS:-backbone transformer}"
      backbone_quant_components="${BACKBONE_QUANT_COMPONENTS:-vit projector}"
      transformer_quant_components="${TRANSFORMER_QUANT_COMPONENTS:-linear matmul softmax layernorm deformable_attention}"
      lwdetr_quant_components="${LWDETR_QUANT_COMPONENTS:-none}"
      ;;
    fully_quantized)
      scenario_label="fully_quantized"
      quant_model_components="${QUANT_MODEL_COMPONENTS:-backbone transformer}"
      backbone_quant_components="${BACKBONE_QUANT_COMPONENTS:-vit projector}"
      transformer_quant_components="${TRANSFORMER_QUANT_COMPONENTS:-linear matmul softmax layernorm deformable_attention}"
      lwdetr_quant_components="${LWDETR_QUANT_COMPONENTS:-linear}"
      ;;
    *)
      echo "Unsupported SCENARIO: $scenario" >&2
      exit 2
      ;;
  esac

  encoder_quant_blocks="${ENCODER_QUANT_BLOCKS:-all}"
  encoder_quant_components="${ENCODER_QUANT_COMPONENTS:-linear conv2d matmul softmax gelu layernorm}"
  projector_quant_components="${PROJECTOR_QUANT_COMPONENTS:-conv2d gelu layernorm}"
}

split_args() {
  local value="$1"
  local -n out_ref="$2"
  read -r -a out_ref <<< "$value"
}

make_quant_block_args() {
  quant_block_args=()
  if [[ "$encoder_quant_blocks" != "all" ]]; then
    read -r -a quant_block_values <<< "$encoder_quant_blocks"
    quant_block_args=(--encoder_quant_blocks "${quant_block_values[@]}")
  fi
}

prepare_common_args() {
  split_args "$quant_model_components" quant_model_component_args
  split_args "$backbone_quant_components" backbone_quant_component_args
  split_args "$encoder_quant_components" encoder_quant_component_args
  split_args "$projector_quant_components" projector_quant_component_args
  split_args "$transformer_quant_components" transformer_quant_component_args
  split_args "$lwdetr_quant_components" lwdetr_quant_component_args
  split_args "$window_block_indexes" window_block_args
  split_args "$out_feature_indexes" out_feature_args
  make_quant_block_args
}

common_model_args=()
build_common_model_args() {
  common_model_args=(
    --encoder "$encoder"
    --vit_encoder_num_layers "$vit_encoder_num_layers"
    --window_block_indexes "${window_block_args[@]}"
    --out_feature_indexes "${out_feature_args[@]}"
    --quant_model_components "${quant_model_component_args[@]}"
    --backbone_quant_components "${backbone_quant_component_args[@]}"
    "${quant_block_args[@]}"
    --encoder_quant_components "${encoder_quant_component_args[@]}"
    --projector_quant_components "${projector_quant_component_args[@]}"
    --transformer_quant_components "${transformer_quant_component_args[@]}"
    --lwdetr_quant_components "${lwdetr_quant_component_args[@]}"
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
    --num_queries "$num_queries"
    --num_select "$num_select"
    --dataset_file coco
    --coco_path "$coco_path"
    --square_resize_div_64
    --easy_quantize
  )
}
