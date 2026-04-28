#!/usr/bin/env bash
set -euo pipefail

export BACKBONE_QUANT_COMPONENTS="${BACKBONE_QUANT_COMPONENTS:-vit}"
export ENCODER_QUANT_BLOCKS="${ENCODER_QUANT_BLOCKS:-all}"
export ENCODER_QUANT_COMPONENTS="${ENCODER_QUANT_COMPONENTS:-linear conv2d matmul softmax gelu layernorm}"
export TRANSFORMER_QUANT_COMPONENTS="${TRANSFORMER_QUANT_COMPONENTS:-linear matmul softmax layernorm deformable_attention}"
export LWDETR_QUANT_COMPONENTS="${LWDETR_QUANT_COMPONENTS:-linear}"
export ACT_OBSERVER="${ACT_OBSERVER:-mean}"

exec bash setup/lwdetr_quant_calib.sh "$@"
