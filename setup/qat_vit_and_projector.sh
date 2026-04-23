#!/usr/bin/env bash
set -euo pipefail

export BACKBONE_QUANT_COMPONENTS="${BACKBONE_QUANT_COMPONENTS:-vit projector}"
export ENCODER_QUANT_BLOCKS="${ENCODER_QUANT_BLOCKS:-all}"
export ENCODER_QUANT_COMPONENTS="${ENCODER_QUANT_COMPONENTS:-linear conv2d matmul softmax gelu layernorm}"
export PROJECTOR_QUANT_COMPONENTS="${PROJECTOR_QUANT_COMPONENTS:-conv2d gelu silu layernorm}"

exec bash setup/lwdetr_quant_qat.sh "$@"
