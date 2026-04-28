# Quantization Setup Scripts

The setup scripts are split by quantization mode:

- `setup/ptq/` for post-training quantization / calibration.
- `setup/qat/` for quantization-aware training.

Each mode has one general script, `run.sh`, plus one wrapper per scenario. In normal use, run the scenario wrapper and pass only:

1. the COCO data directory
2. a checkpoint file or a checkpoint directory

The default model size is `tiny`. Set `MODEL_SIZE=small` or `MODEL_SIZE=medium` to use another model.

## Scenarios

| Scenario | Script name | Quantized parts | Output suffix |
| --- | --- | --- | --- |
| ViT only | `vit_only.sh` | ViT encoder only | `vit_only` |
| ViT + projector | `vit_projector.sh` | ViT encoder and projector | `vit_projector` |
| ViT + projector + transformer, float heads | `vit_projector_transformer_float_head.sh` | Backbone and transformer. LW-DETR prediction heads stay float, including final-layer bias. | `vit_projector_transformer_float_head` |
| Fully quantized | `fully_quantized.sh` | Backbone, transformer, and LW-DETR prediction heads | `fully_quantized` |

## PTQ Commands

```bash
bash setup/ptq/vit_only.sh /path/to/coco /path/to/checkpoints
bash setup/ptq/vit_projector.sh /path/to/coco /path/to/checkpoints
bash setup/ptq/vit_projector_transformer_float_head.sh /path/to/coco /path/to/checkpoints
bash setup/ptq/fully_quantized.sh /path/to/coco /path/to/checkpoints
```

PTQ output directories default to:

```text
output/<model_name>_ptq_<scenario>_<model_size>
```

## QAT Commands

```bash
bash setup/qat/vit_only.sh /path/to/coco /path/to/checkpoints
bash setup/qat/vit_projector.sh /path/to/coco /path/to/checkpoints
bash setup/qat/vit_projector_transformer_float_head.sh /path/to/coco /path/to/checkpoints
bash setup/qat/fully_quantized.sh /path/to/coco /path/to/checkpoints
```

QAT output directories default to:

```text
output/<model_name>_qat_<scenario>_<model_size>
```

## Checkpoint Resolution

The second argument can be either a checkpoint file or a directory.

If you pass a file, it is used as the resume checkpoint. For QAT, you should also provide `PRETRAINED_ENCODER` and `PRETRAIN_WEIGHTS` unless the needed files are in the same directory.

If you pass a directory, scripts look for these filenames:

| Model size | Resume checkpoint | Pretrained encoder | Pretrain weights |
| --- | --- | --- | --- |
| tiny | `LWDETR_tiny_60e_coco.pth` | `caev2_tiny_300e_objects365.pth` | `LWDETR_tiny_30e_objects365.pth` |
| small | `LWDETR_small_60e_coco.pth` | `caev2_tiny_300e_objects365.pth` | `LWDETR_small_60e_coco.pth` |
| medium | `LWDETR_medium_60e_coco.pth` | `caev2_small_300e_objects365.pth` | `LWDETR_medium_60e_coco.pth` |

## Common Overrides

```bash
MODEL_SIZE=small bash setup/ptq/fully_quantized.sh /path/to/coco /path/to/checkpoints
OUTPUT_ROOT=/tmp/lwdetr_runs bash setup/qat/vit_projector.sh /path/to/coco /path/to/checkpoints
OUTPUT_DIR=output/custom_run bash setup/ptq/vit_only.sh /path/to/coco /path/to/checkpoints
```

## Fine-Grained Quantization Knobs

The scenario scripts only set defaults. You can still turn quantization on or off for blocks/components with environment variables.

| Variable | Applies to | Values |
| --- | --- | --- |
| `ENCODER_QUANT_BLOCKS` | ViT block selection | `all` or block indexes like `"0 3 5"` |
| `ENCODER_QUANT_COMPONENTS` | Ops inside selected ViT blocks | `linear conv2d matmul softmax gelu layernorm` |
| `PROJECTOR_QUANT_COMPONENTS` | Ops inside the projector | `conv2d gelu silu layernorm` |
| `TRANSFORMER_QUANT_COMPONENTS` | Ops inside the transformer | `linear matmul softmax layernorm deformable_attention` |
| `LWDETR_QUANT_COMPONENTS` | LW-DETR prediction heads | `linear` or `none` |
| `BACKBONE_QUANT_COMPONENTS` | Backbone submodules | `vit`, `projector`, or both |
| `QUANT_MODEL_COMPONENTS` | Top-level modules | `backbone`, `transformer`, or both |

Use `none` for `TRANSFORMER_QUANT_COMPONENTS` or `LWDETR_QUANT_COMPONENTS` to keep that part float.

Examples:

```bash
ENCODER_QUANT_BLOCKS="0 3 5" \
ENCODER_QUANT_COMPONENTS="linear softmax layernorm" \
bash setup/ptq/vit_only.sh /path/to/coco /path/to/checkpoints

PROJECTOR_QUANT_COMPONENTS="conv2d layernorm" \
bash setup/qat/vit_projector.sh /path/to/coco /path/to/checkpoints

TRANSFORMER_QUANT_COMPONENTS="linear layernorm" \
LWDETR_QUANT_COMPONENTS="none" \
bash setup/ptq/vit_projector_transformer_float_head.sh /path/to/coco /path/to/checkpoints

TRANSFORMER_QUANT_COMPONENTS="linear matmul softmax layernorm deformable_attention" \
LWDETR_QUANT_COMPONENTS="linear" \
bash setup/qat/fully_quantized.sh /path/to/coco /path/to/checkpoints
```

Other useful overrides:

```bash
BATCH_SIZE=2
NPROC_PER_NODE=8
ACT_OBSERVER=mean
ACT_PERCENTILE=99
RESUME=/path/to/resume.pth
PRETRAINED_ENCODER=/path/to/caev2_tiny_300e_objects365.pth
PRETRAIN_WEIGHTS=/path/to/LWDETR_tiny_30e_objects365.pth
```

Root-level scripts such as `setup/only_quantize_vit.sh` and `setup/qat_vit_and_projector.sh` are kept as compatibility wrappers. Prefer the new `setup/ptq/` and `setup/qat/` entrypoints for new runs.
