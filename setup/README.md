# Quantization Setup

This directory contains the recommended entrypoints for LW-DETR quantization.

- `setup/ptq/` runs PTQ calibration and optional PTQ reconstruction.
- `setup/qat/` runs QAT training.
- `setup/common.sh` owns the shared model-size, scenario, and component-selection settings.
- Root-level files such as `setup/lwdetr_quant_calib.sh`, `setup/lwdetr_quant_qat.sh`, `setup/only_quantize_vit.sh`, and `setup/qat_vit_and_projector.sh` are compatibility wrappers. Prefer `setup/ptq/<scenario>.sh` and `setup/qat/<scenario>.sh` for new runs.

All scenario scripts use this form:

```bash
bash setup/<ptq|qat>/<scenario>.sh /path/to/coco /path/to/checkpoint_or_checkpoint_dir
```

Set `MODEL_SIZE=tiny|small|medium` to select the model. The default is `tiny`.

## Scenarios

| Scenario | Script | Default quantized parts |
| --- | --- | --- |
| ViT only | `vit_only.sh` | Backbone ViT encoder |
| ViT + projector | `vit_projector.sh` | Backbone ViT encoder and projector |
| ViT + projector + transformer, float heads | `vit_projector_transformer_float_head.sh` | Backbone and transformer; LW-DETR prediction heads stay float |
| Fully quantized | `fully_quantized.sh` | Backbone, transformer, and LW-DETR prediction heads |

Scenario wrappers only set defaults. You can override almost every quantized component with environment variables.

## Checkpoints

The second argument can be a checkpoint file or a directory.

If it is a file, the scripts use that file as `--resume`. For QAT, the script also needs the pretrained encoder and pretrain weights. If they are not in the same directory as the resume file, set `PRETRAINED_ENCODER` and `PRETRAIN_WEIGHTS`.

If it is a directory, the scripts look for these files:

| Model size | Resume checkpoint | Pretrained encoder | Pretrain weights |
| --- | --- | --- | --- |
| `tiny` | `LWDETR_tiny_60e_coco.pth` | `caev2_tiny_300e_objects365.pth` | `LWDETR_tiny_30e_objects365.pth` |
| `small` | `LWDETR_small_60e_coco.pth` | `caev2_tiny_300e_objects365.pth` | `LWDETR_small_60e_coco.pth` |
| `medium` | `LWDETR_medium_60e_coco.pth` | `caev2_small_300e_objects365.pth` | `LWDETR_medium_60e_coco.pth` |

Override paths explicitly when needed:

```bash
RESUME=/path/to/resume.pth \
PRETRAINED_ENCODER=/path/to/caev2_tiny_300e_objects365.pth \
PRETRAIN_WEIGHTS=/path/to/LWDETR_tiny_30e_objects365.pth \
bash setup/qat/vit_projector.sh /path/to/coco /path/to/checkpoint_dir
```

## PTQ

PTQ loads a float/easy-quant checkpoint, calibrates observers, optionally evaluates, and saves:

```text
checkpoint_calibrated.pth
```

Run common PTQ scenarios:

```bash
bash setup/ptq/vit_only.sh /path/to/coco /path/to/checkpoints
bash setup/ptq/vit_projector.sh /path/to/coco /path/to/checkpoints
bash setup/ptq/vit_projector_transformer_float_head.sh /path/to/coco /path/to/checkpoints
bash setup/ptq/fully_quantized.sh /path/to/coco /path/to/checkpoints
```

Useful PTQ overrides:

```bash
MODEL_SIZE=small
OUTPUT_DIR=output/my_ptq_run
NPROC_PER_NODE=4
BATCH_SIZE=4
CALIB_BATCHES=500
CALIB_EVAL=true
ACT_OBSERVER=mean
ACT_PERCENTILE=99
```

Example:

```bash
MODEL_SIZE=small \
CALIB_BATCHES=500 \
ACT_OBSERVER=mean \
ACT_PERCENTILE=99 \
OUTPUT_DIR=output/small_ptq_full \
bash setup/ptq/fully_quantized.sh /path/to/coco /path/to/checkpoints
```

Skip evaluation after calibration:

```bash
CALIB_EVAL=false bash setup/ptq/fully_quantized.sh /path/to/coco /path/to/checkpoints
```

Run PTQ reconstruction:

```bash
PTQ_RECONSTRUCT=true \
PTQ_RECONSTRUCT_MODULES=auto \
PTQ_RECONSTRUCT_BATCHES=16 \
PTQ_RECONSTRUCT_LR=1e-5 \
bash setup/ptq/fully_quantized.sh /path/to/coco /path/to/checkpoints
```

When reconstruction is enabled, the saved file is `checkpoint_reconstructed.pth`.

## QAT

QAT loads a checkpoint with `--resume`, builds the quantized model, and trains it. If you start from a PTQ checkpoint, pass the calibrated checkpoint as `RESUME`.

```bash
RESUME=/path/to/checkpoint_calibrated.pth \
MODEL_SIZE=tiny \
EPOCHS=20 \
OUTPUT_DIR=output/tiny_qat_full \
bash setup/qat/fully_quantized.sh /path/to/coco /path/to/checkpoint_dir
```

Useful QAT overrides:

```bash
LR=2e-5
LR_ENCODER=2e-5
WEIGHT_DECAY=1e-4
EPOCHS=20
LR_DROP=25
QAT_CALIB_BATCHES=0
QAT_FREEZE_OBSERVERS=true
ACT_OBSERVER=ema_minmax
ACT_PERCENTILE=99.9
RESUME_OPTIMIZER=false
```

Use `QAT_FREEZE_OBSERVERS=true` when resuming from a calibrated PTQ checkpoint and you want fixed quantization ranges during QAT.

### Resume Interrupted QAT

For normal QAT initialization, `RESUME` loads model weights only. To continue an interrupted QAT run from `checkpoint.pth`, also set `RESUME_OPTIMIZER=true`.

```bash
RESUME=/path/to/qat_output/checkpoint.pth \
RESUME_OPTIMIZER=true \
OUTPUT_DIR=/path/to/qat_output \
EPOCHS=20 \
bash setup/qat/fully_quantized.sh /path/to/coco /path/to/checkpoint_dir
```

`EPOCHS` is the final total number of epochs, not the number of extra epochs. If the checkpoint saved `epoch=7`, `EPOCHS=20` resumes at epoch 8 and trains through epoch 19.

## Component Selection

Use the scenario as a starting point, then override these variables to quantize only the parts you want.

Top-level selection:

| Variable | Values | Meaning |
| --- | --- | --- |
| `QUANT_MODEL_COMPONENTS` | `backbone transformer` | Top-level components to quantize |
| `BACKBONE_QUANT_COMPONENTS` | `vit projector` | Backbone submodules to quantize |
| `LWDETR_QUANT_COMPONENTS` | `linear` or `none` | Prediction head quantization |

Backbone selection:

| Variable | Values |
| --- | --- |
| `ENCODER_QUANT_BLOCKS` | `all` or block indexes such as `"0 2 4"` |
| `ENCODER_QUANT_COMPONENTS` | `linear conv2d matmul softmax gelu layernorm` |
| `PROJECTOR_QUANT_COMPONENTS` | `linear conv2d matmul softmax gelu layernorm silu` |

Transformer selection:

| Variable | Values |
| --- | --- |
| `TRANSFORMER_QUANT_COMPONENTS` | `self_attn cross_attn ffn norm refpoint two_stage decoder_output_norm`, or `none` |
| `DECODER_QUANT_BLOCKS` | `all` or decoder layer indexes such as `"0 2"` |
| `DECODER_QUANT_NORMS` | `norm1 norm2 norm3`, or `none` |
| `DECODER_SELF_ATTN_QUANT_COMPONENTS` | `linear matmul softmax`, or `none` |
| `DECODER_CROSS_ATTN_QUANT_COMPONENTS` | `value_proj sampling_offsets attention_weights softmax grid_sample_core weighted_sum output_proj core all`, or `none` |
| `DECODER_FFN_QUANT_COMPONENTS` | `linear`, or `none` |
| `DECODER_REFPOINT_QUANT_COMPONENTS` | `linear sine_embed coord_proj`, or `none` |
| `DECODER_TWO_STAGE_QUANT_COMPONENTS` | `proposal enc_output enc_output_norm class_head bbox_head bbox_reparam selected_inputs`, or `none` |
| `DECODER_OUTPUT_NORM_QUANT_COMPONENTS` | `layernorm output_qact`, or `none` |

Other quantization behavior:

| Variable | Values | Meaning |
| --- | --- | --- |
| `TRANSFORMER_EXP_MODE` | `ivit` or `float` | Exponential path in transformer bbox reparameterization |
| `LWDETR_EXP_MODE` | `ivit` or `float` | Exponential path in LW-DETR bbox heads |
| `DECODER_CROSS_ATTN_FLOAT` | `true` or `false` | Keep deformable cross-attention on a float fallback path |
| `DECODER_CROSS_ATTN_VALUE_PER_CHANNEL` | `true` or `false` | Use per-channel quantization for cross-attention value path |
| `DECODER_TWO_STAGE_FLOAT` | `true` or `false` | Keep the two-stage proposal/top-k path float |
| `DECODER_NORM_OUTPUT_QACT` | `true` or `false` | Add output activation quantization after decoder norms |
| `DECODER_REFPOINT_HEAD_FLOAT` | `true` or `false` | PTQ-only runner flag to keep refpoint head float |

Examples:

Quantize only ViT linear and LayerNorm ops:

```bash
QUANT_MODEL_COMPONENTS=backbone \
BACKBONE_QUANT_COMPONENTS=vit \
ENCODER_QUANT_COMPONENTS="linear layernorm" \
bash setup/ptq/vit_only.sh /path/to/coco /path/to/checkpoints
```

Quantize only selected ViT blocks:

```bash
ENCODER_QUANT_BLOCKS="0 3 5" \
bash setup/ptq/vit_only.sh /path/to/coco /path/to/checkpoints
```

Quantize projector only:

```bash
QUANT_MODEL_COMPONENTS=backbone \
BACKBONE_QUANT_COMPONENTS=projector \
PROJECTOR_QUANT_COMPONENTS="conv2d silu layernorm" \
bash setup/ptq/vit_projector.sh /path/to/coco /path/to/checkpoints
```

Quantize only transformer self-attention softmax:

```bash
QUANT_MODEL_COMPONENTS=transformer \
TRANSFORMER_QUANT_COMPONENTS=self_attn \
DECODER_SELF_ATTN_QUANT_COMPONENTS=softmax \
LWDETR_QUANT_COMPONENTS=none \
bash setup/ptq/fully_quantized.sh /path/to/coco /path/to/checkpoints
```

Quantize transformer and keep prediction heads float:

```bash
TRANSFORMER_QUANT_COMPONENTS="self_attn cross_attn ffn norm refpoint two_stage decoder_output_norm" \
LWDETR_QUANT_COMPONENTS=none \
bash setup/ptq/vit_projector_transformer_float_head.sh /path/to/coco /path/to/checkpoints
```

Run QAT with a narrower component set:

```bash
RESUME=/path/to/checkpoint_calibrated.pth \
TRANSFORMER_QUANT_COMPONENTS="self_attn ffn norm" \
DECODER_QUANT_BLOCKS="0 1" \
LWDETR_QUANT_COMPONENTS=none \
bash setup/qat/fully_quantized.sh /path/to/coco /path/to/checkpoint_dir
```

## Sweeps

The PTQ sweep scripts run multiple PTQ jobs sequentially and write summaries.

Sweep one ViT component at a time:

```bash
MODEL_SIZE=small \
COMPONENTS="linear conv2d matmul softmax gelu layernorm" \
bash setup/ptq/sweep_vit_components.sh /path/to/coco /path/to/checkpoints
```

Sweep one decoder block family at a time:

```bash
BLOCKS="self_attn cross_attn ffn norm refpoint two_stage decoder_output_norm" \
bash setup/ptq/sweep_decoder_blocks.sh /path/to/coco /path/to/checkpoints
```

Sweep individual component families across ViT, projector, transformer, and heads:

```bash
CASES="vit:linear projector:conv2d transformer_self_attn:softmax lwdetr:linear" \
bash setup/ptq/sweep_single_components.sh /path/to/coco /path/to/checkpoints
```

Outputs default under:

```text
output/ptq_vit_component_sweep/<model_size>
output/ptq_decoder_block_sweep/<model_size>
output/ptq_single_component_sweep/<model_size>
```

## Slurm Wrappers

The `scripts/slurm/` directory has cluster-specific wrappers. The newer quantization wrappers call the same `setup/ptq` and `setup/qat` entrypoints, so the same environment variables work.

Fully quantized tiny PTQ + QAT:

```bash
OUTPUT_DIR=output/lwdetr_tiny_qat_fully_quantized \
PTQ_OUTPUT_DIR=output/lwdetr_tiny_ptq_fully_quantized \
QAT_FREEZE_OBSERVERS=true \
sbatch scripts/slurm/lwdetr_tiny_coco_qat_fully_quantized.sh /path/to/coco /path/to/checkpoint.pth
```

Fully quantized small PTQ + QAT:

```bash
OUTPUT_DIR=output/lwdetr_small_qat_fully_quantized \
PTQ_OUTPUT_DIR=output/lwdetr_small_ptq_fully_quantized \
sbatch scripts/slurm/lwdetr_small_coco_qat_fully_quantized.sh /path/to/coco /path/to/checkpoint.pth
```

Resume interrupted QAT under Slurm:

```bash
QAT_RESUME=/path/to/qat_output/checkpoint.pth \
RESUME_OPTIMIZER=true \
OUTPUT_DIR=/path/to/qat_output \
EPOCHS=20 \
sbatch scripts/slurm/lwdetr_tiny_coco_qat_fully_quantized.sh /path/to/coco /path/to/original_checkpoint.pth
```

The fully-quantized Slurm wrappers run a PTQ phase before QAT. For a pure QAT resume, either use a QAT-only wrapper such as `scripts/slurm/lwdetr_tiny_coco_qat_vit_proj.sh`, or adjust the wrapper to skip PTQ when your calibrated checkpoint already exists.

Older Slurm files that call `main.py` directly are useful examples, but they do not expose the full component-selection interface as cleanly as the `setup/` wrappers.

