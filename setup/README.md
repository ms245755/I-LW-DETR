# LW-DETR Setup Guide

The `setup/` directory is the main entry point for training, evaluation, PTQ,
and QAT. The same workflow scripts run locally or through the generic Slurm
wrapper.

Run all commands from the repository root.

## Workflow map

| Task | Local entry point | Slurm entry point |
| --- | --- | --- |
| Float fine-tuning | `setup/finetune/run.sh` | `setup/slurm/run.sh finetune` |
| Float/QR/PTQ/QAT evaluation | `setup/eval/run.sh` | `setup/slurm/run.sh eval` |
| PTQ | `setup/ptq/<scenario>.sh` | `setup/slurm/run.sh ptq <scenario>` |
| QAT | `setup/qat/<scenario>.sh` | `setup/slurm/run.sh qat <scenario>` |

Model size is always one of `tiny`, `small`, or `medium`. COCO is the primary
dataset and uses `DATASET_FILE=coco`. Other COCO-format datasets use
`DATASET_FILE=custom_coco`.

## 1. Prepare the dataset

### COCO

Point the runners at the standard COCO root:

```text
COCO/
├── train2017/
├── val2017/
└── annotations/
    ├── instances_train2017.json
    └── instances_val2017.json
```

Example path used below:

```bash
COCO_PATH=../../data/COCO
```

No conversion is needed for COCO.

### Custom COCO-format datasets

A custom dataset must use the same directory and annotation filenames. Set
`DATASET_FILE=custom_coco`; the loader infers the classification-head size
from the category IDs and verifies that train and validation IDs agree.

VisDrone is one example. To preserve its ten native classes:

```bash
python util/convert_visdrone_to_coco.py \
  /home/data/dataset/Objects365/VisDrone2019-DET-test-dev \
  /home/data/dataset/Objects365/VisDrone2019-DET-coco \
  --train-root /home/data/dataset/Objects365/VisDrone2019-DET-train/VisDrone2019-DET-train \
  --class-mode visdrone
```

Add `--overwrite` to regenerate an existing output. Native VisDrone category
IDs are `1` through `10`, so the model uses an 11-logit head with unused ID
`0`. Do not mix a native VisDrone checkpoint with the converter's COCO-mapped
mode.

## 2. Fine-tune a float model

The local interface is:

```bash
bash setup/finetune/run.sh <tiny|small|medium> <dataset_root> [checkpoint.pth]
```

The checkpoint is optional. Without it, the matching checkpoint under
`pretrain_weights/<model-size>/` is used. If checkpoints are grouped by
dataset, pass the file explicitly as shown below.

### COCO fine-tuning

COCO uses the existing COCO classification heads:

```bash
DATASET_FILE=coco \
KEEP_CLASS_HEAD=true \
CLASS_HEAD_WARMUP_EPOCHS=0 \
OUTPUT_DIR=output/coco_tiny_finetune \
EPOCHS=40 \
bash setup/finetune/run.sh \
  tiny \
  ../../data/COCO \
  pretrain_weights/coco/tiny/LWDETR_tiny_60e_coco.pth
```

### Custom-dataset fine-tuning

For a new class set, leave `KEEP_CLASS_HEAD` unset. The COCO classification
heads are replaced and trained alone for the warm-up period:

```bash
DATASET_FILE=custom_coco \
OUTPUT_DIR=output/visdrone_tiny \
EPOCHS=100 \
CLASS_HEAD_WARMUP_EPOCHS=5 \
bash setup/finetune/run.sh \
  tiny \
  /home/data/dataset/Objects365/VisDrone2019-DET-coco
```

To continue from a checkpoint already trained on the same custom classes:

```bash
DATASET_FILE=custom_coco \
KEEP_CLASS_HEAD=true \
CLASS_HEAD_WARMUP_EPOCHS=0 \
OUTPUT_DIR=output/visdrone_small_continue \
bash setup/finetune/run.sh \
  small \
  /home/data/dataset/Objects365/VisDrone2019-DET-coco \
  /path/to/visdrone_checkpoint.pth
```

This initializes a new run from the checkpoint weights; it does not restore
the previous optimizer or scheduler.

Training writes `checkpoint.pth`, periodic checkpoints, `log.txt`, and
evaluation artifacts under `OUTPUT_DIR`. Use a distinct output directory for
each experiment to avoid replacing files.

Common overrides:

| Variable | Default | Meaning |
| --- | --- | --- |
| `DATASET_FILE` | `custom_coco` | Set `coco` for the primary COCO dataset |
| `OUTPUT_DIR` | generated name | Checkpoints and training logs |
| `EPOCHS` | `40` | Total number of epochs |
| `BATCH_SIZE` | size-specific | Batch size per GPU process |
| `NPROC_PER_NODE` | size-specific | Number of GPU processes |
| `LR` | `1e-5` | Detector learning rate |
| `LR_ENCODER` | `1.5e-5` | Encoder learning rate |
| `LR_CLASS_HEAD` | `1e-4` | New classification-head learning rate |
| `CLASS_HEAD_WARMUP_EPOCHS` | `3` | Head-only epochs for a new class set |
| `EASY_QUANTIZE` | `false` | Use quantization-friendly float operations |
| `LR_SCHEDULER` | `step` | `step` or `delayed_cosine` |
| `LR_DROP` | 80% of epochs | Step-scheduler decay epoch |
| `LR_DECAY_START_EPOCH` | 25% of epochs | Start of delayed cosine decay |
| `LR_MIN_FACTOR` | `0.01` | Final LR / initial LR for cosine |
| `KEEP_CLASS_HEAD` | `false` | Keep heads when checkpoint classes match |
| `DRY_RUN` | `false` | Print the generated command only |

### Fine-tune an existing checkpoint for easy quantization

Easy quantization is float fine-tuning with quantization-friendly operations:
learned decoder reference-position encoding and nearest-neighbor deformable
sampling. It is normally done before PTQ or QAT.

When the checkpoint already uses the target dataset classes, keep its
classification heads and disable head-only warm-up:

```bash
DATASET_FILE=custom_coco \
EASY_QUANTIZE=true \
KEEP_CLASS_HEAD=true \
CLASS_HEAD_WARMUP_EPOCHS=0 \
OUTPUT_DIR=output/visdrone_small_easy_quant \
EPOCHS=60 \
LR_SCHEDULER=delayed_cosine \
LR_DECAY_START_EPOCH=10 \
LR_MIN_FACTOR=0.01 \
LR=5e-5 \
LR_ENCODER=1e-5 \
BATCH_SIZE=1 \
bash setup/finetune/run.sh \
  small \
  /home/data/dataset/Objects365/VisDrone2019-DET-coco \
  pretrain_weights/visdrone/small/lwdetr_small_visdrone_original.pth
```

The existing classification and box heads are loaded. The newly introduced
decoder reference-position MLP is initialized and learned during this run.

## 3. Run through Slurm

The Slurm wrapper dispatches the same local workflows. Create the parent
directory used by `sbatch -o` before submitting.

COCO example:

```bash
mkdir -p logs/coco

DATASET_FILE=coco \
KEEP_CLASS_HEAD=true \
CLASS_HEAD_WARMUP_EPOCHS=0 \
OUTPUT_DIR=output/coco_small_finetune \
EPOCHS=40 \
sbatch \
  -o logs/coco/small_finetune_%j.txt \
  setup/slurm/run.sh finetune \
  small \
  ../../data/COCO \
  pretrain_weights/coco/small/LWDETR_small_60e_coco.pth
```

Custom-dataset example:

```bash
mkdir -p logs/vis_drone

DATASET_FILE=custom_coco \
OUTPUT_DIR=output/visdrone_small \
EPOCHS=100 \
CLASS_HEAD_WARMUP_EPOCHS=5 \
sbatch \
  -o logs/vis_drone/small_%j.txt \
  setup/slurm/run.sh finetune \
  small \
  /home/data/dataset/Objects365/VisDrone2019-DET-coco
```

Override resources with normal `sbatch` options:

```bash
sbatch --partition=gpu80G --gres=gpu:8 setup/slurm/run.sh ...
```

The wrapper activates `CONDA_ENV` (`lwdetr2` by default), derives
`NPROC_PER_NODE` from the allocation, combines stderr with stdout, and runs
Python unbuffered. It also accepts `REPO_ROOT`, `CONDA_SH`, `FACTORY_SETUP`,
and `SLURM_WRAPPER_DRY_RUN=true`.

## 4. Evaluate float and quantized checkpoints

### Evaluate COCO checkpoints

Evaluate the standard checkpoints discovered under `pretrain_weights`:

```bash
bash setup/eval/run.sh \
  --model-size tiny \
  --variant original \
  --dataset-file coco \
  --checkpoint-root pretrain_weights/coco \
  --coco-path ../../data/COCO \
  --output-root output/eval_coco
```

Evaluate all configured model sizes and checkpoint variants:

```bash
bash setup/eval/run.sh \
  --model-size all \
  --variant all \
  --dataset-file coco \
  --checkpoint-root pretrain_weights/coco \
  --coco-path ../../data/COCO \
  --output-root output/eval_coco_all
```

This requires every checkpoint expected by `setup/eval/run.sh` to exist.

### Evaluate an explicit custom-dataset checkpoint

```bash
bash setup/eval/run.sh \
  --model-size small \
  --variant original \
  --dataset-file custom_coco \
  --checkpoint pretrain_weights/visdrone/small/lwdetr_small_visdrone_original.pth \
  --coco-path /home/data/dataset/Objects365/VisDrone2019-DET-coco \
  --output-root output/eval_visdrone
```

Submit any evaluation by inserting the Slurm dispatcher:

```bash
sbatch setup/slurm/run.sh eval \
  --model-size tiny \
  --variant original \
  --dataset-file coco \
  --checkpoint-root pretrain_weights/coco \
  --coco-path ../../data/COCO \
  --output-root output/eval_coco
```

Variants are `original`, `qr`, `ptq`, and `qat`. An explicit `--checkpoint`
requires one explicit model size and one explicit variant.

The `ptq` and `qat` variants are quantized-checkpoint evaluation only. They
build the quantized model with `--quant` and do not run PTQ calibration or QAT
training.

Each run writes per-model `eval.log` and `eval.pth` files, plus
`summary.csv` and `summary.xlsx` under the output root. Metrics are ordered:

```text
AP, AP50, AP75, AP-small, AP-medium, AP-large,
AR@1, AR@10, AR@100, AR-small, AR-medium, AR-large
```

The size-specific AP values are conditional measurements over different
object-size subsets; they are not components that must be below overall AP.

## 5. Visualize a custom-dataset checkpoint

The current side-by-side visualization utility is for `custom_coco`
checkpoints. It plots ground truth on the left and predictions on the right:

```bash
python demo/visualize_custom_detections.py \
  --model-size small \
  --checkpoint pretrain_weights/visdrone/small/lwdetr_small_visdrone_original.pth \
  --data-path /home/data/dataset/Objects365/VisDrone2019-DET-coco \
  --num-images 10 \
  --score-threshold 0.5
```

Outputs default to `observation/visdrone_images/<model-size>/`. Use the same
`--seed` to compare identical images across model sizes, or use
`--image-names` to select exact files. A threshold of `0.5` gives cleaner
figures; `0.3` is useful for inspecting weak detections.

## Quantization scenarios

The quantization directories contain the recommended entry points:

- `setup/ptq/` runs PTQ calibration.
- `setup/qat/` runs QAT training.
- `setup/common.sh` owns shared model-size, scenario, and component settings.
- Root-level quantization scripts are compatibility wrappers. Prefer
  `setup/ptq/<scenario>.sh` and `setup/qat/<scenario>.sh` for new runs.

All scenario scripts use this form:

```bash
bash setup/<ptq|qat>/<scenario>.sh /path/to/coco /path/to/checkpoint_or_checkpoint_dir
```

Set `MODEL_SIZE=tiny|small|medium` to select the model. The default is `tiny`.

COCO is the default dataset. Set `DATASET_FILE=custom_coco` for VisDrone or
another COCO-format dataset with a different class set.

The Python flags now separate model type from workflow:

```text
--eval             evaluate the float model
--eval --quant     evaluate a quantized checkpoint
--quant --calib    run PTQ calibration and optional evaluation
--qat              run QAT training with the quantized model
```

`--calib` and `--qat` imply quantized model construction internally, but the
setup scripts pass `--quant --calib` for PTQ to keep the intent explicit. QAT
does not run a pre-training calibration pass; run PTQ first and resume from
`checkpoint_calibrated.pth` when fixed calibrated ranges are desired.

### COCO PTQ example

```bash
MODEL_SIZE=small \
DATASET_FILE=coco \
OUTPUT_DIR=output/coco_small_ptq \
CALIB_BATCHES=500 \
bash setup/ptq/fully_quantized.sh \
  ../../data/COCO \
  pretrain_weights/coco/small/LWDETR_small_60e_coco.pth
```

The calibrated checkpoint is written to:

```text
output/coco_small_ptq/checkpoint_calibrated.pth
```

### COCO QAT example

```bash
MODEL_SIZE=small \
DATASET_FILE=coco \
SKIP_PRETRAIN=true \
RESUME=output/coco_small_ptq/checkpoint_calibrated.pth \
QAT_FREEZE_OBSERVERS=true \
EPOCHS=20 \
LR_DROP=15 \
OUTPUT_DIR=output/coco_small_qat \
bash setup/qat/fully_quantized.sh \
  ../../data/COCO \
  output/coco_small_ptq/checkpoint_calibrated.pth
```

Use the same commands on Slurm:

```bash
MODEL_SIZE=small \
DATASET_FILE=coco \
OUTPUT_DIR=output/coco_small_ptq \
sbatch setup/slurm/run.sh ptq fully_quantized \
  ../../data/COCO \
  pretrain_weights/coco/small/LWDETR_small_60e_coco.pth

MODEL_SIZE=small \
DATASET_FILE=coco \
SKIP_PRETRAIN=true \
RESUME=output/coco_small_ptq/checkpoint_calibrated.pth \
QAT_FREEZE_OBSERVERS=true \
EPOCHS=20 \
LR_DROP=15 \
OUTPUT_DIR=output/coco_small_qat \
sbatch setup/slurm/run.sh qat fully_quantized \
  ../../data/COCO \
  output/coco_small_ptq/checkpoint_calibrated.pth
```

### Custom-dataset PTQ example

Calibrate a fully quantized small model from the float checkpoint:

```bash
MODEL_SIZE=small \
DATASET_FILE=custom_coco \
OUTPUT_DIR=output/visdrone_small_ptq \
CALIB_BATCHES=500 \
bash setup/ptq/fully_quantized.sh \
  /home/data/dataset/Objects365/VisDrone2019-DET-coco \
  pretrain_weights/visdrone/small/lwdetr_small_visdrone_original.pth
```

The default output checkpoint is:

```text
output/visdrone_small_ptq/checkpoint_calibrated.pth
```

Run it through Slurm:

```bash
MODEL_SIZE=small \
DATASET_FILE=custom_coco \
OUTPUT_DIR=output/visdrone_small_ptq \
sbatch \
  -o logs/vis_drone/ptq_small_%j.txt \
  setup/slurm/run.sh ptq fully_quantized \
  /home/data/dataset/Objects365/VisDrone2019-DET-coco \
  pretrain_weights/visdrone/small/lwdetr_small_visdrone_original.pth
```

### Custom-dataset QAT example

Start QAT from the calibrated PTQ checkpoint:

```bash
MODEL_SIZE=small \
DATASET_FILE=custom_coco \
SKIP_PRETRAIN=true \
RESUME=output/visdrone_small_ptq/checkpoint_calibrated.pth \
QAT_FREEZE_OBSERVERS=true \
EPOCHS=20 \
LR_DROP=15 \
OUTPUT_DIR=output/visdrone_small_qat \
bash setup/qat/fully_quantized.sh \
  /home/data/dataset/Objects365/VisDrone2019-DET-coco \
  output/visdrone_small_ptq/checkpoint_calibrated.pth
```

Slurm uses the same variables:

```bash
MODEL_SIZE=small \
DATASET_FILE=custom_coco \
SKIP_PRETRAIN=true \
RESUME=output/visdrone_small_ptq/checkpoint_calibrated.pth \
QAT_FREEZE_OBSERVERS=true \
EPOCHS=20 \
LR_DROP=15 \
OUTPUT_DIR=output/visdrone_small_qat \
sbatch \
  -o logs/vis_drone/qat_small_%j.txt \
  setup/slurm/run.sh qat fully_quantized \
  /home/data/dataset/Objects365/VisDrone2019-DET-coco \
  output/visdrone_small_ptq/checkpoint_calibrated.pth
```

### Scenarios

| Scenario | Script | Default quantized parts |
| --- | --- | --- |
| ViT only | `vit_only.sh` | Backbone ViT encoder |
| ViT + projector | `vit_projector.sh` | Backbone ViT encoder and projector |
| ViT + projector + transformer, float heads | `vit_projector_transformer_float_head.sh` | Backbone and transformer; LW-DETR prediction heads stay float |
| Fully quantized | `fully_quantized.sh` | Backbone, transformer, and LW-DETR prediction heads |

Scenario wrappers only set defaults. You can override almost every quantized component with environment variables.

### Checkpoints

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

### PTQ

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

### QAT

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
QAT_FREEZE_OBSERVERS=true
ACT_OBSERVER=ema_minmax
ACT_PERCENTILE=99.9
RESUME_OPTIMIZER=false
```

Use `QAT_FREEZE_OBSERVERS=true` when resuming from a calibrated PTQ checkpoint and you want fixed quantization ranges during QAT.

#### Resume interrupted QAT

For normal QAT initialization, `RESUME` loads model weights only. To continue an interrupted QAT run from `checkpoint.pth`, also set `RESUME_OPTIMIZER=true`.

```bash
RESUME=/path/to/qat_output/checkpoint.pth \
RESUME_OPTIMIZER=true \
OUTPUT_DIR=/path/to/qat_output \
EPOCHS=20 \
bash setup/qat/fully_quantized.sh /path/to/coco /path/to/checkpoint_dir
```

`EPOCHS` is the final total number of epochs, not the number of extra epochs. If the checkpoint saved `epoch=7`, `EPOCHS=20` resumes at epoch 8 and trains through epoch 19.

### Component selection

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

### Sweeps

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

## Legacy scripts

Model-specific files under `scripts/slurm/` and root-level quantization scripts
under `setup/` are retained for older experiments. New runs should use the
generic workflow entry points documented above.
