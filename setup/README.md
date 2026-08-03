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
| ONNX/TensorRT export | `setup/export/run.sh` | - |
| Fully quantized PTQ | `setup/ptq/run.sh` | `setup/slurm/run.sh ptq` |
| Fully quantized QAT | `setup/qat/run.sh` | `setup/slurm/run.sh qat` |

Model size is always one of `tiny`, `small`, or `medium`. COCO is the primary
dataset and uses `DATASET_FILE=coco`. Other COCO-format datasets use
`DATASET_FILE=custom_coco`.

Architecture presets are defined once in `setup/common.sh`. The maintained
fine-tuning, evaluation, inference, PTQ, QAT, and export runners all consume
that shared definition. Files under `scripts/` and `scripts_pretrain/` are
legacy experiment snapshots and are not the source of truth for model presets.

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
python3 util/convert_visdrone_to_coco.py \
  /path/to/detection/VisDrone_original/VisDrone2019-DET-test-dev \
  /path/to/detection/VisDrone-coco-style \
  --train-root /path/to/detection/VisDrone_original/VisDrone2019-DET-train/VisDrone2019-DET-train \
  --class-mode visdrone
```

When the adapted converter is stored directly under `detection/`, run
`python3 convert_visdrone_to_coco.py --overwrite` from that directory to use
its relative defaults.

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
EASY_QUANTIZE=false \
KEEP_CLASS_HEAD=true \
CLASS_HEAD_WARMUP_EPOCHS=0 \
OUTPUT_DIR=output/coco_tiny_finetune \
EPOCHS=40 \
bash setup/finetune/run.sh \
  tiny \
  ../../data/COCO \
  pretrain_weights/coco/tiny/lwdetr_fp32.pth
```

### Custom-dataset original fine-tuning

For COCO -> VisDrone original fine-tuning, disable easy quantization and use
the original custom-dataset schedule. Leave `KEEP_CLASS_HEAD` unset so the
COCO classification heads are replaced and trained alone for the warm-up
period:

```bash
DATASET_FILE=custom_coco \
EASY_QUANTIZE=false \
OUTPUT_DIR=output/visdrone_medium_finetune_from_coco_cosine \
EPOCHS=200 \
LR_SCHEDULER=delayed_cosine \
LR_DECAY_START_EPOCH=40 \
LR_MIN_FACTOR=0.01 \
LR=1e-4 \
LR_ENCODER=1e-5 \
LR_CLASS_HEAD=2e-4 \
CLASS_HEAD_WARMUP_EPOCHS=5 \
BATCH_SIZE=2 \
bash setup/finetune/run.sh \
  medium \
  /path/to/detection/VisDrone-coco-style \
  pretrain_weights/coco/medium/lwdetr_fp32.pth
```

To continue from a checkpoint already trained on the same custom classes:

```bash
DATASET_FILE=custom_coco \
EASY_QUANTIZE=false \
KEEP_CLASS_HEAD=true \
CLASS_HEAD_WARMUP_EPOCHS=0 \
OUTPUT_DIR=output/visdrone_small_continue \
bash setup/finetune/run.sh \
  small \
  /path/to/detection/VisDrone-coco-style \
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
| `EPOCHS` | `60` | Total number of epochs for default easy-quant finetune |
| `BATCH_SIZE` | `1` | Batch size per GPU process for default easy-quant finetune |
| `NPROC_PER_NODE` | size-specific | Number of GPU processes |
| `LR` | `5e-5` | Detector learning rate for default easy-quant finetune |
| `LR_ENCODER` | `1e-5` | Encoder learning rate for default easy-quant finetune |
| `LR_CLASS_HEAD` | `1e-4` | New classification-head learning rate |
| `CLASS_HEAD_WARMUP_EPOCHS` | `0` | Head-only epochs for default easy-quant finetune |
| `EASY_QUANTIZE` | `true` | Use quantization-friendly float operations |
| `LR_SCHEDULER` | `delayed_cosine` | `step` or `delayed_cosine` |
| `LR_DROP` | 80% of epochs | Step-scheduler decay epoch |
| `LR_DECAY_START_EPOCH` | `10` | Start of delayed cosine decay |
| `LR_MIN_FACTOR` | `0.01` | Final LR / initial LR for cosine |
| `KEEP_CLASS_HEAD` | `true` | Keep heads when checkpoint classes match |
| `DRY_RUN` | `false` | Print the generated command only |

### Fine-tune an existing checkpoint for easy quantization

Easy quantization is float fine-tuning with quantization-friendly operations:
learned decoder reference-position encoding and nearest-neighbor deformable
sampling. It is normally done before PTQ or QAT.

When the checkpoint already uses the target dataset classes, keep its
classification heads. These are now the default finetune settings:

```bash
DATASET_FILE=custom_coco \
OUTPUT_DIR=output/visdrone_small_easy_quant \
bash setup/finetune/run.sh \
  small \
  /path/to/detection/VisDrone-coco-style \
  pretrain_weights/visdrone/small/lwdetr_fp32.pth
```

The existing classification and box heads are loaded. The newly introduced
decoder reference-position MLP is initialized and learned during this run.

## 3. Export ONNX or TensorRT

Use the generic export wrapper instead of the model-specific scripts:

```bash
bash setup/export/run.sh \
  tiny \
  ../../data/COCO \
  pretrain_weights/coco/tiny/lwdetr_fp32.pth
```

Pass normal `export_model` options after the checkpoint:

```bash
bash setup/export/run.sh small ../../data/COCO /path/to/checkpoint.pth --simplify
bash setup/export/run.sh medium ../../data/COCO /path/to/checkpoint.pth --simplify --trt
```

For checkpoints trained with easy quantization, export with the same model
variant:

```bash
EASY_QUANTIZE=true \
bash setup/export/run.sh tiny ../../data/COCO /path/to/qrdetr_fp32.pth --simplify
```

For fully quantized checkpoints, use the quant export wrapper:

```bash
EASY_QUANTIZE=true \
QUANT=true bash setup/export/run.sh \
  tiny \
  ../../data/COCO \
  /path/to/checkpoint_quantized.pth \
  --simplify
```

You can also drive the same path through `run.sh`:

```bash
QUANT=true \
EASY_QUANTIZE=true \
bash setup/export/run.sh tiny ../../data/COCO /path/to/checkpoint_quantized.pth --simplify
```

The quantized export path quantizes the backbone, transformer, and LW-DETR
prediction heads. Selecting individual blocks, layers, or operation families
is not supported.

The default output is `output/lwdetr_<size>_<dataset>/inference_model.onnx`.
Set `OUTPUT_DIR=/path/to/export_dir` to choose a different destination.

## 4. Run through Slurm

The Slurm wrapper dispatches the same local workflows. Create the parent
directory used by `sbatch -o` before submitting.

COCO example:

```bash
mkdir -p logs/coco

DATASET_FILE=coco \
EASY_QUANTIZE=false \
KEEP_CLASS_HEAD=true \
CLASS_HEAD_WARMUP_EPOCHS=0 \
OUTPUT_DIR=output/coco_small_finetune \
EPOCHS=40 \
sbatch \
  -o logs/coco/small_finetune_%j.txt \
  setup/slurm/run.sh finetune \
  small \
  ../../data/COCO \
  pretrain_weights/coco/small/lwdetr_fp32.pth
```

Custom-dataset example:

```bash
mkdir -p logs/vis_drone

DATASET_FILE=custom_coco \
EASY_QUANTIZE=false \
OUTPUT_DIR=output/visdrone_medium_finetune_from_coco_cosine \
EPOCHS=200 \
LR_SCHEDULER=delayed_cosine \
LR_DECAY_START_EPOCH=40 \
LR_MIN_FACTOR=0.01 \
LR=1e-4 \
LR_ENCODER=1e-5 \
LR_CLASS_HEAD=2e-4 \
CLASS_HEAD_WARMUP_EPOCHS=5 \
BATCH_SIZE=2 \
sbatch \
  -o logs/vis_drone/medium_finetune_%j.txt \
  setup/slurm/run.sh finetune \
  medium \
  /path/to/detection/VisDrone-coco-style \
  pretrain_weights/coco/medium/lwdetr_fp32.pth
```

Override resources with normal `sbatch` options:

```bash
sbatch --partition=gpu80G --gres=gpu:8 setup/slurm/run.sh ...
```

The wrapper activates `CONDA_ENV` (`lwdetr2` by default), derives
`NPROC_PER_NODE` from the allocation, combines stderr with stdout, and runs
Python unbuffered. It also accepts `REPO_ROOT`, `CONDA_SH`, `FACTORY_SETUP`,
and `SLURM_WRAPPER_DRY_RUN=true`.

## 5. Evaluate float and quantized checkpoints

### Evaluate COCO checkpoints

Pass the checkpoint file explicitly. The evaluation script does not infer a
checkpoint from the model size or variant:

```bash
DATASET_FILE=coco OUTPUT_DIR=output/eval_coco \
bash setup/eval/run.sh tiny original \
  ../../data/COCO \
  pretrain_weights/coco/tiny/lwdetr_fp32.pth
```

Evaluate a QAT checkpoint by selecting the matching quantized variant and
passing the exact checkpoint produced by that run:

```bash
DATASET_FILE=coco OUTPUT_DIR=output/eval_coco_tiny_qat \
bash setup/eval/run.sh tiny qat \
  ../../data/COCO \
  output/coco_tiny_qat/checkpoint.pth
```

Run the command once per checkpoint when comparing model sizes or variants.
Requiring an explicit path prevents an evaluation from silently selecting a
different checkpoint with a conventional filename.

### Evaluate an explicit custom-dataset checkpoint

```bash
DATASET_FILE=custom_coco OUTPUT_DIR=output/eval_visdrone \
bash setup/eval/run.sh small original \
  /path/to/detection/VisDrone-coco-style \
  pretrain_weights/visdrone/small/lwdetr_fp32.pth
```

Submit any evaluation by inserting the Slurm dispatcher:

```bash
sbatch setup/slurm/run.sh eval \
  tiny original \
  ../../data/COCO \
  pretrain_weights/coco/tiny/lwdetr_fp32.pth
```

`--checkpoint` is required and must name a checkpoint file. Variants are
`original`, `qr`, `ptq`, and `qat`; each invocation evaluates one model size
and one variant.

The `ptq` and `qat` variants are quantized-checkpoint evaluation only. They
build the quantized model with `--quant` and do not run PTQ calibration or QAT
training.

Each run writes `eval.log` and `eval.pth` under
`<output-root>/<model-size>_<variant>/`. The log contains these metrics:

```text
AP, AP50, AP75, AP-small, AP-medium, AP-large,
AR@1, AR@10, AR@100, AR-small, AR-medium, AR-large
```

The size-specific AP values are conditional measurements over different
object-size subsets; they are not components that must be below overall AP.

## 6. Visualize a custom-dataset checkpoint

The current side-by-side visualization utility is for `custom_coco`
checkpoints. It plots ground truth on the left and predictions on the right:

```bash
python demo/visualize_custom_detections.py \
  --model-size small \
  --checkpoint pretrain_weights/visdrone/small/lwdetr_fp32.pth \
  --data-path /path/to/detection/VisDrone-coco-style \
  --num-images 10 \
  --score-threshold 0.5
```

Outputs default to `observation/visdrone_images/<model-size>/`. Use the same
`--seed` to compare identical images across model sizes, or use
`--image-names` to select exact files. A threshold of `0.5` gives cleaner
figures; `0.3` is useful for inspecting weak detections.

## Fully quantized PTQ and QAT

The quantization directories contain one supported quantized model path:

- `setup/ptq/` runs PTQ calibration.
- `setup/qat/` runs QAT training.
- `setup/common.sh` owns shared model-size and workflow settings.
- `setup/ptq/run.sh` and `setup/qat/run.sh` are the direct quantization entry points.

The PTQ and QAT scripts use this form:

```bash
bash setup/<ptq|qat>/run.sh <tiny|small|medium> /path/to/coco /path/to/checkpoint_or_checkpoint_dir
```

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
DATASET_FILE=coco \
OUTPUT_DIR=output/coco_small_ptq \
CALIB_BATCHES=500 \
bash setup/ptq/run.sh \
  small \
  ../../data/COCO \
  pretrain_weights/coco/small/lwdetr_fp32.pth
```

The calibrated checkpoint is written to:

```text
output/coco_small_ptq/checkpoint_calibrated.pth
```

### COCO QAT example

```bash
DATASET_FILE=coco \
SKIP_PRETRAIN=true \
RESUME=output/coco_small_ptq/checkpoint_calibrated.pth \
OUTPUT_DIR=output/coco_small_qat \
bash setup/qat/run.sh \
  small \
  ../../data/COCO \
  output/coco_small_ptq/checkpoint_calibrated.pth
```

Use the same commands on Slurm:

```bash
DATASET_FILE=coco \
OUTPUT_DIR=output/coco_small_ptq \
sbatch setup/slurm/run.sh ptq small \
  ../../data/COCO \
  pretrain_weights/coco/small/lwdetr_fp32.pth

DATASET_FILE=coco \
SKIP_PRETRAIN=true \
RESUME=output/coco_small_ptq/checkpoint_calibrated.pth \
OUTPUT_DIR=output/coco_small_qat \
sbatch setup/slurm/run.sh qat small \
  ../../data/COCO \
  output/coco_small_ptq/checkpoint_calibrated.pth
```

### Custom-dataset PTQ example

Calibrate a fully quantized small model from the float checkpoint:

```bash
DATASET_FILE=custom_coco \
OUTPUT_DIR=output/visdrone_small_ptq \
CALIB_BATCHES=500 \
bash setup/ptq/run.sh \
  small \
  /path/to/detection/VisDrone-coco-style \
  pretrain_weights/visdrone/small/lwdetr_fp32.pth
```

The default output checkpoint is:

```text
output/visdrone_small_ptq/checkpoint_calibrated.pth
```

Run it through Slurm:

```bash
DATASET_FILE=custom_coco \
OUTPUT_DIR=output/visdrone_small_ptq \
sbatch \
  -o logs/vis_drone/ptq_small_%j.txt \
  setup/slurm/run.sh ptq small \
  /path/to/detection/VisDrone-coco-style \
  pretrain_weights/visdrone/small/lwdetr_fp32.pth
```

### Custom-dataset QAT example

Start QAT from the calibrated PTQ checkpoint:

```bash
DATASET_FILE=custom_coco \
SKIP_PRETRAIN=true \
RESUME=output/visdrone_small_ptq/checkpoint_calibrated.pth \
OUTPUT_DIR=output/visdrone_small_qat \
bash setup/qat/run.sh \
  small \
  /path/to/detection/VisDrone-coco-style \
  output/visdrone_small_ptq/checkpoint_calibrated.pth
```

Slurm uses the same variables:

```bash
DATASET_FILE=custom_coco \
SKIP_PRETRAIN=true \
RESUME=output/visdrone_small_ptq/checkpoint_calibrated.pth \
OUTPUT_DIR=output/visdrone_small_qat \
sbatch \
  -o logs/vis_drone/qat_small_%j.txt \
  setup/slurm/run.sh qat small \
  /path/to/detection/VisDrone-coco-style \
  output/visdrone_small_ptq/checkpoint_calibrated.pth
```

The quantized model always includes the ViT encoder, projector, transformer
decoder, and LW-DETR prediction heads.

### Checkpoints

The second argument can be a checkpoint file or a directory.

If it is a file, the scripts use that file as `--resume`. For QAT, the script also needs the pretrained encoder and pretrain weights. If they are not in the same directory as the resume file, set `PRETRAINED_ENCODER` and `PRETRAIN_WEIGHTS`.

If it is a directory, the scripts look for these files:

| Model size | Resume checkpoint | Pretrained encoder | Pretrain weights |
| --- | --- | --- | --- |
| `tiny` | `lwdetr_fp32.pth` | `caev2_tiny_300e_objects365.pth` | `LWDETR_tiny_30e_objects365.pth` |
| `small` | `lwdetr_fp32.pth` | `caev2_tiny_300e_objects365.pth` | `lwdetr_fp32.pth` |
| `medium` | `lwdetr_fp32.pth` | `caev2_small_300e_objects365.pth` | `lwdetr_fp32.pth` |

Override paths explicitly when needed:

```bash
RESUME=/path/to/resume.pth \
PRETRAINED_ENCODER=/path/to/caev2_tiny_300e_objects365.pth \
PRETRAIN_WEIGHTS=/path/to/LWDETR_tiny_30e_objects365.pth \
bash setup/qat/run.sh tiny /path/to/coco /path/to/checkpoint_dir
```

### PTQ

PTQ loads a float/easy-quant checkpoint, calibrates observers, optionally evaluates, and saves:

```text
checkpoint_calibrated.pth
```

Run PTQ:

```bash
bash setup/ptq/run.sh tiny /path/to/coco /path/to/checkpoints
```

Useful PTQ overrides:

```bash
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
CALIB_BATCHES=500 \
ACT_OBSERVER=mean \
ACT_PERCENTILE=99 \
OUTPUT_DIR=output/small_ptq_full \
bash setup/ptq/run.sh small /path/to/coco /path/to/checkpoints
```

Skip evaluation after calibration:

```bash
CALIB_EVAL=false bash setup/ptq/run.sh tiny /path/to/coco /path/to/checkpoints
```

### QAT

QAT loads a checkpoint with `--resume`, builds the quantized model, and trains it. If you start from a PTQ checkpoint, pass the calibrated checkpoint as `RESUME`.

```bash
RESUME=/path/to/checkpoint_calibrated.pth \
OUTPUT_DIR=output/tiny_qat_full \
bash setup/qat/run.sh tiny /path/to/coco /path/to/checkpoint_dir
```

Useful QAT defaults and overrides:

```bash
LR=5e-7
LR_ENCODER=5e-7
WEIGHT_DECAY=0
EPOCHS=20
LR_DROP=5
QAT_FREEZE_OBSERVERS=true
ACT_OBSERVER=ema_minmax
ACT_PERCENTILE=99.9
RESUME_OPTIMIZER=false
```

`QAT_FREEZE_OBSERVERS=true` is now the default, which keeps quantization
ranges fixed when resuming from a calibrated PTQ checkpoint.

#### Resume interrupted QAT

For normal QAT initialization, `RESUME` loads model weights only. To continue an interrupted QAT run from `checkpoint.pth`, also set `RESUME_OPTIMIZER=true`.

```bash
RESUME=/path/to/qat_output/checkpoint.pth \
RESUME_OPTIMIZER=true \
OUTPUT_DIR=/path/to/qat_output \
EPOCHS=20 \
bash setup/qat/run.sh tiny /path/to/coco /path/to/checkpoint_dir
```

`EPOCHS` is the final total number of epochs, not the number of extra epochs. If the checkpoint saved `epoch=7`, `EPOCHS=20` resumes at epoch 8 and trains through epoch 19.

### Sweeps

The supported sweep compares fully quantized model sizes and implementation
settings. It does not disable individual quantized components.

```bash
CHECKPOINT_ROOT=/path/to/checkpoints \
bash setup/ptq/sweep_fully_quantized_sizes.sh /path/to/coco
```

Outputs default under:

```text
output/cumulative_ptq/<case>/<model_size>/
```

## Legacy scripts

Model-specific files under `scripts/slurm/` and root-level quantization scripts
under `setup/` are retained for older experiments. New runs should use the
generic workflow entry points documented above.
