# Fully Quantized LW-DETR

This repository develops a fully quantized I-LW-DETR pipeline for efficient object detection. It supports float fine-tuning, post-training quantization (PTQ), quantization-aware training (QAT), evaluation, single-image inference, and ONNX/TensorRT export on COCO and COCO-format datasets such as VisDrone.

The quantized model covers the complete detection graph: the ViT encoder, multi-scale projector, transformer encoder/decoder, and prediction heads. The project intentionally provides one fully quantized configuration instead of per-layer or per-component quantization choices.

This repository accompanies our upcoming paper on fully quantized LW-DETR.
The paper title, link, and citation will be added when it is published.

Our work builds on the official
[LW-DETR](https://github.com/Atten4Vis/LW-DETR) implementation.

<p align="center">
  <a href="figures/simplified_quantized_lwdetr.pdf">
    <img
      src="figures/simplified_quantized_lwdetr.png"
      width="80%"
      alt="Fully quantized LW-DETR architecture"
    >
  </a>
</p>

## Installation

Create a Python environment, install PyTorch for your CUDA version, and then
install the project dependencies:

```bash
conda create -n lwdetr python=3.8.19
conda activate lwdetr

# Install the appropriate PyTorch and torchvision builds first.
pip install -r requirements.txt

cd models/ops
python setup.py build install
python test.py
cd ../..
```

The original development environment used Python 3.8, PyTorch 1.13, and CUDA
11.6. TensorRT deployment dependencies are listed in
`deploy/requirements.txt`.

## Dataset layout

COCO and custom datasets use the same directory layout:

```text
DATASET_ROOT/
├── train2017/
├── val2017/
└── annotations/
    ├── instances_train2017.json
    └── instances_val2017.json
```

Use `DATASET_FILE=coco` for COCO. Use `DATASET_FILE=custom_coco` for VisDrone or another dataset converted to COCO format. Instructions for converting VisDrone are available in [setup/README.md](setup/README.md).

## Quick start

Run the following commands from the repository root. The examples use the tiny model; `tiny`, `small`, and `medium` are supported by the workflow scripts.

### Fine-tune

There are two different fine-tuning cases.

#### Adapt to a different dataset

Fine-tune on VisDrone or another COCO-format dataset with a new class set:

```bash
DATASET_FILE=custom_coco \
OUTPUT_DIR=output/visdrone_tiny_finetune \
bash setup/finetune/run.sh \
  tiny \
  /path/to/VisDrone-COCO \
  /path/to/LWDETR_tiny_60e_coco.pth
```

The COCO classification heads are replaced automatically. By default, the new heads are trained alone for a short warm-up before the complete model is fine-tuned.

#### Create an easy-quantization checkpoint

Before PTQ or QAT, fine-tune a checkpoint that is already trained on the target dataset using the quantization-friendly float graph:

```bash
DATASET_FILE=coco \
EASY_QUANTIZE=true \
KEEP_CLASS_HEAD=true \
CLASS_HEAD_WARMUP_EPOCHS=0 \
OUTPUT_DIR=output/coco_tiny_easy_quant \
bash setup/finetune/run.sh \
  tiny \
  /path/to/COCO \
  /path/to/LWDETR_tiny_60e_coco.pth
```

This keeps the COCO classification heads and produces a float checkpoint using the graph expected by the fully quantized workflows. For VisDrone or another COCO-format dataset, use `DATASET_FILE=custom_coco` and pass a checkpoint already trained on that dataset.

### Post-training quantization

Calibrate and evaluate the fully quantized model:

```bash
MODEL_SIZE=tiny \
DATASET_FILE=coco \
OUTPUT_DIR=output/coco_tiny_ptq \
bash setup/ptq/fully_quantized.sh \
  /path/to/COCO \
  /path/to/checkpoint_easy_quant.pth
```

The calibrated model is saved as
`output/coco_tiny_ptq/checkpoint_calibrated.pth`.

### Quantization-aware training

Start QAT from the calibrated PTQ checkpoint:

```bash
MODEL_SIZE=tiny \
DATASET_FILE=coco \
SKIP_PRETRAIN=true \
RESUME=output/coco_tiny_ptq/checkpoint_calibrated.pth \
QAT_FREEZE_OBSERVERS=true \
OUTPUT_DIR=output/coco_tiny_qat \
bash setup/qat/fully_quantized.sh \
  /path/to/COCO \
  output/coco_tiny_ptq/checkpoint_calibrated.pth
```

### Inference

Run fully quantized single-image inference with a QAT checkpoint:

```bash
bash setup/infer/run.sh \
  tiny \
  qat \
  /path/to/qat_checkpoint.pth \
  /path/to/image.jpg
```

Results are written under `output/infer/tiny_qat/` by default. For a VisDrone checkpoint, prefix the command with `DATASET_FILE=custom_coco`.

## Inference examples

### COCO

> **Image placeholder:** add a representative COCO inference result here.

<!-- Example when the asset is available:
<p align="center"><img src="figures/inference_coco.jpg" width="80%" alt="COCO inference result"></p>
-->

### VisDrone

> **Image placeholder:** add a representative VisDrone inference result here.

<!-- Example when the asset is available:
<p align="center"><img src="figures/inference_visdrone.jpg" width="80%" alt="VisDrone inference result"></p>
-->

## More options

See [setup/README.md](setup/README.md) for:

- detailed COCO and VisDrone preparation;
- model-size and training overrides;
- PTQ observer and calibration settings;
- QAT resume and observer-freezing behavior;
- evaluation, export, Slurm, and sweep commands.

## Citation

### Our work

Our paper will be published soon. Its BibTeX citation will be added here when the final publication metadata is available.

<!--
@article{TODO,
  title={TODO: Paper title},
  author={TODO: Authors},
  journal={TODO: Venue},
  year={TODO: Year}
}
-->

### Acknowledgements

This project builds upon LW-DETR, Group DETR, ViTDet, Deformable DETR, Conditional DETR, AlignDETR, CAE, I-ViT, and I-Segmenter. We thank the authors of these projects for making their work publicly available.
