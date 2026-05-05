# Demo Quant Inference

Run one-image inference with the same model-size and quantization-scenario setup used by the repo-level `setup/` scripts:

```bash
bash demo/setup/run_one_image_quant_infer.sh \
  fully_quantized \
  tiny \
  /path/to/checkpoint.pth \
  demo/images/catdog.jpg \
  output/demo_quant_tiny
```

The first argument is the quantization scenario:

- `vit_only`
- `vit_projector`
- `vit_projector_transformer_float_head`
- `fully_quantized`

`vit_only` only quantizes the ViT encoder, but the observation comparison still includes both ViT and projector tensors.

The second argument is the model size: `tiny`, `small`, or `medium`.

The script sources `setup/common.sh`, so these environment overrides match training/calibration:

```bash
ENCODER_QUANT_BLOCKS="0 3 5" \
ACT_OBSERVER=mean \
CONFIDENCE_THRESHOLD=0.4 \
DEVICE=cuda \
bash demo/setup/run_one_image_quant_infer.sh fully_quantized tiny /path/to/checkpoint.pth demo/images/catdog.jpg output/demo_quant_tiny
```

To save observed tensors during the same inference:

```bash
OBSERVE_STAT=1 \
bash demo/setup/run_one_image_quant_infer.sh fully_quantized tiny /path/to/checkpoint.pth demo/images/catdog.jpg output/demo_quant_tiny observed_tensor
```

The visualization is written to `<output_dir>/visualize.jpg`.

## Float, Quant, Compare, Plot

To run the whole observation comparison flow:

```bash
bash demo/setup/run_one_image_observe_compare.sh \
  fully_quantized \
  tiny \
  /path/to/float_checkpoint.pth \
  /path/to/quant_checkpoint.pth \
  demo/images/catdog.jpg \
  observation/tiny_fully_quantized
```

This writes:

- float visualization: `observation/tiny_fully_quantized/observe_output/float/visualize.jpg`
- quant visualization: `observation/tiny_fully_quantized/observe_output/quant/visualize.jpg`
- observed tensors: `observation/tiny_fully_quantized/observed_tensor`
- tensor comparisons: `observation/tiny_fully_quantized/observed_tensor/comparison`
- summary plots: `observation/tiny_fully_quantized/observed_tensor/plots`
