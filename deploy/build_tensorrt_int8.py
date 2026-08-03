# ------------------------------------------------------------------------
# LW-DETR
# ------------------------------------------------------------------------

"""
Build a TensorRT INT8 engine from an exported LW-DETR ONNX model.

This uses COCO images for PTQ calibration. The resulting .engine can be
evaluated with deploy/benchmark.py.
"""

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image

import torch
import tensorrt as trt


def parse_args():
    parser = argparse.ArgumentParser("LW-DETR TensorRT INT8 builder")
    parser.add_argument("--onnx", required=True, help="Path to exported ONNX model.")
    parser.add_argument("--engine", required=True, help="Path to save TensorRT engine.")
    parser.add_argument("--coco_path", default="data/coco", help="COCO root with images/val2017 and annotations.")
    parser.add_argument("--ann_file", default=None, help="Calibration annotation JSON. Defaults to instances_val2017.json.")
    parser.add_argument("--image_dir", default=None, help="Calibration images. Defaults to coco_path/val2017 or coco_path/images/val2017.")
    parser.add_argument("--image_size", type=int, default=640, help="Square input size used during ONNX export.")
    parser.add_argument("--calib_count", type=int, default=500, help="Number of images for INT8 calibration.")
    parser.add_argument("--batch_size", type=int, default=1, help="Calibration and engine batch size.")
    parser.add_argument("--device", type=int, default=0, help="CUDA device id used for TensorRT calibration.")
    parser.add_argument("--cache", default=None, help="INT8 calibration cache path.")
    parser.add_argument("--workspace_gb", type=float, default=4.0, help="TensorRT workspace size in GiB.")
    parser.add_argument("--fp16", action="store_true", help="Allow FP16 fallback for non-INT8 layers.")
    parser.add_argument(
        "--scenario",
        choices=(
            "default",
            "safe_linear",
            "safe_vit_attention",
            "safe_all_attention",
            "safe_all_attention_norm_softmax",
        ),
        default="default",
        help=(
            "INT8 scenario. default lets TensorRT decide; safe_linear adds Conv plus FFN/head "
            "linear layers; safe_vit_attention adds ViT attention; safe_all_attention adds "
            "decoder attention; safe_all_attention_norm_softmax also requests norm/softmax."
        ),
    )
    parser.add_argument(
        "--precision_constraints",
        choices=("none", "prefer", "obey"),
        default="prefer",
        help="How strongly TensorRT should follow requested per-layer INT8 precision.",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def resolve_image_dir(coco_path, image_dir=None):
    coco_path = Path(coco_path)
    if image_dir:
        return Path(image_dir)
    direct = coco_path / "val2017"
    if direct.is_dir():
        return direct
    return coco_path / "images" / "val2017"


def load_coco_image_paths(coco_path, ann_file, image_dir, limit):
    coco_path = Path(coco_path)
    ann_file = Path(ann_file) if ann_file else coco_path / "annotations" / "instances_val2017.json"
    image_dir = resolve_image_dir(coco_path, image_dir)

    with ann_file.open("r") as f:
        annotations = json.load(f)

    paths = []
    for image in annotations["images"]:
        path = image_dir / image["file_name"]
        if path.is_file():
            paths.append(path)
        if len(paths) >= limit:
            break

    if not paths:
        raise FileNotFoundError(f"No calibration images found under {image_dir}")
    return paths


def preprocess_image(path, image_size):
    image = Image.open(path).convert("RGB")
    image = image.resize((image_size, image_size), Image.BILINEAR)
    array = np.asarray(image, dtype=np.float32) / 255.0
    array = (array - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array(
        [0.229, 0.224, 0.225], dtype=np.float32
    )
    return np.ascontiguousarray(np.transpose(array, (2, 0, 1)))


class CocoEntropyCalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, image_paths, input_shape, cache_file):
        super().__init__()
        self.image_paths = image_paths
        self.input_shape = tuple(input_shape)
        self.cache_file = Path(cache_file)
        self.batch_size = self.input_shape[0]
        self.image_size = self.input_shape[-1]
        self.index = 0
        self.host_batch = np.empty(self.input_shape, dtype=np.float32)
        self.device_batch = torch.empty(self.input_shape, dtype=torch.float32, device="cuda")

    def get_batch_size(self):
        return self.batch_size

    def get_batch(self, names):
        if self.index >= len(self.image_paths):
            return None

        batch_paths = self.image_paths[self.index : self.index + self.batch_size]
        if len(batch_paths) < self.batch_size:
            return None

        for batch_index, path in enumerate(batch_paths):
            self.host_batch[batch_index] = preprocess_image(path, self.image_size)

        self.index += self.batch_size
        self.device_batch.copy_(torch.from_numpy(self.host_batch), non_blocking=False)
        return [int(self.device_batch.data_ptr())]

    def read_calibration_cache(self):
        if self.cache_file.is_file():
            return self.cache_file.read_bytes()
        return None

    def write_calibration_cache(self, cache):
        self.cache_file.parent.mkdir(parents=True, exist_ok=True)
        self.cache_file.write_bytes(cache)


def set_memory_pool_limit(config, workspace_gb):
    bytes_size = int(workspace_gb * (1 << 30))
    if hasattr(config, "set_memory_pool_limit"):
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, bytes_size)
    else:
        config.max_workspace_size = bytes_size


def set_precision_constraints(config, mode):
    if mode == "none":
        return
    flag_name = "PREFER_PRECISION_CONSTRAINTS" if mode == "prefer" else "OBEY_PRECISION_CONSTRAINTS"
    if hasattr(trt.BuilderFlag, flag_name):
        config.set_flag(getattr(trt.BuilderFlag, flag_name))
        print(f"TensorRT precision constraints: {mode}")
    else:
        print(f"TensorRT does not expose {flag_name}; continuing without precision constraint flag")


def should_force_int8(layer, scenario):
    if scenario == "default":
        return False

    convolution = getattr(trt.LayerType, "CONVOLUTION", None)
    matrix_multiply = getattr(trt.LayerType, "MATRIX_MULTIPLY", None)
    fully_connected = getattr(trt.LayerType, "FULLY_CONNECTED", None)
    layer_type = layer.type
    name = layer.name

    is_conv = convolution is not None and layer_type == convolution
    is_linear = (
        (matrix_multiply is not None and layer_type == matrix_multiply)
        or (fully_connected is not None and layer_type == fully_connected)
        or "MatMul" in name
        or "Gemm" in name
    )
    is_attention = (
        "/attn/" in name
        or "self_attn" in name
        or "cross_attn" in name
        or "attention_weights" in name
        or "sampling_offsets" in name
        or "value_proj" in name
        or "output_proj" in name
    )
    is_vit_attention = "/backbone/backbone.0/encoder/blocks." in name and "/attn/" in name and is_linear
    is_decoder_attention = (
        "/transformer/decoder/" in name
        and (
            "self_attn" in name
            or "cross_attn" in name
            or "attention_weights" in name
            or "sampling_offsets" in name
            or "value_proj" in name
            or "output_proj" in name
        )
        and is_linear
    )
    is_all_attention = is_vit_attention or is_decoder_attention
    softmax_type = getattr(trt.LayerType, "SOFTMAX", None)
    is_softmax = (softmax_type is not None and layer_type == softmax_type) or "Softmax" in name
    is_layernorm = (
        "LayerNormalization" in name
        or "/norm" in name
        or ".norm" in name
        or "norm1" in name
        or "norm2" in name
        or "norm3" in name
    )
    is_norm_softmax = is_softmax or is_layernorm
    is_safe_linear = is_linear and not is_attention and (
        "/mlp/fc" in name
        or "/linear1/" in name
        or "/linear2/" in name
        or "ref_point_head" in name
        or "enc_output" in name
        or "enc_out_class_embed" in name
        or "enc_out_bbox_embed" in name
        or "class_embed" in name
        or "bbox_embed" in name
    )

    if scenario == "safe_linear":
        return is_conv or is_safe_linear
    if scenario == "safe_vit_attention":
        return is_conv or is_safe_linear or is_vit_attention
    if scenario == "safe_all_attention":
        return is_conv or is_safe_linear or is_all_attention
    if scenario == "safe_all_attention_norm_softmax":
        return is_conv or is_safe_linear or is_all_attention or is_norm_softmax
    return False


def force_int8_layers(network, scenario):
    if scenario == "default":
        print("Scenario 'default': TensorRT will choose INT8 layers from calibration.")
        return []

    requested = []
    skipped = []
    for layer_index in range(network.num_layers):
        layer = network.get_layer(layer_index)
        if not should_force_int8(layer, scenario):
            continue
        try:
            layer.precision = trt.int8
            for output_index in range(layer.num_outputs):
                output = layer.get_output(output_index)
                if output is not None:
                    layer.set_output_type(output_index, trt.int8)
            requested.append(layer.name)
        except Exception as exc:
            skipped.append((layer.name, str(exc)))

    if requested:
        print(f"Requested INT8 precision for {len(requested)} layers using scenario '{scenario}'")
    if skipped:
        print(f"Skipped INT8 precision request for {len(skipped)} layers")
        for name, reason in skipped[:10]:
            print(f"  {name}: {reason}")
    return requested


def print_cuda_environment(device):
    print(f"TensorRT version: {trt.__version__}")
    print(f"PyTorch CUDA build: {torch.version.cuda}")
    print(f"PyTorch CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device {device}: {torch.cuda.get_device_name(device)}")
    try:
        result = subprocess.run(
            ["nvidia-smi"],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0:
            first_lines = "\n".join(result.stdout.splitlines()[:4])
            print(first_lines)
        else:
            print("nvidia-smi failed:", result.stderr.strip())
    except OSError as exc:
        print(f"nvidia-smi is not available: {exc}")


def build_engine(args):
    if not torch.cuda.is_available():
        raise RuntimeError("TensorRT INT8 calibration requires a CUDA-capable GPU.")
    torch.cuda.set_device(args.device)
    print_cuda_environment(args.device)

    logger = trt.Logger(trt.Logger.VERBOSE if args.verbose else trt.Logger.INFO)
    trt.init_libnvinfer_plugins(logger, "")

    onnx_path = Path(args.onnx)
    engine_path = Path(args.engine)
    cache_path = Path(args.cache) if args.cache else engine_path.with_suffix(".calib.cache")
    engine_path.parent.mkdir(parents=True, exist_ok=True)

    image_paths = load_coco_image_paths(args.coco_path, args.ann_file, args.image_dir, args.calib_count)
    print(f"Using {len(image_paths)} calibration images")

    explicit_batch = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    try:
        builder = trt.Builder(logger)
    except TypeError as exc:
        raise RuntimeError(
            "TensorRT could not initialize CUDA. CUDA error 35 usually means the "
            "installed NVIDIA driver is too old for the CUDA runtime used by your "
            "TensorRT Python package. Update the NVIDIA driver, or install a "
            "TensorRT build that matches your driver/CUDA runtime."
        ) from exc

    with builder, \
            builder.create_network(explicit_batch) as network, \
            trt.OnnxParser(network, logger) as parser, \
            builder.create_builder_config() as config:

        set_memory_pool_limit(config, args.workspace_gb)
        config.set_flag(trt.BuilderFlag.INT8)
        set_precision_constraints(config, args.precision_constraints)
        if hasattr(trt, "ProfilingVerbosity"):
            config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
        if args.fp16:
            config.set_flag(trt.BuilderFlag.FP16)

        input_shape = (args.batch_size, 3, args.image_size, args.image_size)
        calibrator = CocoEntropyCalibrator(image_paths, input_shape, cache_path)
        config.int8_calibrator = calibrator

        with onnx_path.open("rb") as f:
            if not parser.parse(f.read()):
                print("ERROR: Failed to parse ONNX model.")
                for index in range(parser.num_errors):
                    print(parser.get_error(index))
                raise RuntimeError("TensorRT ONNX parsing failed")

        for input_index in range(network.num_inputs):
            tensor = network.get_input(input_index)
            shape = tuple(tensor.shape)
            print(f"Input {tensor.name}: {shape}")
            if any(dim < 0 for dim in shape):
                profile = builder.create_optimization_profile()
                profile.set_shape(tensor.name, input_shape, input_shape, input_shape)
                config.add_optimization_profile(profile)

        force_int8_layers(network, args.scenario)

        print("Building TensorRT INT8 engine. This can take a while...")
        if hasattr(builder, "build_serialized_network"):
            serialized_engine = builder.build_serialized_network(network, config)
        else:
            engine = builder.build_engine(network, config)
            serialized_engine = engine.serialize() if engine is not None else None

        if serialized_engine is None:
            raise RuntimeError("TensorRT engine build failed")

        engine_path.write_bytes(bytes(serialized_engine))
        print(f"Saved TensorRT engine: {engine_path}")
        print(f"Calibration cache: {cache_path}")


if __name__ == "__main__":
    build_engine(parse_args())
