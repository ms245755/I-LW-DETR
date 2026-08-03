# ------------------------------------------------------------------------
# LW-DETR
# ------------------------------------------------------------------------

"""
Quantize an exported LW-DETR ONNX model with ONNX Runtime.

The static path uses COCO images as representative calibration data and writes
a QDQ INT8 model that can be evaluated by deploy/benchmark.py.
"""

import argparse
import inspect
import json
from pathlib import Path

import onnx
import numpy as np
from PIL import Image
from onnxruntime.quantization import (
    CalibrationMethod,
    CalibrationDataReader,
    QuantFormat,
    QuantType,
    quantize_dynamic,
    quantize_static,
)


def parse_args():
    parser = argparse.ArgumentParser("LW-DETR ONNX Runtime quantization")
    parser.add_argument("--input", required=True, help="Path to the FP32 ONNX model.")
    parser.add_argument("--output", required=True, help="Path to save the quantized ONNX model.")
    parser.add_argument("--coco_path", default="data/coco", help="COCO root with val2017 and annotations.")
    parser.add_argument("--ann_file", default=None, help="Calibration annotation JSON. Defaults to COCO val2017.")
    parser.add_argument("--image_dir", default=None, help="Calibration image directory. Defaults to COCO val2017.")
    parser.add_argument("--input_name", default="input", help="ONNX model input name.")
    parser.add_argument("--image_size", type=int, default=640, help="Square calibration resize size.")
    parser.add_argument("--calib_count", type=int, default=500, help="Number of calibration images to use.")
    parser.add_argument("--batch_size", type=int, default=1, help="Calibration batch size.")
    parser.add_argument(
        "--calibration_providers",
        nargs="*",
        default=["CUDAExecutionProvider", "CPUExecutionProvider"],
        help="ONNX Runtime providers used during calibration inference.",
    )
    parser.add_argument("--method", choices=("static", "dynamic"), default="static")
    parser.add_argument(
        "--preset",
        choices=("accuracy", "balanced", "aggressive", "full"),
        default="accuracy",
        help=(
            "Quantization preset. accuracy=Conv only; balanced=Conv/Gemm/MatMul but excludes "
            "decoder nodes; aggressive=Conv/Gemm/MatMul everywhere; full=let ORT quantize all supported ops."
        ),
    )
    parser.add_argument(
        "--calibration_method",
        choices=("minmax", "entropy", "percentile"),
        default="percentile",
        help="Static calibration method.",
    )
    parser.add_argument(
        "--format",
        choices=("auto", "qdq", "qoperator"),
        default="auto",
        help="Quantized ONNX format. auto uses qoperator for the Conv-only accuracy preset and qdq otherwise.",
    )
    parser.add_argument("--activation_type", choices=("uint8", "int8"), default="uint8")
    parser.add_argument("--weight_type", choices=("uint8", "int8"), default="int8")
    parser.add_argument(
        "--per_channel",
        action="store_true",
        help=(
            "Use per-channel weight quantization. This can make transformer MatMul "
            "nodes fail on ONNX Runtime CPU with QLinearMatMul zero-point shape errors."
        ),
    )
    parser.add_argument("--reduce_range", action="store_true", help="Use 7-bit weight range for better CPU compatibility.")
    parser.add_argument("--op_types", nargs="*", default=None, help="Optional ONNX op types to quantize.")
    parser.add_argument("--nodes_to_exclude", nargs="*", default=None, help="Optional ONNX node names to keep in FP32.")
    parser.add_argument("--skip_preprocess", action="store_true", help="Skip ONNX initializer cleanup before quantization.")
    parser.add_argument(
        "--force_quantize_no_input_check",
        action="store_true",
        help="Let ORT skip input range adjustment. Useful when calibration of Softmax tensors is too memory-heavy.",
    )
    return parser.parse_args()


def load_coco_image_paths(coco_path, ann_file, image_dir, limit):
    coco_path = Path(coco_path)
    ann_file = Path(ann_file) if ann_file else coco_path / "annotations" / "instances_val2017.json"
    if image_dir:
        image_dir = Path(image_dir)
    elif (coco_path / "val2017").is_dir():
        image_dir = coco_path / "val2017"
    else:
        image_dir = coco_path / "images" / "val2017"

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
    return np.transpose(array, (2, 0, 1))


class CocoCalibrationDataReader(CalibrationDataReader):
    def __init__(self, image_paths, input_name, image_size, batch_size):
        self.image_paths = image_paths
        self.input_name = input_name
        self.image_size = image_size
        self.batch_size = batch_size
        self.index = 0

    def get_next(self):
        if self.index >= len(self.image_paths):
            return None

        batch_paths = self.image_paths[self.index : self.index + self.batch_size]
        self.index += self.batch_size
        batch = np.stack([preprocess_image(path, self.image_size) for path in batch_paths], axis=0)
        return {self.input_name: batch.astype(np.float32)}


def quant_type(name):
    return QuantType.QUInt8 if name == "uint8" else QuantType.QInt8


def calibration_method(name):
    methods = {
        "minmax": CalibrationMethod.MinMax,
        "entropy": CalibrationMethod.Entropy,
        "percentile": getattr(CalibrationMethod, "Percentile", CalibrationMethod.MinMax),
    }
    return methods[name]


def remove_softmax_clip_nodes(model):
    producer_by_output = {}
    for node in model.graph.node:
        for output_name in node.output:
            producer_by_output[output_name] = node

    output_replacements = {}
    removed_node_indexes = set()
    for node_index, node in enumerate(model.graph.node):
        if node.op_type != "Clip" or not node.input:
            continue
        producer = producer_by_output.get(node.input[0])
        if producer is None or producer.op_type != "Softmax":
            continue
        for output_name in node.output:
            output_replacements[output_name] = node.input[0]
        removed_node_indexes.add(node_index)

    if not output_replacements:
        return 0

    for node_index, node in enumerate(model.graph.node):
        if node_index in removed_node_indexes:
            continue
        for index, input_name in enumerate(node.input):
            if input_name in output_replacements:
                node.input[index] = output_replacements[input_name]

    kept_nodes = [
        node for node_index, node in enumerate(model.graph.node)
        if node_index not in removed_node_indexes
    ]
    del model.graph.node[:]
    model.graph.node.extend(kept_nodes)
    return len(removed_node_indexes)


def preprocess_onnx(input_path, output_path):
    model = onnx.load(input_path)
    initializer_names = {initializer.name for initializer in model.graph.initializer}
    original_input_count = len(model.graph.input)
    for index in reversed(range(len(model.graph.input))):
        if model.graph.input[index].name in initializer_names:
            del model.graph.input[index]
    removed = original_input_count - len(model.graph.input)
    if removed:
        print(f"Removed {removed} initializers from graph inputs")
    removed_softmax_clips = remove_softmax_clip_nodes(model)
    if removed_softmax_clips:
        print(f"Removed {removed_softmax_clips} redundant Clip nodes after Softmax")
    onnx.save(model, output_path)
    return str(output_path)


def collect_nodes_to_exclude(model_path, preset, user_excludes):
    excludes = set(user_excludes or [])
    if preset == "full":
        return list(excludes)

    model = onnx.load(model_path)
    for node in model.graph.node:
        if preset == "accuracy" and node.op_type != "Conv" and node.name:
            excludes.add(node.name)
        elif preset in {"balanced", "aggressive"} and node.op_type not in {"Conv", "Gemm", "MatMul"} and node.name:
            excludes.add(node.name)
        elif preset == "balanced" and (
            "/transformer/decoder/" in node.name or node.name.startswith("/transformer/decoder/")
        ):
            excludes.add(node.name)
    return list(excludes)


def resolve_op_types(args):
    if args.op_types is not None:
        return args.op_types
    if args.preset == "accuracy":
        return ["Conv"]
    if args.preset in {"balanced", "aggressive"}:
        return ["Conv", "Gemm", "MatMul"]
    return None


def resolve_quant_format(args):
    if args.format == "qoperator":
        return QuantFormat.QOperator
    if args.format == "qdq":
        return QuantFormat.QDQ
    if args.preset == "accuracy":
        return QuantFormat.QOperator
    return QuantFormat.QDQ


def call_quantizer(fn, **kwargs):
    parameters = inspect.signature(fn).parameters
    supported_kwargs = {key: value for key, value in kwargs.items() if key in parameters}
    skipped = sorted(set(kwargs) - set(supported_kwargs))
    if skipped:
        print(f"Skipping unsupported ONNX Runtime quantizer args: {skipped}")
    return fn(**supported_kwargs)


def main(args):
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    model_input = args.input

    if args.per_channel:
        print(
            "Warning: --per_channel can produce QLinearMatMul nodes that fail on "
            "ONNX Runtime CPU for transformer attention MatMul operations. If eval "
            "fails, re-quantize without --per_channel."
        )

    if not args.skip_preprocess:
        preprocessed = output.with_suffix(".preprocessed.onnx")
        model_input = preprocess_onnx(args.input, preprocessed)

    op_types = resolve_op_types(args)
    nodes_to_exclude = collect_nodes_to_exclude(model_input, args.preset, args.nodes_to_exclude)
    if op_types:
        print(f"Quantizing op types: {op_types}")
    elif args.preset == "accuracy":
        print("Quantization preset: accuracy (Conv nodes only)")
    elif args.preset == "balanced":
        print("Quantization preset: balanced (Conv/Gemm/MatMul except decoder)")
    elif args.preset == "aggressive":
        print("Quantization preset: aggressive (Conv/Gemm/MatMul everywhere)")
    quant_format = resolve_quant_format(args)
    print(f"Quantization format: {getattr(quant_format, 'name', quant_format)}")
    if nodes_to_exclude:
        print(f"Keeping {len(nodes_to_exclude)} nodes in FP32")

    if args.method == "dynamic":
        call_quantizer(
            quantize_dynamic,
            model_input=model_input,
            model_output=str(output),
            per_channel=args.per_channel,
            reduce_range=args.reduce_range,
            weight_type=quant_type(args.weight_type),
            op_types_to_quantize=op_types,
            nodes_to_exclude=nodes_to_exclude,
        )
    else:
        image_paths = load_coco_image_paths(
            args.coco_path,
            args.ann_file,
            args.image_dir,
            args.calib_count,
        )
        print(f"Calibrating with {len(image_paths)} images")
        reader = CocoCalibrationDataReader(
            image_paths,
            args.input_name,
            args.image_size,
            args.batch_size,
        )
        quantize_kwargs = {
            "model_input": model_input,
            "model_output": str(output),
            "calibration_data_reader": reader,
            "quant_format": quant_format,
            "calibrate_method": calibration_method(args.calibration_method),
            "activation_type": quant_type(args.activation_type),
            "weight_type": quant_type(args.weight_type),
            "per_channel": args.per_channel,
            "reduce_range": args.reduce_range,
            "op_types_to_quantize": op_types,
            "nodes_to_exclude": nodes_to_exclude,
            "execution_providers": args.calibration_providers,
            "extra_options": {
                "ForceQuantizeNoInputCheck": args.force_quantize_no_input_check,
            },
        }
        try:
            call_quantizer(quantize_static, **quantize_kwargs)
        except RuntimeError as exc:
            message = str(exc)
            if "Only an existing tensor can be modified" in message:
                print("ORT range adjustment touched an uncalibrated tensor; retrying with ForceQuantizeNoInputCheck.")
                reader = CocoCalibrationDataReader(
                    image_paths,
                    args.input_name,
                    args.image_size,
                    args.batch_size,
                )
                quantize_kwargs["calibration_data_reader"] = reader
                quantize_kwargs["extra_options"] = {
                    **quantize_kwargs.get("extra_options", {}),
                    "ForceQuantizeNoInputCheck": True,
                }
                call_quantizer(quantize_static, **quantize_kwargs)
                print(f"Saved quantized ONNX model: {output}")
                return
            if quant_format == QuantFormat.QOperator:
                raise
            print("QDQ quantization failed while adjusting an uncalibrated tensor range; retrying as QOperator.")
            reader = CocoCalibrationDataReader(
                image_paths,
                args.input_name,
                args.image_size,
                args.batch_size,
            )
            quantize_kwargs["calibration_data_reader"] = reader
            quantize_kwargs["quant_format"] = QuantFormat.QOperator
            call_quantizer(quantize_static, **quantize_kwargs)

    print(f"Saved quantized ONNX model: {output}")


if __name__ == "__main__":
    main(parse_args())