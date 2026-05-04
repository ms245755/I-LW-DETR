# ------------------------------------------------------------------------
# LW-DETR
# ------------------------------------------------------------------------

"""
Inspect a TensorRT engine and summarize layer precisions.
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import tensorrt as trt


def parse_args():
    parser = argparse.ArgumentParser("Inspect TensorRT engine")
    parser.add_argument("--engine", required=True, help="Path to TensorRT .engine file.")
    parser.add_argument("--output", default=None, help="Optional JSON report path.")
    parser.add_argument("--show_layers", action="store_true", help="Print each layer from the engine inspector.")
    parser.add_argument(
        "--show_quantized_layers",
        action="store_true",
        help="Print every layer that appears to use INT8 tensors or INT8 tactics.",
    )
    return parser.parse_args()


def infer_layer_precision(layer):
    if isinstance(layer, str):
        return "unknown"

    explicit = layer.get("Precision") or layer.get("precision")
    if explicit:
        return str(explicit)

    weight = layer.get("Weights")
    if isinstance(weight, dict) and "Int8" in str(weight.get("Type", "")):
        return "INT8"

    tensors = list(layer.get("Inputs", [])) + list(layer.get("Outputs", []))
    if any("Int8" in tensor.get("Format/Datatype", "") for tensor in tensors):
        return "INT8"

    tactic = str(layer.get("TacticName", "")).lower()
    if "int8" in tactic:
        return "INT8"
    if "fp16" in tactic or "f16" in tactic:
        return "FP16"
    if "f32" in tactic or "tf32" in tactic:
        return "FP32/TF32"

    if any("Float" in tensor.get("Format/Datatype", "") for tensor in tensors):
        return "FP32"
    return "unknown"


COMPUTE_LAYER_TYPES = {
    "CaskConvolution",
    "CaskGemmMatrixMultiply",
    "Convolution",
    "Deconvolution",
    "FullyConnected",
    "MatrixMultiply",
    "Gemm",
    "PointWise",
    "PointWiseV2",
    "Pooling",
    "Reduce",
    "Scale",
    "SoftMax",
    "Softmax",
    "LayerNorm",
    "Normalization",
}

GLUE_LAYER_TYPES = {
    "Reformat",
    "Shuffle",
    "NoOp",
    "Constant",
    "Shape",
    "Slice",
    "Gather",
    "TopK",
    "Resize",
    "Concatenation",
    "Identity",
    "Cast",
}


def get_layer_type(layer):
    if isinstance(layer, str):
        return "unknown"
    return layer.get("LayerType") or layer.get("layerType") or layer.get("Type") or "unknown"


def get_layer_name(layer):
    if isinstance(layer, str):
        return layer
    return layer.get("Name") or layer.get("name") or "<unnamed>"


def classify_layer_kind(layer):
    layer_type = get_layer_type(layer)
    name = get_layer_name(layer)
    if layer_type in GLUE_LAYER_TYPES:
        return "glue/helper"
    if layer_type in COMPUTE_LAYER_TYPES:
        return "compute"
    if any(token in name for token in ("MatMul", "Conv", "Gemm", "Add", "Mul", "Div", "Relu", "Sigmoid", "Softmax", "LayerNormalization")):
        return "compute"
    if any(token in name for token in ("Reformat", "reshape", "Reshape", "Transpose", "copy", "Shuffle", "Unsqueeze", "Squeeze")):
        return "glue/helper"
    return "other/unknown"


def classify_model_area(layer):
    name = get_layer_name(layer)
    if "patch_embed" in name:
        return "patch_embed"
    if "projector" in name:
        return "projector/neck conv"
    if "/encoder/blocks." in name and "/mlp/" in name:
        return "ViT encoder MLP"
    if "/encoder/blocks." in name and "/attn/" in name:
        return "ViT encoder attention"
    if "/transformer/decoder/layers." in name and "/linear" in name:
        return "decoder FFN linear"
    if "class_embed" in name or "bbox_embed" in name:
        return "prediction heads"
    if "input_proj" in name:
        return "input projection"
    if "self_attn" in name:
        return "decoder self-attention"
    if any(token in name for token in ("cross_attn", "sampling_offsets", "attention_weights", "value_proj", "output_proj", "out_proj")):
        return "decoder cross/deformable attention"
    if "LayerNormalization" in name or "norm" in name:
        return "layer norm"
    if "Softmax" in name:
        return "softmax"
    if any(token in name for token in ("Reformat", "copy", "reshape", "Reshape", "Transpose", "Shuffle")):
        return "reformat/reshape glue"
    return "other"


def summarize_layers(layers):
    precision_counts = Counter()
    type_counts = Counter()
    kind_precision_counts = Counter()
    int8_type_counts = Counter()
    int8_kind_counts = Counter()
    int8_area_counts = Counter()
    enriched_layers = []

    for layer in layers:
        layer_type = get_layer_type(layer)
        precision = infer_layer_precision(layer)
        kind = classify_layer_kind(layer)
        area = classify_model_area(layer)

        precision_counts[str(precision)] += 1
        type_counts[str(layer_type)] += 1
        kind_precision_counts[f"{kind}:{precision}"] += 1

        if precision == "INT8":
            int8_type_counts[str(layer_type)] += 1
            int8_kind_counts[kind] += 1
            int8_area_counts[area] += 1

        enriched_layers.append({
            "name": get_layer_name(layer),
            "type": layer_type,
            "precision": precision,
            "kind": kind,
            "model_area": area,
        })

    return {
        "precision_counts": dict(precision_counts),
        "type_counts": dict(type_counts),
        "kind_precision_counts": dict(kind_precision_counts),
        "int8_type_counts": dict(int8_type_counts),
        "int8_kind_counts": dict(int8_kind_counts),
        "int8_area_counts": dict(int8_area_counts),
        "enriched_layers": enriched_layers,
    }


def inspect_engine(args):
    logger = trt.Logger(trt.Logger.INFO)
    trt.init_libnvinfer_plugins(logger, "")

    engine_path = Path(args.engine)
    with engine_path.open("rb") as f, trt.Runtime(logger) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())

    if engine is None:
        raise RuntimeError(f"Failed to load TensorRT engine: {engine_path}")

    report = {
        "engine": str(engine_path),
        "bindings": [],
        "layer_precision_counts": {},
        "layer_type_counts": {},
        "layer_kind_precision_counts": {},
        "int8_layer_type_counts": {},
        "int8_layer_kind_counts": {},
        "int8_model_area_counts": {},
        "layer_summary": [],
        "layers": [],
    }

    for index in range(engine.num_io_tensors):
        name = engine.get_tensor_name(index)
        report["bindings"].append({
            "name": name,
            "mode": str(engine.get_tensor_mode(name)),
            "dtype": str(engine.get_tensor_dtype(name)),
            "shape": tuple(engine.get_tensor_shape(name)),
        })

    inspector = engine.create_engine_inspector()
    layer_json = None
    if hasattr(trt, "LayerInformationFormat"):
        try:
            layer_json = inspector.get_engine_information(trt.LayerInformationFormat.JSON)
        except TypeError:
            layer_json = None

    if layer_json:
        layers = json.loads(layer_json)
        if isinstance(layers, dict) and "Layers" in layers:
            layers = layers["Layers"]
        if isinstance(layers, list):
            report["layers"] = layers
        elif isinstance(layers, str):
            report["layers"] = [{"Name": line} for line in layers.splitlines() if line.strip()]
        else:
            report["layers"] = []

        summary = summarize_layers(report["layers"])
        report["layer_precision_counts"] = summary["precision_counts"]
        report["layer_type_counts"] = summary["type_counts"]
        report["layer_kind_precision_counts"] = summary["kind_precision_counts"]
        report["int8_layer_type_counts"] = summary["int8_type_counts"]
        report["int8_layer_kind_counts"] = summary["int8_kind_counts"]
        report["int8_model_area_counts"] = summary["int8_area_counts"]
        report["layer_summary"] = summary["enriched_layers"]

    print("Bindings:")
    for binding in report["bindings"]:
        print(f"  {binding['mode']} {binding['name']}: {binding['dtype']} {binding['shape']}")

    if report["layer_precision_counts"]:
        print("\nLayer precision counts:")
        for precision, count in sorted(report["layer_precision_counts"].items()):
            print(f"  {precision}: {count}")

        print("\nINT8 real compute vs glue/helper:")
        for kind, count in sorted(report["int8_layer_kind_counts"].items()):
            print(f"  {kind}: {count}")

        print("\nINT8 compute layer types:")
        int8_compute_types = Counter()
        for layer in report["layer_summary"]:
            if layer["precision"] == "INT8" and layer["kind"] == "compute":
                int8_compute_types[layer["type"]] += 1
        if int8_compute_types:
            for layer_type, count in sorted(int8_compute_types.items()):
                print(f"  {layer_type}: {count}")
        else:
            print("  none")

        print("\nINT8 model areas:")
        for area, count in sorted(report["int8_model_area_counts"].items(), key=lambda item: (-item[1], item[0])):
            print(f"  {area}: {count}")
    else:
        print("\nLayer precision details are not available from this TensorRT build.")

    if args.show_quantized_layers and report["layer_summary"]:
        print("\nQuantized layers:")
        for layer in report["layer_summary"]:
            if layer["precision"] == "INT8":
                print(f"  {layer['kind']:>13} {layer['type']:>22} {layer['model_area']:>34} {layer['name']}")

    if args.show_layers:
        print("\nLayers:")
        if report["layers"]:
            for layer in report["layers"]:
                name = get_layer_name(layer)
                layer_type = get_layer_type(layer)
                precision = infer_layer_precision(layer)
                print(f"  {precision:>8} {layer_type:>16} {name}")
        else:
            try:
                print(inspector.get_engine_information(trt.LayerInformationFormat.ONELINE))
            except Exception:
                pass

    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nSaved report: {output}")


if __name__ == "__main__":
    inspect_engine(parse_args())
