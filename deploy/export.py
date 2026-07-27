# ------------------------------------------------------------------------
# LW-DETR
# Copyright (c) 2024 Baidu. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""
export ONNX model and TensorRT engine for deployment
"""
import os
import ast
import random
import argparse
import subprocess
from pathlib import Path

import onnx
import torch
import onnxsim
import numpy as np
from PIL import Image

import util.misc as utils
import datasets.transforms as T
from models import build_model
from quant_models import build_model_quant
from quant_models.fake_quantized.model_utils import configure_softmax_m_calibration
from workflows.common import (
    adapt_checkpoint_keys_for_model,
    fuse_bn_into_conv_in_state_dict,
    print_load_state,
)
from main import get_args_parser
from deploy._onnx import OnnxOptimizer


def run_command_shell(command, dry_run:bool = False) -> int:
    if dry_run:
        print("")
        print(f"CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']} {command}")
        print("")
        status = 0
    else:
        status = subprocess.call(command, shell=True)
    return status


def make_infer_image(args, device="cuda"):
    if args.infer_dir is None:
        dummy = np.random.randint(0, 256, (args.shape[0], args.shape[1], 3), dtype=np.uint8)
        image = Image.fromarray(dummy, mode="RGB")
    else:
        image = Image.open(args.infer_dir).convert("RGB")

    transforms = T.Compose([
        T.SquareResize([args.shape[0]]),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    inps, _ = transforms(image, None)
    inps = inps.to(device)
    # inps = utils.nested_tensor_from_tensor_list([inps for _ in range(args.batch_size)])
    inps = torch.stack([inps for _ in range(args.batch_size)])
    return inps


def export_onnx(model, args, input_names, input_tensors, output_names, dynamic_axes):
    export_name = "inference_model"
    output_file = os.path.join(args.output_dir, f"{export_name}.onnx")
    
    # dry_run
    if args.aidge_export and hasattr(model, "aidge_export"):
        model.aidge_export()
    else:
        model.export()
    # outputs = model(input_tensors)

    torch.onnx.export(
        model,
        input_tensors,
        output_file,
        input_names=input_names,
        output_names=output_names,
        export_params=True,
        keep_initializers_as_inputs=False,
        do_constant_folding=True,
        verbose=args.verbose,
        opset_version=args.opset_version,
        dynamic_axes=dynamic_axes)

    print(f'Successfully exported ONNX model: {output_file}')
    return output_file


def onnx_simplify(onnx_dir:str, input_names, input_tensors, force=False):
    sim_onnx_dir = onnx_dir.replace(".onnx", ".sim.onnx")
    if os.path.isfile(sim_onnx_dir) and not force:
        return sim_onnx_dir
    
    if isinstance(input_tensors, torch.Tensor):
        input_tensors = [input_tensors]
    
    print(f'start simplify ONNX model: {onnx_dir}')
    opt = OnnxOptimizer(onnx_dir)
    opt.info('Model: original')
    opt.common_opt()
    opt.info('Model: optimized')
    opt.save_onnx(sim_onnx_dir)
    return sim_onnx_dir
    input_dict = {name: tensor.detach().cpu().numpy() for name, tensor in zip(input_names, input_tensors)}
    model_opt, check_ok = onnxsim.simplify(
        onnx_dir,
        check_n = 3,
        input_data=input_dict,
        dynamic_input_shape=False)
    if check_ok:
        onnx.save(model_opt, sim_onnx_dir)
    else:
        raise RuntimeError("Failed to simplify ONNX model.")
    print(f'Successfully simplified ONNX model: {sim_onnx_dir}')
    return sim_onnx_dir


def trtexec(onnx_dir:str, args) -> None:
    engine_dir = onnx_dir.replace(".onnx", f".engine")
    addition = "--useCudaGraph --useSpinWait --warmUp=500 --avgRuns=1000"
    verbose = "--verbose" if args.verbose else ""
    command = " ".join([
        "trtexec",
            f"--onnx={onnx_dir}",
            f"--saveEngine={engine_dir}",
            f"--workspace=4096 --fp16",
            f"{addition}",
            f"{verbose}"])

    status = run_command_shell(command, args.dry_run)
    assert status == 0, f"error({status}) in infer command: {command}"
    print(f'Successfully serialized TensorRT engine: {engine_dir}')
    return engine_dir


def build_export_model(args):
    if args.quant or args.calib or args.qat:
        model, criterion, postprocessors = build_model_quant(args)
        softmax_count = configure_softmax_m_calibration(
            model,
            calibrate_m=args.softmax_calibrate_m,
            m_guard_bits=args.softmax_m_safe_guard,
            m_calib_mode=args.softmax_m_calib_mode,
            m_percentile=args.softmax_m_percentile,
            m_ema_momentum=args.softmax_m_ema_momentum,
        )
        print(
            "Configured quantized export model: "
            f"softmax_count={softmax_count}, "
            f"softmax_calibrate_m={args.softmax_calibrate_m}"
        )
        return model, criterion, postprocessors

    return build_model(args)


def load_export_checkpoint(model, args):
    checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False)
    state_dict = checkpoint['model'] if isinstance(checkpoint, dict) and 'model' in checkpoint else checkpoint

    if args.quant or args.qat or args.calib:
        state_dict = fuse_bn_into_conv_in_state_dict(state_dict)

    state_dict = adapt_checkpoint_keys_for_model(state_dict, model)
    resize_model_buffers_for_checkpoint(model, state_dict)
    state_dict = reshape_compatible_checkpoint_tensors(state_dict, model)
    load_state = model.load_state_dict(state_dict, strict=False)
    print_load_state("export checkpoint load", load_state)
    print(f"load checkpoints {args.resume}")


def resize_model_buffers_for_checkpoint(model, state_dict):
    resizable_suffixes = (
        "act_scaling_factor",
        "out_scaling_factor",
        "act_scale_mantissa",
        "act_scale_shift",
        "min_val",
        "max_val",
    )
    modules = dict(model.named_modules())
    resized = []

    for key, tensor in state_dict.items():
        if not key.endswith(resizable_suffixes) or not hasattr(tensor, "shape"):
            continue

        module_name, sep, buffer_name = key.rpartition(".")
        if not sep:
            continue
        module = modules.get(module_name)
        if module is None or buffer_name not in module._buffers:
            continue

        current = module._buffers[buffer_name]
        if current is None or current.shape == tensor.shape:
            continue
        if current.numel() == 1 and tensor.numel() > 1:
            module._buffers[buffer_name] = current.new_empty(tensor.shape)
            resized.append((key, tuple(current.shape), tuple(tensor.shape)))

    if resized:
        print(
            "Resized export model quant buffers from checkpoint: "
            f"{len(resized)} buffer(s)."
        )
        print(f"First resized buffers: {resized[:20]}")


def reshape_compatible_checkpoint_tensors(state_dict, model):
    model_state = model.state_dict()
    reshaped = {}
    changed = []
    for key, tensor in state_dict.items():
        target = model_state.get(key)
        if (
            target is not None
            and hasattr(tensor, "shape")
            and tensor.shape != target.shape
            and tensor.numel() == target.numel()
        ):
            tensor = tensor.reshape(target.shape)
            changed.append((key, tuple(tensor.shape)))
        reshaped[key] = tensor

    if changed:
        print(
            "Reshaped compatible checkpoint tensors: "
            f"{len(changed)} scalar/vector buffer(s)."
        )
        print(f"First reshaped tensors: {changed[:20]}")

    return reshaped


def main(args):
    print("git:\n  {}\n".format(utils.get_sha()))
    print(args)
    # convert device to device_id
    if args.device == 'cuda':
        device_id = "0"
    elif args.device == 'cpu':
        device_id = ""
    else:
        device_id = str(int(args.device))
        args.device = f"cuda:{device_id}"

    # device for export onnx
    # TODO: export onnx with cuda failed with onnx error
    device = torch.device("cpu")
    os.environ["CUDA_VISIBLE_DEVICES"] = device_id

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model, criterion, postprocessors = build_export_model(args)

    if args.resume:
        load_export_checkpoint(model, args)

    model.to(device)

    input_tensors = make_infer_image(args, device)
    input_names = ['input']
    output_names = ['dets', 'labels']
    dynamic_axes = None

    output_file = export_onnx(model, args, input_names, input_tensors, output_names, dynamic_axes)
    
    if args.simplify:
        output_file = onnx_simplify(output_file, input_names, input_tensors)

    if args.tensorrt:
        output_file = trtexec(output_file, args)
