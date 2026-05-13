# ------------------------------------------------------------------------
# LW-DETR
# Copyright (c) 2024 Baidu. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Conditional DETR
# Copyright (c) 2021 Microsoft. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Copied from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------

"""
Train and eval functions used in main.py
"""
import fnmatch
import math
import re
import sys
from typing import Iterable
import random
from torch.utils.data import Subset, DataLoader

import torch
import util.misc as utils
from datasets.coco_eval import CocoEvaluator
from quant_models.fake_quantized.model_utils import *


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float=0,
                    ema_m: torch.nn.Module=None, schedules: dict={}, 
                    num_training_steps_per_epoch=None, vit_encoder_num_layers=None, args=None):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10
    start_steps=epoch * num_training_steps_per_epoch

    for data_iter_step, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        it = start_steps + data_iter_step
        if 'dp' in schedules:
            if args.distributed:
                model.module.update_drop_path(schedules['dp'][it], vit_encoder_num_layers)
            else:
                model.update_drop_path(schedules['dp'][it], vit_encoder_num_layers)
        if 'do' in schedules:
            if args.distributed:
                model.module.update_dropout(schedules['do'][it])
            else:
                model.update_dropout(schedules['do'][it])
        
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        outputs = model(samples, targets)
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            
        for i, group in enumerate(optimizer.param_groups):
            for p in group["params"]:
                if p.grad is None:
                    continue
                st = optimizer.state.get(p, {})
                if "exp_avg" in st and st["exp_avg"].shape != p.grad.shape:
                    print("MISMATCH!")
                    print("param shape:", tuple(p.shape))
                    print("grad shape :", tuple(p.grad.shape))
                    print("exp_avg    :", tuple(st["exp_avg"].shape))
                    raise SystemExit
        optimizer.step()
        if ema_m is not None:
            if epoch >= 0:
                ema_m.update(model)
        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(model, criterion, postprocessors, data_loader, base_ds, device, args = None):
    model.eval()
    if args.fp16_eval:
        model.half()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
    coco_evaluator = CocoEvaluator(base_ds, iou_types)

    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        if args.fp16_eval:
            samples.tensors = samples.tensors.half()
        
        outputs = model(samples)

        if args.fp16_eval:
            for key in outputs.keys():
                if key == 'enc_outputs':
                    for sub_key in outputs[key].keys():
                        outputs[key][sub_key] = outputs[key][sub_key].float()
                elif key == 'aux_outputs':
                    for idx in range(len(outputs[key])):
                        for sub_key in outputs[key][idx].keys():
                            outputs[key][idx][sub_key] = outputs[key][idx][sub_key].float()
                else:
                    outputs[key] = outputs[key].float()

        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
                             **loss_dict_reduced_scaled,
                             **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors['bbox'](outputs, orig_target_sizes)
        res = {target['image_id'].item(): output for target, output in zip(targets, results)}
        if coco_evaluator is not None:
            coco_evaluator.update(res)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if coco_evaluator is not None:
        if 'bbox' in postprocessors.keys():
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
        if 'segm' in postprocessors.keys():
            stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()
    return stats, coco_evaluator

import torch

@torch.no_grad()
def calibrate(
    model,
    data_loader,
    device,
    args=None,
    calib_batches: int = 200,
):
    """
    Run PTQ-style calibration for the quantized model by collecting activation stats
    (e.g., IViTAct.running_stat=True) on representative data.

    This mirrors the structure of evaluate(), but:
      - no criterion / postprocessors
      - no losses / coco eval
      - only forward passes
      - freezes stats at the end

    Args:
        model: quantized model (already loaded with fp32 checkpoint)
        data_loader: calibration data loader (can reuse val loader)
        device: torch.device
        args: should have fp16_eval optionally (same as evaluate)
        calib_batches: number of batches to run for calibration
    """
    model.eval()

    unfreeze_model(model=model)

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Calibrate:"

    for i, (samples, targets) in enumerate(metric_logger.log_every(data_loader, 1, header)):
        if i >= calib_batches:
            break

        samples = samples.to(device)

        # targets are not used, but keep the same device move pattern
        if targets is not None:
            _ = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # if args is not None and getattr(args, "fp16_eval", False):
        #     samples.tensors = samples.tensors.half()
        
        _ = model(samples)

        # You can log something lightweight if desired (optional)
        metric_logger.update(calib_batches=i + 1)

    metric_logger.synchronize_between_processes()
    print("Calibration done. Processed batches:", calib_batches)

    freeze_model(model)
    softmax_m_summary = get_softmax_m_summary(model)
    if softmax_m_summary is not None:
        print(
            "Calibrated IViTSoftmax m: "
            f"count={softmax_m_summary['count']}, "
            f"min={softmax_m_summary['min']}, "
            f"max={softmax_m_summary['max']}"
        )
    print_softmax_m(model)

    return model


def _unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def _flatten_tensors(value):
    if torch.is_tensor(value):
        return [value]
    if isinstance(value, (list, tuple)):
        tensors = []
        for item in value:
            tensors.extend(_flatten_tensors(item))
        return tensors
    if isinstance(value, dict):
        tensors = []
        for item in value.values():
            tensors.extend(_flatten_tensors(item))
        return tensors
    if hasattr(value, "tensors") and torch.is_tensor(value.tensors):
        return [value.tensors]
    return []


def _pair_output_tensors(student_output, teacher_output):
    student_tensors = _flatten_tensors(student_output)
    teacher_tensors = _flatten_tensors(teacher_output)
    pairs = []
    used_teacher = set()
    for student_tensor in student_tensors:
        if student_tensor.numel() == 0:
            continue
        for teacher_idx, teacher_tensor in enumerate(teacher_tensors):
            if teacher_idx in used_teacher:
                continue
            if student_tensor.shape == teacher_tensor.shape:
                pairs.append((student_tensor.float(), teacher_tensor.detach().float()))
                used_teacher.add(teacher_idx)
                break
    return pairs


def _normalized_mse(student_tensor, teacher_tensor, eps=1e-6):
    denom = teacher_tensor.detach().float().pow(2).mean().clamp_min(eps)
    return torch.nn.functional.mse_loss(student_tensor.float(), teacher_tensor.detach().float()) / denom


def _select_reconstruction_modules(student_model, teacher_model, patterns):
    student_modules = dict(_unwrap_model(student_model).named_modules())
    teacher_modules = dict(_unwrap_model(teacher_model).named_modules())
    common_names = sorted(set(student_modules).intersection(teacher_modules))
    selected = []

    if patterns == ["auto"]:
        auto_patterns = [
            re.compile(r"backbone\.0\.encoder\.blocks\.\d+"),
            re.compile(r"backbone\.0\.projector\.stages\.\d+"),
            re.compile(r"backbone\.0\.projector\.stages_sampling\.\d+\.\d+"),
            re.compile(r"transformer\.decoder\.layers\.\d+"),
        ]
        for name in common_names:
            if any(pattern.fullmatch(name) for pattern in auto_patterns):
                selected.append(name)
        return selected

    for pattern in patterns:
        selected.extend(name for name in common_names if fnmatch.fnmatch(name, pattern))
    return sorted(dict.fromkeys(selected))


def _set_trainable_module(model, module_name):
    model = _unwrap_model(model)
    for param in model.parameters():
        param.requires_grad = False
    module = dict(model.named_modules())[module_name]
    params = []
    for param in module.parameters(recurse=True):
        param.requires_grad = True
        params.append(param)
    return params


def _run_reconstruction_step(student_model, teacher_model, module_name, samples):
    student_cache = {}
    teacher_cache = {}
    student_module = dict(_unwrap_model(student_model).named_modules())[module_name]
    teacher_module = dict(_unwrap_model(teacher_model).named_modules())[module_name]

    def student_hook(_module, _inputs, output):
        student_cache["output"] = output

    def teacher_hook(_module, _inputs, output):
        teacher_cache["output"] = output

    student_handle = student_module.register_forward_hook(student_hook)
    teacher_handle = teacher_module.register_forward_hook(teacher_hook)
    try:
        with torch.no_grad():
            _ = teacher_model(samples)
        _ = student_model(samples)
    finally:
        student_handle.remove()
        teacher_handle.remove()

    pairs = _pair_output_tensors(student_cache.get("output"), teacher_cache.get("output"))
    if not pairs:
        return None
    return sum(_normalized_mse(student, teacher) for student, teacher in pairs) / len(pairs)


def _output_distill_loss(student_output, teacher_output, logit_weight=1.0, box_weight=5.0, temperature=2.0):
    losses = []
    if isinstance(student_output, dict) and isinstance(teacher_output, dict):
        if "pred_logits" in student_output and "pred_logits" in teacher_output:
            student_logits = student_output["pred_logits"].float()
            teacher_logits = teacher_output["pred_logits"].detach().float()
            if student_logits.shape == teacher_logits.shape:
                log_student = torch.nn.functional.log_softmax(student_logits / temperature, dim=-1)
                prob_teacher = torch.nn.functional.softmax(teacher_logits / temperature, dim=-1)
                losses.append(
                    logit_weight
                    * temperature
                    * temperature
                    * torch.nn.functional.kl_div(log_student, prob_teacher, reduction="batchmean")
                )
        if "pred_boxes" in student_output and "pred_boxes" in teacher_output:
            student_boxes = student_output["pred_boxes"].float()
            teacher_boxes = teacher_output["pred_boxes"].detach().float()
            if student_boxes.shape == teacher_boxes.shape:
                losses.append(box_weight * torch.nn.functional.l1_loss(student_boxes, teacher_boxes))
        if "aux_outputs" in student_output and "aux_outputs" in teacher_output:
            for student_aux, teacher_aux in zip(student_output["aux_outputs"], teacher_output["aux_outputs"]):
                aux_loss = _output_distill_loss(
                    student_aux,
                    teacher_aux,
                    logit_weight=logit_weight,
                    box_weight=box_weight,
                    temperature=temperature,
                )
                if aux_loss is not None:
                    losses.append(0.5 * aux_loss)
        if "enc_outputs" in student_output and "enc_outputs" in teacher_output:
            enc_loss = _output_distill_loss(
                student_output["enc_outputs"],
                teacher_output["enc_outputs"],
                logit_weight=logit_weight,
                box_weight=box_weight,
                temperature=temperature,
            )
            if enc_loss is not None:
                losses.append(0.5 * enc_loss)
    if not losses:
        return None
    return sum(losses) / len(losses)


def reconstruct_output_distill(
    student_model,
    teacher_model,
    data_loader,
    device,
    batches=0,
    lr=1e-6,
    weight_decay=0.0,
    max_norm=0.1,
    logit_weight=1.0,
    box_weight=5.0,
    temperature=2.0,
):
    if batches <= 0:
        return {}

    student_model.eval()
    teacher_model.eval()
    freeze_model(student_model)
    for param in teacher_model.parameters():
        param.requires_grad = False
    for param in _unwrap_model(student_model).parameters():
        param.requires_grad = True

    optimizer = torch.optim.AdamW(
        [param for param in _unwrap_model(student_model).parameters() if param.requires_grad],
        lr=lr,
        weight_decay=weight_decay,
    )
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "PTQ output distill:"
    losses = []

    for batch_idx, (samples, _targets) in enumerate(metric_logger.log_every(data_loader, 1, header)):
        if batch_idx >= batches:
            break
        samples = samples.to(device)
        with torch.no_grad():
            teacher_output = teacher_model(samples)
        student_output = student_model(samples)
        loss = _output_distill_loss(
            student_output,
            teacher_output,
            logit_weight=logit_weight,
            box_weight=box_weight,
            temperature=temperature,
        )
        if loss is None:
            print("PTQ output distill skipped: no matching output tensors")
            break
        if not math.isfinite(loss.item()):
            raise RuntimeError(f"PTQ output distill loss is not finite: {loss.item()}")

        optimizer.zero_grad()
        loss.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(_unwrap_model(student_model).parameters(), max_norm)
        optimizer.step()

        loss_value = loss.item()
        losses.append(loss_value)
        metric_logger.update(ptq_output_distill_loss=loss_value)

    student_model.eval()
    freeze_model(student_model)
    metric_logger.synchronize_between_processes()
    if losses:
        avg_loss = sum(losses) / len(losses)
        print(f"PTQ output distill: loss={avg_loss:.6f}")
        return {"output_distill": avg_loss}
    return {}


def reconstruct_ptq(
    student_model,
    teacher_model,
    data_loader,
    device,
    args=None,
    module_patterns=None,
    batches_per_module=16,
    lr=1e-5,
    weight_decay=0.0,
    max_norm=0.1,
):
    """
    Block-wise unlabeled PTQ reconstruction.

    The quantized student is optimized to match the frozen float teacher at selected
    module outputs. Activation observers should already be calibrated and fixed.
    """
    student_model.eval()
    teacher_model.eval()
    freeze_model(student_model)
    for param in teacher_model.parameters():
        param.requires_grad = False

    patterns = module_patterns or ["auto"]
    module_names = _select_reconstruction_modules(student_model, teacher_model, patterns)
    if not module_names:
        print("PTQ reconstruction skipped: no matched modules for patterns:", patterns)
        return {}

    print("PTQ reconstruction modules:", module_names)
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "PTQ reconstruct:"
    stats = {}

    for module_idx, module_name in enumerate(module_names):
        params = _set_trainable_module(student_model, module_name)
        if not params:
            print(f"PTQ reconstruction skip {module_name}: no trainable parameters")
            continue

        optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
        module_losses = []
        for batch_idx, (samples, _targets) in enumerate(metric_logger.log_every(data_loader, 1, header)):
            if batch_idx >= batches_per_module:
                break
            samples = samples.to(device)
            loss = _run_reconstruction_step(student_model, teacher_model, module_name, samples)
            if loss is None:
                if batch_idx == 0:
                    print(f"PTQ reconstruction skip {module_name}: no same-shaped tensor outputs")
                break
            if not math.isfinite(loss.item()):
                raise RuntimeError(f"PTQ reconstruction loss is not finite for {module_name}: {loss.item()}")

            optimizer.zero_grad()
            loss.backward()
            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(params, max_norm)
            optimizer.step()

            loss_value = loss.item()
            module_losses.append(loss_value)
            metric_logger.update(ptq_reconstruct_loss=loss_value, ptq_reconstruct_module=module_idx)

        if module_losses:
            stats[module_name] = sum(module_losses) / len(module_losses)
            print(f"PTQ reconstruction {module_name}: loss={stats[module_name]:.6f}")

    for param in _unwrap_model(student_model).parameters():
        param.requires_grad = True
    student_model.eval()
    freeze_model(student_model)
    metric_logger.synchronize_between_processes()
    return stats

def make_random_calib_loader(original_dataset, batch_size, seed, n_samples=800, **loader_kwargs):
    # Sample a deterministic, coverage-oriented subset instead of a purely random one.
    # This makes calibration less sensitive to which exact images were drawn.
    dataset_len = len(original_dataset)
    if n_samples >= dataset_len:
        indices = list(range(dataset_len))
    else:
        rng = random.Random(seed)
        stride = dataset_len / float(n_samples)
        indices = []
        for i in range(n_samples):
            start = int(i * stride)
            end = int((i + 1) * stride)
            end = min(max(end, start + 1), dataset_len)
            indices.append(rng.randrange(start, end))

    subset = Subset(original_dataset, indices)
    
    # Return a new loader that only uses this random subset
    return DataLoader(subset, batch_size=batch_size, shuffle=False, **loader_kwargs)
