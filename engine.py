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
import math
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

    return model

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
