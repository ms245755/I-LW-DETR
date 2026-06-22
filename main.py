# ------------------------------------------------------------------------
# LW-DETR
# Copyright (c) 2024 Baidu. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from Conditional DETR (https://github.com/Atten4Vis/ConditionalDETR)
# Copyright (c) 2021 Microsoft. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------

"""
cleaned main file
"""
import argparse
import datetime
import json
import random
import time
import ast
import copy
import fnmatch
import math
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler
from torch.amp import GradScaler

from datasets import build_dataset, get_coco_api_from_dataset, resolve_num_classes
from engine import (
    evaluate,
    make_random_calib_loader,
    train_one_epoch,
    calibrate,
    reconstruct_ptq,
    reconstruct_output_distill,
)
from models import build_model
from quant_models import build_model_quant
from quant_models.fake_quantized.model_utils import (
    configure_softmax_m_calibration,
    freeze_model,
    unfreeze_model,
)
from util.drop_scheduler import drop_scheduler
from util.get_param_dicts import get_param_dict
import util.misc as utils
from util.utils import ModelEma, BestMetricHolder, clean_state_dict
from util.benchmark import benchmark


def sync_model_state_from_main(model):
    if not utils.is_dist_avail_and_initialized():
        return
    for tensor in model.state_dict().values():
        torch.distributed.broadcast(tensor, src=0)


def adapt_checkpoint_keys_for_model(state_dict, model):
    """Load float self-attention weights into quant q/k/v projections."""
    model_keys = set(model.state_dict().keys())
    adapted = dict(state_dict)
    for key, value in state_dict.items():
        if key.endswith("self_attn.in_proj_weight"):
            prefix = key[: -len("in_proj_weight")]
            q, k, v = value.chunk(3, dim=0)
            converted = False
            for name, tensor in (("q_proj.weight", q), ("k_proj.weight", k), ("v_proj.weight", v)):
                target_key = f"{prefix}{name}"
                if target_key in model_keys and target_key not in adapted:
                    adapted[target_key] = tensor
                    converted = True
            if converted and key not in model_keys:
                adapted.pop(key, None)
            continue
        if key.endswith("self_attn.in_proj_bias"):
            prefix = key[: -len("in_proj_bias")]
            q, k, v = value.chunk(3, dim=0)
            converted = False
            for name, tensor in (("q_proj.bias", q), ("k_proj.bias", k), ("v_proj.bias", v)):
                target_key = f"{prefix}{name}"
                if target_key in model_keys and target_key not in adapted:
                    adapted[target_key] = tensor
                    converted = True
            if converted and key not in model_keys:
                adapted.pop(key, None)
            continue
    return adapted


def print_load_state(prefix, load_state):
    if not utils.is_main_process():
        return
    missing = getattr(load_state, "missing_keys", [])
    unexpected = getattr(load_state, "unexpected_keys", [])
    missing_params = [k for k in missing if k.endswith(".weight") or k.endswith(".bias")]
    print(f"{prefix}: missing_keys={len(missing)}, unexpected_keys={len(unexpected)}")
    print(f"{prefix}: missing trainable weight/bias keys={len(missing_params)}")
    if missing_params:
        print(f"{prefix}: first missing trainable keys: {missing_params[:20]}")
    if missing:
        print(f"{prefix}: first missing keys: {missing[:20]}")
    if unexpected:
        print(f"{prefix}: first unexpected keys: {unexpected[:20]}")


def is_classification_head_parameter(name):
    return (
        name.startswith("class_embed.")
        or name.startswith("transformer.enc_out_class_embed.")
    )


def report_class_head_only_training(model):
    head_tensors = 0
    frozen_tensors = 0
    for name, parameter in model.named_parameters():
        is_head = is_classification_head_parameter(name)
        if is_head:
            head_tensors += 1
        elif parameter.requires_grad:
            frozen_tensors += 1
    print(
        "Class-head warm-up enabled: "
        f"updating {head_tensors} classification tensors; "
        f"suppressing updates for {frozen_tensors} other tensors."
    )


def build_lr_scheduler(optimizer, args):
    if args.lr_scheduler == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)

    decay_start = args.lr_decay_start_epoch
    decay_steps = max(1, args.epochs - decay_start - 1)
    min_factor = args.lr_min_factor

    def delayed_cosine(epoch):
        if epoch <= decay_start:
            return 1.0
        progress = min(1.0, (epoch - decay_start) / decay_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_factor + (1.0 - min_factor) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, delayed_cosine)


def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_encoder', default=1.5e-4, type=float)
    parser.add_argument('--lr_class_head', default=None, type=float,
                        help='Learning rate for class_embed and two-stage encoder '
                             'classification heads. Defaults to --lr.')
    parser.add_argument('--class_head_warmup_epochs', default=0, type=int,
                        help='Train only classification heads for the first N epochs, '
                             'then automatically unfreeze the full detector.')
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=12, type=int)
    parser.add_argument('--lr_drop', default=11, type=int)
    parser.add_argument('--lr_scheduler', default='step',
                        choices=('step', 'delayed_cosine'))
    parser.add_argument('--lr_decay_start_epoch', default=0, type=int,
                        help='Epoch where delayed cosine decay begins.')
    parser.add_argument('--lr_min_factor', default=0.01, type=float,
                        help='Final LR as a fraction of each parameter group LR.')
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')
    parser.add_argument('--lr_vit_layer_decay', default=0.8, type=float)
    parser.add_argument('--lr_component_decay', default=1.0, type=float)

    # drop args 
    # dropout and stochastic depth drop rate; set at most one to non-zero
    parser.add_argument('--dropout', type=float, default=0,
                        help='Drop path rate (default: 0.0)')
    parser.add_argument('--drop_path', type=float, default=0,
                        help='Drop path rate (default: 0.0)')

    # early / late dropout and stochastic depth settings
    parser.add_argument('--drop_mode', type=str, default='standard',
                        choices=['standard', 'early', 'late'], help='drop mode')
    parser.add_argument('--drop_schedule', type=str, default='constant',
                        choices=['constant', 'linear'],
                        help='drop schedule for early dropout / s.d. only')
    parser.add_argument('--cutoff_epoch', type=int, default=0,
                        help='if drop_mode is early / late, this is the epoch where dropout ends / starts')

    # Model parameters
    parser.add_argument('--pretrained_encoder', type=str, default=None, 
                        help="Path to the pretrained encoder.")
    parser.add_argument('--pretrain_weights', type=str, default=None, 
                        help="Path to the pretrained model.")
    parser.add_argument('--pretrain_exclude_keys', type=str, default=None, nargs='+', 
                        help="Keys you do not want to load.")
    parser.add_argument('--pretrain_keys_modify_to_load', type=str, default=None, nargs='+',
                        help="Keys you want to modify to load. Only used when loading objects365 pre-trained weights.")

    # * Backbone
    parser.add_argument('--encoder', default='vit_tiny', type=str,
                        help="Name of the transformer or convolutional encoder to use")
    parser.add_argument('--vit_encoder_num_layers', default=12, type=int,
                        help="Number of layers used in ViT encoder")
    parser.add_argument('--window_block_indexes', default=None, type=int, nargs='+')
    parser.add_argument('--position_embedding', default='sine', type=str, 
                        choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")
    parser.add_argument('--out_feature_indexes', default=[-1], type=int, nargs='+', help='only for vit now')
    parser.add_argument('--quant_model_components', default=None, type=str, nargs='+',
                        choices=('backbone', 'transformer'),
                        help='Top-level model components to quantize when --calib/--qat is used.')
    parser.add_argument('--backbone_quant_components', default=['vit', 'projector'], type=str, nargs='+',
                        choices=('vit', 'projector'),
                        help='Backbone components to quantize when --calib/--qat is used.')
    parser.add_argument('--encoder_quant_blocks', default=None, type=int, nargs='+',
                        help='ViT encoder block indexes to quantize. Default: all blocks.')
    parser.add_argument('--encoder_quant_components',
                        default=['linear', 'conv2d', 'matmul', 'softmax', 'gelu', 'layernorm'],
                        type=str, nargs='+',
                        choices=('linear', 'conv2d', 'matmul', 'softmax', 'gelu', 'layernorm'),
                        help='Operation families to quantize inside quantized ViT blocks.')
    parser.add_argument('--projector_quant_components',
                        default=['conv2d', 'gelu', 'layernorm'],
                        type=str, nargs='+',
                        choices=('linear', 'conv2d', 'matmul', 'softmax', 'gelu', 'layernorm', 'silu'),
                        help='Operation families to quantize inside the projector. Only conv2d, gelu, and layernorm are used by the projector today.')
    parser.add_argument('--transformer_quant_components',
                        default=['self_attn', 'cross_attn', 'ffn', 'norm', 'refpoint', 'two_stage', 'decoder_output_norm'],
                        type=str, nargs='+',
                        choices=(
                            'self_attn', 'cross_attn', 'ffn', 'norm', 'refpoint', 'two_stage', 'decoder_output_norm',
                            'none'
                        ),
                        help='Large transformer decoder blocks to quantize.')
    parser.add_argument('--decoder_quant_blocks', default=None, type=int, nargs='+',
                        help='Decoder layer indexes to quantize. Default: all decoder layers.')
    parser.add_argument('--decoder_quant_norms', default=None, type=str, nargs='+',
                        choices=('norm1', 'norm2', 'norm3', 'none'),
                        help='Decoder LayerNorm modules to quantize when transformer layernorm quantization is enabled. Default: all decoder norms.')
    parser.add_argument('--decoder_self_attn_quant_components',
                        default=None,
                        type=str, nargs='+',
                        choices=('linear', 'matmul', 'softmax', 'none'),
                        help='Subcomponents to quantize inside decoder self-attention when self_attn is enabled.')
    parser.add_argument('--decoder_cross_attn_quant_components',
                        default=None,
                        type=str, nargs='+',
                        choices=('value_proj', 'sampling_offsets', 'attention_weights', 'softmax', 'grid_sample_core', 'weighted_sum', 'output_proj', 'core', 'all', 'none'),
                        help='Subcomponents to quantize inside decoder deformable cross-attention when cross_attn is enabled.')
    parser.add_argument('--decoder_ffn_quant_components',
                        default=None,
                        type=str, nargs='+',
                        choices=('linear', 'none'),
                        help='Subcomponents to quantize inside decoder FFN when ffn is enabled.')
    parser.add_argument('--decoder_refpoint_quant_components',
                        default=None,
                        type=str, nargs='+',
                        choices=('linear', 'sine_embed', 'coord_proj', 'none'),
                        help='Subcomponents to quantize inside decoder reference-point/query-position path when refpoint is enabled.')
    parser.add_argument('--decoder_two_stage_quant_components',
                        default=None,
                        type=str, nargs='+',
                        choices=('proposal', 'enc_output', 'enc_output_norm', 'class_head', 'bbox_head', 'bbox_reparam', 'selected_inputs', 'none'),
                        help='Subcomponents to quantize inside the optional two-stage decoder proposal path when two_stage is enabled.')
    parser.add_argument('--decoder_output_norm_quant_components',
                        default=None,
                        type=str, nargs='+',
                        choices=('layernorm', 'output_qact', 'none'),
                        help='Subcomponents to quantize in the final decoder output norm when decoder_output_norm is enabled.')
    parser.add_argument('--lwdetr_quant_components',
                        default=['linear'],
                        type=str, nargs='+',
                        choices=('linear', 'none'),
                        help='Operation families to quantize inside LW-DETR prediction heads.')
    parser.add_argument('--transformer_exp_mode', default='ivit', type=str,
                        choices=('float', 'ivit'),
                        help='Use float exp or IViTExp in quantized transformer bbox reparameterization paths.')
    parser.add_argument('--lwdetr_exp_mode', default='ivit', type=str,
                        choices=('float', 'ivit'),
                        help='Use float exp or IViTExp in quantized LW-DETR bbox head reparameterization.')
    parser.add_argument('--act_observer', default='ema_minmax', type=str,
                        choices=('ema_minmax', 'percentile', 'mean'),
                        help='Activation observer used by IViTAct during calibration/QAT.')
    parser.add_argument('--act_percentile', default=99.9, type=float,
                        help='Percentile used when --act_observer percentile is selected.')
    parser.add_argument('--softmax_m_safe_guard', default=4, type=int,
                        help='Guard bits added when calibrating IViTSoftmax m.')
    parser.add_argument('--softmax_m_calib_mode', default='max', type=str,
                        choices=('max', 'percentile', 'ema_percentile'),
                        help='Statistic used to calibrate IViTSoftmax m.')
    parser.add_argument('--softmax_m_percentile', default=99.9, type=float,
                        help='Percentile used when --softmax_m_calib_mode percentile is selected.')
    parser.add_argument('--softmax_m_ema_momentum', default=0.99, type=float,
                        help='EMA momentum used when --softmax_m_calib_mode ema_percentile is selected.')
    parser.add_argument('--no_softmax_calibrate_m', dest='softmax_calibrate_m',
                        action='store_false',
                        help='Disable calibration of IViTSoftmax m during calibration/QAT observer collection.')
    parser.set_defaults(softmax_calibrate_m=True)


    # * Transformer
    parser.add_argument('--dec_layers', default=3, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--sa_nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's self-attentions")
    parser.add_argument('--ca_nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's cross-attentions")
    parser.add_argument('--num_queries', default=300, type=int,
                        help="Number of query slots")
    parser.add_argument('--group_detr', default=13, type=int,
                        help="Number of groups to speed up detr training")
    parser.add_argument('--two_stage', action='store_true')
    parser.add_argument('--projector_scale', default='P4', type=str, nargs='+', choices=('P3', 'P4', 'P5', 'P6'))
    parser.add_argument('--lite_refpoint_refine', action='store_true', help='lite refpoint refine mode for speed-up')
    parser.add_argument('--num_select', default=100, type=int,
                        help='the number of predictions selected for evaluation')
    parser.add_argument('--dec_n_points', default=4, type=int,
                        help='the number of sampling points')
    parser.add_argument('--decoder_norm', default='LN', type=str)
    parser.add_argument('--bbox_reparam', action='store_true')
    parser.add_argument('--easy_quantize', default=False, action='store_true',
                        help='use quantization-friendly float components: learned ref pos encoding and nearest deformable sampling')
    parser.add_argument('--decoder_cross_attn_float', default=False, action='store_true',
                        help='keep quantized decoder deformable cross-attention in a float fallback path')
    parser.add_argument('--decoder_cross_attn_value_per_channel', default=False, action='store_true',
                        help='use per-channel activation scaling for quantized decoder deformable-attention value projection')
    parser.add_argument('--decoder_two_stage_float', default=False, action='store_true',
                        help='keep two-stage proposal heads and top-k selection in float before requantizing selected decoder inputs')
    parser.add_argument('--decoder_norm_output_qact', default=False, action='store_true',
                        help='add an activation quantizer after the final quantized decoder norm')
    parser.add_argument('--decoder_refpoint_head_float', default=False, action='store_true',
                        help='keep decoder reference-point MLP heads in float even when transformer linear quantization is enabled')

    # * Matcher
    parser.add_argument('--set_cost_class', default=2, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")

    # * Loss coefficients
    parser.add_argument('--cls_loss_coef', default=2, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--focal_alpha', default=0.25, type=float)
    
    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")
    parser.add_argument('--sum_group_losses', action='store_true',
                        help="To sum losses across groups or mean losses.")
    parser.add_argument('--use_varifocal_loss', action='store_true')
    parser.add_argument('--use_position_supervised_loss', action='store_true')
    parser.add_argument('--ia_bce_loss', action='store_true')

    # dataset parameters
    parser.add_argument('--dataset_file', default='coco',
                        choices=('coco', 'custom_coco', 'o365'),
                        help='Dataset loader/metadata type. custom_coco infers the '
                             'classification head from COCO-format categories.')
    parser.add_argument('--coco_path', type=str)
    parser.add_argument('--num_classes', default=None, type=int,
                        help='Optional classification head override. Normally inferred '
                             'from --dataset_file metadata.')
    parser.add_argument('--square_resize_div_64', action='store_true')

    parser.add_argument('--output_dir', default='output',
                        help='path where to save, empty for no saving')
    parser.add_argument('--checkpoint_interval', default=10, type=int,
                        help='epoch interval to save checkpoint')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--resume_optimizer', default=False, action='store_true',
                        help='restore optimizer, scheduler, and start epoch from --resume checkpoint; use only to continue the same training run')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--use_ema', action='store_true')
    parser.add_argument('--calib', action='store_true')
    parser.add_argument('--calib_batches', default=500, type=int,
                        help='Number of images/batches used for activation calibration.')
    parser.add_argument('--no_calib_eval', dest='calib_eval', action='store_false',
                        help='Skip evaluation after calibration.')
    parser.add_argument('--no_calib_save', dest='calib_save', action='store_false',
                        help='Skip saving the calibrated checkpoint.')
    parser.set_defaults(calib_eval=True, calib_save=True)
    parser.add_argument('--qat', action='store_true')
    parser.add_argument('--qat_calib_batches', default=0, type=int,
                        help='Run PTQ-style activation calibration for this many batches before QAT training. 0 disables.')
    parser.add_argument('--qat_freeze_observers', action='store_true',
                        help='Keep quantization activation/softmax observers frozen during QAT. Useful when resuming from a calibrated PTQ checkpoint.')
    parser.add_argument('--ptq_reconstruct', action='store_true',
                        help='After PTQ calibration, run unlabeled block-wise reconstruction against a float teacher.')
    parser.add_argument('--ptq_reconstruct_modules', default=['auto'], type=str, nargs='+',
                        help='Module name glob patterns to reconstruct. Default auto targets ViT blocks, projector stages, and transformer decoder layers.')
    parser.add_argument('--ptq_reconstruct_batches', default=16, type=int,
                        help='Calibration batches used per reconstructed module.')
    parser.add_argument('--ptq_reconstruct_lr', default=1e-5, type=float,
                        help='Learning rate for PTQ reconstruction.')
    parser.add_argument('--ptq_reconstruct_weight_decay', default=0.0, type=float,
                        help='Weight decay for PTQ reconstruction.')
    parser.add_argument('--ptq_reconstruct_clip_max_norm', default=0.1, type=float,
                        help='Gradient clipping max norm for PTQ reconstruction.')
    parser.add_argument('--ptq_output_distill_batches', default=16, type=int,
                        help='Extra final-output distillation batches after block reconstruction. Set 0 to disable.')
    parser.add_argument('--ptq_output_distill_lr', default=1e-6, type=float,
                        help='Learning rate for final-output PTQ distillation.')
    parser.add_argument('--ptq_output_distill_logit_weight', default=1.0, type=float,
                        help='Logit KL weight for final-output PTQ distillation.')
    parser.add_argument('--ptq_output_distill_box_weight', default=5.0, type=float,
                        help='Box L1 weight for final-output PTQ distillation.')
    parser.add_argument('--ptq_output_distill_temperature', default=2.0, type=float,
                        help='Logit distillation temperature.')
    parser.add_argument('--ema_decay', default=0.9997, type=float)
    parser.add_argument('--num_workers', default=2, type=int)

    # distributed training parameters
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', 
                        help='url used to set up distributed training')
    parser.add_argument('--sync_bn', default=True, type=bool,
                        help='setup synchronized BatchNorm for distributed training')
    parser.add_argument('--find_unused_parameters', default=False, action='store_true',
                        help='enable unused parameter detection in DistributedDataParallel')
    
    # fp16
    parser.add_argument('--fp16_eval', default=False, action='store_true',
                        help='evaluate in fp16 precision.')

    # subparsers
    subparsers = parser.add_subparsers(title='sub-commands', dest='subcommand',
        description='valid subcommands', help='additional help')

    # subparser for export model
    parser_export = subparsers.add_parser('export_model', help='LWDETR model export')
    parser_export.add_argument('--shape', type=int, nargs=2, default=(640, 640), help="input shape (width, height)")
    parser_export.add_argument('--infer_dir', type=str, default=None)
    parser_export.add_argument('--verbose', type=ast.literal_eval, default=False, nargs="?", const=True)
    parser_export.add_argument('--opset_version', type=int, default=17)
    parser_export.add_argument('--simplify', action='store_true', help="Simplify onnx model")
    parser_export.add_argument('--tensorrt', '--trtexec', '--trt', action='store_true',
                               help="build tensorrt engine")
    parser_export.add_argument('--dry-run', '--test', '-t', action='store_true', help="just print command")
    
    # analysis
    parser.add_argument('--observe_stat', default=False, action='store_true', help="save activation tensor for analysis")
    
    return parser


def main(args):
    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))
    args.num_classes = resolve_num_classes(args)
    print(
        f"Dataset: {args.dataset_file}; classification head size: "
        f"{args.num_classes}"
    )
    print(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank() + 20
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    if args.calib or args.qat:
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
            "Configured IViTSoftmax m calibration: "
            f"count={softmax_count}, "
            f"calibrate_m={args.softmax_calibrate_m}, "
            f"safe_guard={args.softmax_m_safe_guard}, "
            f"mode={args.softmax_m_calib_mode}, "
            f"percentile={args.softmax_m_percentile}, "
            f"ema_momentum={args.softmax_m_ema_momentum}"
        )
        print(model)
        # return
    else:
        model, criterion, postprocessors = build_model(args)
    model.to(device)
    teacher_checkpoint_state = None

    if args.use_ema:
        ema_m = ModelEma(model, decay=args.ema_decay)
    else:
        ema_m = None
    model_without_ddp = model
    if args.distributed:
        if args.sync_bn:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.gpu],
            find_unused_parameters=args.find_unused_parameters,
        )
        model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)
    param_dicts = get_param_dict(args, model_without_ddp)

    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr, 
                                weight_decay=args.weight_decay)
    lr_scheduler = build_lr_scheduler(optimizer, args)
    # scaler = GradScaler('cuda') if torch.cuda.is_available() else None

    dataset_train = build_dataset(image_set='train', args=args)
    dataset_val = build_dataset(image_set='val', args=args)

    if args.distributed:
        sampler_train = DistributedSampler(dataset_train)
        sampler_val = DistributedSampler(dataset_val, shuffle=False)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True)

    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                collate_fn=utils.collate_fn, num_workers=0)
    data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val,
                                drop_last=False, collate_fn=utils.collate_fn, 
                                num_workers=0)

    base_ds = get_coco_api_from_dataset(dataset_val)

    if args.pretrain_weights is not None:
        checkpoint = torch.load(args.pretrain_weights, map_location='cpu', weights_only=False)
        # if args.qat or args.calib:
        #     checkpoint['model'] = fuse_bn_into_conv_in_state_dict(checkpoint['model'])
        # add support to exclude_keys
        # e.g., when load object365 pretrain, do not load `class_embed.[weight, bias]`
        if args.pretrain_exclude_keys is not None:
            assert isinstance(args.pretrain_exclude_keys, list)
            excluded_keys = []
            for exclude_pattern in args.pretrain_exclude_keys:
                matched_keys = [
                    key for key in list(checkpoint['model'].keys())
                    if fnmatch.fnmatch(key, exclude_pattern)
                ]
                for matched_key in matched_keys:
                    checkpoint['model'].pop(matched_key)
                excluded_keys.extend(matched_keys)
            if excluded_keys:
                print(f"Excluded {len(excluded_keys)} pretrain keys:")
                for excluded_key in excluded_keys:
                    print(f"  {excluded_key}")
        if args.pretrain_keys_modify_to_load is not None:
            from util.obj365_to_coco_model import get_coco_pretrain_from_obj365
            assert isinstance(args.pretrain_keys_modify_to_load, list)
            for modify_key_to_load in args.pretrain_keys_modify_to_load:
                checkpoint['model'][modify_key_to_load] = get_coco_pretrain_from_obj365(
                    model_without_ddp.state_dict()[modify_key_to_load],
                    checkpoint['model'][modify_key_to_load]
                )
        teacher_checkpoint_state = dict(checkpoint['model'])
        checkpoint['model'] = adapt_checkpoint_keys_for_model(checkpoint['model'], model_without_ddp)
        stat = model_without_ddp.load_state_dict(checkpoint['model'], strict=False)
        print_load_state("pretrain load", stat)
        if args.use_ema:
            del ema_m
            ema_m = ModelEma(model_without_ddp)
    output_dir = Path(args.output_dir)
    
    #### this took a lot of gpu resource
    # if  utils.is_main_process():
    #     print("Get benchmark")
    #     benchmark_model = copy.deepcopy(model_without_ddp)
    #     bm = benchmark(benchmark_model.float(), dataset_val, output_dir)
    #     print(json.dumps(bm, indent=2))
    #     del benchmark_model

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False)
        teacher_checkpoint_state = dict(checkpoint['model'])
        quant_model_components = set(args.quant_model_components or [])
        backbone_quant_components = set(args.backbone_quant_components or [])
        if (
            (args.qat or args.calib)
            and 'backbone' in quant_model_components
            and 'projector' in backbone_quant_components
        ):
            checkpoint['model'] = fuse_bn_into_conv_in_state_dict(checkpoint['model'])
        checkpoint['model'] = adapt_checkpoint_keys_for_model(checkpoint['model'], model_without_ddp)
        stat = model_without_ddp.load_state_dict(checkpoint['model'], strict=False)
        print_load_state("resume load", stat)
        # return

        if args.use_ema:
            if 'ema_model' in checkpoint:
                ema_m.module.load_state_dict(clean_state_dict(checkpoint['ema_model']))
            else:
                del ema_m
                ema_m = ModelEma(model) 
        resume_training_state = not args.eval and args.resume_optimizer
        if resume_training_state and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            checkpoint['optimizer']["param_groups"] = optimizer.state_dict()["param_groups"]
            optimizer.load_state_dict(checkpoint['optimizer'])
            checkpoint_args = checkpoint.get('args')
            checkpoint_scheduler = getattr(
                checkpoint_args, 'lr_scheduler', 'step'
            )
            if checkpoint_scheduler == args.lr_scheduler:
                lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            else:
                print(
                    "Skipping LR scheduler state because scheduler changed: "
                    f"{checkpoint_scheduler} -> {args.lr_scheduler}"
                )
            args.start_epoch = checkpoint['epoch'] + 1

    if args.eval:
        if args.calib or args.qat:
            print("Freezing quantization observers for standalone evaluation.")
            freeze_model(model)
        test_stats, coco_evaluator = evaluate(
            model, criterion, postprocessors, data_loader_val, base_ds, device, args)
        if args.output_dir:
            utils.save_on_master(coco_evaluator.coco_eval["bbox"].eval, output_dir / "eval.pth")
        return
    
    if args.calib:
        # 1) calibrate using val set (or train set). val is usually fine.
        num_calib_batch = args.calib_batches
        calib_loader = make_random_calib_loader(
            dataset_val,
            batch_size=1,
            seed=seed,                  # <--- fixed seed for reproducibility!
            n_samples=num_calib_batch * 1,
            collate_fn=utils.collate_fn,
            num_workers=0
        )
        calibrate(
            model,
            calib_loader,
            device,
            args=args,
            calib_batches=num_calib_batch,
        )

        if args.ptq_reconstruct:
            if teacher_checkpoint_state is None:
                raise RuntimeError("--ptq_reconstruct requires --resume or --pretrain_weights so the float teacher can be initialized.")
            teacher_args = copy.deepcopy(args)
            teacher_args.calib = False
            teacher_args.qat = False
            teacher_args.observe_stat = False
            teacher_model, _, _ = build_model(teacher_args)
            teacher_model.to(device)
            teacher_state = adapt_checkpoint_keys_for_model(teacher_checkpoint_state, teacher_model)
            teacher_stat = teacher_model.load_state_dict(teacher_state, strict=False)
            print_load_state("ptq teacher load", teacher_stat)
            recon_samples = args.ptq_reconstruct_batches * max(1, len(args.ptq_reconstruct_modules))
            if args.ptq_reconstruct_modules == ["auto"]:
                recon_samples = args.ptq_reconstruct_batches * max(1, args.vit_encoder_num_layers + len(args.projector_scale) + args.dec_layers)
            recon_loader = make_random_calib_loader(
                dataset_val,
                batch_size=1,
                seed=seed + 1009,
                n_samples=recon_samples,
                collate_fn=utils.collate_fn,
                num_workers=0
            )
            reconstruct_ptq(
                model_without_ddp,
                teacher_model,
                recon_loader,
                device,
                args=args,
                module_patterns=args.ptq_reconstruct_modules,
                batches_per_module=args.ptq_reconstruct_batches,
                lr=args.ptq_reconstruct_lr,
                weight_decay=args.ptq_reconstruct_weight_decay,
                max_norm=args.ptq_reconstruct_clip_max_norm,
            )
            if args.ptq_output_distill_batches > 0:
                output_distill_loader = make_random_calib_loader(
                    dataset_val,
                    batch_size=1,
                    seed=seed + 2027,
                    n_samples=args.ptq_output_distill_batches,
                    collate_fn=utils.collate_fn,
                    num_workers=0
                )
                reconstruct_output_distill(
                    model_without_ddp,
                    teacher_model,
                    output_distill_loader,
                    device,
                    batches=args.ptq_output_distill_batches,
                    lr=args.ptq_output_distill_lr,
                    weight_decay=args.ptq_reconstruct_weight_decay,
                    max_norm=args.ptq_reconstruct_clip_max_norm,
                    logit_weight=args.ptq_output_distill_logit_weight,
                    box_weight=args.ptq_output_distill_box_weight,
                    temperature=args.ptq_output_distill_temperature,
                )
            sync_model_state_from_main(model_without_ddp)
            del teacher_model

        # 2) (optional) save calibrated quant checkpoint
        if args.output_dir and utils.is_main_process() and getattr(args, "calib_save", True):
            calib_path = output_dir / ("checkpoint_reconstructed.pth" if args.ptq_reconstruct else "checkpoint_calibrated.pth")
            utils.save_on_master(
                {
                    "model": model_without_ddp.state_dict(),
                    "args": args,
                },
                calib_path,
            )
            print("Saved calibrated checkpoint to:", calib_path)

        # 3) (optional) evaluate immediately after calibration
        if getattr(args, "calib_eval", True):
            test_stats, coco_evaluator = evaluate(
                model, criterion, postprocessors, data_loader_val, base_ds, device, args
            )
            if args.output_dir and utils.is_main_process():
                utils.save_on_master(coco_evaluator.coco_eval["bbox"].eval, output_dir / "eval_calib.pth")

        return
        
    
    # for drop
    total_batch_size = args.batch_size * utils.get_world_size()
    num_training_steps_per_epoch = (len(dataset_train) + total_batch_size - 1) // total_batch_size
    schedules = {}
    if args.dropout > 0:
        schedules['do'] = drop_scheduler(
            args.dropout, args.epochs, num_training_steps_per_epoch,
            args.cutoff_epoch, args.drop_mode, args.drop_schedule)
        print("Min DO = %.7f, Max DO = %.7f" % (min(schedules['do']), max(schedules['do'])))

    if args.drop_path > 0:
        schedules['dp'] = drop_scheduler(
            args.drop_path, args.epochs, num_training_steps_per_epoch,
            args.cutoff_epoch, args.drop_mode, args.drop_schedule)
        print("Min DP = %.7f, Max DP = %.7f" % (min(schedules['dp']), max(schedules['dp'])))

    print("Start training")
    
    start_time = time.time()
    best_map_holder = BestMetricHolder(use_ema=args.use_ema)
    class_head_warmup_active = (
        args.class_head_warmup_epochs > 0
        and args.start_epoch < args.class_head_warmup_epochs
    )
    if class_head_warmup_active:
        report_class_head_only_training(model_without_ddp)
    if args.qat and args.qat_calib_batches > 0 and args.start_epoch == 0:
        qat_calib_loader = make_random_calib_loader(
            dataset_val,
            batch_size=1,
            seed=seed,
            n_samples=args.qat_calib_batches,
            collate_fn=utils.collate_fn,
            num_workers=0
        )
        calibrate(
            model,
            qat_calib_loader,
            device,
            args=args,
            calib_batches=args.qat_calib_batches,
        )
        qat_calib_stats, qat_calib_coco_evaluator = evaluate(
            model, criterion, postprocessors, data_loader_val, base_ds, device, args
        )
        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps({
                    "phase": "pre_qat_calib",
                    "qat_calib_batches": args.qat_calib_batches,
                    **{f"pre_qat_calib_{k}": v for k, v in qat_calib_stats.items()},
                }) + "\n")
            if qat_calib_coco_evaluator is not None:
                utils.save_on_master(
                    qat_calib_coco_evaluator.coco_eval["bbox"].eval,
                    output_dir / "eval_qat_calib.pth",
                )
    if args.qat and args.qat_freeze_observers:
        print("QAT observers are frozen; training weights with fixed quantization ranges.")
        freeze_model(model)

    for epoch in range(args.start_epoch, args.epochs):
        if class_head_warmup_active and epoch >= args.class_head_warmup_epochs:
            print("Class-head warm-up complete: updating the full detector.")
            class_head_warmup_active = False
        if args.qat and not args.qat_freeze_observers:
            unfreeze_model(model)
        epoch_start_time = time.time()
        if args.distributed:
            sampler_train.set_epoch(epoch)
        train_stats = train_one_epoch(
            model, criterion, data_loader_train, optimizer, device, epoch,
            args.clip_max_norm, ema_m=ema_m, schedules=schedules, 
            num_training_steps_per_epoch=num_training_steps_per_epoch,
            vit_encoder_num_layers=args.vit_encoder_num_layers, args=args)
        train_epoch_time = time.time() - epoch_start_time
        train_epoch_time_str = str(datetime.timedelta(seconds=int(train_epoch_time)))
        if args.qat:
            freeze_model(model)
        lr_scheduler.step()
        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            # extra checkpoint before LR drop and every `checkpoint_interval` epochs
            scheduler_event = (
                args.lr_scheduler == "step"
                and (epoch + 1) % args.lr_drop == 0
            ) or (
                args.lr_scheduler == "delayed_cosine"
                and (epoch + 1) == args.lr_decay_start_epoch
            )
            if scheduler_event or (epoch + 1) % args.checkpoint_interval == 0:
                checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}.pth')
            for checkpoint_path in checkpoint_paths:
                weights = {
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }
                if args.use_ema:
                    weights.update({
                        'ema_model': ema_m.module.state_dict(),
                    })
                utils.save_on_master(weights, checkpoint_path)


        test_stats, coco_evaluator = evaluate(
            model, criterion, postprocessors, data_loader_val, base_ds, device, args=args
        )


        map_regular = test_stats['coco_eval_bbox'][0]
        _isbest = best_map_holder.update(map_regular, epoch, is_ema=False)
        if _isbest:
            checkpoint_path = output_dir / 'checkpoint_best_regular.pth'
            utils.save_on_master({
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
                'args': args,
            }, checkpoint_path)
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                    **{f'test_{k}': v for k, v in test_stats.items()},
                    'epoch': epoch,
                    'n_parameters': n_parameters}
        
        if args.use_ema:
            ema_test_stats, _ = evaluate(
                ema_m.module, criterion, postprocessors, data_loader_val, base_ds, device, args=args
            )

            log_stats.update({f'ema_test_{k}': v for k,v in ema_test_stats.items()})
            map_ema = ema_test_stats['coco_eval_bbox'][0]
            _isbest = best_map_holder.update(map_ema, epoch, is_ema=True)
            if _isbest:
                checkpoint_path = output_dir / 'checkpoint_best_ema.pth'
                utils.save_on_master({
                    'model': ema_m.module.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }, checkpoint_path)
        log_stats.update(best_map_holder.summary())
        
        # epoch parameters
        ep_paras = {
                'epoch': epoch,
                'n_parameters': n_parameters
            }
        log_stats.update(ep_paras)
        try:
            log_stats.update({'now_time': str(datetime.datetime.now())})
        except:
            pass
        log_stats['train_epoch_time'] = train_epoch_time_str
        epoch_time = time.time() - epoch_start_time
        epoch_time_str = str(datetime.timedelta(seconds=int(epoch_time)))
        log_stats['epoch_time'] = epoch_time_str
        
        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

            # for evaluation logs
            if coco_evaluator is not None:
                (output_dir / 'eval').mkdir(exist_ok=True)
                if "bbox" in coco_evaluator.coco_eval:
                    filenames = ['latest.pth']
                    if epoch % 50 == 0:
                        filenames.append(f'{epoch:03}.pth')
                    for name in filenames:
                        torch.save(coco_evaluator.coco_eval["bbox"].eval,
                                   output_dir / "eval" / name)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

@torch.no_grad()
def fuse_bn_into_conv_in_state_dict(
    state_dict: dict,
    eps: float = 1e-5,
    conv_suffix: str = ".conv",
    bn_suffix: str = ".bn",
    verbose: bool = True,
):
    """
    Fuse BatchNorm params into Conv params directly in a checkpoint state_dict.

    Expects ConvX-like naming where conv and bn live under the same prefix:
      <prefix>.conv.weight
      <prefix>.conv.bias (optional)
      <prefix>.bn.weight
      <prefix>.bn.bias
      <prefix>.bn.running_mean
      <prefix>.bn.running_var
      <prefix>.bn.num_batches_tracked (optional)

    Returns a NEW dict (doesn't mutate the input).
    """
    sd = dict(state_dict)
    keys = list(sd.keys())

    # Find all BN weight keys and infer the block prefix from them.
    bn_weight_keys = [k for k in keys if k.endswith(f"{bn_suffix}.weight")]

    fused = 0
    skipped_no_conv = 0

    for bn_w_key in bn_weight_keys:
        prefix = bn_w_key[: -len(f"{bn_suffix}.weight")]  # strip ".bn.weight"

        # BN keys
        bn_b_key  = prefix + f"{bn_suffix}.bias"
        bn_rm_key = prefix + f"{bn_suffix}.running_mean"
        bn_rv_key = prefix + f"{bn_suffix}.running_var"

        if not all(k in sd for k in (bn_w_key, bn_b_key, bn_rm_key, bn_rv_key)):
            continue

        # Matching conv keys
        conv_w_key = prefix + f"{conv_suffix}.weight"
        conv_b_key = prefix + f"{conv_suffix}.bias"

        if conv_w_key not in sd:
            skipped_no_conv += 1
            if verbose:
                print(f"[skip] BN found but no conv weight for prefix: {prefix}")
            continue

        W = sd[conv_w_key]
        if conv_b_key in sd:
            b = sd[conv_b_key]
        else:
            b = torch.zeros(W.size(0), device=W.device, dtype=W.dtype)

        gamma = sd[bn_w_key].to(dtype=W.dtype, device=W.device)
        beta  = sd[bn_b_key].to(dtype=W.dtype, device=W.device)
        mean  = sd[bn_rm_key].to(dtype=W.dtype, device=W.device)
        var   = sd[bn_rv_key].to(dtype=W.dtype, device=W.device)

        # a = gamma / sqrt(var + eps)
        a = gamma / torch.sqrt(var + eps)

        # Fuse
        if W.ndim != 4:
            raise ValueError(f"Expected Conv2d weight (4D) at {conv_w_key}, got shape {tuple(W.shape)}")

        W_fused = W * a.reshape(-1, 1, 1, 1)
        b_fused = a * (b - mean) + beta

        sd[conv_w_key] = W_fused
        sd[conv_b_key] = b_fused

        # Remove BN entries so they won't show up as unexpected later
        for k in (
            bn_w_key,
            bn_b_key,
            bn_rm_key,
            bn_rv_key,
            prefix + f"{bn_suffix}.num_batches_tracked",
        ):
            if k in sd:
                del sd[k]

        fused += 1
        if verbose:
            print(f"[fused] {prefix}{conv_suffix} <= {prefix}{bn_suffix}")

    if verbose:
        print(f"Fused blocks: {fused}")
        print(f"Skipped (BN but missing conv): {skipped_no_conv}")

    return sd

if __name__ == '__main__':
    parser = argparse.ArgumentParser('LWDETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    if args.subcommand is None:
        main(args)
    elif args.subcommand == 'export_model':
        from deploy.export import main
        if args.batch_size != 1:
            args.batch_size = 1
            print(f"Only batch_size 1 is supported for onnx export, \
                 but got batchsize = {args.batch_size}. batch_size is forcibly set to 1.")
        main(args)
