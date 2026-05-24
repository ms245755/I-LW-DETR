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
inference demo
"""
import os
import sys
current_directory = os.getcwd()
print(f"Current directory: {current_directory}")
sys.path.append(os.getcwd())

import argparse
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from models import build_model
from quant_models import build_model_quant
from util.get_param_dicts import get_param_dict
import util.misc as utils
from torchvision import transforms
from util.misc import nested_tensor_from_tensor_list
from quant_models.fake_quantized.model_utils import *

# COCO classes
COCO_CLASSES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana',
    'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut',
    'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A', 'N/A',
    'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]


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

def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    # Model parameters
    parser.add_argument('--weights', type=str, default=None, required=True,
                        help="Path to the model parameters.")
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--pretrained_encoder', type=str, default=None, 
                        help="Path to the pretrained encoder.")
    
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
    parser.add_argument('--quant_model_components', default=['backbone'], type=str, nargs='+',
                        choices=('backbone', 'transformer'),
                        help='Top-level model components to quantize in quant demo mode.')
    parser.add_argument('--backbone_quant_components', default=['vit', 'projector'], type=str, nargs='+',
                        choices=('vit', 'projector'),
                        help='Backbone components to quantize in quant demo mode.')
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
                        help='Operation families to quantize inside the projector.')
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
                        help='Decoder LayerNorm modules to quantize when transformer norm quantization is enabled.')
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
    parser.add_argument('--act_observer', default='ema_minmax', type=str,
                        choices=('ema_minmax', 'percentile', 'mean'),
                        help='Activation observer used by IViTAct.')
    parser.add_argument('--act_percentile', default=99.9, type=float,
                        help='Percentile used when --act_observer percentile is selected.')

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
    parser.add_argument('--easy_quantize', action='store_true',
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

    # * Dataset infomation
    parser.add_argument('--dataset_file', default='coco')

    # * Matcher
    parser.add_argument('--set_cost_class', default=2, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")

    # * Learning rate
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_encoder', default=1.5e-4, type=float)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=12, type=int)
    parser.add_argument('--lr_drop', default=11, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')
    parser.add_argument('--lr_vit_layer_decay', default=0.8, type=float)
    parser.add_argument('--lr_component_decay', default=1.0, type=float)
    
    # * Drop args
    parser.add_argument('--dropout', type=float, default=0,
                        help='Drop path rate (default: 0.0)')
    parser.add_argument('--drop_path', type=float, default=0,
                        help='Drop path rate (default: 0.0)')

    # * Loss coefficients
    parser.add_argument('--cls_loss_coef', default=2, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--focal_alpha', default=0.25, type=float)
    
    # * Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")
    parser.add_argument('--sum_group_losses', action='store_true',
                        help="To sum losses across groups or mean losses.")
    parser.add_argument('--use_varifocal_loss', action='store_true')
    parser.add_argument('--use_position_supervised_loss', action='store_true')
    parser.add_argument('--ia_bce_loss', action='store_true')

    # * Input and output
    parser.add_argument('--input', default=None, required=True,
                        help='"Path to image file."')
    parser.add_argument('--output_dir', default='output',
                        help='Directory to save output visualizations.')
    parser.add_argument('--confidence_threshold', type=float, default=0.5,
                        help='Minimum score for instance predictions to be shown')
    
    # Observation
    parser.add_argument("--observe_stat", action="store_true")
    parser.add_argument("--mode", type=str, default="float")

    return parser

def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")
    orig_image_size = torch.tensor(image.size[::-1])

    normalize = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    transform = transforms.Compose([
            transforms.Resize([640, 640]),
            normalize,
        ])
    image = transform(image)
    return image, orig_image_size

def visualize_detections(image, boxes, labels, scores, conf_thresh, output_path):
    draw = ImageDraw.Draw(image)
    image_w, image_h = image.size
    line_width = max(2, min(image_w, image_h) // 180)
    font_size = max(12, min(22, min(image_w, image_h) // 22))
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", font_size)
    except OSError:
        font = ImageFont.load_default()

    def text_size(text):
        if hasattr(draw, "textbbox"):
            left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
            return right - left, bottom - top
        return draw.textsize(text, font=font)

    occupied_labels = []

    for box, label, score in zip(boxes, labels, scores):
        if score > conf_thresh:
            xmin, ymin, xmax, ymax = map(int, box)
            xmin = max(0, min(xmin, image_w - 1))
            ymin = max(0, min(ymin, image_h - 1))
            xmax = max(0, min(xmax, image_w - 1))
            ymax = max(0, min(ymax, image_h - 1))
            if xmax <= xmin or ymax <= ymin:
                continue

            draw.rectangle([xmin, ymin, xmax, ymax], outline="green", width=line_width)

            text = f"{COCO_CLASSES[label]} {score:.2f}"
            text_w, text_h = text_size(text)
            pad_x = max(2, line_width)
            pad_y = max(1, line_width // 2)
            label_w = text_w + 2 * pad_x
            label_h = text_h + 2 * pad_y

            label_x = max(0, min(xmin, image_w - label_w))
            label_y = ymin - label_h - line_width
            if label_y < 0:
                label_y = ymin + line_width
            label_y = max(0, min(label_y, image_h - label_h))

            label_box = [label_x, label_y, label_x + label_w, label_y + label_h]
            for _ in range(8):
                overlaps = any(
                    label_box[0] < old[2]
                    and label_box[2] > old[0]
                    and label_box[1] < old[3]
                    and label_box[3] > old[1]
                    for old in occupied_labels
                )
                if not overlaps:
                    break
                next_y = label_box[3] + line_width
                if next_y + label_h > image_h:
                    break
                label_box = [label_box[0], next_y, label_box[2], next_y + label_h]

            draw.rectangle(label_box, fill="green")
            draw.text(
                (label_box[0] + pad_x, label_box[1] + pad_y),
                text,
                fill="white",
                font=font,
            )
            occupied_labels.append(label_box)

    image.save(output_path)

def main(args):
    utils.init_distributed_mode(args)
    print(args)

    device = torch.device(args.device)

    if args.mode == 'quant':
        print("Build quant model")
        model, _, postprocessors = build_model_quant(args)
    else:
        print("Build float model")
        model, _, postprocessors = build_model(args)
    
    model.to(device)
    model.eval()
    freeze_model(model)

    param_dicts = get_param_dict(args, model)

    output_path = Path(args.output_dir) /  "visualize.jpg"

    if args.weights:
        checkpoint = torch.load(args.weights, map_location='cpu')
        state_dict = checkpoint['model']
        if args.mode == 'quant':
            state_dict = adapt_checkpoint_keys_for_model(state_dict, model)
        load_state = model.load_state_dict(state_dict, strict=False)
        print(
            f"Loaded weights: missing_keys={len(load_state.missing_keys)}, "
            f"unexpected_keys={len(load_state.unexpected_keys)}"
        )

    # preprocess
    image, orig_image_size = preprocess_image(args.input)
    image = image.to(device)
    orig_image_size = orig_image_size.to(device)

    images = nested_tensor_from_tensor_list([image])
    orig_image_sizes = torch.stack([orig_image_size])
    # forward
    with torch.no_grad():
        outputs = model(images)

    # postprocess
    predictions = postprocessors['bbox'](outputs, orig_image_sizes)

    # visualize
    boxes = predictions[0]['boxes'].cpu().numpy()
    labels = predictions[0]['labels'].cpu().numpy()
    scores = predictions[0]['scores'].cpu().numpy()

    original_image = Image.open(args.input).convert("RGB")
    visualize_detections(
        original_image,
        boxes,
        labels,
        scores,
        args.confidence_threshold,
        output_path)

    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser('LWDETR infer script', parents=[get_args_parser()])
    args = parser.parse_args()

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    main(args)
