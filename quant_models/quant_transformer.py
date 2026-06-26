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
Transformer class
"""
import math
import copy
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn, Tensor

from .quant_blocks import *
from quant_models.fake_quantized.layers import *
from util.decorators import save_tensor_from_pathname


def _component_set(components):
    if components is None:
        return {"linear", "matmul", "softmax", "layernorm"}
    return {component for component in components if component != "none"}


_DECODER_BLOCK_COMPONENTS = {
    "self_attn",
    "cross_attn",
    "ffn",
    "norm",
    "refpoint",
    "two_stage",
    "decoder_output_norm",
}


def _component_list(components, default):
    if components is None:
        return set(default)
    return {component for component in components if component != "none"}


def _decoder_block_components(components):
    if components is None:
        return set(_DECODER_BLOCK_COMPONENTS)

    components = {component for component in components if component != "none"}
    if not components:
        return set()

    unknown = components - _DECODER_BLOCK_COMPONENTS
    if unknown:
        raise ValueError("unknown transformer_quant_components: {}".format(sorted(unknown)))
    return components


def _quant_block_set(quant_blocks, num_blocks, block_name):
    if quant_blocks is None:
        return set(range(num_blocks))
    quant_blocks = set(quant_blocks)
    invalid_blocks = sorted(block for block in quant_blocks if block < 0 or block >= num_blocks)
    if invalid_blocks:
        raise ValueError(
            "{} indexes out of range for {} blocks: {}".format(
                block_name, num_blocks, invalid_blocks
            )
        )
    return quant_blocks


def _decoder_norm_set(norms):
    if norms is None:
        return {"norm1", "norm2", "norm3"}
    norms = {norm for norm in norms if norm != "none"}
    valid_norms = {"norm1", "norm2", "norm3"}
    invalid_norms = sorted(norms - valid_norms)
    if invalid_norms:
        raise ValueError("unknown decoder_quant_norms: {}".format(invalid_norms))
    return norms


def _decoder_cross_attn_components(components):
    components = _component_list(
        components,
        {
            "value_proj",
            "sampling_offsets",
            "attention_weights",
            "softmax",
            "grid_sample_core",
            "weighted_sum",
            "output_proj",
        },
    )
    if "all" in components:
        components.update(
            {
                "value_proj",
                "sampling_offsets",
                "attention_weights",
                "softmax",
                "grid_sample_core",
                "weighted_sum",
                "output_proj",
            }
        )
    if "core" in components:
        components.update({"grid_sample_core", "weighted_sum"})
    valid = {
        "value_proj",
        "sampling_offsets",
        "attention_weights",
        "softmax",
        "grid_sample_core",
        "weighted_sum",
        "output_proj",
        "core",
        "all",
    }
    invalid = sorted(components - valid)
    if invalid:
        raise ValueError("unknown decoder_cross_attn_quant_components: {}".format(invalid))
    return components & {
        "value_proj",
        "sampling_offsets",
        "attention_weights",
        "softmax",
        "grid_sample_core",
        "weighted_sum",
        "output_proj",
    }


def _scale_mul(scale, value):
    return None if scale is None else scale * value


def _add_tensor_scale(x, scale_x, y, scale_y):
    if scale_x is None or scale_y is None:
        return x + y, None
    return add_tensor_scale(x, scale_x, y, scale_y)


def _concat_optional_tensor_scale(tensor_list, dim=0):
    if any(scale is None for _, scale in tensor_list):
        return torch.cat([tensor for tensor, _ in tensor_list], dim=dim), None
    return concat_list_tensor_scale(tensor_list, dim=dim)


def _multiply_tensor_scale(x, scale_x, y, scale_y):
    if scale_x is None or scale_y is None:
        return x * y, None
    return multiply_tensor_scale(x, scale_x, y, scale_y)


def _exp_tensor_scale(exp_module, x, scale):
    if scale is None:
        return x.clamp(min=-20.0, max=20.0).exp(), None
    return exp_module(x, scale)


def _sanitize_boxes(boxes):
    return torch.nan_to_num(boxes, nan=0.0, posinf=1.0, neginf=0.0)


class ScaleLinear(nn.Linear):
    def forward(self, x, act_scaling_factor=None):
        return super().forward(x), None


class ScaleLayerNorm(nn.LayerNorm):
    def __init__(self, normalized_shape, eps=1e-6, elementwise_affine=True):
        super().__init__(normalized_shape, eps=eps, elementwise_affine=elementwise_affine)

    def forward(self, x, act_scaling_factor=None):
        return super().forward(x), None


class ScaleMatMul(nn.Module):
    def forward(self, A, pre_act_scaling_factor_A, B, pre_act_scaling_factor_B):
        return A @ B, None


class ScaleSoftmax(nn.Module):
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim

    def forward(self, x, act_scaling_factor=None):
        return F.softmax(x, dim=self.dim), None


def _linear_float(layer, x):
    return F.linear(x, layer.weight, layer.bias)


def _layer_norm_float(norm, x):
    return F.layer_norm(x, norm.normalized_shape, norm.weight, norm.bias, norm.eps)


def _mlp_float(mlp, x):
    for i, layer in enumerate(mlp.layers):
        x = _linear_float(layer, x)
        if i < mlp.num_layers - 1:
            x = F.relu(x)
    return x


class ScaleMSDeformAttn(MSDeformAttn):
    def __init__(self, *args, use_float_path=False, value_per_channel=False, observe_stat=False, block_name="", **kwargs):
        super().__init__(*args, **kwargs)
        self.observe_stat = observe_stat
        self.block_name = block_name

    def forward(
        self,
        query,
        reference_points,
        input_flatten,
        input_spatial_shapes,
        input_level_start_index,
        input_padding_mask=None,
        query_scaling_factor=None,
        input_scaling_factor=None,
    ):
        save_tensor_from_pathname(query, self.block_name + "0_query", observe_stat=self.observe_stat)
        save_tensor_from_pathname(reference_points, self.block_name + "1_reference_points", observe_stat=self.observe_stat)
        save_tensor_from_pathname(input_flatten, self.block_name + "2_input_flatten", observe_stat=self.observe_stat)
        output = super().forward(
            query,
            reference_points,
            input_flatten,
            input_spatial_shapes,
            input_level_start_index,
            input_padding_mask,
        )
        save_tensor_from_pathname(output, self.block_name + "9_output_proj", observe_stat=self.observe_stat)
        return output, None


class IMLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, bias_bit=32, quant_components=None):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.quant_components = _component_set(quant_components)
        self.quant_linear = "linear" in self.quant_components
        linear_cls = IViTLinear if self.quant_linear else ScaleLinear
        self.layers = nn.ModuleList(
            linear_cls(n, k, bias_bit=bias_bit) if linear_cls is IViTLinear else linear_cls(n, k)
            for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.qact_input_bridge = IViTAct()
        self.qact_post_linears = nn.ModuleList(IViTAct() for _ in range(num_layers))
        self.qact_post_relu = nn.ModuleList(IViTAct() for _ in range(num_layers - 1))

    def _qact_channel_last(self, qact, x, act_scaling_factor):
        if x.dim() != 4:
            return qact(x, act_scaling_factor)

        original_shape = x.shape
        x = x.reshape(-1, original_shape[-2], original_shape[-1])
        x, act_scaling_factor = qact(x, act_scaling_factor)
        return x.reshape(original_shape), act_scaling_factor

    def forward(self, x, act_scaling_factor = None):
        if not self.quant_linear:
            for i, layer in enumerate(self.layers):
                x, _ = layer(x, None)
                if i < self.num_layers - 1:
                    x = F.relu(x)
            return x, None

        for i, layer in enumerate(self.layers):
            if i == 0 and (act_scaling_factor is None or act_scaling_factor.numel() != 1):
                x, act_scaling_factor = self._qact_channel_last(self.qact_input_bridge, x, None)
            x, act_scaling_factor = layer(x, act_scaling_factor)
            x, act_scaling_factor = self._qact_channel_last(self.qact_post_linears[i], x, act_scaling_factor)
            if i < self.num_layers - 1:
                x = F.relu(x)
                x, act_scaling_factor = self._qact_channel_last(self.qact_post_relu[i], x, act_scaling_factor)
        return x, act_scaling_factor


def gen_sineembed_for_position(pos_tensor, dim=128):
    # n_query, bs, _ = pos_tensor.size()
    # sineembed_tensor = torch.zeros(n_query, bs, 256)
    scale = 2 * math.pi
    dim_t = torch.arange(dim, dtype=torch.float32, device=pos_tensor.device)
    dim_t = 10000 ** (2 * (dim_t // 2) / dim)
    x_embed = pos_tensor[:, :, 0] * scale
    y_embed = pos_tensor[:, :, 1] * scale
    pos_x = x_embed[:, :, None] / dim_t
    pos_y = y_embed[:, :, None] / dim_t
    pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
    pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2)
    if pos_tensor.size(-1) == 2:
        pos = torch.cat((pos_y, pos_x), dim=2)
    elif pos_tensor.size(-1) == 4:
        w_embed = pos_tensor[:, :, 2] * scale
        pos_w = w_embed[:, :, None] / dim_t
        pos_w = torch.stack((pos_w[:, :, 0::2].sin(), pos_w[:, :, 1::2].cos()), dim=3).flatten(2)

        h_embed = pos_tensor[:, :, 3] * scale
        pos_h = h_embed[:, :, None] / dim_t
        pos_h = torch.stack((pos_h[:, :, 0::2].sin(), pos_h[:, :, 1::2].cos()), dim=3).flatten(2)

        pos = torch.cat((pos_y, pos_x, pos_w, pos_h), dim=2)
    else:
        raise ValueError("Unknown pos_tensor shape(-1):{}".format(pos_tensor.size(-1)))
    return pos


def gen_encoder_output_proposals(memory, memory_padding_mask, spatial_shapes, unsigmoid=True):
    """
    Input:
        - memory: bs, \sum{hw}, d_model
        - memory_scale: scalar
        - memory_padding_mask: bs, \sum{hw}
        - spatial_shapes: nlevel, 2
    Output:
        - output_memory: bs, \sum{hw}, d_model
        - output_proposals: bs, \sum{hw}, 4
    """
    N_, S_, C_ = memory.shape
    base_scale = 4.0
    proposals = []
    _cur = 0
    for lvl, (H_, W_) in enumerate(spatial_shapes):
        if memory_padding_mask is not None:
            mask_flatten_ = memory_padding_mask[:, _cur:(_cur + H_ * W_)].view(N_, H_, W_, 1)
            valid_H = torch.sum(~mask_flatten_[:, :, 0, 0], 1)
            valid_W = torch.sum(~mask_flatten_[:, 0, :, 0], 1)
        else:
            valid_H = torch.tensor([H_ for _ in range(N_)], device=memory.device)
            valid_W = torch.tensor([W_ for _ in range(N_)], device=memory.device)

        grid_y, grid_x = torch.meshgrid(torch.linspace(0, H_ - 1, H_, dtype=torch.float32, device=memory.device),
                                        torch.linspace(0, W_ - 1, W_, dtype=torch.float32, device=memory.device))
        grid = torch.cat([grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)], -1) # H_, W_, 2

        scale = torch.cat([valid_W.unsqueeze(-1), valid_H.unsqueeze(-1)], 1).view(N_, 1, 1, 2)
        grid = (grid.unsqueeze(0).expand(N_, -1, -1, -1) + 0.5) / scale

        wh = torch.ones_like(grid) * 0.05 * (2.0 ** lvl)

        proposal = torch.cat((grid, wh), -1).view(N_, -1, 4)
        proposals.append(proposal)
        _cur += (H_ * W_)

    output_proposals = torch.cat(proposals, 1)
    output_proposals_valid = ((output_proposals > 0.01) & (output_proposals < 0.99)).all(-1, keepdim=True)

    # bbox_reparam = False
    if unsigmoid:
        output_proposals = torch.log(output_proposals / (1 - output_proposals)) # unsigmoid
        if memory_padding_mask is not None:
            output_proposals = output_proposals.masked_fill(memory_padding_mask.unsqueeze(-1), float('inf'))
        output_proposals = output_proposals.masked_fill(~output_proposals_valid, float('inf'))
    else:
        if memory_padding_mask is not None:
            output_proposals = output_proposals.masked_fill(memory_padding_mask.unsqueeze(-1), float(0))
        output_proposals = output_proposals.masked_fill(~output_proposals_valid, float(0))

    output_memory = memory
    if memory_padding_mask is not None:
        output_memory = output_memory.masked_fill(memory_padding_mask.unsqueeze(-1), float(0))
    output_memory = output_memory.masked_fill(~output_proposals_valid, float(0))

    return output_memory.to(memory.dtype), output_proposals.to(memory.dtype)


class Transformer(nn.Module):

    def __init__(self, d_model=512, sa_nhead=8, ca_nhead=8, num_queries=300,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.0,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False, group_detr=1, 
                 two_stage=False,
                 num_feature_levels=4, dec_n_points=4,
                 lite_refpoint_refine=False,
                 easy_quantize=False,
                 decoder_norm_type='LN',
                 bbox_reparam=False,
                 decoder_cross_attn_float=False,
                 decoder_cross_attn_value_per_channel=False,
                 decoder_two_stage_float=False,
                 decoder_norm_output_qact=False,
                 decoder_refpoint_head_float=False,
                 transformer_quant_components=None,
                 decoder_quant_blocks=None,
                 decoder_quant_norms=None,
                 decoder_self_attn_quant_components=None,
                 decoder_cross_attn_quant_components=None,
                 decoder_ffn_quant_components=None,
                 decoder_refpoint_quant_components=None,
                 decoder_two_stage_quant_components=None,
                 decoder_output_norm_quant_components=None,
                 exp_mode="ivit",
                 observe_stat=False,
                 block_name="observed_tensor/quant/transformer/"):
        super().__init__()
        if exp_mode not in {"float", "ivit"}:
            raise ValueError(f"unknown transformer exp_mode: {exp_mode}")
        self.encoder = None
        self.decoder_block_components = _decoder_block_components(transformer_quant_components)
        self.quant_components = set()
        self.decoder_quant_blocks = _quant_block_set(decoder_quant_blocks, num_decoder_layers, "decoder_quant_blocks")
        self.decoder_quant_norms = _decoder_norm_set(decoder_quant_norms)
        self.self_attn_quant_components = _component_list(decoder_self_attn_quant_components, {"linear", "matmul", "softmax"})
        self.cross_attn_quant_components = _decoder_cross_attn_components(decoder_cross_attn_quant_components)
        self.ffn_quant_components = _component_list(decoder_ffn_quant_components, {"linear"})
        self.refpoint_quant_components = _component_list(decoder_refpoint_quant_components, {"linear", "sine_embed", "coord_proj"})
        self.two_stage_quant_components = _component_list(
            decoder_two_stage_quant_components,
            {"proposal", "enc_output", "enc_output_norm", "class_head", "bbox_head", "bbox_reparam", "selected_inputs"},
        )
        self.output_norm_quant_components = _component_list(decoder_output_norm_quant_components, {"layernorm"})
        if decoder_two_stage_float:
            self.two_stage_quant_components = {"selected_inputs"}
        if decoder_refpoint_head_float:
            self.refpoint_quant_components.discard("linear")
        self.observe_stat = observe_stat
        self.block_name = block_name
        self.decoder_two_stage_float = decoder_two_stage_float
        self.exp_mode = exp_mode

        decoder_layers = nn.ModuleList([
            TransformerDecoderLayer(
                d_model, sa_nhead, ca_nhead, dim_feedforward,
                dropout, activation, normalize_before,
                group_detr=group_detr,
                num_feature_levels=num_feature_levels,
                dec_n_points=dec_n_points,
                easy_quantize=easy_quantize,
                decoder_cross_attn_float=decoder_cross_attn_float,
                decoder_cross_attn_value_per_channel=decoder_cross_attn_value_per_channel,
                skip_self_attn=False,
                quant_components=[],
                quant_blocks=self.decoder_block_components if layer_idx in self.decoder_quant_blocks else [],
                decoder_self_attn_quant_components=self.self_attn_quant_components,
                decoder_cross_attn_quant_components=self.cross_attn_quant_components,
                decoder_ffn_quant_components=self.ffn_quant_components,
                decoder_quant_norms=self.decoder_quant_norms,
                observe_stat=observe_stat,
                block_name=block_name + f"2_decoder/layer{layer_idx}/",
            )
            for layer_idx in range(num_decoder_layers)
        ])
        assert decoder_norm_type in ['LN', 'Identity']
        quant_output_norm = (
            "decoder_output_norm" in self.decoder_block_components
            and "layernorm" in self.output_norm_quant_components
        )
        norm = { 
            "LN": lambda channels: IViTLayerNorm(channels) if quant_output_norm else ScaleLayerNorm(channels),
            "Identity": lambda channels: nn.Identity(),
        }
        decoder_norm = norm[decoder_norm_type](d_model)

        self.decoder = TransformerDecoder(decoder_layers, num_decoder_layers, decoder_norm,
                                          return_intermediate=return_intermediate_dec,
                                          d_model=d_model,
                                          lite_refpoint_refine=lite_refpoint_refine,
                                          easy_quantize=easy_quantize,
                                          bbox_reparam=bbox_reparam,
                                          norm_output_qact=decoder_norm_output_qact or (
                                              "decoder_output_norm" in self.decoder_block_components
                                              and "output_qact" in self.output_norm_quant_components
                                          ),
                                          refpoint_head_float=decoder_refpoint_head_float,
                                          quant_components=self.refpoint_quant_components if "refpoint" in self.decoder_block_components else [],
                                          exp_mode=exp_mode,
                                          observe_stat=observe_stat,
                                          block_name=block_name + "2_decoder/")
        
        self.int_exp = IViTExp()
        self.qact_output_proposal = IViTAct(activation_bit=16)
        self.qact_memory_cat = IViTAct()
        self.qact_src_input_bridge = IViTAct()
        self.qact_two_stage_memory = IViTAct()
        self.qact_two_stage_refpoints = IViTAct(activation_bit=16)
        self.qact_two_stage_boxes = IViTAct()

        self.two_stage = two_stage
        if two_stage:
            linear_cls = IViTLinear if (
                "two_stage" in self.decoder_block_components and "enc_output" in self.two_stage_quant_components
            ) else ScaleLinear
            norm_cls = IViTLayerNorm if (
                "two_stage" in self.decoder_block_components and "enc_output_norm" in self.two_stage_quant_components
            ) else ScaleLayerNorm
            self.enc_output = nn.ModuleList([linear_cls(d_model, d_model) for _ in range(group_detr)])
            self.enc_output_norm = nn.ModuleList([norm_cls(d_model) for _ in range(group_detr)])
            self.qact_post_linear_output = nn.ModuleList([IViTAct() for _ in range(group_detr)])
            self.qact_post_norm_output = nn.ModuleList([IViTAct() for _ in range(group_detr)])

        self._reset_parameters()

        self.num_queries = num_queries
        self.d_model = d_model
        self.dec_layers = num_decoder_layers
        self.group_detr = group_detr
        self.num_feature_levels = num_feature_levels
        self.bbox_reparam = bbox_reparam

        self._export = False

    def _exp(self, x, scale):
        if self.exp_mode == "float":
            return x.exp(), None
        return _exp_tensor_scale(self.int_exp, x, scale)
    
    def export(self):
        self._export = True

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, (MSDeformAttn, IMSDeformAttn)):
                m._reset_parameters()
    def get_valid_ratio(self, mask):
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    def forward(self, srcs, masks, pos_embeds, refpoint_embed, refpoint_scale, query_feat, query_scale):
        src_flatten = []
        mask_flatten = [] if masks is not None else None
        spatial_shapes = []
        valid_ratios = [] if masks is not None else None

        srcs = [
            self.qact_src_input_bridge(src, None) if scale is None else (src, scale)
            for src, scale in srcs
        ]
        max_scale = torch.stack([scale for (_, scale) in srcs]).max()
        for lvl, (src, scale) in enumerate(srcs):
            bs, c, h, w = src.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)

            src = src.flatten(2).transpose(1, 2)                # bs, hw, c
            src_int = round_ste.apply(src / scale)
            src = src_int * max_scale
            src_flatten.append(src)
            if masks is not None:
                mask = masks[lvl].flatten(1)                    # bs, hw
                mask_flatten.append(mask)
            
        memory = torch.cat(src_flatten, 1)    # bs, \sum{hxw}, c # Cong: should we cat them? because the scale would be lost
        memory, memory_scale = self.qact_memory_cat(memory, max_scale)   

        save_tensor_from_pathname(memory, self.block_name + "0_memory", observe_stat=self.observe_stat)
        if masks is not None:
            mask_flatten = torch.cat(mask_flatten, 1)   # bs, \sum{hxw}
            valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=memory.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))
        
        if self.two_stage:
            output_memory, output_proposals = gen_encoder_output_proposals(
                memory, mask_flatten, spatial_shapes, unsigmoid=not self.bbox_reparam)
            save_tensor_from_pathname(output_memory, self.block_name + "1_two_stage/0_output_memory", observe_stat=self.observe_stat)
            save_tensor_from_pathname(output_proposals, self.block_name + "1_two_stage/1_output_proposals", observe_stat=self.observe_stat)
            
            quant_two_stage = "two_stage" in self.decoder_block_components
            if not quant_two_stage or "proposal" not in self.two_stage_quant_components:
                output_proposals_scale = None
            else:
                output_proposals, output_proposals_scale = self.qact_output_proposal(output_proposals, None)
            # group detr for first stage
            refpoint_embed_ts, memory_ts, boxes_ts = [], [], []
            group_detr = self.group_detr if self.training else 1
            for g_idx in range(group_detr):
                if not quant_two_stage or (
                    "enc_output" not in self.two_stage_quant_components
                    and "enc_output_norm" not in self.two_stage_quant_components
                ):
                    output_memory_gidx = _linear_float(self.enc_output[g_idx], output_memory)
                    output_memory_gidx = _layer_norm_float(self.enc_output_norm[g_idx], output_memory_gidx)
                    output_memory_gidx_scale = None
                else:
                    # output_memory_gidx = self.enc_output_norm[g_idx](self.enc_output[g_idx](output_memory, memory_scale))
                    output_memory_gidx, output_memory_gidx_scale = self.enc_output[g_idx](output_memory, memory_scale)
                    output_memory_gidx, output_memory_gidx_scale = self.qact_post_linear_output[g_idx](
                        output_memory_gidx, output_memory_gidx_scale)
                    output_memory_gidx, output_memory_gidx_scale = self.enc_output_norm[g_idx](
                        output_memory_gidx, output_memory_gidx_scale)
                    output_memory_gidx, output_memory_gidx_scale = self.qact_post_norm_output[g_idx](
                        output_memory_gidx, output_memory_gidx_scale)
                save_tensor_from_pathname(
                    output_memory_gidx,
                    self.block_name + f"1_two_stage/2_group{g_idx}/0_output_memory",
                    observe_stat=self.observe_stat,
                )
    
                if not quant_two_stage or "class_head" not in self.two_stage_quant_components:
                    enc_outputs_class_unselected_gidx = _linear_float(
                        self.enc_out_class_embed[g_idx], output_memory_gidx
                    )
                    enc_outputs_class_unselected_gidx_scale = None
                else:
                    enc_outputs_class_unselected_gidx, enc_outputs_class_unselected_gidx_scale  = self.enc_out_class_embed[g_idx](output_memory_gidx, output_memory_gidx_scale)
                save_tensor_from_pathname(
                    enc_outputs_class_unselected_gidx,
                    self.block_name + f"1_two_stage/2_group{g_idx}/1_class_logits",
                    observe_stat=self.observe_stat,
                )
                if self.bbox_reparam:
                    if not quant_two_stage or "bbox_head" not in self.two_stage_quant_components:
                        enc_outputs_coord_delta_gidx = _mlp_float(
                            self.enc_out_bbox_embed[g_idx], output_memory_gidx
                        )
                        enc_outputs_coord_delta_gidx_scale = None
                    else:
                        enc_outputs_coord_delta_gidx, enc_outputs_coord_delta_gidx_scale = self.enc_out_bbox_embed[g_idx](output_memory_gidx, output_memory_gidx_scale)
                    tmp, tmp_scale = _multiply_tensor_scale(
                        enc_outputs_coord_delta_gidx[..., :2],
                        enc_outputs_coord_delta_gidx_scale,
                        output_proposals[..., 2:],
                        output_proposals_scale,
                    )

                    enc_outputs_coord_cxcy_gidx, enc_outputs_coord_cxcy_gidx_scale = _add_tensor_scale(
                        tmp,
                        tmp_scale,
                        output_proposals[..., :2],
                        output_proposals_scale,
                    )

                    if quant_two_stage and "bbox_reparam" in self.two_stage_quant_components:
                        tmp, tmp_scale = self._exp(
                            enc_outputs_coord_delta_gidx[..., 2:],
                            enc_outputs_coord_delta_gidx_scale,
                        )
                    else:
                        tmp = enc_outputs_coord_delta_gidx[..., 2:].clamp(min=-20.0, max=20.0).exp()
                        tmp_scale = None
                    enc_outputs_coord_wh_gidx, enc_outputs_coord_wh_gidx_scale = _multiply_tensor_scale(tmp, tmp_scale, output_proposals[..., 2:], output_proposals_scale)

                    concat_list = [(enc_outputs_coord_cxcy_gidx, enc_outputs_coord_cxcy_gidx_scale), (enc_outputs_coord_wh_gidx, enc_outputs_coord_wh_gidx_scale)]
                    enc_outputs_coord_unselected_gidx, enc_outputs_coord_unselected_gidx_scale = _concat_optional_tensor_scale(concat_list, dim=-1)

                else:
                    if not quant_two_stage or "bbox_head" not in self.two_stage_quant_components:
                        tmp = _mlp_float(self.enc_out_bbox_embed[g_idx], output_memory_gidx)
                        tmp_scale = None
                    else:
                        tmp, tmp_scale = self.enc_out_bbox_embed[g_idx](output_memory_gidx, output_memory_gidx_scale)
                    enc_outputs_coord_unselected_gidx, enc_outputs_coord_unselected_gidx_scale = _add_tensor_scale(tmp, tmp_scale, output_proposals, output_proposals_scale)
                enc_outputs_coord_unselected_gidx = _sanitize_boxes(enc_outputs_coord_unselected_gidx)

                topk = self.num_queries
                # Return the index of top k, not the value
                topk_proposals_gidx = torch.topk(enc_outputs_class_unselected_gidx.max(-1)[0], topk, dim=1)[1] # bs, nq
                
                refpoint_embed_gidx_undetach = torch.gather(
                    enc_outputs_coord_unselected_gidx, 1, topk_proposals_gidx.unsqueeze(-1).repeat(1, 1, 4)) # unsigmoid
                
                # for decoder layer, detached as initial ones, (bs, nq, 4)
                refpoint_embed_gidx = refpoint_embed_gidx_undetach.detach()
                refpoint_embed_gidx_scale = enc_outputs_coord_unselected_gidx_scale
                
                # get memory tgt
                tgt_undetach_gidx = torch.gather(
                    output_memory_gidx, 1, topk_proposals_gidx.unsqueeze(-1).repeat(1, 1, self.d_model))
                
                refpoint_embed_ts.append((refpoint_embed_gidx, refpoint_embed_gidx_scale))
                memory_ts.append((tgt_undetach_gidx, output_memory_gidx_scale))
                boxes_ts.append((refpoint_embed_gidx_undetach, refpoint_embed_gidx_scale))
            
            # concat on dim=1, the nq dimension, (bs, nq, d) --> (bs, nq, d)
            refpoint_embed_ts, refpoint_embed_ts_scale = _concat_optional_tensor_scale(refpoint_embed_ts, dim=1)
            # (bs, nq, d)
            memory_ts, memory_ts_scale = _concat_optional_tensor_scale(memory_ts, dim=1)#.transpose(0, 1)
            boxes_ts, boxes_ts_scale = _concat_optional_tensor_scale(boxes_ts, dim=1)#.transpose(0, 1)
            if quant_two_stage and "selected_inputs" in self.two_stage_quant_components:
                refpoint_embed_ts, refpoint_embed_ts_scale = self.qact_two_stage_refpoints(refpoint_embed_ts, None)
                boxes_ts, boxes_ts_scale = self.qact_two_stage_boxes(boxes_ts, None)
                memory_ts, memory_ts_scale = self.qact_two_stage_memory(memory_ts, None)
            save_tensor_from_pathname(memory_ts, self.block_name + "1_two_stage/3_memory_topk", observe_stat=self.observe_stat)
            save_tensor_from_pathname(boxes_ts, self.block_name + "1_two_stage/4_boxes_topk", observe_stat=self.observe_stat)
        
        tgt = query_feat.unsqueeze(0).repeat(bs, 1, 1)
        refpoint_embed = refpoint_embed.unsqueeze(0).repeat(bs, 1, 1)
        if self.two_stage:
            if self.bbox_reparam:
                tmp, tmp_scale = _multiply_tensor_scale(refpoint_embed[..., :2], refpoint_scale, refpoint_embed_ts[..., 2:], refpoint_embed_ts_scale)
                refpoint_embed_cxcy, refpoint_embed_cxcy_scale = _add_tensor_scale(tmp, tmp_scale, refpoint_embed_ts[..., :2], refpoint_embed_ts_scale)

                tmp, tmp_scale = self._exp(refpoint_embed[..., 2:], refpoint_scale)
                refpoint_embed_wh, refpoint_embed_wh_scale = _multiply_tensor_scale(tmp, tmp_scale, refpoint_embed_ts[..., 2:], refpoint_embed_ts_scale)
                refpoint_embed, refpoint_scale = _concat_optional_tensor_scale([(refpoint_embed_cxcy, refpoint_embed_cxcy_scale), (refpoint_embed_wh, refpoint_embed_wh_scale)], dim=-1)
                refpoint_embed = _sanitize_boxes(refpoint_embed)
            else:
                refpoint_embed, refpoint_scale = _add_tensor_scale(refpoint_embed, refpoint_scale, refpoint_embed_ts, refpoint_embed_ts_scale)

        hs, hs_scale, references, references_scale = self.decoder(tgt, query_scale, memory, memory_scale, memory_key_padding_mask=mask_flatten,
                          pos=None, refpoints_unsigmoid=refpoint_embed, refpoint_scale=refpoint_scale,
                          level_start_index=level_start_index, 
                          spatial_shapes=spatial_shapes,
                          valid_ratios=valid_ratios.to(memory.dtype) if valid_ratios is not None else valid_ratios)
        save_tensor_from_pathname(hs, self.block_name + "3_hs", observe_stat=self.observe_stat)
        save_tensor_from_pathname(references, self.block_name + "4_references", observe_stat=self.observe_stat)
        if self.two_stage:
            if self.bbox_reparam:
                return hs, hs_scale, references, references_scale, memory_ts, memory_ts_scale, _sanitize_boxes(boxes_ts)
            else:
                return hs, hs_scale, references, references_scale, memory_ts, memory_ts_scale, boxes_ts.sigmoid()
        return hs, hs_scale, references, references_scale, None, None, None


class TransformerDecoder(nn.Module):

    def __init__(self,
                 decoder_layer,
                 num_layers,
                 norm=None, 
                 return_intermediate=False,
                 d_model=256,
                 lite_refpoint_refine=False,
                 easy_quantize=False,
                 bbox_reparam=False,
                 norm_output_qact=False,
                 refpoint_head_float=False,
                 quant_components=None,
                 exp_mode="ivit",
                 observe_stat=False,
                 block_name=""):
        super().__init__()
        if exp_mode not in {"float", "ivit"}:
            raise ValueError(f"unknown decoder exp_mode: {exp_mode}")
        self.quant_components = _component_set(quant_components)
        if isinstance(decoder_layer, nn.ModuleList):
            self.layers = decoder_layer
        elif isinstance(decoder_layer, (list, tuple)):
            self.layers = nn.ModuleList(decoder_layer)
        else:
            self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.d_model = d_model
        self.norm = norm
        self.return_intermediate = return_intermediate
        self.lite_refpoint_refine = lite_refpoint_refine
        self.easy_quantize = easy_quantize
        self.bbox_reparam = bbox_reparam
        self.norm_output_qact = norm_output_qact
        self.refpoint_head_float = refpoint_head_float
        self.quant_sine_embed = "sine_embed" in self.quant_components
        self.exp_mode = exp_mode
        self.int_exp = IViTExp()
        self.observe_stat = observe_stat
        self.block_name = block_name

        # Keep the original ref_point_head width so pretrained weights remain reusable,
        # and learn a quantization-friendly coordinate expansion in front of it.
        refpoint_quant_components = set(self.quant_components)
        if self.refpoint_head_float:
            refpoint_quant_components.discard("linear")
        self.ref_point_head = IMLP(2 * d_model, d_model, d_model, 2, quant_components=refpoint_quant_components)
        self.ref_point_coord_proj = IMLP(4, d_model, 2 * d_model, 2, quant_components=refpoint_quant_components) if easy_quantize else None
        self.qact_sine_embed = IViTAct()
        self.qact_norm_input = IViTAct()
        self.qact_norm_output = IViTAct()
        self._export = False
    
    def export(self):
        self._export = True

    def _exp(self, x, scale):
        if self.exp_mode == "float":
            return x.clamp(min=-20.0, max=20.0).exp(), None
        return _exp_tensor_scale(self.int_exp, x, scale)

    def _apply_norm(self, x, scale):
        if self.norm is None:
            return x, scale
        if isinstance(self.norm, IViTLayerNorm) and scale is None:
            x, scale = self.qact_norm_input(x, None)
        if isinstance(self.norm, (IViTLayerNorm, ScaleLayerNorm)):
            x, scale = self.norm(x, scale)
        else:
            x = self.norm(x)
        if self.norm_output_qact:
            return self.qact_norm_output(x, scale)
        return x, scale

    def refpoints_refine(self, refpoints_unsigmoid, refpoints_scale, new_refpoints_delta, new_refpoints_delta_scale):
        if self.bbox_reparam:
            tmp, tmp_scale = _multiply_tensor_scale(
                new_refpoints_delta[..., :2],
                new_refpoints_delta_scale,
                refpoints_unsigmoid[..., 2:],
                refpoints_scale,
            )
            new_refpoints_cxcy, new_refpoints_cxcy_scale = _add_tensor_scale(
                tmp,
                tmp_scale,
                refpoints_unsigmoid[..., :2],
                refpoints_scale,
            )
            tmp, tmp_scale = self._exp(
                new_refpoints_delta[..., 2:],
                new_refpoints_delta_scale,
            )
            new_refpoints_wh, new_refpoints_wh_scale = _multiply_tensor_scale(
                tmp,
                tmp_scale,
                refpoints_unsigmoid[..., 2:],
                refpoints_scale,
            )
            new_refpoints_unsigmoid, new_refpoints_scale = _concat_optional_tensor_scale(
                [(new_refpoints_cxcy, new_refpoints_cxcy_scale), (new_refpoints_wh, new_refpoints_wh_scale)],
                dim=-1,
            )
            new_refpoints_unsigmoid = _sanitize_boxes(new_refpoints_unsigmoid)
        else:
            new_refpoints_unsigmoid, new_refpoints_scale = _add_tensor_scale(
                refpoints_unsigmoid,
                refpoints_scale,
                new_refpoints_delta,
                new_refpoints_delta_scale,
            )
        return new_refpoints_unsigmoid, new_refpoints_scale

    def forward(self, tgt, query_scale, memory, memory_scale,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                refpoints_unsigmoid: Optional[Tensor] = None, refpoint_scale = None,
                # for memory
                level_start_index: Optional[Tensor] = None, # num_levels
                spatial_shapes: Optional[Tensor] = None, # bs, num_levels, 2
                valid_ratios: Optional[Tensor] = None):
        output = tgt
        output_scale = query_scale

        intermediate = []
        intermediate_scales = []
        hs_refpoints_unsigmoid = [refpoints_unsigmoid]
        hs_refpoints_scales = [refpoint_scale]
        
        def get_reference(refpoints, refpoint_scale):
            obj_center = refpoints[..., :4]
            
            if self.easy_quantize:
                if self._export:
                    refpoints_input = obj_center[:, :, None] # bs, nq, 1, 4
                    query_pos_input = obj_center
                else:
                    refpoints_input = obj_center[:, :, None] \
                                            * torch.cat([valid_ratios, valid_ratios], -1)[:, None] # bs, nq, nlevel, 4
                    query_pos_input = refpoints_input[:, :, 0, :]
                if "coord_proj" in self.quant_components:
                    query_pos_embed, query_pos_embed_scale = self.ref_point_coord_proj(query_pos_input, refpoint_scale)
                else:
                    query_pos_embed = _mlp_float(self.ref_point_coord_proj, query_pos_input)
                    query_pos_embed_scale = None
                query_pos, query_pos_scale = self.ref_point_head(query_pos_embed, query_pos_embed_scale)
                query_sine_embed = query_pos_input
            else:
                if self._export:
                    query_sine_embed = gen_sineembed_for_position(obj_center, self.d_model // 2)
                    refpoints_input = obj_center[:, :, None] # bs, nq, 1, 4
                else:
                    refpoints_input = obj_center[:, :, None] \
                                            * torch.cat([valid_ratios, valid_ratios], -1)[:, None] # bs, nq, nlevel, 4
                    query_sine_embed = gen_sineembed_for_position(
                        refpoints_input[:, :, 0, :], self.d_model // 2)
                if self.quant_sine_embed:
                    query_sine_embed, query_sine_embed_scale = self.qact_sine_embed(query_sine_embed, None)
                else:
                    query_sine_embed_scale = None
                query_pos, query_pos_scale = self.ref_point_head(query_sine_embed, query_sine_embed_scale)
            return obj_center, refpoints_input, query_pos, query_pos_scale, query_sine_embed
        
        # always use init refpoints
        if self.lite_refpoint_refine:
            if self.bbox_reparam:
                obj_center, refpoints_input, query_pos, query_pos_scale, query_sine_embed = get_reference(refpoints_unsigmoid, refpoint_scale)
            else:
                obj_center, refpoints_input, query_pos, query_pos_scale, query_sine_embed = get_reference(refpoints_unsigmoid.sigmoid(), refpoint_scale)

        for layer_id, layer in enumerate(self.layers):
            # iter refine each layer
            if not self.lite_refpoint_refine:
                if self.bbox_reparam:
                    obj_center, refpoints_input, query_pos, query_pos_scale, query_sine_embed = get_reference(refpoints_unsigmoid, refpoint_scale)
                else:
                    obj_center, refpoints_input, query_pos, query_pos_scale, query_sine_embed = get_reference(refpoints_unsigmoid.sigmoid(), refpoint_scale)
            save_tensor_from_pathname(query_pos, self.block_name + f"1_layer{layer_id}/0_query_pos", observe_stat=self.observe_stat)

            # For the first decoder layer, we do not apply transformation over p_s
            pos_transformation = 1

            query_pos = query_pos * pos_transformation
            if hasattr(layer, "block_name"):
                layer.block_name = self.block_name + f"1_layer{layer_id}/"
            
            output, output_scale = layer(output, output_scale, memory, memory_scale, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos, query_pos_scale=query_pos_scale, query_sine_embed=query_sine_embed, 
                           is_first=(layer_id == 0),
                           reference_points=refpoints_input, refpoint_scale = refpoint_scale,
                           spatial_shapes=spatial_shapes,
                           level_start_index=level_start_index)

            if not self.lite_refpoint_refine:
                # box iterative update
                new_refpoints_delta, new_refpoints_delta_scale = self.bbox_embed(output, output_scale)
                new_refpoints_unsigmoid, new_refpoints_scale = self.refpoints_refine(
                    refpoints_unsigmoid,
                    refpoint_scale,
                    new_refpoints_delta,
                    new_refpoints_delta_scale,
                )
                if layer_id != self.num_layers - 1:
                    hs_refpoints_unsigmoid.append(new_refpoints_unsigmoid)
                    hs_refpoints_scales.append(new_refpoints_scale)
                refpoints_unsigmoid = new_refpoints_unsigmoid.detach()
                refpoint_scale = new_refpoints_scale

            if self.return_intermediate:
                normed_output, normed_output_scale = self._apply_norm(output, output_scale)
                intermediate.append(normed_output)
                intermediate_scales.append(normed_output_scale)

        # if self.norm is not None:
        #     output, output_scale = self._apply_norm(output, output_scale)
        #     if self.return_intermediate:
        #         intermediate.pop()
        #         intermediate_scales.pop()
        #         intermediate.append(output)
        #         intermediate_scales.append(output_scale)

        if self.return_intermediate:
            if self._export:
                # to shape: B, N, C
                hs = intermediate[-1]
                hs_scale = intermediate_scales[-1]
                if self.bbox_embed is not None:
                    ref = hs_refpoints_unsigmoid[-1]
                    ref_scale = hs_refpoints_scales[-1]
                else:
                    ref = refpoints_unsigmoid
                    ref_scale = refpoint_scale
                return hs, hs_scale, ref, ref_scale
            # box iterative update
            hs, hs_scale = _concat_optional_tensor_scale(
                [(x.unsqueeze(0), scale) for x, scale in zip(intermediate, intermediate_scales)],
                dim=0,
            )
            if self.bbox_embed is not None:
                references, references_scale = _concat_optional_tensor_scale(
                    [(x.unsqueeze(0), scale) for x, scale in zip(hs_refpoints_unsigmoid, hs_refpoints_scales)],
                    dim=0,
                )
                return hs, hs_scale, references, references_scale
            else:
                return hs, hs_scale, refpoints_unsigmoid.unsqueeze(0), refpoint_scale
        
        return output.unsqueeze(0), output_scale, refpoints_unsigmoid.unsqueeze(0), refpoint_scale


class QuantDecoderSelfAttention(nn.Module):

    def __init__(self, d_model, nhead, quant_components=None):
        super().__init__()
        self.quant_components = _component_set(quant_components)
        self.quant_linear = "linear" in self.quant_components
        self.quant_matmul = "matmul" in self.quant_components
        self.quant_softmax = "softmax" in self.quant_components
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.scale = self.head_dim ** -0.5

        linear_cls = IViTLinear if self.quant_linear else ScaleLinear
        self.q_proj = linear_cls(d_model, d_model)
        self.k_proj = linear_cls(d_model, d_model)
        self.v_proj = linear_cls(d_model, d_model)
        self.out_proj = linear_cls(d_model, d_model)

        self.qact_qk_input = IViTAct()
        self.qact_v_input = IViTAct()
        self.qact_q = IViTAct()
        self.qact_k = IViTAct()
        self.qact_v = IViTAct()
        self.qact_attn = IViTAct()
        self.qact_softmax_out = IViTAct()
        self.qact_out = IViTAct()
        self.qact_proj_drop = IViTAct()
        self.matmul = IViTMatMul() if self.quant_matmul else ScaleMatMul()
        self.softmax = IViTSoftmaxV2(output_bit=16, dim=-1, m=27) if self.quant_softmax else ScaleSoftmax(dim=-1)

    def forward(self, tgt, tgt_scale, query_pos, query_pos_scale):
        qk_input, qk_scale = _add_tensor_scale(tgt, tgt_scale, query_pos, query_pos_scale)
        v_input, v_scale = tgt, tgt_scale

        if isinstance(self.q_proj, IViTLinear) and qk_scale is None:
            qk_input, qk_scale = self.qact_qk_input(qk_input, None)
        if isinstance(self.v_proj, IViTLinear) and v_scale is None:
            v_input, v_scale = self.qact_v_input(v_input, None)

        q, q_scale = self.q_proj(qk_input, qk_scale)
        k, k_scale = self.k_proj(qk_input, qk_scale)
        v, v_scale = self.v_proj(v_input, v_scale)
        if self.quant_matmul:
            q, q_scale = self.qact_q(q, q_scale)
            k, k_scale = self.qact_k(k, k_scale)
            v, v_scale = self.qact_v(v, v_scale)

        bs, num_queries, _ = q.shape
        q = q.view(bs, num_queries, self.nhead, self.head_dim).transpose(1, 2)
        k = k.view(bs, num_queries, self.nhead, self.head_dim).transpose(1, 2)
        v = v.view(bs, num_queries, self.nhead, self.head_dim).transpose(1, 2)

        attn, attn_scale = self.matmul(q, q_scale, k.transpose(-2, -1), k_scale)
        attn = attn * self.scale
        attn_scale = _scale_mul(attn_scale, self.scale)
        if self.quant_softmax and attn_scale is None:
            attn, attn_scale = self.qact_attn(attn, None)
        attn, attn_scale = self.softmax(attn, attn_scale)
        if self.quant_matmul and attn_scale is None:
            attn, attn_scale = self.qact_softmax_out(attn, None)

        out, out_scale = self.matmul(attn, attn_scale, v, v_scale)
        out = out.transpose(1, 2).reshape(bs, num_queries, self.d_model)
        if self.quant_linear and out_scale is None:
            out, out_scale = self.qact_out(out, None)
        out, out_scale = self.out_proj(out, out_scale)
        if self.quant_matmul or self.quant_softmax:
            out, out_scale = self.qact_proj_drop(out, out_scale)
        return out, out_scale


class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, sa_nhead, ca_nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False, group_detr=1, 
                 num_feature_levels=4, dec_n_points=4,
                 easy_quantize=False,
                 decoder_cross_attn_float=False,
                 decoder_cross_attn_value_per_channel=False,
                 skip_self_attn=False,
                 quant_components=None,
                 quant_blocks=None,
                 decoder_self_attn_quant_components=None,
                 decoder_cross_attn_quant_components=None,
                 decoder_ffn_quant_components=None,
                 decoder_quant_norms=None,
                 observe_stat=False,
                 block_name=""):
        super().__init__()
        self.quant_components = _component_set(quant_components)
        self.quant_blocks = set(quant_blocks) if quant_blocks is not None else set()
        self.self_attn_quant_components = (
            _component_list(decoder_self_attn_quant_components, {"linear", "matmul", "softmax"})
            if "self_attn" in self.quant_blocks else set()
        )
        self.cross_attn_quant_components = (
            _decoder_cross_attn_components(decoder_cross_attn_quant_components)
            if "cross_attn" in self.quant_blocks else set()
        )
        self.ffn_quant_components = (
            _component_list(decoder_ffn_quant_components, {"linear"})
            if "ffn" in self.quant_blocks else set()
        )
        self.quant_ffn_linear = "linear" in self.ffn_quant_components
        self.quant_cross_attn = bool(self.cross_attn_quant_components)
        self.quant_layernorm = "norm" in self.quant_blocks
        self.decoder_quant_norms = _decoder_norm_set(decoder_quant_norms)
        self.quant_norm1 = self.quant_layernorm and "norm1" in self.decoder_quant_norms
        self.quant_norm2 = self.quant_layernorm and "norm2" in self.decoder_quant_norms
        self.quant_norm3 = self.quant_layernorm and "norm3" in self.decoder_quant_norms
        linear_cls = IViTLinear if self.quant_ffn_linear else ScaleLinear
        norm1_cls = IViTLayerNorm if self.quant_norm1 else ScaleLayerNorm
        norm2_cls = IViTLayerNorm if self.quant_norm2 else ScaleLayerNorm
        norm3_cls = IViTLayerNorm if self.quant_norm3 else ScaleLayerNorm
        # Decoder Self-Attention
        self.self_attn = QuantDecoderSelfAttention(d_model, sa_nhead, quant_components=self.self_attn_quant_components)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = norm1_cls(d_model)

        # Decoder Cross-Attention
        cross_attn_cls = IMSDeformAttn if self.quant_cross_attn else ScaleMSDeformAttn
        cross_attn_kwargs = {}
        if self.quant_cross_attn:
            cross_attn_kwargs["quant_components"] = self.cross_attn_quant_components
        self.cross_attn = cross_attn_cls(
            d_model, n_levels=num_feature_levels, n_heads=ca_nhead, n_points=dec_n_points,
            sampling_mode="nearest" if easy_quantize else "bilinear",
            use_float_path=decoder_cross_attn_float,
            value_per_channel=decoder_cross_attn_value_per_channel,
            observe_stat=observe_stat,
            block_name=block_name + "2_cross_attn/",
            **cross_attn_kwargs)

        self.nhead = ca_nhead

        # Implementation of Feedforward model
        self.linear1 = linear_cls(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = linear_cls(dim_feedforward, d_model)

        self.norm2 = norm2_cls(d_model)
        self.norm3 = norm3_cls(d_model)
        
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = F.relu
        self.normalize_before = normalize_before
        self.group_detr = group_detr
        self.observe_stat = observe_stat
        self.block_name = block_name
        self.qact_norm1_input = IViTAct()
        self.qact_norm2_input = IViTAct()
        self.qact_norm3_input = IViTAct()
        self.qact_ffn_1 = IViTAct()
        self.qact_ffn_relu = IViTAct()
        self.qact_ffn_2 = IViTAct()
        self.qact_self_attn_block = IViTAct()
        self.qact_cross_attn_block = IViTAct()
        self.qact_ffn_block = IViTAct()
        

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def _apply_norm(self, norm, qact, x, scale):
        if isinstance(norm, IViTLayerNorm) and scale is None:
            x, scale = qact(x, None)
        if isinstance(norm, (IViTLayerNorm, ScaleLayerNorm)):
            return norm(x, scale)
        return norm(x), scale

    def _qact_before_norm(self, norm, qact, x, scale):
        if isinstance(norm, IViTLayerNorm):
            return qact(x, scale)
        return x, scale

    def _maybe_qact(self, qact, x, scale, enabled):
        if enabled:
            return qact(x, scale)
        return x, scale

    def forward_post(self, tgt, query_scale, memory, memory_scale,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None,
                     query_pos_scale = None,
                     query_sine_embed = None,
                     is_first = False,
                     reference_points = None, refpoint_scale = None,
                     spatial_shapes=None,
                     level_start_index=None,
                     ):
        bs, num_queries, _ = tgt.shape
        
        # ========== Begin of Self-Attention =============
        # Apply projections here
        # shape: batch_size x num_queries x 256
        if self.training:
            group_queries = num_queries // self.group_detr
            tgt_sa = torch.cat(tgt.split(group_queries, dim=1), dim=0)
            query_pos_sa = torch.cat(query_pos.split(group_queries, dim=1), dim=0)
        else:
            tgt_sa = tgt
            query_pos_sa = query_pos

        tgt2, tgt2_scale = self.self_attn(tgt_sa, query_scale, query_pos_sa, query_pos_scale)
        
        if self.training:
            tgt2 = torch.cat(tgt2.split(bs, dim=0), dim=1)
        save_tensor_from_pathname(tgt2, self.block_name + "0_self_attn", observe_stat=self.observe_stat)
        # ========== End of Self-Attention =============

        tgt, query_scale = _add_tensor_scale(tgt, query_scale, self.dropout1(tgt2), tgt2_scale)
        tgt, query_scale = self._qact_before_norm(self.norm1, self.qact_norm1_input, tgt, query_scale)
        tgt, query_scale = self._apply_norm(self.norm1, self.qact_norm1_input, tgt, query_scale)
        tgt, query_scale = self._maybe_qact(
            self.qact_self_attn_block,
            tgt,
            query_scale,
            self.quant_cross_attn or self.quant_norm1 or self.quant_ffn_linear,
        )
        save_tensor_from_pathname(tgt, self.block_name + "1_norm1", observe_stat=self.observe_stat)

        # ========== Begin of Cross-Attention =============
        query_with_pos, query_with_pos_scale = _add_tensor_scale(tgt, query_scale, query_pos, query_pos_scale)
        save_tensor_from_pathname(query_with_pos, self.block_name + "2_cross_attn/0_pre_call_query", observe_stat=self.observe_stat)
        if hasattr(self.cross_attn, "block_name"):
            self.cross_attn.block_name = self.block_name + "2_cross_attn/"
        tgt2, tgt2_scale = self.cross_attn(
            query_with_pos,
            reference_points,
            memory,
            spatial_shapes,
            level_start_index,
            memory_key_padding_mask,
            query_with_pos_scale,
            memory_scale,
        )
        save_tensor_from_pathname(tgt2, self.block_name + "2_cross_attn_output", observe_stat=self.observe_stat)
        # ========== End of Cross-Attention =============

        tgt, query_scale = _add_tensor_scale(tgt, query_scale, self.dropout2(tgt2), tgt2_scale)
        save_tensor_from_pathname(tgt, self.block_name + "3_norm2/0_residual_input", observe_stat=self.observe_stat)
        tgt, query_scale = self._qact_before_norm(self.norm2, self.qact_norm2_input, tgt, query_scale)
        tgt, query_scale = self._apply_norm(self.norm2, self.qact_norm2_input, tgt, query_scale)
        tgt, query_scale = self._maybe_qact(
            self.qact_cross_attn_block,
            tgt,
            query_scale,
            self.quant_norm2 or self.quant_ffn_linear,
        )
        save_tensor_from_pathname(tgt, self.block_name + "3_norm2/1_output", observe_stat=self.observe_stat)
        tgt2, tgt2_scale = self.linear1(tgt, query_scale)
        tgt2, tgt2_scale = self._maybe_qact(self.qact_ffn_1, tgt2, tgt2_scale, self.quant_ffn_linear)
        tgt2 = self.activation(tgt2)
        tgt2, tgt2_scale = self._maybe_qact(self.qact_ffn_relu, tgt2, tgt2_scale, self.quant_ffn_linear)
        tgt2, tgt2_scale = self.linear2(tgt2, tgt2_scale)
        tgt2, tgt2_scale = self._maybe_qact(self.qact_ffn_2, tgt2, tgt2_scale, self.quant_ffn_linear)
        save_tensor_from_pathname(tgt2, self.block_name + "4_ffn", observe_stat=self.observe_stat)
        tgt, query_scale = _add_tensor_scale(tgt, query_scale, self.dropout3(tgt2), tgt2_scale)
        tgt, query_scale = self._qact_before_norm(self.norm3, self.qact_norm3_input, tgt, query_scale)
        tgt, query_scale = self._apply_norm(self.norm3, self.qact_norm3_input, tgt, query_scale)
        tgt, query_scale = self._maybe_qact(self.qact_ffn_block, tgt, query_scale, self.quant_norm3 or self.quant_ffn_linear)
        save_tensor_from_pathname(tgt, self.block_name + "5_norm3", observe_stat=self.observe_stat)
        return tgt, query_scale

    def forward(self, tgt, query_scale, memory, memory_scale,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None,
                query_pos_scale = None,
                query_sine_embed = None,
                is_first = False,
                reference_points = None, refpoint_scale = None,
                spatial_shapes=None,
                level_start_index=None):
        return self.forward_post(tgt, query_scale, memory, memory_scale, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos, query_pos_scale,
                                 query_sine_embed, is_first,
                                 reference_points, refpoint_scale, spatial_shapes, level_start_index)


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def build_transformer(args):
    
    try:
        two_stage = args.two_stage
    except:
        two_stage = False

    return Transformer(
        d_model=args.hidden_dim,
        sa_nhead=args.sa_nheads,
        ca_nhead=args.ca_nheads,
        num_queries=args.num_queries,
        dropout=args.dropout,
        dim_feedforward=args.dim_feedforward,
        num_decoder_layers=args.dec_layers,
        return_intermediate_dec=True,
        group_detr=args.group_detr,
        two_stage=two_stage,
        num_feature_levels=args.num_feature_levels,
        dec_n_points=args.dec_n_points,
        lite_refpoint_refine=args.lite_refpoint_refine,
        easy_quantize=args.easy_quantize,
        decoder_norm_type=args.decoder_norm,
        bbox_reparam=args.bbox_reparam,
        decoder_cross_attn_float=getattr(args, "decoder_cross_attn_float", False),
        decoder_cross_attn_value_per_channel=getattr(args, "decoder_cross_attn_value_per_channel", False),
        decoder_two_stage_float=getattr(args, "decoder_two_stage_float", False),
        decoder_norm_output_qact=getattr(args, "decoder_norm_output_qact", False),
        decoder_refpoint_head_float=getattr(args, "decoder_refpoint_head_float", False),
        transformer_quant_components=getattr(args, "transformer_quant_components", None),
        decoder_quant_blocks=getattr(args, "decoder_quant_blocks", None),
        decoder_quant_norms=getattr(args, "decoder_quant_norms", None),
        decoder_self_attn_quant_components=getattr(args, "decoder_self_attn_quant_components", None),
        decoder_cross_attn_quant_components=getattr(args, "decoder_cross_attn_quant_components", None),
        decoder_ffn_quant_components=getattr(args, "decoder_ffn_quant_components", None),
        decoder_refpoint_quant_components=getattr(args, "decoder_refpoint_quant_components", None),
        decoder_two_stage_quant_components=getattr(args, "decoder_two_stage_quant_components", None),
        decoder_output_norm_quant_components=getattr(args, "decoder_output_norm_quant_components", None),
        exp_mode=getattr(args, "transformer_exp_mode", "ivit"),
        observe_stat=getattr(args, "observe_stat", False),
    )
