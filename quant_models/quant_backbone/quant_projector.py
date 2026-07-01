# ------------------------------------------------------------------------
# LW-DETR
# Copyright (c) 2024 Baidu. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from ViTDet (https://github.com/facebookresearch/detectron2/tree/main/projects/ViTDet)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------

"""
Projector
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.backbone.projector import LayerNorm as FloatLayerNorm
from quant_models.fake_quantized.layers import *
from quant_models.fake_quantized.utils import *
from util.decorators import save_tensor_from_pathname


def _component_set(components):
    if components is None:
        return {"conv2d", "gelu", "layernorm"}
    return set(components)


def _as_quant_input(x, act_scaling_factor, qact_bridge):
    if act_scaling_factor is None:
        return qact_bridge(x=x, act_scaling_factor=None)
    return x, act_scaling_factor


def _get_activation(name, inplace=False, quant=False):
    if name == "gelu":
        return ISegGELU() if quant else nn.GELU()
    if name == "silu":
        return IViTSiLU(lambda_scaler=8) if quant else nn.SiLU()
    if name == "relu":
        return nn.ReLU(inplace=inplace)
    if name in ["LeakyReLU", "leakyrelu", "lrelu"]:
        return nn.LeakyReLU(0.1, inplace=inplace)
    if name is None:
        return nn.Identity()
    raise AttributeError("Unsupported act type: {}".format(name))


def _is_conv(layer):
    return isinstance(layer, (nn.Conv2d, nn.ConvTranspose2d, IViTConv2d, IViTConvTranspose2d))


def _is_quant_conv(layer):
    return isinstance(layer, (IViTConv2d, IViTConvTranspose2d))


def _is_norm(layer):
    return isinstance(layer, (FloatLayerNorm, IViTLayerNorm))


def _is_quant_norm(layer):
    return isinstance(layer, IViTLayerNorm)


def _is_quant_activation(layer):
    return isinstance(layer, (ISegGELU, IViTSiLU))


def _forward_projector_layer(layer, x, act_scaling_factor, qact_bridge=None, qact_post=None):
    if isinstance(layer, nn.Identity):
        return x, act_scaling_factor

    if _is_quant_conv(layer):
        x, act_scaling_factor = _as_quant_input(x, act_scaling_factor, qact_bridge)
        x, act_scaling_factor = layer(x, act_scaling_factor)
        return qact_post(x, act_scaling_factor)

    if _is_quant_norm(layer):
        x, act_scaling_factor = _as_quant_input(x, act_scaling_factor, qact_bridge)
        x, act_scaling_factor = layer(x, act_scaling_factor)
        return qact_post(x, act_scaling_factor)

    if _is_quant_activation(layer):
        x, act_scaling_factor = _as_quant_input(x, act_scaling_factor, qact_bridge)
        x, act_scaling_factor = layer(x, act_scaling_factor)
        return x, act_scaling_factor

    if _is_conv(layer) or _is_norm(layer) or isinstance(layer, (nn.GELU, nn.SiLU, nn.ReLU, nn.LeakyReLU)):
        return layer(x), None

    return layer(x, act_scaling_factor)


def _forward_projector_sequence(layers, qact_bridge, qact_post, x, act_scaling_factor):
    for idx, layer in enumerate(layers):
        x, act_scaling_factor = _forward_projector_layer(
            layer,
            x,
            act_scaling_factor,
            qact_bridge[idx],
            qact_post[idx],
        )
    return x, act_scaling_factor


def _make_conv2d(quant_components, *args, transpose=False, **kwargs):
    if "conv2d" in _component_set(quant_components):
        conv_cls = IViTConvTranspose2d if transpose else IViTConv2d
        return conv_cls(*args, **kwargs)
    conv_cls = nn.ConvTranspose2d if transpose else nn.Conv2d
    float_kwargs = {k: v for k, v in kwargs.items() if k not in {"weight_bit", "bias_bit"}}
    return conv_cls(*args, **float_kwargs)


def _make_norm(channels, quant_components):
    if "layernorm" in _component_set(quant_components):
        return IViTLayerNorm(channels, dim=1)
    return FloatLayerNorm(channels)


class IConvX(nn.Module):
    """Conv-activation module with direct component modules for checkpoint compatibility."""

    def __init__(
        self,
        in_planes,
        out_planes,
        kernel=3,
        stride=1,
        groups=1,
        dilation=1,
        act="relu",
        observe_stat=False,
        block_name="",
        quant_components=None,
        split_input_channels=None,
    ):
        super().__init__()
        quant_components = _component_set(quant_components)
        self.conv = _make_conv2d(
            quant_components,
            in_planes,
            out_planes,
            kernel_size=kernel,
            stride=stride,
            padding=kernel // 2,
            groups=groups,
            dilation=dilation,
            bias=True,
            weight_bit=8,
            bias_bit=32,
        )
        self.act = _get_activation(
            act,
            inplace=True,
            quant=(act in quant_components),
        )
        self.qact_conv_input_bridge = IViTAct(
            observer_mode="percentile",
            percentile=99.9,
        )
        self.qact_split_input_bridges = nn.ModuleList(
            [
                IViTAct(observer_mode="percentile", percentile=99.9)
                for _ in (split_input_channels or [])
            ]
        )
        split_count = len(split_input_channels or [])
        self.qact_split_output_bridges = nn.ModuleList(
            [
                IViTAct(observer_mode="percentile", percentile=99.9)
                for _ in range(split_count)
            ]
        )
        self.register_buffer("split_branch_output_scaling_factors", torch.ones(split_count))
        self.register_buffer("split_out_scaling_factor", torch.tensor(1.0))
        self.qact_post_conv = IViTAct(
            observer_mode="percentile",
            percentile=99.9,
        )
        self.qact_act_input_bridge = IViTAct()
        self.qact_post_act = IViTAct()
        self.block_name = block_name
        self.observe_stat = observe_stat

    def forward(self, x, act_scaling_factor=None):
        x, act_scaling_factor = _forward_projector_layer(
            self.conv,
            x,
            act_scaling_factor,
            self.qact_conv_input_bridge,
            self.qact_post_conv,
        )
        save_tensor_from_pathname(x, self.block_name + "0_conv", observe_stat=self.observe_stat)
        x, act_scaling_factor = _forward_projector_layer(
            self.act,
            x,
            act_scaling_factor,
            self.qact_act_input_bridge,
            self.qact_post_act,
        )
        save_tensor_from_pathname(x, self.block_name + "1_act", observe_stat=self.observe_stat)
        return x, act_scaling_factor

    def forward_split_inputs(self, inputs):
        conv_out, conv_scale = self._forward_split_conv(inputs)
        conv_out, conv_scale = self.qact_post_conv(conv_out, conv_scale)
        save_tensor_from_pathname(conv_out, self.block_name + "0_conv", observe_stat=self.observe_stat)
        conv_out, conv_scale = _forward_projector_layer(
            self.act,
            conv_out,
            conv_scale,
            self.qact_act_input_bridge,
            self.qact_post_act,
        )
        save_tensor_from_pathname(conv_out, self.block_name + "1_act", observe_stat=self.observe_stat)
        return conv_out, conv_scale

    def _forward_split_conv(self, inputs):
        """Apply one Conv2d to multiple input-channel blocks without concatenating them first."""
        if isinstance(self.conv, IViTConv2d):
            return self._forward_split_quant_conv2d(inputs)

        if isinstance(self.conv, nn.Conv2d):
            return self._forward_split_float_conv2d(inputs)

        raise TypeError("split projector conv only supports Conv2d, got {}".format(type(self.conv)))

    def _forward_split_quant_conv2d(self, inputs):
        weight_integer, conv_scaling_factor = self.conv.quantized_weight()
        partials = []
        partial_scales = []
        start = 0

        for input_idx, (feat, scale) in enumerate(inputs):
            channels = feat.shape[1]
            end = start + channels
            weight_slice = weight_integer[:, start:end]

            if scale is None:
                qact_bridge = (
                    self.qact_split_input_bridges[input_idx]
                    if input_idx < len(self.qact_split_input_bridges)
                    else self.qact_conv_input_bridge
                )
                feat, scale = qact_bridge(feat, None)
            if scale.numel() != 1:
                scale = scale.max()

            x_int = round_ste.apply(feat / scale)
            y_int = F.conv2d(
                x_int,
                weight_slice,
                None,
                self.conv.stride,
                self.conv.padding,
                self.conv.dilation,
                self.conv.groups,
            )
            partial_scale = conv_scaling_factor * scale
            partial = y_int * partial_scale.view(1, -1, 1, 1)
            if input_idx < len(self.qact_split_output_bridges):
                partial, partial_scale = self.qact_split_output_bridges[input_idx](
                    partial,
                    partial_scale,
                )
                partial_scale = partial_scale.reshape(-1).max()
                if input_idx < self.split_branch_output_scaling_factors.numel():
                    self.split_branch_output_scaling_factors[input_idx].copy_(partial_scale.detach())
            partials.append(partial)
            partial_scales.append(partial_scale)
            start = end

        out_scale = torch.stack([scale.max() for scale in partial_scales]).max()
        self.split_out_scaling_factor.copy_(out_scale.detach())
        out_int = sum(round_ste.apply(partial / out_scale) for partial in partials)
        if self.conv.bias is not None:
            bias_int = round_ste.apply(self.conv.bias / out_scale)
            self.conv.bias_integer = bias_int
            out_int = out_int + bias_int.view(1, -1, 1, 1)
        return out_int * out_scale, out_scale

    def _forward_split_float_conv2d(self, inputs):
        outputs = []
        start = 0

        for feat, _ in inputs:
            channels = feat.shape[1]
            end = start + channels
            weight_slice = self.conv.weight[:, start:end]
            outputs.append(
                F.conv2d(
                    feat,
                    weight_slice,
                    None,
                    self.conv.stride,
                    self.conv.padding,
                    self.conv.dilation,
                    self.conv.groups,
                )
            )
            start = end

        out = sum(outputs)
        if self.conv.bias is not None:
            out = out + self.conv.bias.view(1, -1, 1, 1)
        return out, None


class IBottleneck(nn.Module):
    """Standard bottleneck."""

    def __init__(
        self,
        c1,
        c2,
        shortcut=True,
        g=1,
        k=(3, 3),
        e=0.5,
        act="silu",
        block_name="",
        observe_stat=False,
        quant_components=None,
    ):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = IConvX(c1, c_, k[0], 1, act=act, block_name=block_name + "cv1/", observe_stat=observe_stat, quant_components=quant_components)
        self.cv2 = IConvX(c_, c2, k[1], 1, groups=g, act=act, block_name=block_name + "cv2/", observe_stat=observe_stat, quant_components=quant_components)
        self.add = shortcut and c1 == c2
        self.block_name = block_name
        self.observe_stat = observe_stat

    def forward(self, x, act_scaling_factor=None):
        x_short = x
        act_scaling_factor_short = act_scaling_factor
        x, act_scaling_factor = self.cv1(x, act_scaling_factor)
        save_tensor_from_pathname(x, self.block_name + "cv1", observe_stat=self.observe_stat)
        x, act_scaling_factor = self.cv2(x, act_scaling_factor)
        save_tensor_from_pathname(x, self.block_name + "cv2", observe_stat=self.observe_stat)
        if self.add:
            if act_scaling_factor is None or act_scaling_factor_short is None:
                x = x + x_short
                act_scaling_factor = None
            else:
                x, act_scaling_factor = add_tensor_scale(x, act_scaling_factor, x_short, act_scaling_factor_short)
        return x, act_scaling_factor


class IC2f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(
        self,
        c1,
        c2,
        n=1,
        shortcut=False,
        g=1,
        e=0.5,
        act="silu",
        observe_stat=False,
        block_name="",
        quant_components=None,
        split_input_channels=None,
    ):
        super().__init__()
        self.observe_stat = observe_stat
        self.block_name = block_name
        self.c = int(c2 * e)
        self.cv1 = IConvX(
            c1,
            2 * self.c,
            1,
            1,
            act=act,
            observe_stat=observe_stat,
            block_name=block_name + "0_cv1/",
            quant_components=quant_components,
            split_input_channels=split_input_channels,
        )
        self.cv2 = IConvX((2 + n) * self.c, c2, 1, act=act, observe_stat=observe_stat, block_name=block_name + "2_cv2/", quant_components=quant_components)
        self.m = nn.ModuleList(
            IBottleneck(self.c, self.c, shortcut, g, k=(3, 3), e=1.0, act=act, block_name=block_name + f"1_bot{i}/", observe_stat=observe_stat, quant_components=quant_components)
            for i in range(n)
        )

    def forward(self, x, act_scaling_factor=None):
        x_out, act_scaling_factor = self.cv1(x, act_scaling_factor)
        y = [(tmp, act_scaling_factor) for tmp in x_out.split((self.c, self.c), 1)]

        for layer in self.m:
            x_out, scale = layer(y[-1][0], y[-1][1])
            y.append((x_out, scale))

        if any(scale is None for _, scale in y):
            concatenated_output = torch.cat([feat for feat, _ in y], 1)
            act_scaling_factor = None
        else:
            concatenated_output, act_scaling_factor = concat_list_tensor_scale(y, 1)

        return self.cv2(concatenated_output, act_scaling_factor)

    def forward_split_inputs(self, inputs):
        x_out, act_scaling_factor = self.cv1.forward_split_inputs(inputs)
        y = [(tmp, act_scaling_factor) for tmp in x_out.split((self.c, self.c), 1)]

        for layer in self.m:
            x_out, scale = layer(y[-1][0], y[-1][1])
            y.append((x_out, scale))

        if any(scale is None for _, scale in y):
            concatenated_output = torch.cat([feat for feat, _ in y], 1)
            act_scaling_factor = None
        else:
            concatenated_output, act_scaling_factor = concat_list_tensor_scale(y, 1)

        return self.cv2(concatenated_output, act_scaling_factor)


class IMultiScaleProjector(nn.Module):
    """
    This module implements MultiScaleProjector in :paper:`lwdetr`.
    It creates pyramid features built on top of the input feature map.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        scale_factors,
        num_blocks=3,
        observe_stat=False,
        block_name="",
        quant_components=None,
    ):
        super().__init__()
        self.scale_factors = scale_factors
        self.observe_stat = observe_stat
        self.block_name = block_name
        self.quant_components = _component_set(quant_components)

        stages_sampling = []
        stages = []
        stages_sampling_qact_bridge = []
        stages_sampling_qact_post = []
        stages_qact_bridge = []
        stages_qact_post = []
        self.use_extra_pool = False
        self.qact_post_fuse = IViTAct()
        for stage_idx, scale in enumerate(scale_factors):
            stage_block_name = block_name + f"stage{stage_idx}/"
            stages_sampling.append([])
            stages_sampling_qact_bridge.append([])
            stages_sampling_qact_post.append([])
            stage_input_channels = []
            for input_idx, in_dim in enumerate(in_channels):
                sampling_block_name = stage_block_name + f"0_sampling/input{input_idx}/"
                out_dim = in_dim
                if scale == 4.0:
                    layers = [
                        _make_conv2d(self.quant_components, in_dim, in_dim // 2, kernel_size=2, stride=2, bias=True, weight_bit=8, bias_bit=32, transpose=True),
                        _make_norm(in_dim // 2, self.quant_components),
                        _get_activation("gelu", quant="gelu" in self.quant_components),
                        _make_conv2d(self.quant_components, in_dim // 2, in_dim // 4, kernel_size=2, stride=2, bias=True, weight_bit=8, bias_bit=32, transpose=True),
                    ]
                    out_dim = in_dim // 4
                elif scale == 2.0:
                    if in_dim > 512:
                        layers = [
                            IConvX(
                                in_dim,
                                in_dim // 2,
                                kernel=1,
                                observe_stat=observe_stat,
                                block_name=sampling_block_name + "scale2/reduce/",
                                quant_components=self.quant_components,
                            ),
                            _make_conv2d(self.quant_components, in_dim // 2, in_dim // 4, kernel_size=2, stride=2, bias=True, weight_bit=8, bias_bit=32, transpose=True),
                        ]
                        out_dim = in_dim // 4
                    else:
                        layers = [
                            _make_conv2d(self.quant_components, in_dim, in_dim // 2, kernel_size=2, stride=2, bias=True, weight_bit=8, bias_bit=32, transpose=True),
                        ]
                        out_dim = in_dim // 2
                elif scale == 1.0:
                    layers = []
                elif scale == 0.5:
                    layers = [
                        IConvX(
                            in_dim,
                            in_dim,
                            3,
                            2,
                            observe_stat=observe_stat,
                            block_name=sampling_block_name + "scale0_5/downsample/",
                            quant_components=self.quant_components,
                        ),
                    ]
                elif scale == 0.25:
                    self.use_extra_pool = True
                    continue
                else:
                    raise NotImplementedError("Unsupported scale_factor:{}".format(scale))
                stage_input_channels.append(out_dim)
                layers = nn.Sequential(*layers)
                stages_sampling[-1].append(layers)
                stages_sampling_qact_bridge[-1].append(nn.ModuleList([IViTAct() for _ in layers]))
                stages_sampling_qact_post[-1].append(nn.ModuleList([IViTAct() for _ in layers]))
            stages_sampling[-1] = nn.ModuleList(stages_sampling[-1])
            stages_sampling_qact_bridge[-1] = nn.ModuleList(stages_sampling_qact_bridge[-1])
            stages_sampling_qact_post[-1] = nn.ModuleList(stages_sampling_qact_post[-1])

            in_dim = sum(stage_input_channels)
            layers = [
                IC2f(
                    in_dim,
                    out_channels,
                    num_blocks,
                    observe_stat=observe_stat,
                    block_name=stage_block_name + "1_c2f/",
                    quant_components=self.quant_components,
                    split_input_channels=stage_input_channels,
                ),
                _make_norm(out_channels, self.quant_components),
            ]
            layers = nn.Sequential(*layers)
            stages.append(layers)
            stages_qact_bridge.append(nn.ModuleList([IViTAct() for _ in layers]))
            stages_qact_post.append(nn.ModuleList([IViTAct() for _ in layers]))

        self.stages_sampling = nn.ModuleList(stages_sampling)
        self.stages = nn.ModuleList(stages)
        self.stages_sampling_qact_bridge = nn.ModuleList(stages_sampling_qact_bridge)
        self.stages_sampling_qact_post = nn.ModuleList(stages_sampling_qact_post)
        self.stages_qact_bridge = nn.ModuleList(stages_qact_bridge)
        self.stages_qact_post = nn.ModuleList(stages_qact_post)

    def _sample_stage_inputs(self, stage_idx, x):
        sampled_inputs = []
        for feat_idx, stage_sampling in enumerate(self.stages_sampling[stage_idx]):
            feat, scale = x[feat_idx] if isinstance(x[feat_idx], tuple) else (x[feat_idx], None)
            sampled_inputs.append(
                _forward_projector_sequence(
                    stage_sampling,
                    self.stages_sampling_qact_bridge[stage_idx][feat_idx],
                    self.stages_sampling_qact_post[stage_idx][feat_idx],
                    feat,
                    scale,
                )
            )
        return sampled_inputs

    def _can_split_first_stage_conv(self, stage, sampled_inputs):
        return (
            len(sampled_inputs) > 1
            and hasattr(stage[0], "forward_split_inputs")
            and isinstance(stage[0].cv1.conv, IViTConv2d)
        )

    def _concat_sampled_inputs(self, sampled_inputs):
        if len(sampled_inputs) == 1:
            return sampled_inputs[0]

        feat_cat = torch.cat([feat for feat, _ in sampled_inputs], dim=1)
        if any(scale is None for _, scale in sampled_inputs):
            return feat_cat, None

        scale_cat = torch.cat(
            [scale.expand(feat.shape[1]) for feat, scale in sampled_inputs],
            dim=0,
        )
        return self.qact_post_fuse(feat_cat, scale_cat)

    def _forward_remaining_stage_layers(self, stage_idx, stage, start_layer, feat, scale):
        for layer_idx in range(start_layer, len(stage)):
            feat, scale = _forward_projector_layer(
                stage[layer_idx],
                feat,
                scale,
                self.stages_qact_bridge[stage_idx][layer_idx],
                self.stages_qact_post[stage_idx][layer_idx],
            )
            save_tensor_from_pathname(
                feat,
                self.block_name + f"stage{stage_idx}/2_stage/layer{layer_idx}",
                observe_stat=self.observe_stat,
            )
        return feat, scale

    def forward(self, x):
        results = []

        for stage_idx, stage in enumerate(self.stages):
            sampled_inputs = self._sample_stage_inputs(stage_idx, x)

            if self._can_split_first_stage_conv(stage, sampled_inputs):
                feat, scale = stage[0].forward_split_inputs(sampled_inputs)
                save_tensor_from_pathname(
                    feat,
                    self.block_name + f"stage{stage_idx}/2_stage/layer0",
                    observe_stat=self.observe_stat,
                )
                feat, scale = self._forward_remaining_stage_layers(stage_idx, stage, 1, feat, scale)
            else:
                feat, scale = self._concat_sampled_inputs(sampled_inputs)
                feat, scale = self._forward_remaining_stage_layers(stage_idx, stage, 0, feat, scale)

            save_tensor_from_pathname(
                feat,
                self.block_name + f"stage{stage_idx}/3_output",
                observe_stat=self.observe_stat,
            )
            results.append((feat, scale))

        if self.use_extra_pool:
            feat, scale = results[-1]
            results.append((F.max_pool2d(feat, kernel_size=1, stride=2, padding=0), scale))
        return results
