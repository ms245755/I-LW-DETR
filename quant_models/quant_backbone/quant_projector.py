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
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from quant_models.fake_quantized.layers import *
    
def get_norm(norm, out_channels):
    """
    Args:
        norm (str or callable): either one of BN, SyncBN, FrozenBN, GN;
            or a callable that takes a channel number and returns
            the normalization layer as a nn.Module.
    Returns:
        nn.Module or None: the normalization layer
    """
    if norm is None:
        return None
    if isinstance(norm, str):
        if len(norm) == 0:
            return None
        norm = {
            "LN": lambda channels: IViTLayerNormChannelFirst(channels),
        }[norm]
    return norm(out_channels)


def get_activation(name, inplace=False):
    """ get activation """
    return ISegGELU()

class IConvX(nn.Module):
    """ Conv-bn module"""
    def __init__(self, in_planes, out_planes, kernel=3, stride=1, groups=1, dilation=1, act='relu'):
        super(IConvX, self).__init__()
        self.conv = IViTConv2d(in_planes, out_planes, kernel_size=kernel,
                              stride=stride, padding=kernel // 2, groups=groups,
                              dilation=dilation, bias=True)
        # self.bn = nn.BatchNorm2d(out_planes)

        self.act = get_activation(act, inplace=True)
        self.qact_post_conv = IViTAct()
        self.qact_post_act = IViTAct()

    def forward(self, x, act_scaling_factor=None):
        """ forward """
        x, act_scaling_factor = self.conv(x, act_scaling_factor)
        x, act_scaling_factor = self.qact_post_conv(x, act_scaling_factor)
        x, act_scaling_factor = self.act(x, act_scaling_factor)
        x, act_scaling_factor = self.qact_post_act(x, act_scaling_factor)
        return x, act_scaling_factor


class IBottleneck(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5, act='silu'):
        """ ch_in, ch_out, shortcut, groups, kernels, expand """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = IConvX(c1, c_, k[0], 1, act=act)
        self.cv2 = IConvX(c_, c2, k[1], 1, groups=g, act=act)
        self.add = shortcut and c1 == c2

    def forward(self, x, act_scaling_factor=None):
        """'forward()' applies the YOLOv5 FPN to input data."""
        x_short = x
        act_scaling_factor_short = act_scaling_factor
        x, act_scaling_factor= self.cv1(x, act_scaling_factor)
        x, act_scaling_factor= self.cv2(x, act_scaling_factor)

        if self.add:
            out_scale = torch.max(act_scaling_factor, act_scaling_factor_short)
            x_short   = round_ste.apply(x_short / out_scale)
            x        = round_ste.apply(x  / out_scale)
            x = (x + x_short) * out_scale
            act_scaling_factor = out_scale

        return x, act_scaling_factor


class IC2f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, act='silu'):
        """ ch_in, ch_out, number, shortcut, groups, expansion """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = IConvX(c1, 2 * self.c, 1, 1, act=act)
        self.cv2 = IConvX((2 + n) * self.c, c2, 1, act=act)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(IBottleneck(self.c, self.c, shortcut, g, k=(3, 3), e=1.0, act=act) for _ in range(n))

    def forward(self, x, act_scaling_factor):
        """Forward pass using split() instead of chunk()."""
        x_out, act_scaling_factor = self.cv1(x, act_scaling_factor)

        y = list(x_out.split((self.c, self.c), 1))

        act_scaling_factors_list = [act_scaling_factor]
        for m in self.m:
            x_out, scale = m(y[-1], act_scaling_factors_list[-1])
            y.append(x_out)
            act_scaling_factors_list.append(scale)
        
        out_scale = torch.stack(act_scaling_factors_list).max()
        for i in range(len(act_scaling_factors_list)):
            y[i] = round_ste.apply(y[i] / act_scaling_factors_list[i])

        concatenated_output = torch.cat(y, 1)
        concatenated_output = concatenated_output * out_scale
        final_output, act_scaling_factor_final = self.cv2(concatenated_output, out_scale)

        return final_output, act_scaling_factor_final


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
        num_blocks=3
    ):
        """
        Args:
            net (Backbone): module representing the subnetwork backbone.
                Must be a subclass of :class:`Backbone`.
            out_channels (int): number of channels in the output feature maps.
            scale_factors (list[float]): list of scaling factors to upsample or downsample
                the input features for creating pyramid features.
        """
        super(IMultiScaleProjector, self).__init__()

        self.scale_factors = scale_factors

        stages_sampling = []
        stages = []
        # use_bias = norm == ""
        use_bias = False
        self.use_extra_pool = False
        self.qact_post_fuse = IViTAct()
        for scale in scale_factors:
            stages_sampling.append([])
            for in_dim in in_channels:
                out_dim = in_dim
                if scale == 4.0:
                    layers = [
                        IViTConvTranspose2d(in_dim, in_dim // 2, kernel_size=2, stride=2),
                        IViTAct(),
                        get_norm('LN', in_dim // 2),
                        IViTAct(),
                        IViTGELU,
                        IViTAct(),
                        IViTConvTranspose2d(in_dim // 2, in_dim // 4, kernel_size=2, stride=2),
                        IViTAct(),
                    ]
                    out_dim = in_dim // 4
                elif scale == 2.0:
                    # a hack to reduce the FLOPs and Params when the dimention of output feature is too large
                    if in_dim > 512:
                        layers = [
                            IConvX(in_dim, in_dim // 2, kernel=1),
                            IViTConvTranspose2d(in_dim // 2, in_dim // 4, kernel_size=2, stride=2),
                            IViTAct(),
                        ]
                        out_dim = in_dim // 4
                    else:
                        layers = [IViTConvTranspose2d(in_dim, in_dim // 2, kernel_size=2, stride=2), IViTAct(),]
                        out_dim = in_dim // 2
                elif scale == 1.0:
                    layers = []
                elif scale == 0.5:
                    layers = [
                        IConvX(in_dim, in_dim, 3, 2),
                        IViTAct(),
                        ]
                elif scale == 0.25:
                    self.use_extra_pool = True
                    continue
                else:
                    raise NotImplementedError("Unsupported scale_factor:{}".format(scale))
                layers = nn.Sequential(*layers)
                stages_sampling[-1].append(layers)
            stages_sampling[-1] = nn.ModuleList(stages_sampling[-1])

            in_dim = out_dim * len(in_channels)
            layers = [
                IC2f(in_dim, out_channels, num_blocks),
                get_norm('LN', out_channels),
                IViTAct()
            ]
            layers = nn.Sequential(*layers)
            stages.append(layers)

        self.stages_sampling = nn.ModuleList(stages_sampling)
        self.stages = nn.ModuleList(stages)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (N,C,H,W). H, W must be a multiple of ``self.size_divisibility``.
        Returns:
            dict[str->Tensor]:
                mapping from feature map name to pyramid feature map tensor
                in high to low resolution order. Returned feature names follow the FPN
                convention: "p<stage>", where stage has stride = 2 ** stage e.g.,
                ["p2", "p3", ..., "p6"].
        """
        # (H, W, B, C) --> (B, C, H, W)
        results = []

        for i, stage in enumerate(self.stages):
            feat_fuse = []
            for j, stage_sampling in enumerate(self.stages_sampling[i]):
                feat_fuse.append(stage_sampling(x[j]))
            if len(feat_fuse) > 1:
                # need to bring all feature layer to one scale
                # max_scale = torch.stack([fs for (_, fs) in feat_fuse]).max()
                # feat_list = []
                # for feat, feat_scale in feat_fuse:
                #     feat_list.append(round_ste.apply(feat / feat_scale))

                # feat_fuse = torch.cat(feat_list, dim=1)
                # feat_fuse = feat_fuse * max_scale
                # feat_fuse, max_scale = self.qact_post_fuse(feat_fuse, max_scale)
                feat_cat = torch.cat([feat for (feat, _) in feat_fuse], dim=1)
                # 2) concat scales in channel order
                scale_cat = torch.cat([fs.expand(feat.shape[1]) for (feat, fs) in feat_fuse], dim=0)  # should be length == C_total
                # print(feat_cat.shape)
                # print(scale_cat.shape)

                feat_fuse, max_scale = self.qact_post_fuse(feat_cat, scale_cat)
            else:
                max_scale = feat_fuse[0][1]
                feat_fuse = feat_fuse[0][0]
            feat = feat_fuse
            scale = max_scale
            # print("fuse scale:", max_scale, "feat_fuse min/max:", feat_fuse.min().item(), feat_fuse.max().item())
            for layer in stage:
                feat, scale = layer(feat, scale)
            results.append((feat, scale))
        if self.use_extra_pool:
            results.append(
                F.max_pool2d(results[-1], kernel_size=1, stride=2, padding=0)
            )
        return results
