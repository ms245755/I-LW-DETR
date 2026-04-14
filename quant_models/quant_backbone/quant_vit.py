# ------------------------------------------------------------------------
# LW-DETR
# Copyright (c) 2024 Baidu. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from ViTDet (https://github.com/facebookresearch/detectron2/tree/main/projects/ViTDet)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------

"""
ViT encoder
"""
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from fairscale.nn.checkpoint import checkpoint_wrapper
from timm.models.layers import DropPath, Mlp, trunc_normal_
from timm.models.vision_transformer import _load_weights

from util.box_ops import box_cxcywh_to_xyxy
from quant_models.fake_quantized.layers import *
from util.utils import init_weights
from quant_models.quant_blocks import *


def get_abs_pos(abs_pos, has_cls_token, hw):
    """
    Calculate absolute positional embeddings. If needed, resize embeddings and remove cls_token
        dimension for the original embeddings.
    Args:
        abs_pos (Tensor): absolute positional embeddings with (1, num_position, C).
        has_cls_token (bool): If true, has 1 embedding in abs_pos for cls token.
        hw (Tuple): size of input image tokens.
    Returns:
        Absolute positional embeddings after processing with shape (1, H, W, C)
    """
    h, w = hw
    if has_cls_token:
        abs_pos = abs_pos[:, 1:]
    xy_num = abs_pos.shape[1]
    size = int(math.sqrt(xy_num))
    assert size * size == xy_num

    if size != h or size != w:
        new_abs_pos = F.interpolate(
            abs_pos.reshape(1, size, size, -1).permute(0, 3, 1, 2),
            size=(h, w),
            mode="bicubic",
            align_corners=False,
        )

        return new_abs_pos.permute(0, 2, 3, 1)
    else:
        return abs_pos.reshape(1, h, w, -1)





class IViT(nn.Module):
    """
    This module implements Vision Transformer (ViT) backbone in :paper:`vitdet`.
    "Exploring Plain Vision Transformer Backbones for Object Detection",
    https://arxiv.org/abs/2203.16527
    """

    def __init__(
        self,
        img_size=1024,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        act_layer=nn.GELU,
        use_abs_pos=True,
        window_block_indexes=(),
        use_act_checkpoint=False,
        pretrain_img_size=224,
        pretrain_use_cls_token=True,
        out_feature_indexes:list=None,
        use_cae=False,
        gelu_n=23,
        gelu_lambda=6,
        dropout=0.1,
        num_bits: int =8
    ):
        """
        Args:
            img_size (int): Input image size.
            patch_size (int): Patch size.
            in_chans (int): Number of input image channels.
            embed_dim (int): Patch embedding dimension.
            depth (int): Depth of ViT.
            num_heads (int): Number of attention heads in each ViT block.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            qkv_bias (bool): If True, add a learnable bias to query, key, value.
            drop_path_rate (float): Stochastic depth rate.
            norm_layer (nn.Module): Normalization layer.
            act_layer (nn.Module): Activation layer.
            use_abs_pos (bool): If True, use absolute positional embeddings.
            use_rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            window_block_indexes (list): Indexes for blocks using window attention.
            residual_block_indexes (list): Indexes for blocks using conv propagation.
            use_act_checkpoint (bool): If True, use activation checkpointing.
            pretrain_img_size (int): input image size for pretraining models.
            pretrain_use_cls_token (bool): If True, pretrainig models use class token.
        """
        super().__init__()
        self.pretrain_use_cls_token = pretrain_use_cls_token

        self.patch_embed = IPatchEmbed(
            kernel_size=(patch_size, patch_size),
            stride=(patch_size, patch_size),
            in_chans=in_chans,
            embed_dim=embed_dim,
            num_bits=num_bits
        )

        if use_abs_pos:
            # Initialize absolute positional embedding with pretrain image size.
            num_patches = (pretrain_img_size // patch_size) * (pretrain_img_size // patch_size)
            num_positions = (num_patches + 1) if pretrain_use_cls_token else num_patches
            self.pos_embed = nn.Parameter(torch.zeros(1, num_positions, embed_dim))
        else:
            self.pos_embed = None

        # stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        self.blocks = nn.ModuleList()
        for i in range(depth):
            block = IBlock(
                dim=embed_dim,
                heads=num_heads,
                mlp_dim=int(mlp_ratio*embed_dim),
                drop_path=dpr[i],
                window=True if i in window_block_indexes else False,
                gelu_n=gelu_n,
                gelu_lambda=gelu_lambda,
                dropout= dropout,
                qkv_bias=qkv_bias,
                num_bits=num_bits,
                use_cae=use_cae
            )
            if use_act_checkpoint:
                block = checkpoint_wrapper(block)
            self.blocks.append(block)

        self.window_block_indexes = window_block_indexes
        out_feature_indexes = [ind if ind >= 0 else ind + depth for ind in out_feature_indexes]
        out_feature_indexes = [ind for ind in range(depth) if ind in out_feature_indexes]

        self._out_features = [True if i in out_feature_indexes else False for i in range(depth)]
        self._out_feature_channels = [embed_dim] * len(out_feature_indexes)
        assert self._out_features[-1] is True

        if self.pos_embed is not None:
            trunc_normal_(self.pos_embed, std=0.02)

        self.apply(self._init_weights)

        self._export = False

        self.qact_input = IViTAct(activation_bit=num_bits)
        self.qact_pos = IViTAct(activation_bit=num_bits)

    def export(self):
        self._export = True
        self.pos_embed_export = get_abs_pos(
            self.pos_embed, self.pretrain_use_cls_token, (40, 40)
        ).detach()

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        x, act_scaling_factor = self.qact_input(x=x, act_scaling_factor=None)

        x, act_scaling_factor = self.patch_embed(x, act_scaling_factor)

        if self.pos_embed is not None:
            x_pos = get_abs_pos(
                    self.pos_embed, self.pretrain_use_cls_token, (x.shape[1], x.shape[2])
                )
            x_pos, pos_scale = self.qact_pos(x_pos, act_scaling_factor=None)
            # out_scale = torch.max(act_scaling_factor, pos_scale)

            x = round_ste.apply(x / act_scaling_factor )
            x_pos = round_ste.apply(x_pos / act_scaling_factor )
            # act_scaling_factor = out_scale
            # x = x + x_pos
            x = x + x_pos
            x *= act_scaling_factor
        B, H, W, C = x.shape
        assert (H % 4 == 0) and (W % 4 == 0)
        h, w = H // 4, W // 4

        x = x.reshape(B, 4, h, 4, w, C).permute(
            0, 1, 3, 2, 4, 5).reshape(B * 16, h * w, C)
        out = []
        for idx, blk in enumerate(self.blocks):
            x, act_scaling_factor = blk(x, mask=None, act_scaling_factor=act_scaling_factor)
            if self._out_features[idx]:
                out.append((x.reshape(B, 4, 4, h, w, C).permute(
                    0, 5, 1, 3, 2, 4).reshape(B, C, H, W), act_scaling_factor))
        return out
