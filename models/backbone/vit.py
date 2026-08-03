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
from timm.models.layers import DropPath, trunc_normal_

from util.box_ops import box_cxcywh_to_xyxy
from util.decorators import save_tensor_from_pathname


class Mlp(nn.Module):
    """MLP as used by timm ViT blocks, with local observation hooks."""

    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
        observe_stat=False,
        block_name="",
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.observe_stat = observe_stat
        self.block_name = block_name

    def forward(self, x):
        x = self.fc1(x)
        save_tensor_from_pathname(x=x, component_path=self.block_name+"0_fc1", observe_stat=self.observe_stat)
        x = self.act(x)
        save_tensor_from_pathname(x=x, component_path=self.block_name+"1_gelu", observe_stat=self.observe_stat)
        x = self.drop(x)
        x = self.fc2(x)
        save_tensor_from_pathname(x=x, component_path=self.block_name+"2_fc2", observe_stat=self.observe_stat)
        x = self.drop(x)
        return x


class MyLayerNorm(nn.Module):
    """
    A LayerNorm variant, popularized by Transformers, that performs point-wise mean and
    variance normalization over the channel dimension for inputs that have shape
    (batch_size, height, width, channel).
    https://github.com/facebookresearch/ConvNeXt/blob/d1fa8f6fef0a165b27399986cc2bdacc92777e40/models/convnext.py#L119
    """

    def __init__(self, normalized_shape, eps=1e-6, observe_stat=False, block_name=""):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.normalized_shape = (normalized_shape,)
        self.block_name = block_name
        self.observe_stat = observe_stat

    def forward(self, x):
        """
        LayerNorm forward
        TODO: this is a hack to avoid overflow when using fp16
        """
        #if x.dtype == torch.half:
        #    x = x / (x.max() + self.eps)
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        std = torch.sqrt(s+self.eps)
        x = (x - u) / std
        x = self.weight[None, None, :] * x + self.bias[None, None, :]
        save_tensor_from_pathname(x, self.block_name+"output", observe_stat=self.observe_stat)
        return x
    
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


class PatchEmbed(nn.Module):
    """
    Image to Patch Embedding.
    """

    def __init__(
        self, kernel_size=(16, 16), stride=(16, 16), padding=(0, 0), in_chans=3, embed_dim=768, observe_stat=False, block_name = ""
    ):
        """
        Args:
            kernel_size (Tuple): kernel size of the projection layer.
            stride (Tuple): stride of the projection layer.
            padding (Tuple): padding size of the projection layer.
            in_chans (int): Number of input image channels.
            embed_dim (int):  embed_dim (int): Patch embedding dimension.
            observe_stat (bool): If True, store activation statistic for analysis.
            block_name (str): store name of block for visualization
        """
        super().__init__()

        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=kernel_size, stride=stride, padding=padding
        )
        self.observe_stat = observe_stat
        self.block_name = block_name

    def forward(self, x):
        x = self.proj(x)
        # B C H W -> B H W C
        x = x.permute(0, 2, 3, 1)
        save_tensor_from_pathname(x=x, component_path=self.block_name+"proj", observe_stat=self.observe_stat)
        return x


class Attention(nn.Module):
    """Multi-head Attention block with relative position embeddings."""

    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=True,
        use_cae=False,
        observe_stat=False,
        block_name = ""
    ):
        """
        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads.
            qkv_bias (bool:  If True, add a learnable bias to query, key, value.
            rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            input_size (int or None): Input resolution for calculating the relative positional
                parameter size.
            observe_stat (bool): If True, store activation statistic for analysis.
        """
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5

        self.use_cae = use_cae
        if use_cae:
            self.qkv = nn.Linear(dim, dim * 3, bias=False)
            self.q_bias = nn.Parameter(torch.zeros(dim))
            self.v_bias = nn.Parameter(torch.zeros(dim))
        else:
            self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.observe_stat = observe_stat
        self.block_name = block_name

    def forward(self, x, mask=None):
        B, N, C = x.shape
        # qkv with shape (B, H, W, 3C)
        if self.use_cae:
            qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))
            qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        else:
            qkv = self.qkv(x)

        save_tensor_from_pathname(qkv, self.block_name+"0_qkv", observe_stat=self.observe_stat)
        # pytorch naive implementation
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = (q * self.scale) @ k.transpose(-2, -1)

        save_tensor_from_pathname(attn, self.block_name+"1_attn", observe_stat=self.observe_stat)
        if mask is not None:
            attn.masked_fill_(mask.reshape(B, 1, 1, N).expand_as(attn), float('-inf'))

        attn = attn.softmax(dim=-1)
        save_tensor_from_pathname(attn, self.block_name+"2_softmax", observe_stat=self.observe_stat)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        save_tensor_from_pathname(x, self.block_name+"3_mulvalue", observe_stat=self.observe_stat)
        x = self.proj(x)
        save_tensor_from_pathname(x, self.block_name+"4_output", observe_stat=self.observe_stat)
        return x


class Block(nn.Module):
    """Transformer blocks with support of window attention and residual propagation blocks"""

    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_path=0.0,
        norm_layer=MyLayerNorm,
        act_layer=nn.GELU,
        window=False,
        use_cae=False,
        observe_stat=False,
        block_name = ""
    ):
        """
        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads in each ViT block.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            qkv_bias (bool): If True, add a learnable bias to query, key, value.
            drop_path (float): Stochastic depth rate.
            norm_layer (nn.Module): Normalization layer.
            act_layer (nn.Module): Activation layer.
            use_rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            use_residual_block (bool): If True, use a residual block after the MLP block.
            input_size (int or None): Input resolution for calculating the relative positional
                parameter size.
            observe_stat (bool): If True, store activation statistic for analysis.
            block_name (str): store name of block for visualization
        """
        super().__init__()
        self.norm1 = norm_layer(dim, observe_stat=observe_stat, block_name=block_name+"0_norm1/")
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            use_cae=use_cae,
            observe_stat = observe_stat,
            block_name = block_name + "1_attn/"
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim, observe_stat=observe_stat, block_name=block_name+"4_norm2/")
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            observe_stat=observe_stat,
            block_name=block_name+"5_mlp/",
        )

        self.window = window

        # for cae
        self.use_cae = use_cae
        if use_cae:
            init_values = 0.1
            self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
            self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
        self.observe_stat = observe_stat
        self.block_name = block_name

    def forward(self, x, mask=None):
        """ Transformer Block forward"""
        B, HW, C = x.shape
        shortcut = x
        x = self.norm1(x)
        
        save_tensor_from_pathname(x=x, component_path=self.block_name+"0_norm1_output", observe_stat=self.observe_stat)

        if not self.window:
            x = x.reshape(B // 16, 16 * HW, C)
            if mask is not None:
                mask = mask.reshape(B // 16, 16 * HW)

        if self.use_cae:
            x = self.attn(x, mask)
            save_tensor_from_pathname(x=x, component_path=self.block_name+"1_attn", observe_stat=self.observe_stat)
            x = self.gamma_1 * x
            # x = self.gamma_1 * self.attn(x, mask)
            save_tensor_from_pathname(x=x, component_path=self.block_name+"2_gamma1", observe_stat=self.observe_stat)
        else:
            x = self.attn(x, mask)
            save_tensor_from_pathname(x=x, component_path=self.block_name+"1_attn", observe_stat=self.observe_stat)

        if not self.window:
            x = x.reshape(B, HW, C)
            if mask is not None:
                mask = mask.reshape(B, HW)

        x = shortcut + self.drop_path(x)
        save_tensor_from_pathname(x=x, component_path=self.block_name+"3_residual1", observe_stat=self.observe_stat)
        if self.use_cae:
            shortcut = x
            x = self.norm2(x)
            save_tensor_from_pathname(x=x, component_path=self.block_name+"4_norm2_output", observe_stat=self.observe_stat)
            x = self.mlp(x)
            save_tensor_from_pathname(x=x, component_path=self.block_name+"5_mlp", observe_stat=self.observe_stat)
            x = x * self.gamma_2
            save_tensor_from_pathname(x=x, component_path=self.block_name+"6_gamma2", observe_stat=self.observe_stat)
            x = self.drop_path(x)
            
            x = x + shortcut
            save_tensor_from_pathname(x=x, component_path=self.block_name+"7_residual2", observe_stat=self.observe_stat)
            # x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        else:
            shortcut = x
            x = self.norm2(x)
            save_tensor_from_pathname(x=x, component_path=self.block_name+"4_norm2_output", observe_stat=self.observe_stat)
            x = self.mlp(x)
            save_tensor_from_pathname(x=x, component_path=self.block_name+"5_mlp", observe_stat=self.observe_stat)
            x = self.drop_path(x)
            x = x + shortcut
            save_tensor_from_pathname(x=x, component_path=self.block_name+"7_residual2", observe_stat=self.observe_stat)
            # x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class ViT(nn.Module):
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
        norm_layer=MyLayerNorm,
        act_layer=nn.GELU,
        use_abs_pos=True,
        window_block_indexes=(),
        use_act_checkpoint=False,
        pretrain_img_size=224,
        pretrain_use_cls_token=True,
        out_feature_indexes:list=None,
        use_cae=False,
        observe_stat = False,
        block_name = ""
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
            observe_stat (bool): If True, store activation statistic for analysis.
        """
        super().__init__()
        self.pretrain_use_cls_token = pretrain_use_cls_token

        self.patch_embed = PatchEmbed(
            kernel_size=(patch_size, patch_size),
            stride=(patch_size, patch_size),
            in_chans=in_chans,
            embed_dim=embed_dim,
            observe_stat = observe_stat,
            block_name = block_name + "1_patchembed/"
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
            block = Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                act_layer=act_layer,
                window=True if i in window_block_indexes else False,
                use_cae=use_cae,
                observe_stat = observe_stat,
                block_name = block_name + f"block{i}/"
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
        self.observe_stat = observe_stat
        self.block_name = block_name

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
        elif isinstance(m, MyLayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        save_tensor_from_pathname(x=x, component_path=self.block_name+"0_input", observe_stat=self.observe_stat)
        x = self.patch_embed(x)
        if self.pos_embed is not None:
            if self._export:
                x = x + self.pos_embed_export
            else:
                x = x + get_abs_pos(
                    self.pos_embed, self.pretrain_use_cls_token, (x.shape[1], x.shape[2])
                )

        B, H, W, C = x.shape
        assert (H % 4 == 0) and (W % 4 == 0)
        h, w = H // 4, W // 4

        x = x.reshape(B, 4, h, 4, w, C).permute(
            0, 1, 3, 2, 4, 5).reshape(B * 16, h * w, C)
        out = []
        for idx, blk in enumerate(self.blocks):
            x = blk(x, mask=None)
            if self._out_features[idx]:
                out.append(x.reshape(B, 4, 4, h, w, C).permute(
                    0, 5, 1, 3, 2, 4).reshape(B, C, H, W))
        return out
