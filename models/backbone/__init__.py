from typing import Dict, List

import torch
from torch import nn

from util.misc import NestedTensor
from ..position_encoding import build_position_encoding
from .backbone import *
from .backbone_dinov2 import BackboneDinoV2


class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)
        self._export = False

    def forward(self, tensor_list: NestedTensor):
        """
        """
        x = self[0](tensor_list)
        pos = []
        for x_ in x:
            pos.append(self[1](x_, align_dim_orders=False).to(x_.tensors.dtype))
        return x, pos

    def export(self):
        self._export = True
        self._forward_origin = self.forward
        self.forward = self.forward_export
    
    def forward_export(self, inputs:torch.Tensor):
        feats, masks = self[0](inputs)
        poss = []
        for feat, mask in zip(feats, masks):
            poss.append(self[1](mask, align_dim_orders=False).to(feat.dtype))
        return feats, None, poss


def build_backbone(args):
    """
    Useful args:
        - encoder: encoder name
        - lr_encoder:
        - dilation
        - use_checkpoint: for swin only for now

    """
    position_embedding = build_position_encoding(args)

    if args.encoder in ['vit_tiny', 'vit_small', 'vit_base', 'res18vd', 'res50vd', 'mobilenetv3_large_1.0']:
        
        backbone = Backbone(
            args.encoder,
            args.vit_encoder_num_layers,
            args.pretrained_encoder,
            window_block_indexes=args.window_block_indexes,
            drop_path=args.drop_path,
            out_channels=args.hidden_dim,
            out_feature_indexes=args.out_feature_indexes,
            projector_scale=args.projector_scale,
        )
    # Congpro
    elif 'dinov2' in args.encoder:
        args_dict = vars(args)
        backbone = BackboneDinoV2(
            args.encoder,
            args.pretrained_encoder,
            window_block_indexes=args_dict.get("window_block_indexes", None),
            out_channels=args_dict.get("hidden_dim", 256),
            out_feature_indexes=args_dict.get("out_feature_indexes", [-1]),
            projector_scale=args_dict.get("projector_scale", "P4"),
            use_cls_token=args_dict.get("use_cls_token", False),
            layer_norm=args_dict.get("layer_norm", False),
            freeze_encoder=args_dict.get("freeze_encoder", True), ### Cong: freeze the Backbone part
        )

    model = Joiner(backbone, position_embedding)
    return model
