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

class ILinearBlock(nn.Module):
    def __init__(self, in_dim, out_dim, has_gelu=True, gelu=None):
        super().__init__()
        if gelu is not None:
            has_gelu = gelu
        self.linear = IViTLinear(in_dim, out_dim)
        self.qact_input_bridge = IViTAct()
        self.qact_post_linear = IViTAct()
        self.has_gelu = has_gelu
        if has_gelu:
            self.act = ISegGELU()
            self.qact_post_gelu = IViTAct()
    
    def _qact_channel_last(self, qact, x, act_scaling_factor):
        if x.dim() != 4:
            return qact(x, act_scaling_factor)

        original_shape = x.shape
        x = x.reshape(-1, original_shape[-2], original_shape[-1])
        x, act_scaling_factor = qact(x, act_scaling_factor)
        return x.reshape(original_shape), act_scaling_factor

    def forward(self, x, act_scaling_factor=None):
        if act_scaling_factor is None or act_scaling_factor.numel() != 1:
            x, act_scaling_factor = self._qact_channel_last(self.qact_input_bridge, x, None)
        x, act_scaling_factor = self.linear(x, act_scaling_factor)
        x, act_scaling_factor = self._qact_channel_last(self.qact_post_linear, x, act_scaling_factor)
        if self.has_gelu:
            x, act_scaling_factor = self.act(x, act_scaling_factor)
            x, act_scaling_factor = self._qact_channel_last(self.qact_post_gelu, x, act_scaling_factor)
        return x, act_scaling_factor

class IMLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(ILinearBlock(n, k, h) for n, k, h in zip([input_dim] + h, h + [output_dim], [True] * (num_layers-1) + [False]))

    def forward(self, x, act_scaling_factor = None):
        for i, layer in enumerate(self.layers):
            x, act_scaling_factor = layer(x, act_scaling_factor)
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


# Cong: this also hard to quantize
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
        # Cong: quantize the log function also
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
                 bbox_reparam=False):
        super().__init__()
        self.encoder = None

        decoder_layer = TransformerDecoderLayer(d_model, sa_nhead, ca_nhead, dim_feedforward,
                                                dropout, activation, normalize_before, 
                                                group_detr=group_detr,
                                                num_feature_levels=num_feature_levels,
                                                dec_n_points=dec_n_points,
                                                skip_self_attn=False,)
        assert decoder_norm_type in ['LN', 'Identity']
        norm = { 
            "LN": lambda channels: IViTLayerNorm(channels),
            "Identity": lambda channels: nn.Identity(),
        }
        decoder_norm = norm[decoder_norm_type](d_model)

        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                          return_intermediate=return_intermediate_dec,
                                          d_model=d_model,
                                          lite_refpoint_refine=lite_refpoint_refine,
                                          easy_quantize=easy_quantize,
                                          bbox_reparam=bbox_reparam)
        
        self.int_exp = IViTExp()
        self.qact_output_proposal = IViTAct()
        self.two_stage = two_stage
        if two_stage:
            self.enc_output = nn.ModuleList([IViTLinear(d_model, d_model) for _ in range(group_detr)])
            self.enc_output_norm = nn.ModuleList([IViTLayerNorm(d_model) for _ in range(group_detr)])
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
    
    def export(self):
        self._export = True

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, (MSDeformAttn, IMSDeformAttn)):
                m._reset_parameters()
    # Cong: keep this as float for now
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
        memory_scale = max_scale
        if masks is not None:
            mask_flatten = torch.cat(mask_flatten, 1)   # bs, \sum{hxw}
            valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=memory.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))
        
        if self.two_stage:
            output_memory, output_proposals = gen_encoder_output_proposals(
                memory, mask_flatten, spatial_shapes, unsigmoid=not self.bbox_reparam)
            
            output_proposals, output_proposals_scale = self.qact_output_proposal(output_proposals, None)
            # group detr for first stage
            refpoint_embed_ts, memory_ts, boxes_ts = [], [], []
            group_detr = self.group_detr if self.training else 1
            for g_idx in range(group_detr):
                # output_memory_gidx = self.enc_output_norm[g_idx](self.enc_output[g_idx](output_memory, memory_scale))
                output_memory_gidx, output_memory_gidx_scale = self.enc_output[g_idx](output_memory, memory_scale)
                output_memory_gidx, output_memory_gidx_scale = self.qact_post_linear_output[g_idx](
                    output_memory_gidx, output_memory_gidx_scale)
                output_memory_gidx, output_memory_gidx_scale = self.enc_output_norm[g_idx](
                    output_memory_gidx, output_memory_gidx_scale)
                output_memory_gidx, output_memory_gidx_scale = self.qact_post_norm_output[g_idx](
                    output_memory_gidx, output_memory_gidx_scale)
    
                enc_outputs_class_unselected_gidx, enc_outputs_class_unselected_gidx_scale  = self.enc_out_class_embed[g_idx](output_memory_gidx, output_memory_gidx_scale)
                if self.bbox_reparam:
                    enc_outputs_coord_delta_gidx, enc_outputs_coord_delta_gidx_scale = self.enc_out_bbox_embed[g_idx](output_memory_gidx, output_memory_gidx_scale)
                    enc_outputs_coord_cxcy_gidx = enc_outputs_coord_delta_gidx[...,
                        :2] * output_proposals[..., 2:] + output_proposals[..., :2]
                    enc_outputs_coord_cxcy_gidx_scale = enc_outputs_coord_delta_gidx_scale * output_proposals_scale

                    # Cong: careful about this exp function, we can also approximate by bit shift (convert to 2^(log e * value))
                    tmp, tmp_scale = self.int_exp(enc_outputs_coord_delta_gidx[..., 2:], enc_outputs_coord_delta_gidx_scale)
                    enc_outputs_coord_wh_gidx, enc_outputs_coord_wh_gidx_scale = multiply_tensor_scale(tmp, tmp_scale, output_proposals[..., 2:], output_proposals_scale)

                    concat_list = [(enc_outputs_coord_cxcy_gidx, enc_outputs_coord_cxcy_gidx_scale), (enc_outputs_coord_wh_gidx, enc_outputs_coord_wh_gidx_scale)]
                    enc_outputs_coord_unselected_gidx, enc_outputs_coord_unselected_gidx_scale = concat_list_tensor_scale(concat_list, dim=-1)

                else:
                    tmp, tmp_scale = self.enc_out_bbox_embed[g_idx](output_memory_gidx, output_memory_gidx_scale)
                    enc_outputs_coord_unselected_gidx, enc_outputs_coord_unselected_gidx_scale = add_tensor_scale(tmp, tmp_scale, output_proposals, output_proposals_scale)

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
            refpoint_embed_ts, refpoint_embed_ts_scale = concat_list_tensor_scale(refpoint_embed_ts, dim=1)
            # (bs, nq, d)
            memory_ts, memory_ts_scale = concat_list_tensor_scale(memory_ts, dim=1)#.transpose(0, 1)
            boxes_ts, boxes_ts_scale = concat_list_tensor_scale(boxes_ts, dim=1)#.transpose(0, 1)
        
        tgt = query_feat.unsqueeze(0).repeat(bs, 1, 1)
        refpoint_embed = refpoint_embed.unsqueeze(0).repeat(bs, 1, 1)
        if self.two_stage:
            if self.bbox_reparam:
                # Cong: dont use this function
                tmp, tmp_scale = multiply_tensor_scale(refpoint_embed[..., :2], refpoint_scale, refpoint_embed_ts[..., 2:], refpoint_embed_ts_scale)
                refpoint_embed_cxcy, refpoint_embed_cxcy_scale = add_tensor_scale(tmp, tmp_scale, refpoint_embed_ts[..., :2], refpoint_embed_ts_scale)

                tmp, tmp_scale = self.int_exp(refpoint_embed[..., 2:], refpoint_scale)
                refpoint_embed_wh, refpoint_embed_wh_scale = multiply_tensor_scale(tmp, tmp_scale, refpoint_embed_ts[..., 2:], refpoint_embed_ts_scale)
                refpoint_embed, refpoint_scale = concat_list_tensor_scale([(refpoint_embed_cxcy, refpoint_embed_cxcy_scale), (refpoint_embed_wh, refpoint_embed_wh_scale)], dim=-1)
            else:
                refpoint_embed, refpoint_scale = add_tensor_scale(refpoint_embed, refpoint_scale, refpoint_embed_ts, refpoint_embed_ts_scale)

        hs, hs_scale, references, references_scale = self.decoder(tgt, query_scale, memory, memory_scale, memory_key_padding_mask=mask_flatten,
                          pos=None, refpoints_unsigmoid=refpoint_embed, refpoint_scale=refpoint_scale,
                          level_start_index=level_start_index, 
                          spatial_shapes=spatial_shapes,
                          valid_ratios=valid_ratios.to(memory.dtype) if valid_ratios is not None else valid_ratios)
        if self.two_stage:
            if self.bbox_reparam:
                return hs, hs_scale, references, references_scale, memory_ts, memory_ts_scale, boxes_ts
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
                 bbox_reparam=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.d_model = d_model
        self.norm = norm
        self.return_intermediate = return_intermediate
        self.lite_refpoint_refine = lite_refpoint_refine
        self.easy_quantize = easy_quantize
        self.bbox_reparam = bbox_reparam

        # Keep the original ref_point_head width so pretrained weights remain reusable,
        # and learn a quantization-friendly coordinate expansion in front of it.
        self.ref_point_head = IMLP(2 * d_model, d_model, d_model, 2)
        self.ref_point_coord_proj = IMLP(4, d_model, 2 * d_model, 2) if easy_quantize else None
        self.qact_sine_embed = IViTAct()
        self._export = False
    
    def export(self):
        self._export = True

    def _apply_norm(self, x, scale):
        if self.norm is None:
            return x, scale
        if isinstance(self.norm, IViTLayerNorm):
            return self.norm(x, scale)
        return self.norm(x), scale

    def refpoints_refine(self, refpoints_unsigmoid, refpoints_scale, new_refpoints_delta, new_refpoints_delta_scale):
        if self.bbox_reparam:
            tmp, tmp_scale = multiply_tensor_scale(
                new_refpoints_delta[..., :2],
                new_refpoints_delta_scale,
                refpoints_unsigmoid[..., 2:],
                refpoints_scale,
            )
            new_refpoints_cxcy, new_refpoints_cxcy_scale = add_tensor_scale(
                tmp,
                tmp_scale,
                refpoints_unsigmoid[..., :2],
                refpoints_scale,
            )
            new_refpoints_wh = new_refpoints_delta[..., 2:].exp() * refpoints_unsigmoid[..., 2:]
            new_refpoints_wh_scale = new_refpoints_delta_scale * refpoints_scale
            new_refpoints_unsigmoid, new_refpoints_scale = concat_list_tensor_scale(
                [(new_refpoints_cxcy, new_refpoints_cxcy_scale), (new_refpoints_wh, new_refpoints_wh_scale)],
                dim=-1,
            )
        else:
            new_refpoints_unsigmoid, new_refpoints_scale = add_tensor_scale(
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
                query_pos_embed, query_pos_embed_scale = self.ref_point_coord_proj(query_pos_input, refpoint_scale)
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
                query_sine_embed, query_sine_embed_scale = self.qact_sine_embed(query_sine_embed, None)
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

            # For the first decoder layer, we do not apply transformation over p_s
            pos_transformation = 1

            query_pos = query_pos * pos_transformation
            
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

        if self.norm is not None:
            output, output_scale = self._apply_norm(output, output_scale)
            if self.return_intermediate:
                intermediate.pop()
                intermediate_scales.pop()
                intermediate.append(output)
                intermediate_scales.append(output_scale)

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
            hs, hs_scale = concat_list_tensor_scale(
                [(x.unsqueeze(0), scale) for x, scale in zip(intermediate, intermediate_scales)],
                dim=0,
            )
            if self.bbox_embed is not None:
                references, references_scale = concat_list_tensor_scale(
                    [(x.unsqueeze(0), scale) for x, scale in zip(hs_refpoints_unsigmoid, hs_refpoints_scales)],
                    dim=0,
                )
                return hs, hs_scale, references, references_scale
            else:
                return hs, hs_scale, refpoints_unsigmoid.unsqueeze(0), refpoint_scale
        
        return output.unsqueeze(0), output_scale, refpoints_unsigmoid.unsqueeze(0), refpoint_scale


class QuantDecoderSelfAttention(nn.Module):

    def __init__(self, d_model, nhead):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.scale = self.head_dim ** -0.5

        self.q_proj = IViTLinear(d_model, d_model)
        self.k_proj = IViTLinear(d_model, d_model)
        self.v_proj = IViTLinear(d_model, d_model)
        self.out_proj = IViTLinear(d_model, d_model)

        self.qact_q = IViTAct()
        self.qact_k = IViTAct()
        self.qact_v = IViTAct()
        self.qact_attn = IViTAct()
        self.qact_out = IViTAct()
        self.qact_proj_drop = IViTAct()
        self.matmul = IViTMatMul()
        self.softmax = IViTSoftmax(output_bit=16, dim=-1)

    def forward(self, tgt, tgt_scale, query_pos, query_pos_scale):
        qk_input, qk_scale = add_tensor_scale(tgt, tgt_scale, query_pos, query_pos_scale)
        v_input, v_scale = tgt, tgt_scale

        q, q_scale = self.q_proj(qk_input, qk_scale)
        q, q_scale = self.qact_q(q, q_scale)
        k, k_scale = self.k_proj(qk_input, qk_scale)
        k, k_scale = self.qact_k(k, k_scale)
        v, v_scale = self.v_proj(v_input, v_scale)
        v, v_scale = self.qact_v(v, v_scale)

        bs, num_queries, _ = q.shape
        q = q.view(bs, num_queries, self.nhead, self.head_dim).transpose(1, 2)
        k = k.view(bs, num_queries, self.nhead, self.head_dim).transpose(1, 2)
        v = v.view(bs, num_queries, self.nhead, self.head_dim).transpose(1, 2)

        attn, attn_scale = self.matmul(q, q_scale, k.transpose(-2, -1), k_scale)
        attn = attn * self.scale
        attn_scale = attn_scale * self.scale
        attn, attn_scale = self.qact_attn(attn, attn_scale)
        attn, attn_scale = self.softmax(attn, attn_scale)

        out, out_scale = self.matmul(attn, attn_scale, v, v_scale)
        out = out.transpose(1, 2).reshape(bs, num_queries, self.d_model)
        out, out_scale = self.qact_out(out, out_scale)
        out, out_scale = self.out_proj(out, out_scale)
        out, out_scale = self.qact_proj_drop(out, out_scale)
        return out, out_scale


class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, sa_nhead, ca_nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False, group_detr=1, 
                 num_feature_levels=4, dec_n_points=4, 
                 skip_self_attn=False):
        super().__init__()
        # Decoder Self-Attention
        self.self_attn = QuantDecoderSelfAttention(d_model, sa_nhead)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = IViTLayerNorm(d_model)

        # Decoder Cross-Attention
        self.cross_attn = IMSDeformAttn(
            d_model, n_levels=num_feature_levels, n_heads=ca_nhead, n_points=dec_n_points)

        self.nhead = ca_nhead

        # Implementation of Feedforward model
        self.linear1 = IViTLinear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = IViTLinear(dim_feedforward, d_model)

        self.norm2 = IViTLayerNorm(d_model)
        self.norm3 = IViTLayerNorm(d_model)
        
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = ISegGELU()
        self.normalize_before = normalize_before
        self.group_detr = group_detr
        self.qact_ffn_1 = IViTAct()
        self.qact_ffn_2 = IViTAct()
        self.qact_self_attn_block = IViTAct()
        self.qact_cross_attn_block = IViTAct()
        self.qact_ffn_block = IViTAct()
        

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

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
        # ========== End of Self-Attention =============

        tgt, query_scale = add_tensor_scale(tgt, query_scale, self.dropout1(tgt2), tgt2_scale)
        tgt, query_scale = self.norm1(tgt, query_scale)
        tgt, query_scale = self.qact_self_attn_block(tgt, query_scale)

        # ========== Begin of Cross-Attention =============
        query_with_pos, query_with_pos_scale = add_tensor_scale(tgt, query_scale, query_pos, query_pos_scale)
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
        # ========== End of Cross-Attention =============

        tgt, query_scale = add_tensor_scale(tgt, query_scale, self.dropout2(tgt2), tgt2_scale)
        tgt, query_scale = self.norm2(tgt, query_scale)
        tgt, query_scale = self.qact_cross_attn_block(tgt, query_scale)
        tgt2, tgt2_scale = self.linear1(tgt, query_scale)
        tgt2, tgt2_scale = self.qact_ffn_1(tgt2, tgt2_scale)
        tgt2, tgt2_scale = self.activation(tgt2, tgt2_scale)
        tgt2, tgt2_scale = self.linear2(tgt2, tgt2_scale)
        tgt2, tgt2_scale = self.qact_ffn_2(tgt2, tgt2_scale)
        tgt, query_scale = add_tensor_scale(tgt, query_scale, self.dropout3(tgt2), tgt2_scale)
        tgt, query_scale = self.norm3(tgt, query_scale)
        tgt, query_scale = self.qact_ffn_block(tgt, query_scale)
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
    )
