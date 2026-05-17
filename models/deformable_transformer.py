# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

import copy
from typing import Optional, List
import math

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.nn.init import xavier_uniform_, constant_, uniform_, normal_

from util.misc import inverse_sigmoid
from models.ops.modules import MSDeformAttn
# from config import *
from functools import partial


def bbox_rel_pos_encoding(boxes: torch.Tensor,
                          T: float = 10000.0,
                          d_re: int = 16,
                          div : float = 2.0, 
                          s: float = 1.0,
                          eps: float = 1e-6) -> torch.Tensor:

    B, N, _ = boxes.shape
    x, y, w, h = boxes.unbind(-1)  # (B, N)

    dx = x[:, :, None] - x[:, None, :]
    dy = y[:, :, None] - y[:, None, :]
    e1 = torch.log(torch.clamp(torch.abs(dx) / (w[:, :, None] + eps) + 1.0, min=eps))
    e2 = torch.log(torch.clamp(torch.abs(dy) / (h[:, :, None] + eps) + 1.0, min=eps))
    e3 = torch.log(torch.clamp(w[:, :, None] / (w[:, None, :] + eps), min=eps))
    e4 = torch.log(torch.clamp(h[:, :, None] / (h[:, None, :] + eps), min=eps))
    E  = torch.stack([e1, e2, e3, e4], dim=-1)  # (B, N, N, 4)

    # 2) frequency term: T^((2k//div)/d_re) for k=0..d_re-1
    device, dtype = boxes.device, boxes.dtype
    k_idx = torch.arange(d_re, device=device, dtype=dtype)
    denom = T ** (2 * (k_idx/div) / d_re)           # (d_re,)

    E_scaled = s * E.unsqueeze(-1)
    args     = E_scaled / denom      
    sin_enc = torch.sin(args)                # (B, N, N, 4, d_re)
    cos_enc = torch.cos(args)                # (B, N, N, 4, d_re)
    #    = (B, N, N, 8*d_re)
    rel_pos = torch.cat([sin_enc, cos_enc], dim=-1)  # (B, N, N, 4, 2*d_re)
    rel_pos = rel_pos.reshape(B, N, N, -1)          # (B, N, N, 8*d_re)

    return rel_pos

class TokenParameterModule(nn.Module):
    def __init__(self, param):
        super().__init__()
        self.param = param
    def forward(self, x=None):
        return self.param

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class DeformableTransformer(nn.Module):
    def __init__(self, d_model=256, nhead=8, num_classes=91,
                 num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=1024, dropout=0.1,
                 activation="relu", return_intermediate_dec=False,
                 num_feature_levels=4, dec_n_points=4,  enc_n_points=4,
                 two_stage=False, two_stage_num_proposals=300, look_forward_twice=False, mixed_selection=False,
                 separate_mode=False,
                 with_aux_class_embed=False, sa_to_ffn=False,
                 use_rel_enc=False,
                 learn_token_bias=False, use_pos_mlp=False, num_token=10):
        super().__init__()

        self.d_model = d_model
        self.nhead = nhead
        self.two_stage = two_stage
        self.two_stage_num_proposals = two_stage_num_proposals
        self.look_forward_twice = look_forward_twice
        self.mixed_selection = mixed_selection
        self.separate_mode = separate_mode
        self.sa_to_ffn = sa_to_ffn
        self.num_classes = num_classes
        self.num_queries = two_stage_num_proposals

        encoder_layer = DeformableTransformerEncoderLayer(d_model, dim_feedforward,
                                                          dropout, activation,
                                                          num_feature_levels, nhead, enc_n_points)
        self.encoder = DeformableTransformerEncoder(encoder_layer, num_encoder_layers)

        if self.separate_mode:
            ca_layer = DeformableTransformerDecoderCALayer(d_model, dim_feedforward,
                                                            dropout, activation,
                                                            num_feature_levels, nhead, dec_n_points, 
                                                            sa_to_ffn=sa_to_ffn)
            sa_layer = DeformableTransformerDecoderSALayer(d_model, dim_feedforward,
                                                            dropout, activation,
                                                            nhead, use_rel_enc=use_rel_enc,
                                                            learn_token_bias=learn_token_bias,
                                                            use_pos_mlp=use_pos_mlp, num_token=num_token)
            self.decoder = DeformableTransformerDecoder([ca_layer, sa_layer], num_decoder_layers,return_intermediate_dec,  num_queries=two_stage_num_proposals, look_forward_twice=look_forward_twice,
                                                            separate_mode=self.separate_mode,
                                                            with_aux_class_embed=with_aux_class_embed,
                                                            use_rel_enc=use_rel_enc,
                                                             num_token=num_token)
        else:
            # MDS-DETR only implemented with seperated decoder layers!
            self.decoder = None

        self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, d_model))

        if two_stage:
            self.enc_output = nn.Linear(d_model, d_model)
            self.enc_output_norm = nn.LayerNorm(d_model)
            self.pos_trans = nn.Linear(d_model * 2, d_model * 2)
            self.pos_trans_norm = nn.LayerNorm(d_model * 2)
        else:
            self.reference_points = nn.Linear(d_model, 2)
        
        self._reset_parameters()
        if learn_token_bias:
            for sa_layer in self.decoder.sa_layers:
                if 'minus' in learn_token_bias:
                    constant_(sa_layer.token_bias.data, -math.log(num_token))
                else:
                    constant_(sa_layer.token_bias.data, 0.)

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()
        if not self.two_stage:
            xavier_uniform_(self.reference_points.weight.data, gain=1.0)
            constant_(self.reference_points.bias.data, 0.)
        normal_(self.level_embed)


    def get_proposal_pos_embed(self, proposals):
        num_pos_feats = 128
        temperature = 10000
        scale = 2 * math.pi

        dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=proposals.device)
        dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)
        # N, L, 4
        proposals = proposals.sigmoid() * scale

        # N, L, 4, 128
        pos = proposals[:, :, :, None] / dim_t
        # N, L, 4, 64, 2
        pos = torch.stack((pos[:, :, :, 0::2].sin(), pos[:, :, :, 1::2].cos()), dim=4).flatten(2)
        return pos

    def gen_encoder_output_proposals(self, memory, memory_padding_mask, spatial_shapes):
        N_, S_, C_ = memory.shape
        base_scale = 4.0
        proposals = []
        _cur = 0
        for lvl, (H_, W_) in enumerate(spatial_shapes):
            mask_flatten_ = memory_padding_mask[:, _cur:(_cur + H_ * W_)].view(N_, H_, W_, 1)
            valid_H = torch.sum(~mask_flatten_[:, :, 0, 0], 1)
            valid_W = torch.sum(~mask_flatten_[:, 0, :, 0], 1)

            grid_y, grid_x = torch.meshgrid(torch.linspace(0, H_ - 1, H_, dtype=torch.float32, device=memory.device),
                                            torch.linspace(0, W_ - 1, W_, dtype=torch.float32, device=memory.device))
            grid = torch.cat([grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)], -1)

            scale = torch.cat([valid_W.unsqueeze(-1), valid_H.unsqueeze(-1)], 1).view(N_, 1, 1, 2)
            grid = (grid.unsqueeze(0).expand(N_, -1, -1, -1) + 0.5) / scale
            wh = torch.ones_like(grid) * 0.05 * (2.0 ** lvl)
            proposal = torch.cat((grid, wh), -1).view(N_, -1, 4)
            proposals.append(proposal)
            _cur += (H_ * W_)
        output_proposals = torch.cat(proposals, 1)
        output_proposals_valid = ((output_proposals > 0.01) & (output_proposals < 0.99)).all(-1, keepdim=True)
        output_proposals = torch.log(output_proposals / (1 - output_proposals))
        output_proposals = output_proposals.masked_fill(memory_padding_mask.unsqueeze(-1), float('inf'))
        output_proposals = output_proposals.masked_fill(~output_proposals_valid, float('inf'))

        output_memory = memory
        output_memory = output_memory.masked_fill(memory_padding_mask.unsqueeze(-1), float(0))
        output_memory = output_memory.masked_fill(~output_proposals_valid, float(0))
        output_memory = self.enc_output_norm(self.enc_output(output_memory))
        return output_memory, output_proposals

    def get_valid_ratio(self, mask):
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    def forward(self, srcs, masks, pos_embeds, query_embed=None, **kwargs):
        assert self.two_stage or query_embed is not None

        # prepare input for encoder
        src_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for lvl, (src, mask, pos_embed) in enumerate(zip(srcs, masks, pos_embeds)):
            bs, c, h, w = src.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)
            src = src.flatten(2).transpose(1, 2)
            mask = mask.flatten(1)
            pos_embed = pos_embed.flatten(2).transpose(1, 2)
            lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            src_flatten.append(src)
            mask_flatten.append(mask)
        src_flatten = torch.cat(src_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)

        # encoder
        memory = self.encoder(src_flatten, spatial_shapes, level_start_index, valid_ratios, lvl_pos_embed_flatten, mask_flatten)

        # prepare input for decoder
        bs, _, c = memory.shape
        if self.two_stage:
            output_memory, output_proposals = self.gen_encoder_output_proposals(memory, mask_flatten, spatial_shapes)
            enc_outputs_class = self.decoder.class_embed[self.decoder.num_layers](output_memory)
            enc_outputs_coord_unact = self.decoder.bbox_embed[self.decoder.num_layers](output_memory) + output_proposals

            topk = self.two_stage_num_proposals

            topk_proposals = torch.topk(enc_outputs_class[..., 0], topk, dim=1)[1]
            topk_coords_unact = torch.gather(enc_outputs_coord_unact, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, 4))
            topk_coords_unact = topk_coords_unact.detach()
            reference_points = topk_coords_unact.sigmoid()
            init_reference_out = reference_points

            pos_trans_out = self.pos_trans_norm(self.pos_trans(self.get_proposal_pos_embed(topk_coords_unact)))

            if not self.mixed_selection:
                query_embed, tgt = torch.split(pos_trans_out, c, dim=2)
            else:
                # tgt: content embedding, query_embed here is the learnable content embedding
                tgt = query_embed.unsqueeze(0).expand(bs, -1, -1)
                # query_embed: position embedding, transformed from the topk proposals
                query_embed, _ = torch.split(pos_trans_out, c, dim=2)
        else:
            query_embed, tgt = torch.split(query_embed, c, dim=1)
            query_embed = query_embed.unsqueeze(0).expand(bs, -1, -1)
            tgt = tgt.unsqueeze(0).expand(bs, -1, -1)
            reference_points = self.reference_points(query_embed).sigmoid()
            init_reference_out = reference_points

        self_attn_mask = None

        hs_o2o, hs_o2m, inter_references = self.decoder(tgt, reference_points, memory,
                                            spatial_shapes, level_start_index, valid_ratios,
                                            query_embed, mask_flatten, self_attn_mask=self_attn_mask) # false : hybrid - o2m layers only!!

        inter_references_out = inter_references
        if self.two_stage:
            return (hs_o2o, hs_o2m, init_reference_out, inter_references_out,
                    enc_outputs_class, enc_outputs_coord_unact, output_proposals.sigmoid())
        return hs_o2o, hs_o2m, init_reference_out, inter_references_out, None, None, output_proposals.sigmoid(),


class DeformableTransformerEncoderLayer(nn.Module):
    def __init__(self,
                 d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4):
        super().__init__()

        # self attention
        self.self_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, src):
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src

    def forward(self, src, pos, reference_points, spatial_shapes, level_start_index, padding_mask=None):
        # self attention
        src2 = self.self_attn(self.with_pos_embed(src, pos), reference_points, src, spatial_shapes, level_start_index, padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        # ffn
        src = self.forward_ffn(src)

        return src

class DeformableTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        reference_points_list = []
        for lvl, (H_, W_) in enumerate(spatial_shapes):

            ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                                          torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device))
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H_)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W_)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points

    def forward(self, src, spatial_shapes, level_start_index, valid_ratios, pos=None, padding_mask=None):
        output = src
        reference_points = self.get_reference_points(spatial_shapes, valid_ratios, device=src.device)
        for _, layer in enumerate(self.layers):
            output = layer(output, pos, reference_points, spatial_shapes, level_start_index, padding_mask)

        return output

class DeformableTransformerDecoderCALayer(nn.Module):
    def __init__(self, d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4, sa_to_ffn=False):
        super().__init__()
        self.d_model = d_model
        # cross attention
        self.cross_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

        self.sa_to_ffn = sa_to_ffn
        
        if sa_to_ffn == 'to_ffn':
            self.sa_linear1 = nn.Linear(d_model, d_ffn)
            self.activation = _get_activation_fn(activation)
            self.sa_dropout3 = nn.Dropout(dropout)
            self.sa_linear2 = nn.Linear(d_ffn, d_model)
            self.sa_dropout4 = nn.Dropout(dropout)
            self.sa_norm3 = nn.LayerNorm(d_model)
        elif sa_to_ffn == 'to_sa':
            self.add_self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
            self.add_dropout = nn.Dropout(dropout)
            self.add_norm = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward(self, tgt, query_pos, reference_points, src, src_spatial_shapes, level_start_index, src_padding_mask=None, self_attn_mask=None):
        # Cross attention
        tgt2 = self.cross_attn(self.with_pos_embed(tgt, query_pos),
                                reference_points, src, src_spatial_shapes, level_start_index, src_padding_mask)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        if self.sa_to_ffn == 'to_ffn':
            tgt2 = self.sa_linear2(self.sa_dropout3(self.activation(self.sa_linear1(tgt))))
            tgt = tgt + self.sa_dropout4(tgt2)
            tgt = self.sa_norm3(tgt)
        elif self.sa_to_ffn == 'to_sa':
            q = k = self.with_pos_embed(tgt, query_pos)
            tgt2 = self.add_self_attn(q.transpose(0, 1), k.transpose(0, 1), tgt.transpose(0, 1), attn_mask=self_attn_mask)[0].transpose(0, 1)
            tgt = tgt + self.add_dropout(tgt2)
            tgt = self.add_norm(tgt)
        # ffn
        tgt_o2m = self.forward_ffn(tgt)
        
        return tgt_o2m

class DeformableTransformerDecoderSALayer(nn.Module):
    def __init__(self, d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_heads=8,  use_rel_enc=False, learn_token_bias=False,
                 use_pos_mlp=False, num_token=10):
        super().__init__()

        # self attention
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

        self.use_rel_enc = use_rel_enc
        self.learn_token_bias = learn_token_bias
        self.use_pos_mlp = use_pos_mlp
        self.vis_mask_attn = None
        self.num_token=num_token
        if self.learn_token_bias:
            self.token_bias = nn.Parameter(torch.zeros(n_heads, 1, num_token))
        self.masked_att = CustomWeightedSelfAttention(self.self_attn, d_model)
        if self.use_rel_enc:
            if use_pos_mlp:
                self.linear_rel = MLP(d_model//2, d_model//2, n_heads, 2)
            else:
                self.linear_rel = nn.Linear(d_model//2, n_heads)
            if self.use_rel_enc == 'pos_allow_relu':
                self.activation_rel = _get_activation_fn(activation)

        self.sup_token_v = nn.Parameter(torch.zeros(self.num_token, d_model))
            
    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt
        
    def create_comparison_matrix(self, P):
        # P shape: (B, N, 1)
        P_i = P.expand(*P.shape[:-1], P.shape[-2])
        P_j = P.transpose(-2, -1).expand(*P.shape[:-1], P.shape[-2])

        ones = P.new_tensor(1.0)
        neg_ones = P.new_tensor(-1.0)
        rank_conf = torch.where(P_i < P_j, ones, neg_ones)  # (..., N, N)
        ## <= must be fixed!!!
        return rank_conf

    def forward(self, tgt, query_pos, intermediate_conf, layer_depth=None, rank_dict=None, rel_pos_emb=None):
        rel_enc_weight=None

        rank_conf, rank_mask = SortAndMask(self.create_comparison_matrix,intermediate_conf, self.num_token)

        if rel_pos_emb is not None:
            assert self.use_rel_enc
            rel_enc_weight = self.linear_rel(rel_pos_emb).permute(-1, 0, 1, 2)
            rel_enc_weight = (rel_enc_weight + rel_enc_weight.permute(0, 1, 3, 2)) * 0.5 ## Constraint Symmetries for relative pos bias

            if self.learn_token_bias:
                rel_enc_weight = torch.cat((
                        rel_enc_weight, self.token_bias.unsqueeze(2).repeat(1, rel_enc_weight.size(1), rel_enc_weight.size(-1), 1)), 3)

            ## Delete absolute position encoding for MDS.
            if self.use_rel_enc == 'rel_only':
                query_pos = None  
            elif self.use_rel_enc == 'pos_allow':
                asdfasdf = 1 # dummy
            else:
                query_pos = None

        q = k = self.with_pos_embed(tgt, query_pos)

        num_queries = tgt.size(1)
        sup_token = rank_dict['sup_token'][layer_depth-1]()
        sup_token = sup_token.repeat(q.size(0), 1, 1)

        q = torch.cat((q, sup_token), 1); k = torch.cat((k, sup_token), 1); 
        tgt = torch.cat((tgt, self.sup_token_v.repeat(q.size(0), 1, 1)), 1)
        if rel_enc_weight is not None:
            if self.learn_token_bias:
                rel_enc_weight = F.pad(rel_enc_weight, (0, 0, 0, self.num_token), mode='constant', value=0)
            else:
                rel_enc_weight = F.pad(rel_enc_weight, (0, self.num_token, 0, self.num_token), mode='constant', value=0)

        tgt2, vis_att, raw_attn_inner= self.masked_att(q.transpose(0, 1), k.transpose(0,1), tgt.transpose(0, 1), rank_mask, rel_enc_weight=rel_enc_weight)
        tgt2 = tgt2.transpose(0, 1)
        # if not self.training:
        #     self.vis_mask_attn = (vis_att, rel_enc_weight, raw_attn_inner)

        tgt2 = tgt2[:, :num_queries, :]
        tgt = tgt[:, :num_queries, :]

        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
    
        # ffn
        tgt_o2o = self.forward_ffn(tgt)
        
        return tgt_o2o


class DeformableTransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, return_intermediate=False, num_queries=300, look_forward_twice=False,
                    separate_mode=False, separate_layers=None, with_aux_class_embed=False,
                    use_rel_enc=False, num_token=10):
        super().__init__()
        self.separate_mode = separate_mode
        self.num_queries = num_queries
        if not self.separate_mode:
            self.layers = _get_clones(decoder_layer, num_layers)
        else:
            self.ca_layers = _get_clones(decoder_layer[0], self.separate_mode[0])
            self.sa_layers = _get_clones(decoder_layer[1], self.separate_mode[1])
        self.num_layers = num_layers
        self.return_intermediate = return_intermediate
        # hack implementation for iterative bounding box refinement and two-stage Deformable DETR
        self.bbox_embed = None
        self.class_embed = None
        self.look_forward_twice = look_forward_twice
        self.with_aux_class_embed=with_aux_class_embed
        self.use_rel_enc=use_rel_enc

        self.rank_dict = nn.ModuleDict()
        self.rank_dict['sup_token'] = nn.ModuleList([TokenParameterModule(nn.Parameter(torch.zeros(num_token, 256))) for _ in range(self.separate_mode[1])])

    def create_comparison_matrix(self, P):
        B, N, _ = P.shape
        # Expand P to compare each pair of elements
        P_i = P.expand(-1, -1, N)  # Shape: (B, N, N)
        P_j = P.transpose(1, 2).expand(-1, N, -1)  # Shape: (B, N, N)
        # Compare P_i and P_j
        rank_conf = torch.where(P_i < P_j, torch.tensor(1.), torch.tensor(-1.))  # Shape: (B, N, N)    
        return rank_conf

    def forward(self, tgt, reference_points, src, src_spatial_shapes, src_level_start_index, src_valid_ratios,
                query_pos=None, src_padding_mask=None, self_attn_mask=None, **kwargs):
        output = tgt
        _, _, src_c = src.shape
        intermediate = []
        intermediate_o2m = []
        intermediate_reference_points = []
        intermediate_o2m_conf = []
        intermediate_o2o = []

        if self.separate_mode:
            for lid, layer in enumerate(self.ca_layers):
                if reference_points.shape[-1] == 4:
                    reference_points_input = reference_points[:, :, None] \
                                            * torch.cat([src_valid_ratios, src_valid_ratios], -1)[:, None]
                else:
                    assert reference_points.shape[-1] == 2
                    reference_points_input = reference_points[:, :, None] * src_valid_ratios[:, None]
 
                output = layer(output, query_pos, reference_points_input, src, src_spatial_shapes, src_level_start_index, src_padding_mask,
                        self_attn_mask=self_attn_mask, **kwargs)


                ## Memory issues
                output_conf = self.class_embed[lid](output)


                # hack implementation for iterative bounding box refinement
                if self.bbox_embed is not None:
                    tmp = self.bbox_embed[lid](output)
                    if reference_points.shape[-1] == 4:
                        new_reference_points = tmp + inverse_sigmoid(reference_points)
                        new_reference_points = new_reference_points.sigmoid()
                    else:
                        assert reference_points.shape[-1] == 2
                        new_reference_points = tmp
                        new_reference_points[..., :2] = tmp[..., :2] + inverse_sigmoid(reference_points)
                        new_reference_points = new_reference_points.sigmoid()
                    reference_points = new_reference_points.detach()

                if self.return_intermediate:
                    intermediate_o2m.append(output)
                    intermediate_reference_points.append(
                        new_reference_points
                        if self.look_forward_twice
                        else reference_points
                    )
                    intermediate_o2m_conf.append(output_conf)

            ## CA layer end.
            
            intermediate_o2m = torch.stack(intermediate_o2m)
            intermediate_reference_points = torch.stack(intermediate_reference_points)
            intermediate_o2m_conf = torch.stack(intermediate_o2m_conf)

            intermediate_o2m_conf = intermediate_o2m_conf[-1, :, :, :].unsqueeze(0).max(-1)[0].unsqueeze(-1) # highest classprob
                

            dec_query_pos = query_pos

            rel_pos_emb = None
            if self.use_rel_enc:
                with torch.no_grad():
                    rel_pos_emb = bbox_rel_pos_encoding(intermediate_reference_points[-1]).detach()

            for lid2, layer in enumerate(self.sa_layers):
                if lid2 == 0:
                    input_conf = intermediate_o2m_conf.detach()
                else:
                    cls_emb_depth = self.separate_mode[0] + lid2 if self.with_aux_class_embed else self.separate_mode[0]-1
                    input_conf = self.class_embed[cls_emb_depth](output).max(-1)[0].unsqueeze(-1)
                    input_conf = input_conf.unsqueeze(0)
                kwargs = {}
                
                output = layer(output, dec_query_pos, input_conf, layer_depth=lid2, rank_dict=self.rank_dict, rel_pos_emb=rel_pos_emb,
                        **kwargs)
                if self.return_intermediate:
                    intermediate_o2o.append(output)
            
            intermediate_o2o = torch.stack(intermediate_o2o)

            return intermediate_o2o, intermediate_o2m, intermediate_reference_points


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
        # return partial(F.relu, inplace=True)
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


class CustomWeightedSelfAttention(nn.Module):
    def __init__(self, attn, d_model=256, headwise_weight=False):
        super(CustomWeightedSelfAttention, self).__init__()
        self.attn = attn
        self.headwise_weight=headwise_weight
        self.logsigmoid = nn.LogSigmoid()

    def forward(self, q, k, v, W, return_att_weight=True, rel_enc_weight=None):
        T, B, d_model = q.size()  # Sequence length, batch size, embedding dimension
        
        q_proj = F.linear(q, self.attn.in_proj_weight[:d_model], self.attn.in_proj_bias[:d_model])
        k_proj = F.linear(k, self.attn.in_proj_weight[d_model:2*d_model], self.attn.in_proj_bias[d_model:2*d_model])
        v_proj = F.linear(v, self.attn.in_proj_weight[2*d_model:], self.attn.in_proj_bias[2*d_model:])        

        q_proj = q_proj.contiguous().view(T, B, self.attn.num_heads, -1).transpose(0, 2)
        k_proj = k_proj.contiguous().view(T, B, self.attn.num_heads, -1).transpose(0, 2)
        v_proj = v_proj.contiguous().view(T, B, self.attn.num_heads, -1).transpose(0, 2)
        
        attn_scores = torch.einsum('hbqc,hbkc->hbqk', q_proj, k_proj)
        attn_scores = attn_scores / ((d_model//self.attn.num_heads) ** 0.5)
        attn_scores_raw = attn_scores

        if rel_enc_weight is not None:
            attn_scores = attn_scores + rel_enc_weight

        if W is not None:
            if not self.headwise_weight:
                W = W.repeat(self.attn.num_heads, 1, 1, 1)
            attn_scores_weighted = attn_scores + self.logsigmoid(W)
        else:
            attn_scores_weighted = attn_scores
        attn_probs = F.softmax(attn_scores_weighted, dim=-1)
        attn_output = torch.einsum('hbqk,hbkc->hbqc', attn_probs, v_proj)
        attn_output = attn_output.permute(2, 1, 0, -1).contiguous().view(T, B, d_model)
        # Apply the final output projection
        attn_output = self.attn.out_proj(attn_output)
        if return_att_weight:
            return attn_output, attn_probs, attn_scores_raw
        return attn_output

def SortAndMask(create_comparison_matrix, intermediate_conf, num_token):
        rank_conf = create_comparison_matrix(intermediate_conf)  
        rank_mask = torch.where(rank_conf == 1, torch.tensor(float('inf')), torch.tensor(float('-inf')))
        rank_mask = F.pad(rank_mask, (0, num_token, 0, num_token), mode='constant', value=float('inf'))

        return rank_conf, rank_mask

def build_deforamble_transformer(args):
    num_classes = 20 if args.dataset_file != 'coco' else 91
    if args.dataset_file == "coco_panoptic":
        num_classes = 250
    return DeformableTransformer(
        d_model=args.hidden_dim,
        nhead=args.nheads,
        num_classes=num_classes,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        activation="relu",
        return_intermediate_dec=True,
        num_feature_levels=args.num_feature_levels,
        dec_n_points=args.dec_n_points,
        enc_n_points=args.enc_n_points,
        two_stage=args.two_stage,
        two_stage_num_proposals=args.num_queries,
        mixed_selection=args.mixed_selection,
        look_forward_twice=args.look_forward_twice,
        separate_mode=args.separate_mode,
        with_aux_class_embed=args.with_aux_class_embed,
        sa_to_ffn=args.sa_to_ffn,
        use_rel_enc=args.use_rel_enc,
        learn_token_bias=args.learn_token_bias,
        use_pos_mlp=args.use_pos_mlp,
        num_token=args.num_token,
    )
