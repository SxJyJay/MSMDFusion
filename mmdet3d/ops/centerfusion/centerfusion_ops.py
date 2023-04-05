import torch
import torch.nn as nn
from ipdb import set_trace
import random
import ipdb
import numpy as np
from mmdet3d.ops import spconv
import torch_scatter
from mmcv.cnn import build_norm_layer
import traceback

def scatter_v2(feat, coors, mode, return_inv=True, min_points=0, unq_inv=None, new_coors=None, sorted=True):
    assert feat.size(0) == coors.size(0)
    if mode == 'avg':
        mode = 'mean'


    if unq_inv is None and min_points > 0:
        new_coors, unq_inv, unq_cnt = torch.unique(coors, sorted=sorted, return_inverse=True, return_counts=True, dim=0)
    elif unq_inv is None:
        new_coors, unq_inv = torch.unique(coors, sorted=sorted, return_inverse=True, return_counts=False, dim=0)
    else:
        assert new_coors is not None, 'please pass new_coors for interface consistency, caller: {}'.format(traceback.extract_stack()[-2][2])


    if min_points > 0:
        cnt_per_point = unq_cnt[unq_inv]
        valid_mask = cnt_per_point >= min_points
        feat = feat[valid_mask]
        coors = coors[valid_mask]
        new_coors, unq_inv, unq_cnt = torch.unique(coors, sorted=sorted, return_inverse=True, return_counts=True, dim=0)

    if mode == 'max':
        new_feat, argmax = torch_scatter.scatter_max(feat, unq_inv, dim=0)
    elif mode in ('mean', 'sum'):
        new_feat = torch_scatter.scatter(feat, unq_inv, dim=0, reduce=mode)
    else:
        raise NotImplementedError

    if not return_inv:
        return new_feat, new_coors
    else:
        return new_feat, new_coors, unq_inv

def build_mlp(in_channel, hidden_dims, norm_cfg, is_head=False, act='relu', bias=False, dropout=0):
    layer_list = []
    last_channel = in_channel
    for i, c in enumerate(hidden_dims):
        act_layer = get_activation_layer(act, c)

        norm_layer = build_norm_layer(norm_cfg, c)[1]
        if i == len(hidden_dims) - 1 and is_head:
            layer_list.append(nn.Linear(last_channel, c, bias=True),)
        else:
            sq = [
                nn.Linear(last_channel, c, bias=bias),
                norm_layer,
                act_layer,
            ]
            if dropout > 0:
                sq.append(nn.Dropout(dropout))
            layer_list.append(
                nn.Sequential(
                    *sq
                )
            )

        last_channel = c
    mlp = nn.Sequential(*layer_list)
    return mlp

def get_activation_layer(act, dim=None):
    """Return an activation function given a string"""
    act = act.lower()
    if act == 'relu':
        act_layer = nn.ReLU(inplace=True)
    elif act == 'gelu':
        act_layer = nn.GELU()
    elif act == 'leakyrelu':
        act_layer = nn.LeakyReLU(inplace=True)
    elif act == 'prelu':
        act_layer = nn.PReLU(num_parameters=dim)
    elif act == 'swish' or act == 'silu':
        act_layer = nn.SiLU(inplace=True)
    elif act == 'glu':
        act_layer = nn.GLU()
    elif act == 'elu':
        act_layer = nn.ELU(inplace=True)
    elif act == 'sigmoid':
        act_layer = nn.Sigmoid()
    else:
        raise NotImplementedError
    return act_layer