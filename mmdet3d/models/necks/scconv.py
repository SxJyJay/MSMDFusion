import time
import numpy as np
import math

import torch

from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet
from torch.nn.modules.batchnorm import _BatchNorm

from mmcv.cnn import (build_conv_layer, build_norm_layer, build_upsample_layer,
                      constant_init, is_norm, kaiming_init, xavier_init)

from mmdet.models import NECKS

# from det3d.torchie.cnn import constant_init, kaiming_init, xavier_init
# from det3d.torchie.trainer import load_checkpoint
# from det3d.models.utils import Empty, GroupNorm, Sequential
# from det3d.models.utils import change_default_args

# from .. import builder
# from ..registry import NECKS
# from ..utils import build_norm_layer



class SCBlock(nn.Module):
    def __init__(self, in_chn, ds_padding):
        super(SCBlock, self).__init__()
        self._norm_cfg = dict(type="BN", eps=1e-3, momentum=0.01)
        h_chn = in_chn // 2
        self.light_branch = nn.ModuleList([
            self._make_conv(in_chn, h_chn, 1),
            self._make_conv(h_chn, h_chn, 3)])

        self.heavy_branch = nn.ModuleList([
            self._make_conv(in_chn, h_chn, 1),
            self._make_conv(h_chn, h_chn, 3),
            self._make_conv(h_chn, h_chn, 3)])
        self.ds_branch = self._make_ds(h_chn, h_chn, padding=ds_padding)
        self.out_conv = self._make_conv(in_chn, in_chn, 3)

    def _make_ds(self, inplanes, planes, stride=4, padding=0):
        block = nn.Sequential(
            nn.AvgPool2d(stride, padding=padding),
            nn.ZeroPad2d(1),
            nn.Conv2d(inplanes, planes, 3, stride=1, bias=False),
            build_norm_layer(self._norm_cfg, planes)[1],
        )
        return block

    def _make_conv(self, inplanes, planes, ks, stride=1):
        if ks == 1:
            block = nn.Sequential(
                nn.Conv2d(inplanes, planes, ks, stride=stride, bias=False),
                build_norm_layer(self._norm_cfg, planes)[1],
            )
        elif ks == 3:
            block = nn.Sequential(
                nn.ZeroPad2d(1),
                nn.Conv2d(inplanes, planes, ks, stride=stride, bias=False),
                build_norm_layer(self._norm_cfg, planes)[1],
            )
        return block

    def forward(self, x):
        light = self.light_branch
        heavy = self.heavy_branch
        ds = self.ds_branch
        out = self.out_conv

        l0 = torch.relu(light[0](x))
        l1 = torch.relu(light[1](l0))

        h0 = torch.relu(heavy[0](x))
        h1 = heavy[1](h0)
        d = F.interpolate(ds(h0), h0.size()[2:]) + h0
        h1 = torch.sigmoid(d) * h1
        h2 = heavy[2](h1)

        o_x = out(torch.cat([h2,l1], dim=1)) + x
        o_x = torch.relu(o_x)
        return o_x

@NECKS.register_module()
class SCConv(nn.Module):
    def __init__(
        self,
        num_input_features=256,
        num_proj_features=128,
        chn_per_segment=[256,256,256],
        blocks_per_segment=[3,3,3],
        ds_rates=[1,2,2],
        us_rates=[1,2,4],
        ds_paddings=[0,0,1],
        norm_cfg=None,
        name="scconv",
        logger=None,
        **kwargs
    ):
        super(SCConv, self).__init__()
        self.in_chn = num_input_features
        self.out_chns = chn_per_segment
        self.in_chns = [self.in_chn,] + chn_per_segment[:-1]
        self.blk_per_seg = blocks_per_segment

        if norm_cfg is None:
            norm_cfg = dict(type="BN", eps=1e-3, momentum=0.01)
        self._norm_cfg = norm_cfg
        self.blocks = []

        self.num_seg = len(chn_per_segment)
        out_convs = []
        for i in range(self.num_seg-1):
            if us_rates[i] > 1:
                out_convs.append(self._make_tconv(self.out_chns[i], num_proj_features, us_rates[i]))
            else:
                out_convs.append(self._make_conv(self.out_chns[i], num_proj_features, 1))

        self.out_convs = nn.ModuleList(out_convs)

        self.segments = []
        for i in range(self.num_seg):
            modules = []

            ds_padding = ds_paddings[i]

            modules.append(self._make_conv(self.in_chns[i], self.out_chns[i], 3, ds_rates[i]))
            for j in range(blocks_per_segment[i]):
                modules.append(SCBlock(self.out_chns[i], ds_padding))

            if i == self.num_seg-1:
                out_chn = num_proj_features
                modules.append(self._make_tconv(self.out_chns[i], out_chn, us_rates[i]))

            self.segments.append(nn.ModuleList(modules))

        self.segments = nn.ModuleList(self.segments)

    @property
    def downsample_factor(self):
        return 1.

    def _make_tconv(self, inplanes, planes, stride):
        block = nn.Sequential(
            nn.ConvTranspose2d(
                inplanes,
                planes,
                stride,
                stride=stride,
                bias=False,
            ),
            build_norm_layer(self._norm_cfg, planes)[1],
        )
        return block

    def _make_conv(self, inplanes, planes, ks, stride=1):
        if ks == 1:
            block = nn.Sequential(
                nn.Conv2d(inplanes, planes, ks, stride=stride, bias=False),
                build_norm_layer(self._norm_cfg, planes)[1],
            )
        elif ks ==3:
            block = nn.Sequential(
                nn.ZeroPad2d(1),
                nn.Conv2d(inplanes, planes, ks, stride=stride, bias=False),
                build_norm_layer(self._norm_cfg, planes)[1],
            )
        return block

    # default init_weights for conv(msra) and norm in ConvModule
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution="uniform")


    def forward(self, x):
        outs = []
        for i in range(self.num_seg):
            modules = self.segments[i]
            for j,m in enumerate(modules):
                if j == 0: # the first conv block
                    x = torch.relu(m(x))
                elif j in list(range(1,1+self.blk_per_seg[i])):
                    x = m(x)
                else: # maybe with an upsampling layer
                    x = torch.relu(m(x))

            if i < self.num_seg-1:
                outs.append(torch.relu(self.out_convs[i](x)))
            else:
                outs.append(x)

        x = torch.cat(outs, dim=1)

        return x, None