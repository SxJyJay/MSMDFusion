import torch
import torch.nn as nn

from mmdet3d.ops.spconv import SparseConv3d, SubMConv3d

from typing import Union
from thop import profile

def count_sparseconv(m: Union[SparseConv3d, SubMConv3d], x, y):
    indice_dict = y.indice_dict[m.indice_key]
    kmap_size = indice_dict[-2].sum().item()
    m.total_ops += kmap_size * x[0].features.shape[1] * y.features.shape[1]

def flops_counter(model, **inputs):
    macs, params = profile(
        model, 
        **inputs, 
        custom_ops={
            # WindowMSA: count_window_msa,
            #ShiftWindowMSA: count_window_msa,
            SparseConv3d: count_sparseconv,
            SubMConv3d: count_sparseconv,
            # MultiheadAttention: count_mha
        },
        verbose=False
    )

    return macs, params