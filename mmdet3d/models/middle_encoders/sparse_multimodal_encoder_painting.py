from mmcv.runner import auto_fp16
from mmdet3d.core import voxel
from torch import nn as nn
from torch.nn import functional as F
import torch
import ipdb
from spconv.pytorch import functional as Fsp

from mmdet3d.ops import SparseBasicBlock, make_sparse_convmodule
from mmdet3d.ops import furthest_point_sample, gather_points, ball_query
# from mmdet3d.ops import spconv as spconv
import spconv.pytorch as spconv
from ..registry import MIDDLE_ENCODERS

def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.
    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist

def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points

def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask].float()
        farthest = torch.max(distance, -1)[1]
    return centroids

def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, N, 3]
        new_xyz: query points, [B, S, 3]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx

@MIDDLE_ENCODERS.register_module()
class SparseMultiModalEncoderPaint(nn.Module):
    def __init__(self,
                 in_channels_3D = (16, 32, 64, 128),
                 in_channels_2D = (259, 259, 259, 259),
                 out_channels=(32, 64, 128, 128),
                 padding=(1, 1, 1, [0, 1, 1]),
                 down_kernel_size=(3, 3, 3, [3, 1, 1]),
                 down_stride=(2, 2, 2, [2, 1, 1]),
                 order=('conv', 'norm', 'act'),
                 norm_cfg=dict(type='BN1d', eps=1e-3, momentum=0.01),
                 block_type='conv_module'):
        super().__init__()
        assert block_type in ['conv_module', 'basicblock']
        self.in_channels_3D = in_channels_3D
        self.in_channels_2D = in_channels_2D
        self.out_channels = out_channels
        self.padding = padding
        self.down_kernel_size = down_kernel_size
        self.down_stride = down_stride
        self.order = order
        self.fp16_enabled = False

        self.make_grouped_sparse_conv_blocks(norm_cfg)
        self.make_aggregation_block(norm_cfg)
        self.make_downscale_block(norm_cfg)

    def make_grouped_sparse_conv_blocks(self, norm_cfg, conv_cfg=dict(type='SubMConv3d')):
        self.grouped_sp_conv_blocks_3D = spconv.SparseSequential()
        self.grouped_sp_conv_blocks_2D = spconv.SparseSequential()
        self.grouped_sp_conv_blocks_mix = spconv.SparseSequential()
        gate_control, cross_gate_control = [], []
        stage_num = len(self.in_channels_3D)
        for i in range(stage_num):
            block_3D = make_sparse_convmodule(
                self.in_channels_3D[i],
                self.in_channels_3D[i],
                3,
                indice_key=f'subm3D_{i+1}',
                norm_cfg=norm_cfg,
                padding=1,
                conv_type='SubMConv3d'
            )
            block_2D = make_sparse_convmodule(
                64,
                64,
                3,
                indice_key=f'block2d_0_{i+1}',
                norm_cfg=norm_cfg,
                padding=1,
                conv_type='SubMConv3d'
            )
            block_mixed = SparseBasicBlock(
                self.in_channels_3D[i]+64,
                self.in_channels_3D[i]+64,
                norm_cfg=norm_cfg,
                conv_cfg=conv_cfg
            )
            gate_control.append(
                nn.Sequential(
                    nn.Linear(self.in_channels_3D[i], self.in_channels_2D[i]),
                    # nn.BatchNorm1d(self.in_channels_2D[i]),
                    nn.ReLU()
                )
            )
            cross_gate_control.append(
                nn.Sequential(
                    nn.Linear(self.in_channels_3D[i], self.in_channels_2D[i]),
                    # nn.BatchNorm1d(self.in_channels_2D[i]),
                    nn.ReLU()
                )
            )
            stage_name = f'stage_{i + 1}'
            self.grouped_sp_conv_blocks_3D.add_module(stage_name, block_3D)
            self.grouped_sp_conv_blocks_2D.add_module(stage_name, block_2D)
            self.grouped_sp_conv_blocks_mix.add_module(stage_name, block_mixed)
            self.gate_control = nn.ModuleList(gate_control)
            self.cross_gate_control = nn.ModuleList(cross_gate_control)

    def make_aggregation_block(self, norm_cfg, conv_cfg=dict(type='SubMConv3d')):
        self.aggregation_blocks = spconv.SparseSequential()
        stage_num = len(self.in_channels_3D)
        for i in range(stage_num):
            agg_block = SparseBasicBlock(
                self.in_channels_3D[i]+64,
                self.in_channels_3D[i]+64,
                norm_cfg=norm_cfg,
                conv_cfg=conv_cfg
            )
            stage_name = f'stage_{i + 1}'
            self.aggregation_blocks.add_module(stage_name, agg_block)
        
    def make_downscale_block(self, norm_cfg):
        self.downscale_blocks = spconv.SparseSequential()
        stage_num = len(self.in_channels_3D)
        for i in range(stage_num):
            ds_block = make_sparse_convmodule(
                self.in_channels_3D[i]+64,
                self.out_channels[i]+64,
                kernel_size=self.down_kernel_size[i],
                indice_key=f'spconv_ds_{i+1}',
                norm_cfg=norm_cfg,
                stride=self.down_stride[i],
                padding=self.padding[i],
                conv_type='SparseConv3d'
            )
            stage_name = f'stage_{i + 1}'
            self.downscale_blocks.add_module(stage_name, ds_block)

    def pad_missing_batch_id(self, indices, features, B, template_indice, template_feature):
        """ deal with corner case of missing certain batch ids during indexing operation
            
            template_indice/feature: must guarantee non-empty
        
        """
        batch_ids = torch.arange(B)
        
        indices_batch_ids = indices[:, 0].unique().cpu()
        for batch_id in batch_ids:
            if batch_id not in indices_batch_ids:
                padded_indices = torch.zeros_like(template_indice).unsqueeze(0)
                padded_indices[0, 0] = batch_id.item()
                padded_features = torch.zeros_like(template_feature).unsqueeze(0)
                indices = torch.cat([indices, padded_indices], dim=0)
                features = torch.cat([features, padded_features], dim=0)

        return indices, features

    def fps_NN(self, query, key, fps_num, radius, max_cluster_samples, dist_thresh):
        """Efficient NN search for huge amounts of query and key (suppose queries are redundant)
        
            Behaivor:
            1. apply FPS on query and generate representative queries (repr_query)
            2. calculate repr_queries' distances with all keys, and get the NN key
            3. apply ball query to assign the same NN key with the group center 

        """

        query_NN_key_idx = torch.zeros_like(query[:, 0]).long() - 1
        query = query[:, 1:].unsqueeze(0)
        key = key[:, 1:].unsqueeze(0)

        if query.shape[1] <= fps_num:
            dist = torch.norm(query.float().unsqueeze(2) - key.float().unsqueeze(1), p=2, dim=-1)
            val, NN_key_idx = dist.squeeze(0).min(-1)
            valid_mask = val < dist_thresh
            query_NN_key_idx[valid_mask] = NN_key_idx[valid_mask]
            return query_NN_key_idx

        else:
            repr_query_idx = farthest_point_sample(query, fps_num)
            repr_query = index_points(query, repr_query_idx)
            
            dist = torch.norm(repr_query.float().unsqueeze(2) - key.float().unsqueeze(1), p=2, dim=-1)
            
            val, NN_key_idx = dist.squeeze(0).min(-1)
            valid_mask = val < dist_thresh

            
            query_group_idx = query_ball_point(radius, max_cluster_samples, query.float(), repr_query.float())
            query_group_idx = query_group_idx.squeeze(0)
            
            # ipdb.set_trace()
            # tmp = query_group_idx.reshape(-1).unique()

            expanded_NN_key_idx = NN_key_idx.unsqueeze(1).repeat(1, max_cluster_samples).reshape(-1)
            expanded_valid_mask = valid_mask.unsqueeze(1).repeat(1, max_cluster_samples).reshape(-1)
            query_group_idx = query_group_idx.reshape(-1)

            # select valid NN key assign
            valid_NN_key_idx = expanded_NN_key_idx[expanded_valid_mask]
            valid_query_group_idx = query_group_idx[expanded_valid_mask]
            
            query_NN_key_idx[valid_query_group_idx] = valid_NN_key_idx

            return query_NN_key_idx

    def fps_NN_fast(self, query, key, fps_num, radius, max_cluster_samples, dist_thresh):
        """Efficient NN search for huge amounts of query and key (suppose queries are redundant)
        
            Behaivor:
            1. apply FPS on query and generate representative queries (repr_query)
            2. calculate repr_queries' distances with all keys, and get the NN key
            3. apply ball query to assign the same NN key with the group center 

        """

        query_NN_key_idx = torch.zeros_like(query[:, 0]).long() - 1
        query = query[:, 1:].unsqueeze(0)
        key = key[:, 1:].unsqueeze(0)

        if query.shape[1] <= fps_num:
            dist = torch.norm(query.float().unsqueeze(2) - key.float().unsqueeze(1), p=2, dim=-1)
            val, NN_key_idx = dist.squeeze(0).min(-1)
            valid_mask = val < dist_thresh
            query_NN_key_idx[valid_mask] = NN_key_idx[valid_mask]
            return query_NN_key_idx

        else:
            repr_query_idx = furthest_point_sample(query.float().contiguous(), fps_num)
            repr_query = query[:, repr_query_idx[0].long(), :]
            # repr_query = gather_points(query.permute(0,2,1).float().contiguous(), repr_query_idx)
            
            dist = torch.norm(repr_query.float().unsqueeze(2) - key.float().unsqueeze(1), p=2, dim=-1)
            
            val, NN_key_idx = dist.squeeze(0).min(-1)
            valid_mask = val < dist_thresh

            query_group_idx = ball_query(0, radius, max_cluster_samples, query.float(), repr_query.float())
            query_group_idx = query_group_idx.squeeze(0).long()
            
            # ipdb.set_trace()
            # tmp = query_group_idx.reshape(-1).unique()

            expanded_NN_key_idx = NN_key_idx.unsqueeze(1).repeat(1, max_cluster_samples).reshape(-1)
            expanded_valid_mask = valid_mask.unsqueeze(1).repeat(1, max_cluster_samples).reshape(-1)
            query_group_idx = query_group_idx.reshape(-1)

            # select valid NN key assign
            valid_NN_key_idx = expanded_NN_key_idx[expanded_valid_mask]
            valid_query_group_idx = query_group_idx[expanded_valid_mask]
            
            query_NN_key_idx[valid_query_group_idx] = valid_NN_key_idx

            return query_NN_key_idx    

    def grouped_sparse_conv(self, voxel_3D, voxel_2D, syn_mix_3D, syn_mix_2D, stage_id, fps_num, radius, max_cluster_samples, dist_thresh):
        """Cascade block for multi-modal branch

            Behaivor: 
            1. gather 3D, 3D-2D mixed and 2D, 2D-3D mixed parts from voxel_3D and voxel_2D, respectively;
            2. aggregate 3D-2D mixed and 2D-3D mixed together;
            3. conduct grouped conv for 3 branches independently;
            4. scatter features from 3 branches to a unified voxel system;
            5. perform unified sparse convolution  

        """
        ## step 1,2: gather and aggregate
        # gather only 3D & 2D voxels
        only_3D_mask = voxel_3D.indices[:, 1] == 0
        only_2D_mask = voxel_2D.indices[:, 1] == 0

        # deal with the corner case of no only 2D voxels for each batch
        voxel_only_2D_indices = voxel_2D.indices[only_2D_mask]
        voxel_only_2D_features = voxel_2D.features[only_2D_mask]
        voxel_only_2D_indices, voxel_only_2D_features = self.pad_missing_batch_id(
            voxel_only_2D_indices, voxel_only_2D_features, voxel_3D.batch_size,
            voxel_2D.indices[0], voxel_2D.features[0]
        )    


        # # select nearest 3D voxels for each only 2D voxels
        voxel_only_2D_indices_tmp = voxel_only_2D_indices[:, [0,2,3,4]]
        voxel_3D_indices = voxel_3D.indices[:, [0,2,3,4]]
        voxel_only_2D_nn_3D_idx = torch.zeros_like(voxel_only_2D_indices_tmp[:, 0]).long() - 1
        B = len(torch.unique(voxel_3D_indices[:, 0]))
        base = 0
        for batch_id in range(B):
            this_batch_mask_2D = voxel_only_2D_indices_tmp[:, 0] == batch_id
            this_batch_mask_3D = voxel_3D_indices[:, 0] == batch_id
            
            this_voxel_only_2D_indices = voxel_only_2D_indices_tmp[this_batch_mask_2D]
            this_voxel_3D_indices = voxel_3D_indices[this_batch_mask_3D]
            # fps_num, radius, max_cluster_samples, dist_thresh = 2048, 6, 200, 13.3
            this_voxel_only_2D_NN_3D_idx = self.fps_NN_fast(this_voxel_only_2D_indices, this_voxel_3D_indices, \
                                                            fps_num, radius, max_cluster_samples, dist_thresh)
            non_empty_mask = (this_voxel_only_2D_NN_3D_idx != -1)
            this_voxel_only_2D_NN_3D_idx[non_empty_mask] = this_voxel_only_2D_NN_3D_idx[non_empty_mask] + base
            voxel_only_2D_nn_3D_idx[this_batch_mask_2D] =  this_voxel_only_2D_NN_3D_idx
            
            base = this_batch_mask_3D.sum()

        # use a randomly initialized embedding for uncovered 2D features
        dummy_embedding = torch.rand(1, voxel_3D.features.shape[1]).to(voxel_3D.features.device)
        cross_gating_in = torch.cat([voxel_3D.features, dummy_embedding], dim=0)    
        cross_gating = self.cross_gate_control[stage_id](cross_gating_in)

        # # deactivate unassigned voxel_2Ds
        voxel_only_2D_features = cross_gating[voxel_only_2D_nn_3D_idx] * voxel_only_2D_features
        
        # Notice: voxel_3D & voxel_2D indices: (batch_id, mix_id, x, y, z), mix_id should be removed before performing spconv3D
        voxel_only_3D = spconv.SparseConvTensor(voxel_3D.features[only_3D_mask],
                                                voxel_3D.indices[only_3D_mask][:, [0,2,3,4]],
                                                voxel_3D.spatial_shape,
                                                voxel_3D.batch_size)

        voxel_only_2D = spconv.SparseConvTensor(voxel_only_2D_features,
                                                voxel_only_2D_indices[:, [0,2,3,4]],
                                                voxel_2D.spatial_shape,
                                                voxel_2D.batch_size)

        # gather 3D+2D mixed voxels from voxel_3D and voxel_2D, and aggregation
        voxel_3D_2D_mixed_feat = voxel_3D.features[syn_mix_3D]
        voxel_2D_3D_mixed_feat = voxel_2D.features[syn_mix_2D]
        assert voxel_3D_2D_mixed_feat.shape[0] == voxel_2D_3D_mixed_feat.shape[0]

        # add a gating mechanism in the mixed features voxels to control noisy 2D voxel features
        gating = self.gate_control[stage_id](voxel_3D_2D_mixed_feat)
        voxel_2D_3D_mixed_feat = gating * voxel_2D_3D_mixed_feat

        

        voxel_mixed_feat = torch.cat([voxel_3D_2D_mixed_feat, voxel_2D_3D_mixed_feat], dim=-1)
        voxel_mixed_indices = voxel_2D.indices[syn_mix_2D]
        voxel_mixed_indices, voxel_mixed_feat = self.pad_missing_batch_id(
            voxel_mixed_indices, voxel_mixed_feat, voxel_3D.batch_size,
            voxel_3D.indices[0], torch.cat([voxel_3D.features[0], voxel_2D.features[0]], dim=-1))

        voxel_mixed = spconv.SparseConvTensor(voxel_mixed_feat, voxel_mixed_indices[:, [0,2,3,4]],
                                            voxel_2D.spatial_shape, voxel_2D.batch_size)

        ## step 3: conduct grouped conv
        stage_name = f'stage_{stage_id + 1}'
        
        voxel_only_3D = getattr(self.grouped_sp_conv_blocks_3D, stage_name)(voxel_only_3D)
        voxel_only_2D = voxel_only_2D.replace_feature(F.pad(voxel_only_2D.features, (self.in_channels_3D[stage_id], 0), mode='constant', value=0))
        voxel_only_3D = voxel_only_3D.replace_feature(F.pad(voxel_only_3D.features, (0, 64), mode='constant', value=0))
        
        assert voxel_only_2D.features.shape[-1] == voxel_only_3D.features.shape[-1]
        assert voxel_only_2D.features.shape[-1] == voxel_mixed.features.shape[-1]

        unified_voxel_feat = torch.cat([voxel_only_3D.features, voxel_only_2D.features, 
                                                            voxel_mixed.features], dim=0)
        unified_voxel_coors = torch.cat([voxel_only_3D.indices, voxel_only_2D.indices,
                                                            voxel_mixed.indices], dim=0)
        unified_voxel = spconv.SparseConvTensor(unified_voxel_feat, unified_voxel_coors,
                                            voxel_mixed.spatial_shape, voxel_2D.batch_size)

        ## step 5: conduct unified sparse conv 
        voxel_out = getattr(self.aggregation_blocks, stage_name)(unified_voxel)

        return voxel_out
                        
    # @auto_fp16(apply_to=('voxel_features', ))
    def forward(self, voxel_3D_list, voxel_2D_list, syn_mix_3D_list, syn_mix_2D_list, fps_num_list, radius_list, max_cluster_samples_list, dist_thresh_list):
        
        stage_outs = []
        for stage_id in range(len(voxel_2D_list)):

            stage_name = f'stage_{stage_id + 1}'
            voxel_3D = voxel_3D_list[stage_id]
            voxel_2D = voxel_2D_list[stage_id]
            syn_mix_3D = syn_mix_3D_list[stage_id]
            syn_mix_2D = syn_mix_2D_list[stage_id]

            fps_num = fps_num_list[stage_id]
            radius = radius_list[stage_id]
            max_cluster_samples = max_cluster_samples_list[stage_id]
            dist_thresh = dist_thresh_list[stage_id]
            
            voxel_stage_out = self.grouped_sparse_conv(voxel_3D, voxel_2D, syn_mix_3D, syn_mix_2D, stage_id, fps_num, radius, max_cluster_samples, dist_thresh)
            
            if stage_id == 0:
                voxel_stage_out_ds = getattr(self.downscale_blocks, stage_name)(voxel_stage_out)
                stage_outs.append(voxel_stage_out_ds)
            else:
                voxel_stage_out = Fsp.sparse_add(voxel_stage_out, stage_outs[stage_id - 1])
                voxel_stage_out_ds = getattr(self.downscale_blocks, stage_name)(voxel_stage_out)
                stage_outs.append(voxel_stage_out_ds)

        return stage_outs
    
    