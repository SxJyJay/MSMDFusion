import mmcv
import torch
import time
from mmcv.parallel import DataContainer as DC
from mmcv.runner import force_fp32
from os import path as osp
from torch import nn as nn
from torch.nn import functional as F

from mmdet3d.core import (Box3DMode, Coord3DMode, bbox3d2result,
                          merge_aug_bboxes_3d, show_result)
from mmdet3d.ops import Voxelization
from mmdet.core import multi_apply
# from mmdet3d.ops import spconv as spconv
import spconv.pytorch as spconv
from mmdet.models import DETECTORS
from .. import builder
from .mvx_two_stage import MVXTwoStageDetector

from numba import jit
import numpy as np

from .tools import rotate_box, get_feats_in_rectangle
import os


@jit(nopython=True)
def type_assign(batch_3D_val, batch_2D_val, batch_3Dtype_idf, batch_2Dtype_idf):

    N, M = batch_3D_val.shape[-1], batch_2D_val.shape[-1]
    ii, jj = 0, 0

    # 0, 1 within batch_3D/2Dtype_idf to indicate only 3D/2D and 2D+3D mixed
    while (ii < N) and (jj < M):
        if batch_3D_val[ii] < batch_2D_val[jj]:
            ii += 1
        elif batch_3D_val[ii] == batch_2D_val[jj]:
            batch_3Dtype_idf[ii] = 1
            batch_2Dtype_idf[jj] = 1
            ii += 1
            jj += 1
        else:
            jj += 1

    return batch_3Dtype_idf, batch_2Dtype_idf

class SPPModule(nn.Module):
    def __init__(self, **kwargs):
        super(SPPModule, self).__init__(**kwargs)

        self.conv1x1 = nn.Sequential(
            nn.Conv2d(384+256, 256, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(256, eps=0.001, momentum=0.01, affine=True, track_running_stats=True),
            nn.ReLU(),
        )
        self.conv3x3 = nn.Sequential(
            nn.Conv2d(384+256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256, eps=0.001, momentum=0.01, affine=True, track_running_stats=True),
            nn.ReLU()
        )
        self.dilated_conv3x3_rate6 = nn.Sequential(
            nn.Conv2d(384+256, 256, kernel_size=3, stride=1, padding=6, dilation=6, bias=False),
            nn.BatchNorm2d(256, eps=0.001, momentum=0.01, affine=True, track_running_stats=True),
            nn.ReLU()
        )
        self.dilated_conv3x3_rate12 = nn.Sequential(
            nn.Conv2d(384+256, 256, kernel_size=3, stride=1, padding=12, dilation=12, bias=False),
            nn.BatchNorm2d(256, eps=0.001, momentum=0.01, affine=True, track_running_stats=True),
            nn.ReLU()
        )
        # self.dilated_conv3x3_rate18 = nn.Sequential(
        #     nn.Conv2d(512+256, 256, kernel_size=3, stride=1, padding=12, dilation=12, bias=False),
        #     nn.BatchNorm2d(256, eps=0.001, momentum=0.01, affine=True, track_running_stats=True),
        #     nn.ReLU()
        # )
        self.fuse = nn.Sequential(
            nn.Conv2d(256*4, 256, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(256, eps=0.001, momentum=0.01, affine=True, track_running_stats=True),
            nn.ReLU(),
        )

    def forward(self, x):
        x1 = self.conv1x1(x)
        x2 = self.conv3x3(x)
        # x2 = self.dilated_conv3x3_rate18(x)
        x3 = self.dilated_conv3x3_rate6(x)
        x4 = self.dilated_conv3x3_rate12(x)
        ret = self.fuse(torch.cat([x1, x2, x3, x4], dim=1))
        
        return ret

@DETECTORS.register_module()
class MSMDFusionDetector(MVXTwoStageDetector):
    """Base class of Multi-modality VoxelNet."""

    def __init__(self, **kwargs):
        super(MSMDFusionDetector, self).__init__(**kwargs)
        
        self.freeze_img = kwargs.get('freeze_img', True)
        self.spatial_shapes = kwargs.get('spatial_shapes')
        self.downscale_factors = kwargs.get('downscale_factors')
        self.fps_num_list = kwargs.get('fps_num_list')
        self.radius_list = kwargs.get('radius_list')
        self.max_cluster_samples_list = kwargs.get('max_cluster_samples_list')
        self.dist_thresh_list = kwargs.get('dist_thresh_list') 
        
         # channel compression for ResNet50
        self.conv1x1_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(256+1, 49, kernel_size=5, stride=1, padding=2, bias=False),
                nn.BatchNorm2d(49, eps=0.001, momentum=0.01, affine=True, track_running_stats=True),
                nn.ReLU(),
            ),
            nn.Sequential(
                nn.Conv2d(256+1, 49, kernel_size=5, stride=1, padding=2, bias=False),
                nn.BatchNorm2d(49, eps=0.001, momentum=0.01, affine=True, track_running_stats=True),
                nn.ReLU(),
            ),
            nn.Sequential(
                nn.Conv2d(256+1, 49, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(49, eps=0.001, momentum=0.01, affine=True, track_running_stats=True),
                nn.ReLU(),
            ),
        ])
        
        self.score_net = nn.Sequential(
            nn.Linear(50+16, 1),
            nn.ReLU()
        )
        
        self.bev_fusion = SPPModule()
        self.init_weights(pretrained=kwargs.get('pretrained', None))

    def init_weights(self, pretrained=None):
        """Initialize model weights."""
        super(MSMDFusionDetector, self).init_weights(pretrained)

        if self.freeze_img:
            if self.with_img_backbone:
                for param in self.img_backbone.parameters():
                    param.requires_grad = False
            if self.with_img_neck:
                for param in self.img_neck.parameters():
                    param.requires_grad = False

    def extract_img_feat(self, img, img_metas):
        """Extract features of images."""
        if self.with_img_backbone and img is not None:
            
            input_shape = img.shape[-2:]
            # update real input shape of each single img
            for img_meta in img_metas:
                img_meta.update(input_shape=input_shape)

            if img.dim() == 5 and img.size(0) == 1:
                img.squeeze_(0)
            elif img.dim() == 5 and img.size(0) > 1:
                B, N, C, H, W = img.size()
                img = img.view(B * N, C, H, W)
            img_feats = self.img_backbone(img.float())
        else:
            return None
        
        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)
        
        return img_feats

    def get_foreground2D(self, img_feats, img_metas):
        """Extract 2D foreground area features from high-dimensional img features
        
            Return batch_fg_pcd_cams = [
                fg_pcd_cams (sample_1) = (
                    <cam1, cam2, ..., cam6>, dim
                )
                fg_pcd_cams (sample_2) = (
                    <cam1, cam2, ..., cam6>, dim
                )
            ]
        """
        downscale_factor =  img_feats.shape[-1] / img_metas[0]['input_shape'][-1]
        B = len(img_metas)
        BN, C, H, W = img_feats.shape
        N = int(BN / B)
        img_feats = img_feats.view(B, N, C, H, W)
        
        batch_fg_pcd_cams = []
        batch_score_in_cams = []
        for sample_idx in range(B):
            
            
            fg_info = img_metas[sample_idx]['foreground2D_info']
            img_feat = img_feats[sample_idx]
            lidar2img = img_metas[sample_idx]['lidar2img']
            device = img_feats[0].device
            
            fg_pcd_cams = []
            score_in_cams = []
            for view_idx in range(N):
                
                img_feat_per_camera = img_feat[view_idx]
                fg_pxl = fg_info['fg_pixels'][view_idx]
                fg_pts = fg_info['fg_points'][view_idx]
                
                sensor_trans = torch.from_numpy(lidar2img[view_idx]).to(device).view(1, -1).contiguous() #(1, 16)

                fg_depth = torch.from_numpy(fg_pxl[:,2]).to(device)
                fg_feat_pxl = fg_pxl * downscale_factor
                fg_feat_pxl = torch.from_numpy(fg_feat_pxl).to(device).long()
                coord_w, coord_h = fg_feat_pxl[:, 0], fg_feat_pxl[:, 1]
                
                fg_feat = img_feat_per_camera.permute(1,2,0)[coord_h, coord_w]
                
                score_in = torch.cat([fg_feat, fg_depth.unsqueeze(1)], dim=1)
                
                sensor_trans = sensor_trans.repeat(score_in.shape[0], 1).float()
                score_in = torch.cat([score_in, sensor_trans], dim=1)
                score_in_cams.append(score_in)
                
                fg_pcd = torch.cat([fg_pts.tensor.to(device), fg_feat], dim=1)
                fg_pcd_cams.append(fg_pcd)

            batch_fg_pcd_cams.append(torch.cat(fg_pcd_cams, dim=0))
            batch_score_in_cams.append(torch.cat(score_in_cams, dim=0))

        all_fg_pcd_cams = torch.cat(batch_fg_pcd_cams, dim=0)
        all_score_in_cams = torch.cat(batch_score_in_cams, dim=0)

        all_scores = self.score_net(all_score_in_cams)
        all_fg_pcd_cams[:, -C:] = all_fg_pcd_cams[:, -C:] * all_scores

        # only suit for bs = 2
        lenA = batch_fg_pcd_cams[0].shape[0]
        batch_fg_pcd_cams[0][:, -C:] = all_fg_pcd_cams[:lenA, -C:]
        if B == 2:
            batch_fg_pcd_cams[1][:, -C:] = all_fg_pcd_cams[lenA:, -C:]
        
        return batch_fg_pcd_cams


    def voxelize_fg_pcd(self, batch_fg_pcd_cams, downscale_factor=1):
        """aggregate foreground points from multiple cameras to a whole lidar scene
           Apply voxelization on the foreground points
        
        """
        # apply voxelization
        fg_voxels, fg_num_points, fg_coors = self.voxelize(batch_fg_pcd_cams, downscale_factor)

        return fg_voxels, fg_num_points, fg_coors

    def voxel_modality_split(self, voxel_3D, voxel_2D, B):
        """Split voxels into 3 groups: only 3D, only 2D, 3D+2D
        
            Input: SparseConvTensor, SparseConvTensor
            
            Return: SparseConvTensor, SparseConvTensor (with mix identifier saved in the 1st dim of indices)
        
        """

        coord_3D = voxel_3D.indices
        coord_2D = voxel_2D.indices
        coord_3D_mix, coord_2D_mix = [], []
        syn_mix_3D, syn_mix_2D = [], []
        last_batch_3D_len, last_batch_2D_len = 0, 0
        for i in range(B):
            batch_3D_mask = coord_3D[:, 0] == i
            batch_2D_mask = coord_2D[:, 0] == i
            batch_coord_3D = coord_3D[batch_3D_mask][:, 1:]
            batch_coord_2D = coord_2D[batch_2D_mask][:, 1:]
            
            batch_coord3D_xyz = batch_coord_3D[:, 0]*1e6 + batch_coord_3D[:, 1]*1e3 + batch_coord_3D[:, 2]
            batch_coord2D_xyz = batch_coord_2D[:, 0]*1e6 + batch_coord_2D[:, 1]*1e3 + batch_coord_2D[:, 2]

            batch_3D_val, batch_3D_ind = torch.sort(batch_coord3D_xyz, dim=-1)
            batch_2D_val, batch_2D_ind = torch.sort(batch_coord2D_xyz, dim=-1)

            batch_3Dtype_idf = torch.zeros_like(batch_3D_val)
            batch_2Dtype_idf = torch.zeros_like(batch_2D_val)
            batch_3Dmix_idf = torch.zeros_like(batch_3D_val)
            batch_2Dmix_idf = torch.zeros_like(batch_2D_val)

            device = batch_3D_val.device
            # 0, 1 within batch_3D/2Dtype_idf to indicate only 3D/2D and 2D+3D mixed
            batch_3Dtype_idf, batch_2Dtype_idf = type_assign(
                batch_3D_val.cpu().numpy(), batch_2D_val.cpu().numpy(), 
                batch_3Dtype_idf.cpu().numpy(), batch_2Dtype_idf.cpu().numpy()
            )
            batch_3Dtype_idf = torch.from_numpy(batch_3Dtype_idf).to(device)
            batch_2Dtype_idf = torch.from_numpy(batch_2Dtype_idf).to(device)

            # save mix voxel indices of 3D and 2D voxels for synchronization & remake index with non-empty voxels from previous batch padded 
            batch_orign_voxel3D_inds = batch_3D_ind[torch.where(batch_3Dtype_idf)[0]]
            batch_orign_voxel2D_inds = batch_2D_ind[torch.where(batch_2Dtype_idf)[0]]
            batch_orign_voxel3D_inds_padded = batch_orign_voxel3D_inds + last_batch_3D_len
            batch_orign_voxel2D_inds_padded = batch_orign_voxel2D_inds + last_batch_2D_len

            syn_mix_3D.append(batch_orign_voxel3D_inds_padded)
            syn_mix_2D.append(batch_orign_voxel2D_inds_padded)

            batch_3Dmix_idf = batch_3Dmix_idf.scatter(dim=-1, index=batch_3D_ind, src=batch_3Dtype_idf)
            batch_2Dmix_idf = batch_2Dmix_idf.scatter(dim=-1, index=batch_2D_ind, src=batch_2Dtype_idf)

            # add mix identifier before xyz, after batch identifier i.e., (mix_id, x, y, z)
            batch_coord_3D_mix = torch.cat([batch_3Dmix_idf.unsqueeze(1), batch_coord_3D], dim=-1)
            batch_coord_2D_mix = torch.cat([batch_2Dmix_idf.unsqueeze(1), batch_coord_2D], dim=-1)
            
            # add batch identifier at the very begining i.e., (batch_id, mix_id, x, y, z)
            batch_coord_3D_mix_pad = F.pad(batch_coord_3D_mix, (1, 0), mode='constant', value=i)
            batch_coord_2D_mix_pad = F.pad(batch_coord_2D_mix, (1, 0), mode='constant', value=i)
            coord_3D_mix.append(batch_coord_3D_mix_pad)
            coord_2D_mix.append(batch_coord_2D_mix_pad)

            last_batch_3D_len = batch_coord_3D.shape[0]
            last_batch_2D_len = batch_coord_2D.shape[0]


        coord_3D_mix = torch.cat(coord_3D_mix, dim=0)
        coord_2D_mix = torch.cat(coord_2D_mix, dim=0)
        syn_mix_3D = torch.cat(syn_mix_3D, dim=0)
        syn_mix_2D = torch.cat(syn_mix_2D, dim=0)

        voxel_3D.indices = coord_3D_mix.int()
        voxel_2D.indices = coord_2D_mix.int()
    
        return voxel_3D, voxel_2D, syn_mix_3D, syn_mix_2D

    def channel_compression(self, feat_list):
        
        out_feat_list = []
        for i in range(3):
            out_feat_list.append(self.conv1x1_blocks[i](feat_list[i]))

        return out_feat_list

    def depth_aware_channel_compression(self, feat_list, img_metas):
        
        # generate sparse depth map
        B = len(img_metas)
        cam_num = 6
        device = feat_list[0].device
        H, W = img_metas[0]['pad_shape'][:2]
        canvas = torch.zeros(B, cam_num, H, W).to(device)

        for i in range(B):
            fg_real_pixels = img_metas[i]['foreground2D_info']['fg_real_pixels']
            fg_pixels = img_metas[i]['foreground2D_info']['fg_pixels']
            
            for j in range(cam_num):
                fg_real_pxl = torch.from_numpy(fg_real_pixels[j]).to(device)
                fg_pxl = torch.from_numpy(fg_pixels[j]).to(device)
                coors = fg_real_pxl[:,:2].long()
                depth = fg_real_pxl[:,2]

                # make an empty canvas
                index = (coors[:,1], coors[:,0])
                canvas[i,j].index_put_(index, depth)
        
        canvas = canvas.view(-1, 1, H, W)        


        out_feat_list = []
        for i in range(3):
            img_feat = feat_list[i]
            h, w = img_feat.shape[-2:]
            sp_depth_map = F.interpolate(canvas, (h,w), mode='bilinear')
            img_feat_depth_aware = torch.cat([img_feat, sp_depth_map], dim=1)
            out_feat_list.append(self.conv1x1_blocks[i](img_feat_depth_aware))

        return out_feat_list

    def fetch_2D_voxels(self, img_feat, img_metas, voxel_size, downscale_factor, B):
        """Fetch 2D foreground points from one image feature and make voxelization"""
        batch_fg_pcd_cams = self.get_foreground2D(img_feat, img_metas)

        # check validity of foreground2D points, if num_pts is 0, then zero padding
        for i in range(B):
            num_pts, dim = batch_fg_pcd_cams[i].shape
            device = batch_fg_pcd_cams[i].device
            if num_pts == 0:
                batch_fg_pcd_cams[i] = torch.zeros(100, dim).to(device)
            
        fg_voxels, fg_num_points, fg_coors = self.voxelize_fg_pcd(batch_fg_pcd_cams, downscale_factor)

        # set feat dim
        feat_dim = fg_voxels.shape[-1]
        self.pts_voxel_encoder.num_features = feat_dim
        fg_voxel_features = self.pts_voxel_encoder(fg_voxels, fg_num_points, fg_coors)
        xyz_normalizer = torch.Tensor([13.5, 13.5, 2.0]).to(device)
        fg_voxel_features[:,:3] = fg_voxel_features[:,:3] / xyz_normalizer[None, :]

        fg_sp_tensor = spconv.SparseConvTensor(fg_voxel_features, fg_coors, voxel_size, B)

        return fg_sp_tensor

    def check_range_validity(self, coors, spatial_shape):
        z_bound, x_bound, y_bound = spatial_shape
        mask = (coors[:,2] >= 0) * (coors[:,2] < x_bound) * (coors[:,3] >= 0) * (coors[:,3] < y_bound)
        return mask

    def extract_multiscale_voxel_feat(self, img_feats, encode_features, img_metas, spatial_shapes, downscale_factors, batch_size):
        """Extract multi-scale modality-specific voxel features"""
        img_feats = self.depth_aware_channel_compression(img_feats, img_metas)
        img_feat_list = [img_feats[0]]
        img_feat_list.extend(img_feats)

        voxel_3D_list, voxel_2D_list = [], []
        syn_mix_3D_list, syn_mix_2D_list = [], []

        for i in range(4):
            voxel_2D = self.fetch_2D_voxels(img_feat_list[i], img_metas, spatial_shapes[i], downscale_factors[i], batch_size)
            voxel_3D = encode_features[i]
            voxel_3D_mix, voxel_2D_mix, syn_mix_3D, syn_mix_2D = self.voxel_modality_split(voxel_3D, voxel_2D, batch_size)
            voxel_2D_list.append(voxel_2D_mix)
            voxel_3D_list.append(voxel_3D_mix)
            syn_mix_3D_list.append(syn_mix_3D)
            syn_mix_2D_list.append(syn_mix_2D)

        return voxel_3D_list, voxel_2D_list, syn_mix_3D_list, syn_mix_2D_list


    def extract_pts_feat(self, pts, img_feats, img_metas):
        """Extract features of points."""
        if not self.with_pts_bbox:
            return None
        
        voxels, num_points, coors = self.voxelize(pts)

        voxel_features = self.pts_voxel_encoder(voxels, num_points, coors)

        batch_size = coors[-1, 0] + 1
        
        x, encode_features = self.pts_middle_encoder(voxel_features, coors, batch_size)

        voxel_3D_list, voxel_2D_list, syn_mix_3D_list, syn_mix_2D_list = self.extract_multiscale_voxel_feat(img_feats, encode_features,\
                    img_metas, self.spatial_shapes, self.downscale_factors, batch_size)
            
        stage_outs = self.multimodal_middle_encoder(voxel_3D_list, voxel_2D_list, syn_mix_3D_list, syn_mix_2D_list, self.fps_num_list,\
                                                     self.radius_list, self.max_cluster_samples_list, self.dist_thresh_list)

        ## Merge multimodal voxels and pure 3D voxels
        multimodal_out_dense = stage_outs[-1].dense()
        N, C, D, H, W = multimodal_out_dense.shape
        x_mm = multimodal_out_dense.view(N, C*D, H, W)
        
        x = self.bev_fusion(torch.cat([x, x_mm], dim=1))

        x = self.pts_backbone(x)
        
        if self.with_pts_neck:
            x = self.pts_neck(x)
        
        return x


    def extract_feat(self, points, img, img_metas):
        """Extract features from images and points."""
        img_feats = self.extract_img_feat(img, img_metas)
        pts_feats = self.extract_pts_feat(points, img_feats, img_metas)
        return (img_feats, pts_feats)


    @torch.no_grad()
    @force_fp32()
    def voxelize(self, points, downscale_factor=1.0):
        """Apply dynamic voxelization to points.

        Args:
            points (list[torch.Tensor]): Points of each sample.

        Returns:
            tuple[torch.Tensor]: Concatenated points, number of points
                per voxel, and coordinates.
        """
        # reset voxel size to [0.075, 0.075, 0.2]
        self.pts_voxel_layer.voxel_size = [0.075, 0.075, 0.2]

        voxels, coors, num_points = [], [], []
        self.pts_voxel_layer.voxel_size = list(map(lambda x: x * downscale_factor, self.pts_voxel_layer.voxel_size))
        for res in points:
            res_voxels, res_coors, res_num_points = self.pts_voxel_layer(res)
            voxels.append(res_voxels)
            coors.append(res_coors)
            num_points.append(res_num_points)
        voxels = torch.cat(voxels, dim=0)
        num_points = torch.cat(num_points, dim=0)
        coors_batch = []
        for i, coor in enumerate(coors):
            coor_pad = F.pad(coor, (1, 0), mode='constant', value=i)
            coors_batch.append(coor_pad)
        coors_batch = torch.cat(coors_batch, dim=0)
        return voxels, num_points, coors_batch

    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img=None,
                      proposals=None,
                      gt_bboxes_ignore=None):
        """Forward training function.

        Args:
            points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[torch.Tensor], optional): Ground truth labels
                of 3D boxes. Defaults to None.
            gt_labels (list[torch.Tensor], optional): Ground truth labels
                of 2D boxes in images. Defaults to None.
            gt_bboxes (list[torch.Tensor], optional): Ground truth 2D boxes in
                images. Defaults to None.
            img (torch.Tensor optional): Images of each sample with shape
                (N, C, H, W). Defaults to None.
            proposals ([list[torch.Tensor], optional): Predicted proposals
                used for training Fast RCNN. Defaults to None.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.

        Returns:
            dict: Losses of different branches.
        """
        
        img_feats, pts_feats = self.extract_feat(
            points, img=img, img_metas=img_metas)
        losses = dict()
        
        if pts_feats:
            losses_pts = self.forward_pts_train(pts_feats, img_feats, gt_bboxes_3d,
                                                gt_labels_3d, img_metas,
                                                gt_bboxes_ignore)
            losses.update(losses_pts)
        
        if img_feats:
            losses_img = self.forward_img_train(
                img_feats,
                img_metas=img_metas,
                gt_bboxes=gt_bboxes,
                gt_labels=gt_labels,
                gt_bboxes_ignore=gt_bboxes_ignore,
                proposals=proposals)
            losses.update(losses_img)
        return losses

    def forward_pts_train(self,
                          pts_feats,
                          img_feats,
                          gt_bboxes_3d,
                          gt_labels_3d,
                          img_metas,
                          gt_bboxes_ignore=None):
        """Forward function for point cloud branch.

        Args:
            pts_feats (list[torch.Tensor]): Features of point cloud branch
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`]): Ground truth
                boxes for each sample.
            gt_labels_3d (list[torch.Tensor]): Ground truth labels for
                boxes of each sampole
            img_metas (list[dict]): Meta information of samples.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                boxes to be ignored. Defaults to None.

        Returns:
            dict: Losses of each branch.
        """
        outs = self.pts_bbox_head(pts_feats, img_feats, img_metas)
        loss_inputs = [gt_bboxes_3d, gt_labels_3d, outs]
        losses = self.pts_bbox_head.loss(*loss_inputs)
        return losses

    def simple_test_pts(self, x, x_img, img_metas, rescale=False):
        """Test function of point cloud branch."""
        outs = self.pts_bbox_head(x, x_img, img_metas)
        bbox_list = self.pts_bbox_head.get_bboxes(
            outs, img_metas, rescale=rescale)
        bbox_results = [
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in bbox_list
        ]
        return bbox_results

    def simple_test(self, points, img_metas, img=None, rescale=False):
        """Test function without augmentaiton."""
        # import time
        # torch.cuda.synchronize()
        # ckpt0 = time.time()
        
        img_feats, pts_feats = self.extract_feat(
            points, img=img, img_metas=img_metas)


        bbox_list = [dict() for i in range(len(img_metas))]
        
        if pts_feats and self.with_pts_bbox:
            bbox_pts = self.simple_test_pts(
                pts_feats, img_feats, img_metas, rescale=rescale)
            for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
                result_dict['pts_bbox'] = pts_bbox
        if img_feats and self.with_img_bbox:
            bbox_img = self.simple_test_img(
                img_feats, img_metas, rescale=rescale)
            for result_dict, img_bbox in zip(bbox_list, bbox_img):
                result_dict['img_bbox'] = img_bbox
        return bbox_list
