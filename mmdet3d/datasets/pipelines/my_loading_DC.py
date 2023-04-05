import mmcv
from mmcv.parallel import DataContainer as DC
import numpy as np
import torch
import os
from functools import partial
import ipdb

from mmdet3d.core.points import BasePoints, get_points_type
from mmdet.datasets.builder import PIPELINES
from mmdet.datasets.pipelines import LoadAnnotations

from mmdet3d.ops import scatter_v2
from scipy.sparse.csgraph import connected_components


def write_ply(points, out_filename):
        """Write points into ``ply`` format for meshlab visualization.

        Args:
            points (np.ndarray): Points in shape (N, dim).
            out_filename (str): Filename to be saved.
        """
        if points is None:
            return
        N = points.shape[0]
        fout = open(out_filename+"_real.obj", 'w')
        for i in range(N):
            if points.shape[1] == 6:
                c = points[i, 3:].astype(int)
                fout.write(
                    'v %f %f %f %d %d %d\n' %
                    (points[i, 0], points[i, 1], points[i, 2], c[0], c[1], c[2]))

            else:
                fout.write('v %f %f %f\n' %
                        (points[i, 0], points[i, 1], points[i, 2]))
        fout.close()


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
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids

@PIPELINES.register_module()
class LoadForeground2D_DC(object):
    """Load foreground info provided by 2D results
    
    The results will be added an item
        saved raw fg info = {
            
            'virtual_pixel_indices' = [
                np.array -> cam_1, ..., np.array -> cam_6
            ]
            'real_pixel_indices' = [
                np.array -> cam_1, ..., np.array -> cam_6
            ]
            note: shape above is (num_fg_pixels, 2), and range of indices is within original image scale 1600*900

            'virtual_points' = [
                np.array -> cam_1, ..., np.array -> cam_6
            ]
            'real_points' = [
                np.array -> cam_1, ..., np.array -> cam_6
            ]
            note: shape above is (num_fg_pixels, 3), virtual/real points are the corresponding points of foreground pixels in LiDAR system
        }
        results["foreground2D_info"] = {
            'fg_pixels' = [
                np.array -> cam_1, ..., np.array -> cam_6
            ]
            'fg_points' = [
                np.array -> cam_1, ..., np.array -> cam_6
            ]
        }

    """
    def __init__(self, dataset='NuScenesDataset', **kwargs):
        self.dataset = dataset

    def _instance_split(self, points, ins_ids, CCL_threshold=0.5, cluster_min_pts_num=20):
        """
        Split multi-instance points mistakenly merged to the same instance due to instance segmentation error
        """

        unique_ins_ids = np.unique(ins_ids)
        ins_num = len(unique_ins_ids)
        base = 0
        components_ids = np.zeros_like(ins_ids)
        abandon_mask = np.zeros_like(ins_ids) + 1

        for i in range(ins_num):
            
            batch_mask = (ins_ids == unique_ins_ids[i])
            this_points = points[batch_mask]
            dist_mat = this_points[:, None, :2] - this_points[None, :, :2] # only care about xy
            dist_mat = (dist_mat ** 2).sum(2) ** 0.5
            adj_mat = dist_mat < CCL_threshold
            c_inds = connected_components(adj_mat, directed=False)[1]
            c_inds = c_inds + base
            base = c_inds.max() + 1
            components_ids[batch_mask] = c_inds

            # discard clusters with points fewer than 20
            cluster_ids = np.unique(c_inds)
            for id in cluster_ids:
                cluster_mask = (c_inds == id)
                pts_num = np.sum(cluster_mask)
                if pts_num < cluster_min_pts_num:
                    abandon_mask[batch_mask] = np.zeros_like(cluster_mask)
                else:
                    continue
        
        return components_ids, abandon_mask.astype("bool")

    def _organize(self, fg_info):
        """
        Private function to select unique foreground pixels (and paired points)
        """
        if self.dataset == 'NuScenesDataset':
            cam_num = len(fg_info['virtual_pixel_indices'])
            fg_pixels, fg_points = [], []
            fg_instance_ids = []
            ins_centroids, ins_ids = [], []
            CCL_threshold = 0.5
            min_cluster_pts_num = 20
            for i in range(cam_num):
                
                fg_pixel_indices = np.concatenate((fg_info['virtual_pixel_indices'][i][:,:2], fg_info['real_pixel_indices'][i][:,:2]), axis=0)
                fg_ins_ids = np.concatenate((fg_info['virtual_pixel_indices'][i][:,2], fg_info['real_pixel_indices'][i][:,2]), axis=0)
                fg_points_set = np.concatenate((fg_info['virtual_points'][i], fg_info['real_points'][i]), axis=0)
                # fg_points_set = np.concatenate((virtual_pts_inds[np.random.randint(num_pts, size=count), :], fg_info['real_points'][i]), axis=0)
                
                fg_ins_ids, abandon_mask = self._instance_split(fg_points_set, fg_ins_ids, CCL_threshold, min_cluster_pts_num)

                # filter noise points
                if abandon_mask.sum() > 0:
                    fg_points_set = fg_points_set[abandon_mask]
                    fg_pixel_indices = fg_pixel_indices[abandon_mask]
                    fg_ins_ids = fg_ins_ids[abandon_mask]
                else:
                    pass

                fg_instance_ids.append(fg_ins_ids)
                fg_pixels.append(fg_pixel_indices)
                fg_points.append(fg_points_set)

                # calculate instance centroids and corresponding instance ids (for sweep instance merge purpose)
                all_ins_xyz, all_ins_ids = torch.from_numpy(fg_points_set[:,:3]), torch.from_numpy(fg_ins_ids)

                centroids_xyz, centroids_ins_ids, inv_ids = scatter_v2(all_ins_xyz, all_ins_ids, mode='avg', return_inv=True)
                ins_centroids.append(centroids_xyz)
                ins_ids.append(centroids_ins_ids)

            return dict(fg_pixels=fg_pixels, fg_points=fg_points, fg_instance_ids=fg_instance_ids, ins_centroids=ins_centroids, ins_ids=ins_ids)
        
        elif self.dataset == 'KittiDataset':
            if len(fg_info.keys()) == 4:
                fg_pixels = np.concatenate((fg_info['virtual_pixel_indices'], fg_info['real_pixel_indices']), axis=0)
                fg_points = np.concatenate((fg_info['virtual_points'], fg_info['real_points']), axis=0)
            else:
                fg_pixels = np.zeros((0,2))
                fg_points = np.zeros((0,6))
            
            return dict(fg_pixels=[fg_pixels], fg_points=[fg_points])

    def _make_point_class(self, fg_info):
        fg_points = fg_info['fg_points']
        cam_num = len(fg_points)
        point_class = get_points_type('LIDAR')
        for i in range(cam_num):
            fg_point = fg_points[i]
            fg_point = point_class(
                fg_point, points_dim=fg_point.shape[-1]
            )
            fg_points[i] = fg_point
        fg_info['fg_points'] = fg_points
        return fg_info

    def __call__(self, results):
        if self.dataset == 'NuScenesDataset':

            pts_filename = results['pts_filename']
            tokens = pts_filename.split('/')
            # might have bug when using absolute path. Just add ```fg_path='/'+fg_path```
            fg_pxl_path = os.path.join(*tokens[:-2], "FOREGROUND_MIXED_6NN_INS", tokens[-1]+'.pkl.npy')
            fg_pts_path = os.path.join(*tokens[:-2], "FOREGROUND_MIXED_6NN", tokens[-1]+'.pkl.npy')
            
            fg_info = np.load(fg_pxl_path, allow_pickle=True).item()
            ori_fg_info = np.load(fg_pts_path, allow_pickle=True).item()
            fg_info["virtual_points"] = ori_fg_info["virtual_points"]
            fg_info["real_points"] = ori_fg_info["real_points"]

            # # aggregate boundary info
            # fg_boundary_path = os.path.join(*tokens[:-2], "FOREGROUND_MIXED_BD_200", tokens[-1]+'.pkl.npy')
            # fg_boundary_info = np.load(fg_boundary_path, allow_pickle=True).item()
            # for key in fg_info:
            #     if 'real' in key:
            #         continue
            #     fg_info[key] = list(map(lambda x,y: np.concatenate((x,y), axis=0), fg_info[key], fg_boundary_info[key]))
                
            fg_info = self._organize(fg_info)

            results["foreground2D_info"] = fg_info

            return results
        
        elif self.dataset == 'KittiDataset':
            
            # find the saved foreground points & pixels file
            pts_filename = results['pts_filename']
            tokens = pts_filename.split('/')
            fg_path = os.path.join(*tokens[:-2], "virtual_1NN", tokens[-1].split('.')[0]+'.npy')
            fg_info = np.load(fg_path, allow_pickle=True).item()
            fg_info = self._organize(fg_info)

            # make mmdet3d point class, as Kitti doesn't have multi-sweep settings
            fg_info = self._make_point_class(fg_info)
            results['foreground2D_info'] = fg_info

            return results

        else:
            raise NotImplementedError("foreground2D info of {} dataset is unavailable!".format(self.dataset))

@PIPELINES.register_module()
class LoadForeground2DFromMultiSweeps_DC(object):
    """Load foreground info provided by 2D results from multiple sweeps
    
    """
    def __init__(self, dataset="NuScenesDataset", sweeps_num=10):
        self.dataset = dataset
        self.sweeps_num = sweeps_num
        self.sweep_iter_num = 0

    def _instance_split(self, points, ins_ids, CCL_threshold=0.5, cluster_min_pts_num=20):
        """
        Split multi-instance points mistakenly merged to the same instance due to instance segmentation error
        """

        unique_ins_ids = np.unique(ins_ids)
        ins_num = len(unique_ins_ids)
        base = 0
        components_ids = np.zeros_like(ins_ids)
        abandon_mask = np.zeros_like(ins_ids) + 1

        for i in range(ins_num):
            
            batch_mask = (ins_ids == unique_ins_ids[i])
            this_points = points[batch_mask]
            dist_mat = this_points[:, None, :2] - this_points[None, :, :2] # only care about xy
            dist_mat = (dist_mat ** 2).sum(2) ** 0.5
            adj_mat = dist_mat < CCL_threshold
            c_inds = connected_components(adj_mat, directed=False)[1]
            c_inds = c_inds + base
            base = c_inds.max() + 1
            components_ids[batch_mask] = c_inds

            # discard clusters with points fewer than 20
            cluster_ids = np.unique(c_inds)
            for id in cluster_ids:
                cluster_mask = (c_inds == id)
                pts_num = np.sum(cluster_mask)
                if pts_num < cluster_min_pts_num:
                    abandon_mask[batch_mask] = np.zeros_like(cluster_mask)
                else:
                    continue
        
        return components_ids, abandon_mask.astype("bool")



    def _organize(self, fg_info):
        """
        Private function to select unique foreground pixels (and paired points)
        """
        cam_num = len(fg_info['virtual_pixel_indices'])
        fg_pixels, fg_points = [], []
        fg_instance_ids = []
        ins_centroids, ins_ids = [], []
        CCL_threshold = 0.5
        min_cluster_pts_num = 20
        for i in range(cam_num):
            
            fg_pixel_indices = np.concatenate((fg_info['virtual_pixel_indices'][i][:,:2], fg_info['real_pixel_indices'][i][:,:2]), axis=0)
            fg_ins_ids = np.concatenate((fg_info['virtual_pixel_indices'][i][:,2], fg_info['real_pixel_indices'][i][:,2]), axis=0)

            fg_points_set = np.concatenate((fg_info['virtual_points'][i], fg_info['real_points'][i]), axis=0)
            # fg_points_set = np.concatenate((virtual_pts_inds[np.random.randint(num_pts, size=count), :], fg_info['real_points'][i]), axis=0)
            
            fg_ins_ids, abandon_mask = self._instance_split(fg_points_set, fg_ins_ids, CCL_threshold, min_cluster_pts_num)

            # filter noise points
            if abandon_mask.sum() > 0:
                fg_points_set = fg_points_set[abandon_mask]
                fg_pixel_indices = fg_pixel_indices[abandon_mask]
                fg_ins_ids = fg_ins_ids[abandon_mask]
            else:
                pass

            fg_instance_ids.append(fg_ins_ids)
            fg_pixels.append(fg_pixel_indices)
            fg_points.append(fg_points_set)

            # calculate instance centroids and corresponding instance ids (for sweep instance merge purpose)
            all_ins_xyz, all_ins_ids = torch.from_numpy(fg_points_set[:,:3]), torch.from_numpy(fg_ins_ids)
            centroids_xyz, centroids_ins_ids, inv_ids = scatter_v2(all_ins_xyz, all_ins_ids, mode='avg', return_inv=True)
            ins_centroids.append(centroids_xyz)
            ins_ids.append(centroids_ins_ids)

        return dict(fg_pixels=fg_pixels, fg_points=fg_points, fg_instance_ids=fg_instance_ids, ins_centroids=ins_centroids, ins_ids=ins_ids)

    def _merge_sweeps(self, fg_info, sweep_fg_info, sweep):
        """
            fg_info and sweep_fg_info: dict like : 
            {
                'fg_pixels' = [
                    np.array --> cam1, ..., np.array --> cam6
                ]
                'fg_points' = [
                    np.array --> cam1, ..., np.array --> cam6
                ]
            }
            sweep: dict of sweep info
        """
        fg_pixels, fg_points = fg_info['fg_pixels'], fg_info['fg_points']
        sweep_fg_pixels, sweep_fg_points = sweep_fg_info['fg_pixels'], sweep_fg_info['fg_points']
        fg_instance_ids, sweep_fg_instance_ids = fg_info['fg_instance_ids'], sweep_fg_info['fg_instance_ids']

        ins_centroids, ins_ids = fg_info['ins_centroids'], fg_info['ins_ids']
        sweep_ins_centroids, sweep_ins_ids = sweep_fg_info['ins_centroids'], sweep_fg_info['ins_ids']

        if len(sweep_fg_points) == len(fg_points):
            cam_num = len(fg_pixels)
            for cam_id in range(cam_num):
                if sweep_ins_centroids[cam_id].shape[0] == 0:
                    continue

                 # align sweep centroids with key-frame centroids
                sweep_ins_centroids[cam_id] = sweep_ins_centroids[cam_id] @ sweep['sensor2lidar_rotation'].T
                sweep_ins_centroids[cam_id] = sweep_ins_centroids[cam_id] + sweep['sensor2lidar_translation']
                sweep_fg_points[cam_id][:,:3] = sweep_fg_points[cam_id][:,:3] @ sweep['sensor2lidar_rotation'].T
                sweep_fg_points[cam_id][:,:3] = sweep_fg_points[cam_id][:,:3] + sweep['sensor2lidar_translation']

                if ins_centroids[cam_id].shape[0] == 0:
                    ins_centroids[cam_id] = torch.cat([ins_centroids[cam_id], sweep_ins_centroids[cam_id]],dim=0)
                    ins_ids[cam_id] = torch.cat([ins_ids[cam_id], sweep_ins_ids[cam_id]],dim=0)
                    fg_pixels[cam_id] = np.concatenate((fg_pixels[cam_id], sweep_fg_pixels[cam_id]), axis=0)
                    fg_points[cam_id] = np.concatenate((fg_points[cam_id], sweep_fg_points[cam_id]), axis=0)
                    fg_instance_ids[cam_id] = np.concatenate((fg_instance_ids[cam_id], sweep_fg_instance_ids[cam_id]), axis=0)
                    continue

                # find the nearest sweep centroids for every key sample centroids for merging instance pts
                dist = sweep_ins_centroids[cam_id].unsqueeze(1) - ins_centroids[cam_id].unsqueeze(0)
                dist = torch.norm(dist, dim=-1, p=2)
                
                # #debug for inspect centroid nn tracking effects
                # base_dir = '/share/home/jiaoyang/code/TransFusion/instance_tracking/'
                # if cam_id == 0:
                #     fg_points_cur_cam = fg_points[cam_id]
                #     fg_instance_ids_cur_cam = fg_instance_ids[cam_id]
                #     ins_ids_cur_cam = ins_ids[cam_id]
                #     tmp, pts_num = [], []
                #     for ii in range(len(ins_ids_cur_cam)):
                #         first_ins_id = ins_ids_cur_cam[ii]
                #         test_mask = (fg_instance_ids_cur_cam == first_ins_id.numpy())
                #         test_ins_pts = fg_points_cur_cam[test_mask][:,:3]
                        
                #         # # visualize pts to check tracking results
                #         # save_path = base_dir + "iter_{}".format(self.sweep_iter_num) + "_instance_{}".format(ii)
                #         # write_ply(test_ins_pts, save_path)
                        
                        
                #         centroid = test_ins_pts.mean(0)
                #         f_cluster = test_ins_pts - centroid.reshape(1, 3)
                #         f_cluster_dist = np.linalg.norm(f_cluster, axis=-1, ord=2)
                #         mean_dist = f_cluster_dist.mean()
                #         tmp.append(mean_dist)
                #         pts_num.append(f_cluster_dist.shape[0])
                        
                #         # ipdb.set_trace()
                #     # ipdb.set_trace()
                #     print(tmp)
                #     print(pts_num)
                #     print(ins_ids_cur_cam)
                #     print("=============================================")
                    
                try:
                    nn_vals, nn_inds = dist.min(1)
                except:
                    ipdb.set_trace()

                # set a distance threshold to filter unproper match
                threshold = 1.0
                valid_nn_mask = nn_vals < threshold
                # nn_vals, nn_inds = nn_vals[valid_nn_mask], nn_inds[valid_nn_mask]

                keyframe_matched_ins_ids = ins_ids[cam_id][nn_inds]
                
                all_mask = np.zeros_like(sweep_fg_instance_ids[cam_id]).astype("bool")
                new_sweep_fg_instance_ids = np.zeros_like(sweep_fg_instance_ids[cam_id]) - 1
                for i in range(len(sweep_ins_ids[cam_id])):
                    if valid_nn_mask[i] == False:
                        continue
                    sweepframe_ins_id = sweep_ins_ids[cam_id][i]
                    keyframe_ins_id = keyframe_matched_ins_ids[i]
                    mask = (sweep_fg_instance_ids[cam_id] == sweepframe_ins_id.numpy())
                    if i == 0:
                        all_mask = mask
                    else:
                        all_mask += mask
                    # # filter noise points
                    # if mask.sum() < 10:
                    #     valid_nn_mask[i] = False
                    #     continue
                    # unify instance id
                    new_sweep_fg_instance_ids[mask] = keyframe_ins_id.numpy()
                        # sweep_fg_instance_ids[cam_id][mask] = keyframe_ins_id.numpy()
                
                max_id = ins_ids[cam_id].max()
                base = max_id + 1
                remain_mask = ~all_mask
                
                # sweep_fg_instance_ids[cam_id][remain_mask] = sweep_fg_instance_ids[cam_id][remain_mask] + base.numpy()
                new_sweep_fg_instance_ids[remain_mask] = sweep_fg_instance_ids[cam_id][remain_mask] + base.numpy()
                
                # merge fg info of sweep info to key-frame
                fg_pixels[cam_id] = np.concatenate((fg_pixels[cam_id], sweep_fg_pixels[cam_id]), axis=0)
                fg_points[cam_id] = np.concatenate((fg_points[cam_id], sweep_fg_points[cam_id]), axis=0)
                # fg_instance_ids[cam_id] = np.concatenate((fg_instance_ids[cam_id], sweep_fg_instance_ids[cam_id]), axis=0)
                fg_instance_ids[cam_id] = np.concatenate((fg_instance_ids[cam_id], new_sweep_fg_instance_ids), axis=0)

                # merge centroids info of sweep into key-frame to faclitate next sweeep calculation
                unused_nn_mask = ~valid_nn_mask
                ins_centroids[cam_id] = torch.cat([ins_centroids[cam_id], sweep_ins_centroids[cam_id][unused_nn_mask]],dim=0)
                ins_ids[cam_id] = torch.cat([ins_ids[cam_id], sweep_ins_ids[cam_id][unused_nn_mask] + base],dim=0)

                ########## use keyframe fg info to query sweepframe fg info may introduce logical bug ############
                ########## Eg: one instance in sweepframe is the nearest neighbor of multiple instances in keyframe ###########
                  
                # dist = ins_centroids[cam_id].unsqueeze(1) - sweep_ins_centroids[cam_id].unsqueeze(0)
                # dist = torch.norm(dist, dim=-1, p=2) # shape: (sample_ins_num, sweep_ins_num)
                # nn_vals, nn_inds = dist.min(1)
                
                # # set a distance threshold to filter unproper match
                # threshold = 3.0
                # ipdb.set_trace()
                # valid_nn_mask = nn_vals < threshold
                # nn_vals, nn_inds = nn_vals[valid_nn_mask], nn_inds[valid_nn_mask]

                # sweep_matched_ins_ids = sweep_ins_ids[cam_id][nn_inds]

                # # substitute instance ids of sweep points and merge them to the key-frame
                # # ipdb.set_trace()
                # for i in range(len(ins_ids[cam_id])):
                #     key_frame_ins_id = ins_ids[cam_id][i]
                #     sweep_frame_ins_id = sweep_matched_ins_ids[i]
                #     mask = (sweep_fg_instance_ids[cam_id] == sweep_frame_ins_id.numpy())
                #     if i == 0:
                #         all_mask = mask
                #     else:
                #         all_mask += mask
                    
                #     if key_frame_ins_id != sweep_frame_ins_id:
                #         # substitute instance ids
                #         sweep_fg_instance_ids[cam_id][mask] = key_frame_ins_id.numpy()
                #     # merge sweep info to key-frame
                #     fg_pixels[cam_id] = np.concatenate((fg_pixels[cam_id], sweep_fg_pixels[cam_id][mask]), axis=0)
                #     fg_points[cam_id] = np.concatenate((fg_points[cam_id], sweep_fg_points[cam_id][mask]), axis=0)
                #     fg_instance_ids[cam_id] = np.concatenate((fg_instance_ids[cam_id], sweep_fg_instance_ids[cam_id][mask]), axis=0)

                # ### a sampling strategy remain to be further investigate
                # #1# aggresive: include all remaining instances
                # #2# solid: exclude remaining instances
                # ### comment out the following part to use solid strategy
                # # distinguish with key sample instance id
                # max_id = ins_ids[cam_id].max()
                # base = max_id + 1
                # remain_mask = ~all_mask
                # fg_pixels[cam_id] = np.concatenate((fg_pixels[cam_id], sweep_fg_pixels[cam_id][remain_mask]), axis=0)
                # fg_points[cam_id] = np.concatenate((fg_points[cam_id], sweep_fg_points[cam_id][remain_mask]), axis=0)
                # fg_instance_ids[cam_id] = np.concatenate((fg_instance_ids[cam_id], sweep_fg_instance_ids[cam_id][remain_mask] + base.numpy()), axis=0)

                # sweep_ins_mask = torch.ones_like(sweep_ins_ids[cam_id]).bool()
                # sweep_ins_mask[nn_inds] = False
                # ins_centroids[cam_id] = torch.cat([ins_centroids[cam_id], sweep_ins_centroids[cam_id][sweep_ins_mask]],dim=0)
                # ins_ids[cam_id] = torch.cat([ins_ids[cam_id], sweep_ins_ids[cam_id][sweep_ins_mask] + base],dim=0)
                # ### finish ###

        else:
            print("##################################################")
            print(len(sweep_fg_points))
            print("##################################################")

        fg_info['fg_pixels'] = fg_pixels
        fg_info['fg_points'] = fg_points
        fg_info['fg_instance_ids'] = fg_instance_ids
        fg_info['ins_centroids'] = ins_centroids
        fg_info['ins_ids'] = ins_ids
        self.sweep_iter_num += 1
        return fg_info

    # ### simple sweep merge without tracking
    # def _merge_sweeps(self, fg_info, sweep_fg_info, sweep):
    #     """
    #         fg_info and sweep_fg_info: dict like : 
    #         {
    #             'fg_pixels' = [
    #                 np.array --> cam1, ..., np.array --> cam6
    #             ]
    #             'fg_points' = [
    #                 np.array --> cam1, ..., np.array --> cam6
    #             ]
    #         }
    #         sweep: dict of sweep info
    #     """
    #     fg_pixels, fg_points = fg_info['fg_pixels'], fg_info['fg_points']
    #     sweep_fg_pixels, sweep_fg_points = sweep_fg_info['fg_pixels'], sweep_fg_info['fg_points']
    #     fg_instance_ids, sweep_fg_instance_ids = fg_info['fg_instance_ids'], sweep_fg_info['fg_instance_ids']

    #     ins_centroids, ins_ids = fg_info['ins_centroids'], fg_info['ins_ids']
    #     sweep_ins_centroids, sweep_ins_ids = sweep_fg_info['ins_centroids'], sweep_fg_info['ins_ids']

    #     if len(sweep_fg_points) == len(fg_points):
    #         cam_num = len(fg_pixels)
    #         for cam_id in range(cam_num):
    #             if sweep_ins_centroids[cam_id].shape[0] == 0:
    #                 continue

    #             # align sweep centroids with key-frame centroids
    #             sweep_ins_centroids[cam_id] = sweep_ins_centroids[cam_id] @ sweep['sensor2lidar_rotation'].T
    #             sweep_ins_centroids[cam_id] = sweep_ins_centroids[cam_id] + sweep['sensor2lidar_translation']
    #             sweep_fg_points[cam_id][:,:3] = sweep_fg_points[cam_id][:,:3] @ sweep['sensor2lidar_rotation'].T
    #             sweep_fg_points[cam_id][:,:3] = sweep_fg_points[cam_id][:,:3] + sweep['sensor2lidar_translation']


    #             if ins_centroids[cam_id].shape[0] == 0:
    #                 ins_centroids[cam_id] = torch.cat([ins_centroids[cam_id], sweep_ins_centroids[cam_id]],dim=0)
    #                 ins_ids[cam_id] = torch.cat([ins_ids[cam_id], sweep_ins_ids[cam_id]],dim=0)
    #                 fg_pixels[cam_id] = np.concatenate((fg_pixels[cam_id], sweep_fg_pixels[cam_id]), axis=0)
    #                 fg_points[cam_id] = np.concatenate((fg_points[cam_id], sweep_fg_points[cam_id]), axis=0)
    #                 fg_instance_ids[cam_id] = np.concatenate((fg_instance_ids[cam_id], sweep_fg_instance_ids[cam_id]), axis=0)
    #                 continue

    #             max_id = ins_ids[cam_id].max()
    #             base = max_id + 1
                
    #             sweep_fg_instance_ids[cam_id] = sweep_fg_instance_ids[cam_id] + base.numpy()
                
    #             # merge fg info of sweep info to key-frame
    #             fg_pixels[cam_id] = np.concatenate((fg_pixels[cam_id], sweep_fg_pixels[cam_id]), axis=0)
    #             fg_points[cam_id] = np.concatenate((fg_points[cam_id], sweep_fg_points[cam_id]), axis=0)
    #             fg_instance_ids[cam_id] = np.concatenate((fg_instance_ids[cam_id], sweep_fg_instance_ids[cam_id]), axis=0)

    #             # merge centroids info of sweep into key-frame to faclitate next sweeep calculation
    #             ins_centroids[cam_id] = torch.cat([ins_centroids[cam_id], sweep_ins_centroids[cam_id]],dim=0)
    #             ins_ids[cam_id] = torch.cat([ins_ids[cam_id], sweep_ins_ids[cam_id] + base],dim=0)

    #     else:
    #         print("##################################################")
    #         print(len(sweep_fg_points))
    #         print("##################################################")

    #     fg_info['fg_pixels'] = fg_pixels
    #     fg_info['fg_points'] = fg_points
    #     fg_info['fg_instance_ids'] = fg_instance_ids
    #     fg_info['ins_centroids'] = ins_centroids
    #     fg_info['ins_ids'] = ins_ids

    #     return fg_info

    def _pseudo_centroids_gen(self, fg_info):
        fg_points = fg_info['fg_points']
        fg_instance_ids = fg_info['fg_instance_ids']
        pseudo_centroids, pseudo_centroids_ins_ids = [], []
        for cam_id in range(len(fg_points)):
            fg_point = fg_points[cam_id]
            fg_instance_id = fg_instance_ids[cam_id]
            unique_instance_id = np.unique(fg_instance_id)

            # random select N (1000) points for fps
            instance_points, instance_centroids = [], []
            instance_labels = []
            fps_limits, fps_num, Pseudo_C = 1000, 100, 10
            for ins_id in unique_instance_id:
                mask = (fg_instance_id == ins_id)
                ins_point = torch.from_numpy(fg_point[mask][:,:3])
                ins_label = torch.from_numpy(fg_point[mask][0,3:])
                pts_num = len(ins_point)
                if pts_num > fps_limits:
                    selected_indices = torch.randperm(pts_num)[:fps_limits]
                else:
                    selected_indices = torch.randperm(pts_num)
                    selected_indices = torch.cat([selected_indices, selected_indices[selected_indices.new_zeros(fps_limits-pts_num)]])
                instance_points.append(ins_point[selected_indices])
                instance_labels.append(ins_label)
            
            instance_points = torch.stack(instance_points, dim=0)
            instance_labels = torch.stack(instance_labels, dim=0)

            instance_centroids = []
            for i in range(Pseudo_C):
                fps_inds = farthest_point_sample(instance_points, fps_num)
                fps_points = index_points(instance_points, fps_inds)
                centroid = fps_points.mean(1)
                instance_centroids.append(centroid)
                
            instance_centroids = torch.stack(instance_centroids, dim=1)
            instance_centroids = torch.cat([instance_centroids, instance_labels.unsqueeze(1).repeat(1, Pseudo_C, 1)], dim=-1)
            pseudo_centroids.append(instance_centroids)
            pseudo_centroids_ins_ids.append(torch.from_numpy(unique_instance_id))

        fg_info['pseudo_centroids'] = pseudo_centroids
        fg_info['pseudo_centroids_ins_ids'] = pseudo_centroids_ins_ids
        fg_info['pseudo_C'] = Pseudo_C

        return fg_info
        
    def _make_point_class(self, fg_info):
        fg_points = fg_info['fg_points']
        cam_num = len(fg_points)
        point_class = get_points_type('LIDAR')
        for i in range(cam_num):
            fg_point = fg_points[i]
            fg_point = point_class(
                fg_point, points_dim=fg_point.shape[-1]
            )
            fg_points[i] = fg_point
        fg_info['fg_points'] = fg_points
        return fg_info

    def __call__(self, results):
        if self.dataset == "NuScenesDataset":
            fg_info = results["foreground2D_info"]

            if len(results['sweeps']) <= self.sweeps_num:
                choices = np.arange(len(results['sweeps']))
            elif self.test_mode:
                choices = np.arange(self.sweeps_num)
            else:
                choices = np.random.choice(
                    len(results['sweeps']), self.sweeps_num, replace=False)
            for idx in choices:
                sweep = results['sweeps'][idx]
                pts_filename = sweep['data_path']
                tokens = pts_filename.split('/')
                sweep_fg_pxl_path = os.path.join(*tokens[:-2], "FOREGROUND_MIXED_6NN_INS", tokens[-1]+'.pkl.npy')
                sweep_fg_pts_path = os.path.join(*tokens[:-2], "FOREGROUND_MIXED_6NN", tokens[-1]+'.pkl.npy')
                
                if os.path.exists(sweep_fg_pts_path):
                    sweep_fg_info = np.load(sweep_fg_pxl_path, allow_pickle=True).item()
                    ori_sweep_fg_info = np.load(sweep_fg_pts_path, allow_pickle=True).item()
                    sweep_fg_info["virtual_points"] = ori_sweep_fg_info["virtual_points"]
                    sweep_fg_info["real_points"] = ori_sweep_fg_info["real_points"]

                    sweep_fg_info = self._organize(sweep_fg_info)
                    # merge sweep_fg_info with sample fg_info
                    fg_info = self._merge_sweeps(fg_info, sweep_fg_info, sweep)
                    
                
                else:
                    continue
            
            # pts = results['points']
            # base_dir = '/share/home/jiaoyang/code/TransFusion/instance_tracking/'
            # save_path = base_dir + "whole_scene"
            
            # write_ply(pts.tensor.numpy(), save_path)
            # ipdb.set_trace()
            
            # print("sweep merge finished")
            # # select pseudo centroids for each instance
            # # caution: since many points can be removed after the following data augmentation synchronization process, 
            # # we will generate pseudo centroids in the detector
            # fg_info = self._pseudo_centroids_gen(fg_info)

            # make mmdet3d LiDARPoint for each foreground 2D points
            fg_info = self._make_point_class(fg_info)

            results['foreground2D_info'] = fg_info

            return results



@PIPELINES.register_module()
class GlobalRotTransFilterForeground2D_DC(object):
    """Apply the same data augmentation process with the input points for consistency of saved 2D foreground points and pcd
       
       Note: must after the whole 3D preprocess pipeline!
    
    """
    def __init__(self, point_cloud_range=None):
        self.pcd_range = np.array(point_cloud_range, dtype=np.float32) if point_cloud_range else None
    
    def _synchronize_pcd_DA(self, input_dict):
        """Private function to synchronize point cloud data augmentation skills

        """

        fg_info = input_dict['foreground2D_info']
        fg_points = fg_info['fg_points']
        fg_pixels = fg_info['fg_pixels']
        fg_instance_ids = fg_info['fg_instance_ids']


        pcd_rotate_mat = input_dict['pcd_rotation'] if 'pcd_rotation' in input_dict else np.eye(3)

        pcd_scale_factor = input_dict['pcd_scale_factor'] if 'pcd_scale_factor' in input_dict else 1.

        pcd_trans_factor = input_dict['pcd_trans'] if 'pcd_trans' in input_dict else np.zeros(3)

        pcd_horizontal_flip = input_dict['pcd_horizontal_flip'] if 'pcd_horizontal_flip' in input_dict else False

        pcd_vertical_flip = input_dict['pcd_vertical_flip'] if 'pcd_vertical_flip' in input_dict else False

        flow = input_dict['transformation_3d_flow'] if 'transformation_3d_flow' in input_dict else []

        
        cam_num = len(fg_points)
        for i in range(cam_num):
            pxl = fg_pixels[i]
            pcd = fg_points[i]
            ins_id = fg_instance_ids[i]
            
            horizontal_flip_func = partial(pcd.flip, bev_direction='horizontal') \
                if pcd_horizontal_flip else lambda: None
            vertical_flip_func = partial(pcd.flip, bev_direction='vertical') \
                if pcd_vertical_flip else lambda: None
            
            scale_func = partial(pcd.scale, scale_factor=pcd_scale_factor)
            translate_func = partial(pcd.translate, trans_vector=pcd_trans_factor)
            rotate_func = partial(pcd.rotate, rotation=pcd_rotate_mat)

            flow_mapping = {
                'T': translate_func,
                'S': scale_func,
                'R': rotate_func,
                'HF': horizontal_flip_func,
                'VF': vertical_flip_func
            }

            for op in flow:
                assert op in flow_mapping, f'This 3D data '\
                    f'transformation op ({op}) is not supported'
                func = flow_mapping[op]
                func()
               
            if isinstance(self.pcd_range, (list,tuple,np.ndarray)):
                points_mask = pcd.in_range_3d(self.pcd_range)
                pcd = pcd[points_mask]
                pxl = pxl[points_mask]
                ins_id = ins_id[points_mask]

            fg_points[i] = pcd
            fg_pixels[i] = pxl
            fg_instance_ids[i] = ins_id
            
        return fg_points, fg_pixels, fg_instance_ids

    def __call__(self, input_dict):
        fg_points, fg_pixels, fg_instance_ids = self._synchronize_pcd_DA(input_dict)
        input_dict['foreground2D_info']['fg_points'] = fg_points
        input_dict['foreground2D_info']['fg_pixels'] = fg_pixels
        input_dict['foreground2D_info']['fg_instance_ids'] = fg_instance_ids

        return input_dict


@PIPELINES.register_module()
class ImgScaleCropFlipForeground2D_DC(object):
    """Apply the same data augmentation process with the input images for consistency of saved 2D foreground pixels and img
       
       Note: must after the whole 2D preprocess pipeline!
    """
    def __init__(self, **kwargs):
        pass

    def __call__(self, input_dict):

        img_scale_factor = input_dict['scale_factor'][:2] if 'scale_factor' in input_dict else [1., 1.]
        img_flip = input_dict['flip'] if 'flip' in input_dict else False
        img_crop_offset = input_dict['img_crop_offset'] if 'img_crop_offset' in input_dict else 0
        img_shape = input_dict['img_shape'][:2]

        pxl = input_dict['foreground2D_info']['fg_pixels']
        cam_num = len(pxl)
        for i in range(cam_num):
            pxl_per_cam = pxl[i]
            pxl_per_cam = pxl_per_cam * img_scale_factor
            pxl_per_cam -= img_crop_offset
            if img_flip:
                orig_h, orig_w = img_shape
                pxl_per_cam[:, 0] = orig_w - pxl_per_cam[:, 0]
            pxl[i] = pxl_per_cam
                
        input_dict['foreground2D_info']['fg_pixels'] = pxl

        return input_dict


@PIPELINES.register_module()
class MyCollect3D_DC(object):
    """Collect data from the loader relevant to the specific task.

    This is usually the last stage of the data loader pipeline. Typically keys
    is set to some subset of "img", "proposals", "gt_bboxes",
    "gt_bboxes_ignore", "gt_labels", and/or "gt_masks".

    The "img_meta" item is always populated.  The contents of the "img_meta"
    dictionary depends on "meta_keys". By default this includes:

        - 'img_shape': shape of the image input to the network as a tuple \
            (h, w, c).  Note that images may be zero padded on the \
            bottom/right if the batch tensor is larger than this shape.
        - 'scale_factor': a float indicating the preprocessing scale
        - 'flip': a boolean indicating if image flip transform was used
        - 'filename': path to the image file
        - 'ori_shape': original shape of the image as a tuple (h, w, c)
        - 'pad_shape': image shape after padding
        - 'lidar2img': transform from lidar to image
        - 'pcd_horizontal_flip': a boolean indicating if point cloud is \
            flipped horizontally
        - 'pcd_vertical_flip': a boolean indicating if point cloud is \
            flipped vertically
        - 'box_mode_3d': 3D box mode
        - 'box_type_3d': 3D box type
        - 'img_norm_cfg': a dict of normalization information:

            - mean: per channel mean subtraction
            - std: per channel std divisor
            - to_rgb: bool indicating if bgr was converted to rgb
        - 'rect': rectification matrix
        - 'Trv2c': transformation from velodyne to camera coordinate
        - 'P2': transformation betweeen cameras
        - 'pcd_trans': point cloud transformations
        - 'sample_idx': sample index
        - 'pcd_scale_factor': point cloud scale factor
        - 'pcd_rotation': rotation applied to point cloud
        - 'pts_filename': path to point cloud file.

    Args:
        keys (Sequence[str]): Keys of results to be collected in ``data``.
        meta_keys (Sequence[str], optional): Meta keys to be converted to
            ``mmcv.DataContainer`` and collected in ``data[img_metas]``.
            Default: ('filename', 'ori_shape', 'img_shape', 'lidar2img', \
            'pad_shape', 'scale_factor', 'flip', 'pcd_horizontal_flip', \
            'pcd_vertical_flip', 'box_mode_3d', 'box_type_3d', \
            'img_norm_cfg', 'rect', 'Trv2c', 'P2', 'pcd_trans', \
            'sample_idx', 'pcd_scale_factor', 'pcd_rotation', 'pts_filename')
    """

    def __init__(self,
                 keys,
                 meta_keys=('filename', 'ori_shape', 'img_shape', 'lidar2img',
                            'pad_shape', 'scale_factor', 'flip',
                            'pcd_horizontal_flip', 'pcd_vertical_flip',
                            'box_mode_3d', 'box_type_3d', 'img_norm_cfg',
                            'rect', 'Trv2c', 'P2', 'pcd_trans', 'sample_idx',
                            'pcd_scale_factor', 'pcd_rotation', 'pts_filename',
                            'transformation_3d_flow', 'foreground2D_info')):
        self.keys = keys
        self.meta_keys = meta_keys

    def __call__(self, results):
        """Call function to collect keys in results. The keys in ``meta_keys``
        will be converted to :obj:`mmcv.DataContainer`.

        Args:
            results (dict): Result dict contains the data to collect.

        Returns:
            dict: The result dict contains the following keys
                - keys in ``self.keys``
                - ``img_metas``
        """ 
        data = {}
        img_metas = {}
        for key in self.meta_keys:
            if key in results:
                img_metas[key] = results[key]

        data['img_metas'] = DC(img_metas, cpu_only=True)
        for key in self.keys:
            data[key] = results[key]

        return data