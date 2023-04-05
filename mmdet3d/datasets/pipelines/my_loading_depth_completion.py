import mmcv
from mmcv.parallel import DataContainer as DC
import numpy as np
import os
from functools import partial

from mmdet3d.core.points import BasePoints, get_points_type
from mmdet.datasets.builder import PIPELINES
from mmdet.datasets.pipelines import LoadAnnotations


@PIPELINES.register_module()
class LoadForeground2D(object):
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

    def _organize(self, fg_info):
        """
        Private function to select unique foreground pixels (and paired points)
        """
        if self.dataset == 'NuScenesDataset':
            cam_num = len(fg_info['virtual_pixel_indices'])
            fg_pixels, fg_points = [], []
            for i in range(cam_num):
                fg_pixel_indices = np.concatenate((fg_info['virtual_pixel_indices'][i], fg_info['real_pixel_indices'][i]), axis=0)
                # virtual_pixel_inds = fg_info['virtual_pixel_indices'][i]
                # virtual_pts_inds = fg_info['virtual_points'][i]
                # num_pts = virtual_pixel_inds.shape[0]
                # count = 1 if num_pts > 1 else num_pts
                
                # fg_pixel_indices = np.concatenate((virtual_pixel_inds[np.random.randint(num_pts, size=count), :], fg_info['real_pixel_indices'][i]), axis=0)
                
                fg_points_set = np.concatenate((fg_info['virtual_points'][i], fg_info['real_points'][i]), axis=0)
                # fg_points_set = np.concatenate((virtual_pts_inds[np.random.randint(num_pts, size=count), :], fg_info['real_points'][i]), axis=0)
                
                fg_pixels.append(fg_pixel_indices)
                fg_points.append(fg_points_set)

            return dict(fg_pixels=fg_pixels, fg_points=fg_points)
        
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
            fg_path = os.path.join(*tokens[:-2], "FOREGROUND_MIXED_6NN", tokens[-1]+'.pkl.npy')
            fg_info = np.load(fg_path, allow_pickle=True).item()

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
class LoadForeground2DFromMultiSweeps(object):
    """Load foreground info provided by 2D results from multiple sweeps
    
    """
    def __init__(self, dataset="NuScenesDataset", sweeps_num=10):
        self.dataset = dataset
        self.sweeps_num = sweeps_num

    def _organize(self, fg_info):
        """
        Private function to select unique foreground pixels (and paired points)
        """
        cam_num = len(fg_info['virtual_pixel_indices'])
        fg_pixels, fg_points = [], []
        for i in range(cam_num):
            fg_pixel_indices = np.concatenate((fg_info['virtual_pixel_indices'][i], fg_info['real_pixel_indices'][i]), axis=0)
            # virtual_pixel_inds = fg_info['virtual_pixel_indices'][i]
            # virtual_pts_inds = fg_info['virtual_points'][i]
            # num_pts = virtual_pixel_inds.shape[0]
            # count = 1 if num_pts > 1 else num_pts 
                
            # fg_pixel_indices = np.concatenate((virtual_pixel_inds[np.random.randint(num_pts, size=count), :], fg_info['real_pixel_indices'][i]), axis=0)

            fg_points_set = np.concatenate((fg_info['virtual_points'][i], fg_info['real_points'][i]), axis=0)
            # fg_points_set = np.concatenate((virtual_pts_inds[np.random.randint(num_pts, size=count), :], fg_info['real_points'][i]), axis=0)
            
            fg_pixels.append(fg_pixel_indices)
            fg_points.append(fg_points_set)

        return dict(fg_pixels=fg_pixels, fg_points=fg_points)

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

        if len(sweep_fg_points) == len(fg_points):
            cam_num = len(fg_pixels)
            for cam_id in range(cam_num):
                # merge fg_pixels and sweep_fg_pixels
                fg_pixel, sweep_fg_pixel = fg_pixels[cam_id], sweep_fg_pixels[cam_id]
                # might be a bug to be fixed in the future, i.e., misalignment between sweep pic and sample pic
                fg_pixel = np.concatenate((fg_pixel, sweep_fg_pixel), axis=0)
                fg_pixels[cam_id] = fg_pixel
                
                # merge fg_points and sweep_fg_points
                fg_point, sweep_fg_point = fg_points[cam_id], sweep_fg_points[cam_id]
                # Note: align sweep points with sample points
                # import ipdb
                # ipdb.set_trace()
                sweep_fg_point[:,:3] = sweep_fg_point[:,:3] @ sweep['sensor2lidar_rotation'].T
                sweep_fg_point[:,:3] = sweep_fg_point[:,:3] + sweep['sensor2lidar_translation']
                fg_point = np.concatenate((fg_point, sweep_fg_point), axis=0)
                fg_points[cam_id] = fg_point

        else:
            print("##################################################")
            print(len(sweep_fg_points))
            print("##################################################")

        fg_info['fg_pixels'] = fg_pixels
        fg_info['fg_points'] = fg_points

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
                sweep_fg_path = os.path.join(*tokens[:-2], "FOREGROUND_MIXED_6NN", tokens[-1]+'.pkl.npy')
                if os.path.exists(sweep_fg_path):
                    sweep_fg_info = np.load(sweep_fg_path, allow_pickle=True).item()
                #     # merge boundary pts into 
                #     sweep_fg_boundary_path = os.path.join(*tokens[:-2], "FOREGROUND_MIXED_BD_200", tokens[-1]+'.pkl.npy')
                #     sweep_fg_boundary_info = np.load(sweep_fg_boundary_path, allow_pickle=True).item()
                #     for key in sweep_fg_info:
                #         if 'real' in key:
                #             continue
                #         sweep_fg_info[key] = list(map(lambda x,y: np.concatenate((x,y), axis=0), sweep_fg_info[key], sweep_fg_boundary_info[key]))

                    sweep_fg_info = self._organize(sweep_fg_info)
                    # merge sweep_fg_info with sample fg_info
                    fg_info = self._merge_sweeps(fg_info, sweep_fg_info, sweep)
                else:
                    continue

            # make mmdet3d LiDARPoint for each foreground 2D points
            fg_info = self._make_point_class(fg_info)

            results['foreground2D_info'] = fg_info

            return results



@PIPELINES.register_module()
class GlobalRotTransFilterForeground2D(object):
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

            fg_points[i] = pcd
            fg_pixels[i] = pxl
            

        return fg_points, fg_pixels

    def __call__(self, input_dict):
        fg_points, fg_pixels = self._synchronize_pcd_DA(input_dict)
        input_dict['foreground2D_info']['fg_points'] = fg_points
        input_dict['foreground2D_info']['fg_pixels'] = fg_pixels
        

        return input_dict


@PIPELINES.register_module()
class ImgScaleCropFlipForeground2D(object):
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
class MyCollect3D(object):
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