import numpy as np
import os
from tqdm import tqdm

def write_ply(points, out_filename):
    """Write points into ``ply`` format for meshlab visualization.

    Args:
        points (np.ndarray): Points in shape (N, dim).
        out_filename (str): Filename to be saved.
    """
    if points is None:
        return
    N = points.shape[0]
    fout = open(out_filename, 'w')
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

def save_obj(filenames, data_root, out_dir, dataset):
    

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    filenames = tqdm(filenames)
    for filename in filenames:
        file_prefix = filename.split('.')[0]
        file_path = os.path.join(data_root, filename)

        file = np.load(file_path, allow_pickle=True).item()

        
        if dataset == 'nuScenes':
            virtual_pts = np.concatenate(file['virtual_points'], axis=0)
            real_pts = np.concatenate(file['real_points'], axis=0)

        elif dataset == 'kitti':
            virtual_pts = file['virtual_points']
            real_pts = file['real_points']

        out_real_filename = os.path.join(out_dir, file_prefix+'_real.obj')
        out_virtual_filename = os.path.join(out_dir, file_prefix+'_virtual.obj')
        
        write_ply(real_pts, out_real_filename)
        write_ply(virtual_pts, out_virtual_filename)

def show_one_sweep(filenames, data_root, out_dir):

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    filenames = tqdm(filenames)
    for filename in filenames:
        file_prefix = filename.split('.')[0]
        file_path = os.path.join(data_root, filename)

        file = np.load(file_path)

        out_filename = os.path.join(out_dir, file_prefix+'_single_sweep.obj')
        write_ply(file, out_filename)

# filenames = ['n008-2018-05-21-11-06-59-0400__LIDAR_TOP__1526915243047392.pcd.bin.pkl.npy',
# 'n008-2018-05-21-11-06-59-0400__LIDAR_TOP__1526915243547836.pcd.bin.pkl.npy',
# 'n008-2018-05-21-11-06-59-0400__LIDAR_TOP__1526915244048280.pcd.bin.pkl.npy',
# 'n008-2018-05-21-11-06-59-0400__LIDAR_TOP__1526915244548143.pcd.bin.pkl.npy',
# 'n008-2018-05-21-11-06-59-0400__LIDAR_TOP__1526915245048008.pcd.bin.pkl.npy',
# 'n008-2018-05-21-11-06-59-0400__LIDAR_TOP__1526915245547336.pcd.bin.pkl.npy',
# 'n008-2018-05-21-11-06-59-0400__LIDAR_TOP__1526915246047755.pcd.bin.pkl.npy',
# 'n008-2018-05-21-11-06-59-0400__LIDAR_TOP__1526915246547642.pcd.bin.pkl.npy',
# 'n008-2018-05-21-11-06-59-0400__LIDAR_TOP__1526915246947455.pcd.bin.pkl.npy',
# 'n008-2018-05-21-11-06-59-0400__LIDAR_TOP__1526915247447307.pcd.bin.pkl.npy']

data_root = '/share/home/jiaoyang/code/TransFusion/data/nuscenes/samples/FOREGROUND_MIXED/'
# data_root = '/share/home/jiaoyang/code/TransFusion/data/nuscenes/samples/FOREGROUND_MIXED_50pts/'
# data_root = '/share/home/jiaoyang/code/TransFusion/data/kitti/training/virtual_1NN/'
# out_dir = '/share/home/jiaoyang/code/TransFusion/virtual_vis_kitti/'
out_dir = '/share/home/jiaoyang/code/TransFusion/virtual_vis_1NN_200pts/'

filenames = os.listdir('/share/home/jiaoyang/code/TransFusion/data/nuscenes/samples/FOREGROUND_MIXED/')
# filenames = os.listdir('/share/home/jiaoyang/code/TransFusion/data/nuscenes/samples/FOREGROUND_MIXED_50pts/')
# filenames = os.listdir('/share/home/jiaoyang/code/TransFusion/data/kitti/training/virtual_1NN/')
datasets = ['nuScenes', 'kitti']

save_obj(filenames, data_root, out_dir, datasets[0])

