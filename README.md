# MSMDFusion

Official implementation of TransFusion for CVPR'2023 paper ["MSMDFusion: Fusing LiDAR and Camera at Multiple Scales with Multi-Depth Seeds for 3D Object Detection"](https://arxiv.org/abs/2209.03102), by Yang Jiao, Zequn Jie, Shaoxiang Chen, Jingjing Chen, Lin Ma, and Yu-Gang Jiang.
![MSMDFusion framework](https://github.com/SxJyJay/MSMDFusion/blob/main/MSMD-Framework.png)


## Introduction

Fusing LiDAR and camera information is essential for achieving accurate and reliable 3D object detection in autonomous driving systems. This is challenging due to the difficulty of combining multi-granularity geometric and semantic features from two drastically different modalities. Recent approaches aim at exploring the semantic densities of camera features through lifting points in 2D camera images (referred to as seeds) into 3D space, and then incorporate 2D semantics via cross-modal interaction or fusion techniques. However, depth information is under-investigated in these approaches when lifting points into 3D space, thus 2D semantics can not be reliably fused with 3D points. Moreover, their multi-modal fusion strategy, which is implemented as concatenation or attention, either can not effectively fuse 2D and 3D information or is unable to perform fine-grained interactions in the voxel space. To this end, we propose a novel framework called MSMDFusion to tackle above problems.


## Getting Started

### Installation

For basic installation, please refer to [getting_started.md](docs/getting_started.md) for installation.

### Data Preparation

**Step 1**: Please refer to the [official site](https://github.com/ADLab-AutoDrive/BEVFusion/blob/main/docs/getting_started.md) for prepare nuscenes data. After data preparation, you will be able to see the following directory structure:
```
mmdetection3d
├── mmdet3d
├── tools
├── configs
├── data
│   ├── nuscenes
│   │   ├── maps
│   │   ├── samples
│   │   ├── sweeps
│   │   ├── v1.0-test
|   |   ├── v1.0-trainval
│   │   ├── nuscenes_database
│   │   ├── nuscenes_infos_train.pkl
│   │   ├── nuscenes_infos_val.pkl
│   │   ├── nuscenes_infos_test.pkl
│   │   ├── nuscenes_dbinfos_train.pkl

```
**Step 2**: Download preprocessed [virtual points samples](https://pan.baidu.com/s/1IxqcGxNCFnmSZw7Dlu3Xig?pwd=9xcb)(extraction code: 9xcb) and [sweeps](https://pan.baidu.com/s/1qUeopFHCWrr35af2MGBSnw?pwd=2eg1)(extraction code: 2eg1) data. And put them under the above folder ```samples``` and ```sweeps```, respectively, and rename them as ```FOREGROUND_MIXED_6NN_WITH_DEPTH```.

### Training and Evaluation

For training, you need to first train a pure LiDAR backbone, such as TransFusion-L. Then, you can merge the checkpoints from pretrained TransFusion-L and ResNet-50 as suggested [here](https://github.com/XuyangBai/TransFusion/issues/7#issuecomment-1115499891). We also provide a merged 1-st stage checkpoint [here](https://pan.baidu.com/s/1Lj35HXc2Ajv0yWEH6H8g_A?pwd=69i7)(extraction code: 69i7)
```
# 1-st stage training
sh ./tools/dist_train.sh ./configs/transfusion_nusc_voxel_L.py 8
# 2-nd stage training
sh ./tools/dist_train.sh ./configs/MSMDFusion_nusc_voxel_LC.py 8
```

For evaluation, you can use the following command:
```
# Evaluation
sh ./tools/dist_test.sh ./configs/MSMDFusion_nusc_voxel_LC.py $ckpt_path$ 8 --eval bbox
```

For testing and making a submission to the leaderboard, please refer to the [official site](https://mmdetection3d.readthedocs.io/en/stable/datasets/nuscenes_det.html)

## Citation
If you find our paper useful, please cite:

```bash
@article{jiao2022msmdfusion,
  title={MSMDFusion: Fusing LiDAR and Camera at Multiple Scales with Multi-Depth Seeds for 3D Object Detection},
  author={Jiao, Yang and Jie, Zequn and Chen, Shaoxiang and Chen, Jingjing and Ma, Lin and Jiang, Yu-Gang},
  journal={arXiv preprint arXiv:2209.03102},
  year={2022}
}
```

## Acknowlegement

We sincerely thank the authors of [mmdetection3d](https://github.com/open-mmlab/mmdetection3d), [CenterPoint](https://github.com/tianweiy/CenterPoint), [TransFusion](https://github.com/XuyangBai/TransFusion), [MVP](https://github.com/tianweiy/MVP), [BEVFusion](https://github.com/ADLab-AutoDrive/BEVFusion) and [BEVFusion](https://github.com/mit-han-lab/bevfusion) for open sourcing their methods.
