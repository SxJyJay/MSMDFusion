# MSMDFusion
MSMDFusion: A Gated Multi-Scale LiDAR-Camera Fusion Framework with Multi-Depth Seeds for 3D Object Detection.

## Paper
- **[MSMDFusion: A Gated Multi-Scale LiDAR-Camera Fusion Framework with Multi-Depth Seeds for 3D Object Detection.](https://arxiv.org/abs/2209.03102)**
  - *Yang Jiao, Zequn Jie, Shaoxiang Chen, Jingjing Chen, Lin Ma, Yu-Gang Jiang*

## Framework
![image](https://github.com/SxJyJay/MSMDFusion/blob/main/MSMD-Framework.png)

## News
- (2022.8.11) Our MSMDFusion ranks 2nd and 1st in the term of NDS and mAP on the nuScenes leaderboard among all methods that don't use TTA and Ensemble. 
- (2022.11.12) Our improved version, MSMDFusion-base, ranks 1st in the term of NDS on the nuScenes leaderboard among all submissions that don't use TTA and Ensemble. MSMDFusion-base also achieve impressive results on the nuScenes tracking task by combining a simple greedy tracker. Please refer the paper [arxiv](https://arxiv.org/abs/2209.03102) for more details.
- (2023.2.1) Simply combining the MSMDFusion-base with scaling and flipping test-time-augmentation, our MSMDFusion-TTA achieves non-trivial improvements (1.8 mAP and 1.1 NDS).
- 🔥(2023.2.28) MSMDFusion has been accepted by CVPR 2023!

## Performances on nuScenes detection track
|  model   | Modality | mAP | NDS | 
|  :----:  | :----:  |  :----:  |  :----:  |
| MSMDFusion (val)  | LC | 69.1 | 71.8 |
| MSMDFusion (test)  | LC | 70.8 | 73.2 |
| MSMDFusion-base (test) | LC | 71.5 | 74.0 |
| MSMDFusion-TTA (test) | LC | 73.3 | 75.1 |

## Performances on nuScenes tracking task
|  model   | Modality | AMOTA | AMOTP | 
|  :----:  | :----:  |  :----:  |  :----:  |
| MSMDFusion-base (test)  | LC | 74.0 | 0.549 |

## Acknowledgement
We sincerely appreciate the following open-source projects for providing valuable and high-quality codes: 
- [mmdetection3d](https://github.com/open-mmlab/mmdetection3d)
- [TrasnFusion](https://github.com/XuyangBai/TransFusion)
- [MVP](https://github.com/tianweiy/MVP)
- [CenterPoint](https://github.com/tianweiy/CenterPoint)
- [BEVFusion](https://github.com/ADLab-AutoDrive/BEVFusion)
- [BEVFusion](https://github.com/mit-han-lab/bevfusion)
## Reference
If you find our paper useful, please kindly cite us via:
```
@article{jiao2022msmdfusion,
  title={MSMDFusion: A Gated Multi-Scale LiDAR-Camera Fusion Framework with Multi-Depth Seeds for 3D Object Detection},
  author={Jiao, Yang and Jie, Zequn and Chen, Shaoxiang and Chen, Jingjing and Ma, Lin and Jiang, Yu-Gang},
  journal={arXiv preprint arXiv:2209.03102},
  year={2022}
}
```
