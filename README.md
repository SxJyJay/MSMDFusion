# MSMDFusion
MSMDFusion: Fusing LiDAR and Camera at Multiple Scales with Multi-Depth Seeds for 3D Object Detection.

## Paper
- **[MSMDFusion: Fusing LiDAR and Camera at Multiple Scales with Multi-Depth Seeds for 3D Object Detection](https://arxiv.org/abs/2209.03102)**
  - *Yang Jiao, Zequn Jie, Shaoxiang Chen, Jingjing Chen, Xiaolin Wei, Lin Ma, Yu-Gang Jiang*


## News
- (2022.8.11) Our MSMDFusion ranks 2nd and 1st in the term of NDS and mAP on the nuScenes leaderboard among all methods that don't use TTA and Ensemble. 
- (2022.9.7) The paper of MSMDFusion is released on the [arxiv](https://arxiv.org/abs/2209.03102).

## Performances on nuScenes
|  model   | Modality | mAP | NDS | 
|  :----:  | :----:  |  :----:  |  :----:  |
| MSMDFusion-T (val)  | LC | 69.06 | 71.77 |
| MSMDFusion-T (test)  | LC | 70.84 | 73.17 | 

## Acknowledgement
We sincerely appreciate the following open-source projects for providing valuable and high-quality codes: 
- [mmdetection3d](https://github.com/open-mmlab/mmdetection3d)
- [TrasnFusion](https://github.com/XuyangBai/TransFusion)
- [MVP](https://github.com/tianweiy/MVP)
- [CenterPoint](https://github.com/tianweiy/CenterPoint)

## Reference
If you find our paper useful, please kindly cite us via:
```
@inproceedings{Jiao2022MSMDFusionFL,
  title={MSMDFusion: Fusing LiDAR and Camera at Multiple Scales with Multi-Depth Seeds for 3D Object Detection},
  author={Yang Jiao and Zequn Jie and Shaoxiang Chen and Jingjing Chen and Xiaolin Wei and Lin Ma and Yu-Gang Jiang},
  year={2022}
}
```
