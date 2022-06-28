# inference model were separated by yk.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import torch
from argparse import ArgumentParser

from xtcocotools.coco import COCO

from mmpose.apis import init_pose_model

# def get_pose_net(device='cuda:0'):
#     backbone_config="/workspace/learnable_triangulation_unification/learnable_triangulation/mmpose/configs/wholebody/2d_kpt_sview_rgb_img/topdown_heatmap/coco-wholebody/hrnet_w48_coco_wholebody_384x288_dark_plus.py"
#     backbone_checkpoint="/workspace/learnable_triangulation_unification/learnable_triangulation/data/pretrained/body/hrnet_w48_coco_wholebody_384x288_dark-f5726563_20200918.pth"
#     pose_model = init_pose_model(backbone_config, backbone_checkpoint)#, device=args.device.lower()
#     pose_model.train()
#     print('Pretrained Body model loaded.')

#     return pose_model

def get_pose_net(device='cuda:0'):
    backbone_config="/workspace/learnable_triangulation/mmpose/configs/wholebody/2d_kpt_sview_rgb_img/topdown_heatmap/coco-wholebody/hrnet_w48_coco_wholebody_384x288_dark_plus.py"
    backbone_checkpoint=None
    pose_model = init_pose_model(backbone_config, backbone_checkpoint)#, device=args.device.lower()
    pose_model.train()
    print('Dummy model loaded.')

    return pose_model