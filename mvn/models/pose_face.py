# inference model were separated by yk.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import torch
from argparse import ArgumentParser

from xtcocotools.coco import COCO

#from mmpose.apis import (inference_top_down_pose_model, init_pose_model, vis_pose_result)
from mmpose.apis import init_pose_model

def get_pose_net(device='cuda:0'):
    backbone_config=\
    "/workspace/learnable_triangulation_unification/learnable_triangulation/mmpose/configs/face/2d_kpt_sview_rgb_img/topdown_heatmap/wflw/hrnetv2_w18_wflw_256x256.py"
    backbone_checkpoint=\
    "/workspace/learnable_triangulation_unification/learnable_triangulation/data/face/hrnetv2_w18_wflw_256x256-2bf032a6_20210125.pth"
    pose_model = init_pose_model(backbone_config, backbone_checkpoint)#, device=args.device.lower()
    pose_model.train()
    print('Pretrained face model loaded.')

    return pose_model