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
    "/workspace/learnable_triangulation_unification/learnable_triangulation/mmpose/configs/hand/2d_kpt_sview_rgb_img/topdown_heatmap/onehand10k/res50_onehand10k_256x256.py"
    backbone_checkpoint=\
    "/workspace/learnable_triangulation_unification/learnable_triangulation/data/hand/res50_onehand10k_256x256-739c8639_20210330.pth"
    backbone_checkpoint=\
    "/workspace/learnable_triangulation_unification/learnable_triangulation/data/hand/rhand_weights.pth"
    
    #/mnt/vision-nas/yeonkunlee/learnable_triangulation_unification/learnable_triangulation/data/hand/rhand_weights.pth
    
    pose_model = init_pose_model(backbone_config, backbone_checkpoint)#, device=args.device.lower()
    pose_model.train()
    print('Pretrained hand model loaded.')
    return pose_model