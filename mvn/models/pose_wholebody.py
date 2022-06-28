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

def get_pose_net(backbone_config, backbone_checkpoint, device='cuda:0'):

    pose_model = init_pose_model(backbone_config, backbone_checkpoint)#, device=args.device.lower()
    
    print('yk : pretrained wholebody model loaded.')
    
    return pose_model