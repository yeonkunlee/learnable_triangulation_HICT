from copy import deepcopy
import numpy as np
import pickle
import random

from scipy.optimize import least_squares

import torch
from torch import nn

from mvn.utils import op, multiview, img, misc, volumetric

from mvn.models.transformer import OCR_2D_Module, OCR_3D_Module

#from mvn.models import pose_resnet
from mvn.models.v2v import V2VModel
from mvn.models import pose_body

def weights_init_uniform_rule(m):
    classname = m.__class__.__name__
    # for every Linear layer in a model..
    if classname.find('Linear') != -1:
        # get the number of the inputs
        n = m.in_features
        y = 1.0/np.sqrt(n)
        m.weight.data.uniform_(-y, y)
        m.bias.data.fill_(0)


def weights_init_xavier_rule(m):
    classname = m.__class__.__name__
    # for every Linear layer in a model..
    if classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)
        

        
        
class VolumetricTriangulationNet_BODY_2D(nn.Module):
    def __init__(self, device='cuda:0'):
        super().__init__()

        # self.num_joints = config.model.backbone.num_joints
        self.pose_body_model = pose_body.get_pose_net(device=device)
        self.backbone = self.pose_body_model.backbone
        self.keypoint_head = self.pose_body_model.keypoint_head

        self.process_features = nn.Sequential(
            # nn.Conv2d(98, 3, 1)
            nn.Conv2d(133, 25, 1)
        )

    def forward(self, images):
        device = images.device
        batch_size, n_views = images.shape[:2]

        # reshape for backbone forward
        images = images.view(-1, *images.shape[2:])

        features = self.backbone(images)
        features = self.keypoint_head(features)
        features = self.process_features(features)
        
        keypoints_2d, features = op.integrate_tensor_2d(features * 100., True)
        
        # reshape back
        images = images.view(batch_size, n_views, *images.shape[1:])
        features = features.view(batch_size, n_views, *features.shape[1:])
        keypoints_2d = keypoints_2d.view(batch_size, n_views, *keypoints_2d.shape[1:])
        
        # calcualte shapes
        image_shape = tuple(images.shape[3:])
        heatmap_shape = tuple(features.shape[3:])

        # upscale keypoints_2d, because image shape != heatmap shape
        keypoints_2d_transformed = torch.zeros_like(keypoints_2d)
        keypoints_2d_transformed[:, :, :, 0] = keypoints_2d[:, :, :, 0] * (image_shape[1] / heatmap_shape[1])
        keypoints_2d_transformed[:, :, :, 1] = keypoints_2d[:, :, :, 1] * (image_shape[0] / heatmap_shape[0])
        keypoints_2d = keypoints_2d_transformed
        
        

        return keypoints_2d, features, images