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
from mvn.models import pose_wholebody
from mvn.models import pose_hand
from mvn.models import pose_body

#KeyError: 'flip_pairs' you must change config file. flip_pairs = False

backbone_config = '/workspace/learnable_triangulation_unification/learnable_triangulation/mmpose/configs/wholebody/2d_kpt_sview_rgb_img/topdown_heatmap/coco-wholebody/hrnet_w48_coco_wholebody_384x288_dark_plus.py'
#backbone_checkpoint = 'https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_coco_wholebody_384x288_dark-f5726563_20200918.pth'
backbone_checkpoint = '/workspace/learnable_triangulation_unification/learnable_triangulation/data/pretrained/coco_wholebody/hrnet_w48_coco_wholebody_384x288_dark-f5726563_20200918.pth'

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



class VolumetricTriangulationNet_Wholebody(nn.Module):
    def __init__(self, config, device='cuda:0'):
        super().__init__()

        self.num_joints = config.model.backbone.num_joints
        self.volume_aggregation_method = config.model.volume_aggregation_method

        # volume
        self.volume_softmax = config.model.volume_softmax
        self.volume_multiplier = config.model.volume_multiplier
        self.volume_size = config.model.volume_size

        self.cuboid_side = config.model.cuboid_side

        self.kind = config.model.kind
        self.use_gt_pelvis = config.model.use_gt_pelvis

        # heatmap
        self.heatmap_softmax = config.model.heatmap_softmax
        self.heatmap_multiplier = config.model.heatmap_multiplier

        # transfer
        self.transfer_cmu_to_human36m = config.model.transfer_cmu_to_human36m if hasattr(config.model, "transfer_cmu_to_human36m") else False

        # modules
        config.model.backbone.alg_confidences = False
        config.model.backbone.vol_confidences = False
        if self.volume_aggregation_method.startswith('conf'):
            config.model.backbone.vol_confidences = True

        #protocol 3
        #self.backbone = pose_wholebody.get_pose_net(backbone_config, backbone_checkpoint, device=device).backbone
        self.pose_wholebody_model = pose_wholebody.get_pose_net(backbone_config, backbone_checkpoint, device=device)
        self.backbone = self.pose_wholebody_model.backbone
        self.keypoint_head = self.pose_wholebody_model.keypoint_head
        del self.pose_wholebody_model

        #protocol 3. with process_features layer.
        self.process_features = nn.Sequential(
            nn.Conv2d(133, 32, 1)
        )
        #protocol 3. with process_features layer.
#         self.process_features = nn.Sequential(
#             nn.Conv2d(48, 32, 1)
#         )
#         # 64
        self.volume_net = V2VModel(32, self.num_joints) # 64
        


    def forward(self, images, proj_matricies, batch):
        device = images.device
        batch_size, n_views = images.shape[:2]
        
        # reshape for backbone forward
        images = images.view(-1, *images.shape[2:])

        # forward backbone
        #heatmaps, features, _, vol_confidences = self.backbone(images)
        
        #protocol 3
        features = self.backbone(images)[0] # torch.Size([batch*4, 48, 96, 72])
        features = self.keypoint_head(features) # torch.Size([batch*4, 133, 96, 72])
        vol_confidences = None

        # reshape back
        images = images.view(batch_size, n_views, *images.shape[1:])
        #heatmaps = heatmaps.view(batch_size, n_views, *heatmaps.shape[1:])
        features = features.view(batch_size, n_views, *features.shape[1:])

        if vol_confidences is not None:
            vol_confidences = vol_confidences.view(batch_size, n_views, *vol_confidences.shape[1:])

        # calcualte shapes
        #image_shape, heatmap_shape = tuple(images.shape[3:]), tuple(heatmaps.shape[3:])
        image_shape, features_shape = tuple(images.shape[3:]), tuple(features.shape[3:])
        n_joints = self.num_joints

        # norm vol confidences
        if self.volume_aggregation_method == 'conf_norm':
            vol_confidences = vol_confidences / vol_confidences.sum(dim=1, keepdim=True)

        # change camera intrinsics
        new_cameras = deepcopy(batch['cameras'])
        for view_i in range(n_views):
            for batch_i in range(batch_size):
                new_cameras[view_i][batch_i].update_after_resize(image_shape, features_shape)

        proj_matricies = torch.stack([torch.stack([torch.from_numpy(camera.projection) for camera in camera_batch], dim=0) for camera_batch in new_cameras], dim=0).transpose(1, 0)  # shape (batch_size, n_views, 3, 4)
        proj_matricies = proj_matricies.float().to(device)

        # build coord volumes
        cuboids = []
        base_points = torch.zeros(batch_size, 3, device=device)
        coord_volumes = torch.zeros(batch_size, self.volume_size, self.volume_size, self.volume_size, 3, device=device)
        for batch_i in range(batch_size):
            # if self.use_precalculated_pelvis:
            if self.use_gt_pelvis:
                keypoints_3d = batch['keypoints_3d'][batch_i]
            else:
                keypoints_3d = batch['pred_keypoints_3d'][batch_i]

            if self.kind == "coco":
                base_point = (keypoints_3d[11, :3] + keypoints_3d[12, :3]) / 2
            elif self.kind == "mpii":
                base_point = keypoints_3d[6, :3]


            base_points[batch_i] = torch.from_numpy(base_point).to(device)

            # build cuboid
            sides = np.array([self.cuboid_side*(3/4), self.cuboid_side*(3/4), self.cuboid_side])
            position = base_point - sides / 2
            cuboid = volumetric.Cuboid3D(position, sides)

            cuboids.append(cuboid)

            # build coord volume
            xxx, yyy, zzz = torch.meshgrid(torch.arange(self.volume_size, device=device), torch.arange(self.volume_size, device=device), torch.arange(self.volume_size, device=device))
            grid = torch.stack([xxx, yyy, zzz], dim=-1).type(torch.float)
            grid = grid.reshape((-1, 3))

            grid_coord = torch.zeros_like(grid)
            grid_coord[:, 0] = position[0] + (sides[0] / (self.volume_size - 1)) * grid[:, 0]
            grid_coord[:, 1] = position[1] + (sides[1] / (self.volume_size - 1)) * grid[:, 1]
            grid_coord[:, 2] = position[2] + (sides[2] / (self.volume_size - 1)) * grid[:, 2]

            coord_volume = grid_coord.reshape(self.volume_size, self.volume_size, self.volume_size, 3)

            # random rotation
            if self.training:
                theta = np.random.uniform(0.0, 2 * np.pi)
            else:
                theta = 0.0

            if self.kind == "coco":
                axis = [0, 1, 0]  # y axis
            elif self.kind == "mpii":
                axis = [0, 0, 1]  # z axis

            center = torch.from_numpy(base_point).type(torch.float).to(device)

            # rotate
            coord_volume = coord_volume - center
            coord_volume = volumetric.rotate_coord_volume(coord_volume, theta, axis)
            coord_volume = coord_volume + center

            # transfer
            if self.transfer_cmu_to_human36m:  # different world coordinates
                coord_volume = coord_volume.permute(0, 2, 1, 3)
                inv_idx = torch.arange(coord_volume.shape[1] - 1, -1, -1).long().to(device)
                coord_volume = coord_volume.index_select(1, inv_idx)

            coord_volumes[batch_i] = coord_volume

        # process features before unprojecting
        features = features.view(-1, *features.shape[2:])
        #protocol 2.
        features = self.process_features(features)
        #protocol 3.
        #features = self.process_features(features)
        features = features.view(batch_size, n_views, *features.shape[1:])

        # lift to volume
        volumes = op.unproject_heatmaps(features, proj_matricies, coord_volumes, volume_aggregation_method=self.volume_aggregation_method, vol_confidences=vol_confidences)

        # integral 3d
        volumes = self.volume_net(volumes)
        vol_keypoints_3d, volumes = op.integrate_tensor_3d_with_coordinates(volumes * self.volume_multiplier, coord_volumes, softmax=self.volume_softmax)

        return vol_keypoints_3d, features, volumes, vol_confidences, cuboids, coord_volumes, base_points
    
    
    
    
    
    
# class VolumetricTriangulationNet_Wholebody_512x512(nn.Module):
#     def __init__(self, config, device='cuda:0'):
#         super().__init__()

#         self.num_joints = config.model.backbone.num_joints
#         self.volume_aggregation_method = config.model.volume_aggregation_method

#         # volume
#         self.volume_softmax = config.model.volume_softmax
#         self.volume_multiplier = config.model.volume_multiplier
#         self.volume_size = config.model.volume_size

#         self.cuboid_side = config.model.cuboid_side

#         self.kind = config.model.kind
#         self.use_gt_pelvis = config.model.use_gt_pelvis

#         # heatmap
#         self.heatmap_softmax = config.model.heatmap_softmax
#         self.heatmap_multiplier = config.model.heatmap_multiplier

#         # transfer
#         self.transfer_cmu_to_human36m = config.model.transfer_cmu_to_human36m if hasattr(config.model, "transfer_cmu_to_human36m") else False

#         # modules
#         config.model.backbone.alg_confidences = False
#         config.model.backbone.vol_confidences = False
#         if self.volume_aggregation_method.startswith('conf'):
#             config.model.backbone.vol_confidences = True

#         #protocol 3
#         #self.backbone = pose_wholebody.get_pose_net(backbone_config, backbone_checkpoint, device=device).backbone
#         self.pose_wholebody_model = pose_body.get_pose_net()
#         self.backbone = self.pose_wholebody_model.backbone
#         self.keypoint_head = self.pose_wholebody_model.keypoint_head
#         del self.pose_wholebody_model

#         #protocol 3. with process_features layer.
#         self.process_features = nn.Sequential(
#             nn.Conv2d(133, 32, 1)
#         )
#         #protocol 3. with process_features layer.
# #         self.process_features = nn.Sequential(
# #             nn.Conv2d(48, 32, 1)
# #         )
# #         # 64
#         self.volume_net = V2VModel(32, self.num_joints) # 64
        


#     def forward(self, images, proj_matricies, batch):
#         device = images.device
#         batch_size, n_views = images.shape[:2]
        
#         # reshape for backbone forward
#         images = images.view(-1, *images.shape[2:])

#         # forward backbone
#         #heatmaps, features, _, vol_confidences = self.backbone(images)
        
#         #protocol 3
#         features = self.backbone(images)[0] # torch.Size([batch*4, 48, 96, 72])
#         features = self.keypoint_head(features) # torch.Size([batch*4, 133, 96, 72])
#         vol_confidences = None

#         # reshape back
#         images = images.view(batch_size, n_views, *images.shape[1:])
#         #heatmaps = heatmaps.view(batch_size, n_views, *heatmaps.shape[1:])
#         features = features.view(batch_size, n_views, *features.shape[1:])

#         if vol_confidences is not None:
#             vol_confidences = vol_confidences.view(batch_size, n_views, *vol_confidences.shape[1:])

#         # calcualte shapes
#         #image_shape, heatmap_shape = tuple(images.shape[3:]), tuple(heatmaps.shape[3:])
#         image_shape, features_shape = tuple(images.shape[3:]), tuple(features.shape[3:])
#         n_joints = self.num_joints

#         # norm vol confidences
#         if self.volume_aggregation_method == 'conf_norm':
#             vol_confidences = vol_confidences / vol_confidences.sum(dim=1, keepdim=True)

#         # change camera intrinsics
#         new_cameras = deepcopy(batch['cameras'])
#         for view_i in range(n_views):
#             for batch_i in range(batch_size):
#                 new_cameras[view_i][batch_i].update_after_resize(image_shape, features_shape)

#         proj_matricies = torch.stack([torch.stack([torch.from_numpy(camera.projection) for camera in camera_batch], dim=0) for camera_batch in new_cameras], dim=0).transpose(1, 0)  # shape (batch_size, n_views, 3, 4)
#         proj_matricies = proj_matricies.float().to(device)

#         # build coord volumes
#         cuboids = []
#         base_points = torch.zeros(batch_size, 3, device=device)
#         coord_volumes = torch.zeros(batch_size, self.volume_size, self.volume_size, self.volume_size, 3, device=device)
#         for batch_i in range(batch_size):
#             # if self.use_precalculated_pelvis:
#             if self.use_gt_pelvis:
#                 keypoints_3d = batch['keypoints_3d'][batch_i]
#             else:
#                 keypoints_3d = batch['pred_keypoints_3d'][batch_i]

#             if self.kind == "coco":
#                 base_point = (keypoints_3d[11, :3] + keypoints_3d[12, :3]) / 2
#             elif self.kind == "mpii":
#                 base_point = keypoints_3d[6, :3]


#             base_points[batch_i] = torch.from_numpy(base_point).to(device)

#             # build cuboid
#             sides = np.array([self.cuboid_side*(3/4), self.cuboid_side*(3/4), self.cuboid_side])
#             position = base_point - sides / 2
#             cuboid = volumetric.Cuboid3D(position, sides)

#             cuboids.append(cuboid)

#             # build coord volume
#             xxx, yyy, zzz = torch.meshgrid(torch.arange(self.volume_size, device=device), torch.arange(self.volume_size, device=device), torch.arange(self.volume_size, device=device))
#             grid = torch.stack([xxx, yyy, zzz], dim=-1).type(torch.float)
#             grid = grid.reshape((-1, 3))

#             grid_coord = torch.zeros_like(grid)
#             grid_coord[:, 0] = position[0] + (sides[0] / (self.volume_size - 1)) * grid[:, 0]
#             grid_coord[:, 1] = position[1] + (sides[1] / (self.volume_size - 1)) * grid[:, 1]
#             grid_coord[:, 2] = position[2] + (sides[2] / (self.volume_size - 1)) * grid[:, 2]

#             coord_volume = grid_coord.reshape(self.volume_size, self.volume_size, self.volume_size, 3)

#             # random rotation
#             if self.training:
#                 theta = np.random.uniform(0.0, 2 * np.pi)
#             else:
#                 theta = 0.0

#             if self.kind == "coco":
#                 axis = [0, 1, 0]  # y axis
#             elif self.kind == "mpii":
#                 axis = [0, 0, 1]  # z axis

#             center = torch.from_numpy(base_point).type(torch.float).to(device)

#             # rotate
#             coord_volume = coord_volume - center
#             coord_volume = volumetric.rotate_coord_volume(coord_volume, theta, axis)
#             coord_volume = coord_volume + center

#             # transfer
#             if self.transfer_cmu_to_human36m:  # different world coordinates
#                 coord_volume = coord_volume.permute(0, 2, 1, 3)
#                 inv_idx = torch.arange(coord_volume.shape[1] - 1, -1, -1).long().to(device)
#                 coord_volume = coord_volume.index_select(1, inv_idx)

#             coord_volumes[batch_i] = coord_volume

#         # process features before unprojecting
#         features = features.view(-1, *features.shape[2:])
#         #protocol 2.
#         features = self.process_features(features)
#         #protocol 3.
#         #features = self.process_features(features)
#         features = features.view(batch_size, n_views, *features.shape[1:])

#         # lift to volume
#         volumes = op.unproject_heatmaps(features, proj_matricies, coord_volumes, volume_aggregation_method=self.volume_aggregation_method, vol_confidences=vol_confidences)

#         # integral 3d
#         volumes = self.volume_net(volumes)
#         vol_keypoints_3d, volumes = op.integrate_tensor_3d_with_coordinates(volumes * self.volume_multiplier, coord_volumes, softmax=self.volume_softmax)

#         return vol_keypoints_3d, features, volumes, vol_confidences, cuboids, coord_volumes, base_points    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
class VolumetricTriangulationNet_Wholebody_TR(VolumetricTriangulationNet_Wholebody):
    def __init__(self, config, device='cuda:0'):
        super().__init__()

        ######## TRANSFORMER SECTION ########
        '''
        _C.MODEL.OCR = CN()
        _C.MODEL.OCR.MID_CHANNELS = 512
        _C.MODEL.OCR.KEY_CHANNELS = 256
        _C.MODEL.OCR.DROPOUT = 0.05
        _C.MODEL.OCR.SCALE = 1
        '''
        self.last_inp_channels = 48
        #self.ocr_mid_channels = 512
        self.ocr_mid_channels = 48 # this is out channel of features.
        self.ocr_key_channels = 48
        
        if self.training:
            self.ocr_dropout = 0.05
        else:
            self.ocr_dropout = 0.00
        
        #config.DATASET.NUM_CLASSES
        #self.num_joints = 17 defined above.
        # batch_size, c, h, w
        self.ocr_2d_transformer = OCR_2D_Module(last_inp_channels=self.last_inp_channels,
                                           ocr_mid_channels=self.ocr_mid_channels,
                                           ocr_key_channels=self.ocr_key_channels,
                                           ocr_dropout=self.ocr_dropout,
                                           num_joints=self.num_joints)

        #self.ocr_2d_transformer.apply(weights_init_positive_uniform_rule)
        self.ocr_2d_transformer.apply(weights_init_uniform_rule) # best results.
        #self.ocr_2d_transformer.apply(weights_init_xavier_rule)


    def forward(self, images, proj_matricies, batch):
        device = images.device
        batch_size, n_views = images.shape[:2]

        # reshape for backbone forward
        images = images.view(-1, *images.shape[2:])

        # forward backbone
        #heatmaps, features, _, vol_confidences = self.backbone(images)
        features = self.backbone(images)[0]
        vol_confidences = None

        # reshape back
        images = images.view(batch_size, n_views, *images.shape[1:])
        #heatmaps = heatmaps.view(batch_size, n_views, *heatmaps.shape[1:])
        features = features.view(batch_size, n_views, *features.shape[1:])

        if vol_confidences is not None:
            vol_confidences = vol_confidences.view(batch_size, n_views, *vol_confidences.shape[1:])

        # calcualte shapes
        #image_shape, heatmap_shape = tuple(images.shape[3:]), tuple(heatmaps.shape[3:])
        image_shape, features_shape = tuple(images.shape[3:]), tuple(features.shape[3:])
        n_joints = self.num_joints

        # norm vol confidences
        if self.volume_aggregation_method == 'conf_norm':
            vol_confidences = vol_confidences / vol_confidences.sum(dim=1, keepdim=True)

        # change camera intrinsics
        new_cameras = deepcopy(batch['cameras'])
        for view_i in range(n_views):
            for batch_i in range(batch_size):
                new_cameras[view_i][batch_i].update_after_resize(image_shape, features_shape)

        proj_matricies = torch.stack([torch.stack([torch.from_numpy(camera.projection) for camera in camera_batch], dim=0) for camera_batch in new_cameras], dim=0).transpose(1, 0)  # shape (batch_size, n_views, 3, 4)
        proj_matricies = proj_matricies.float().to(device)

        # build coord volumes
        cuboids = []
        base_points = torch.zeros(batch_size, 3, device=device)
        coord_volumes = torch.zeros(batch_size, self.volume_size, self.volume_size, self.volume_size, 3, device=device)
        for batch_i in range(batch_size):
            # if self.use_precalculated_pelvis:
            if self.use_gt_pelvis:
                keypoints_3d = batch['keypoints_3d'][batch_i]
            else:
                keypoints_3d = batch['pred_keypoints_3d'][batch_i]

            if self.kind == "coco":
                base_point = (keypoints_3d[11, :3] + keypoints_3d[12, :3]) / 2
            elif self.kind == "mpii":
                base_point = keypoints_3d[6, :3]
                
                
            # just for experiment.    
            #base_point = np.array([0., 0., self.cuboid_side / 2])
                
                

            base_points[batch_i] = torch.from_numpy(base_point).to(device)

            # build cuboid
            sides = np.array([self.cuboid_side, self.cuboid_side, self.cuboid_side])
            position = base_point - sides / 2
            cuboid = volumetric.Cuboid3D(position, sides)

            cuboids.append(cuboid)

            # build coord volume
            xxx, yyy, zzz = torch.meshgrid(torch.arange(self.volume_size, device=device), torch.arange(self.volume_size, device=device), torch.arange(self.volume_size, device=device))
            grid = torch.stack([xxx, yyy, zzz], dim=-1).type(torch.float)
            grid = grid.reshape((-1, 3))

            grid_coord = torch.zeros_like(grid)
            grid_coord[:, 0] = position[0] + (sides[0] / (self.volume_size - 1)) * grid[:, 0]
            grid_coord[:, 1] = position[1] + (sides[1] / (self.volume_size - 1)) * grid[:, 1]
            grid_coord[:, 2] = position[2] + (sides[2] / (self.volume_size - 1)) * grid[:, 2]

            coord_volume = grid_coord.reshape(self.volume_size, self.volume_size, self.volume_size, 3)

            # random rotation
            if self.training:
                theta = np.random.uniform(0.0, 2 * np.pi)
            else:
                theta = 0.0

            if self.kind == "coco":
                axis = [0, 1, 0]  # y axis
            elif self.kind == "mpii":
                axis = [0, 0, 1]  # z axis

            center = torch.from_numpy(base_point).type(torch.float).to(device)

            # rotate
            coord_volume = coord_volume - center
            coord_volume = volumetric.rotate_coord_volume(coord_volume, theta, axis)
            coord_volume = coord_volume + center

            # transfer
            if self.transfer_cmu_to_human36m:  # different world coordinates
                coord_volume = coord_volume.permute(0, 2, 1, 3)
                inv_idx = torch.arange(coord_volume.shape[1] - 1, -1, -1).long().to(device)
                coord_volume = coord_volume.index_select(1, inv_idx)

            coord_volumes[batch_i] = coord_volume

        # process features before unprojecting
        features = features.view(-1, *features.shape[2:])
        features = self.process_features(features)
        features = features.view(batch_size, n_views, *features.shape[1:])

        # lift to volume
        volumes = op.unproject_heatmaps(features, proj_matricies, coord_volumes, volume_aggregation_method=self.volume_aggregation_method, vol_confidences=vol_confidences)

        # integral 3d
        volumes = self.volume_net(volumes)
        vol_keypoints_3d, volumes = op.integrate_tensor_3d_with_coordinates(volumes * self.volume_multiplier, coord_volumes, softmax=self.volume_softmax)

        return vol_keypoints_3d, features, volumes, vol_confidences, cuboids, coord_volumes, base_points
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
class VolumetricTriangulationNet_HANDSONLY(nn.Module):
    def __init__(self, config, device='cuda:0'):
        super().__init__()

        self.num_joints = config.model.backbone.num_joints
        self.volume_aggregation_method = config.model.volume_aggregation_method

        # volume
        self.volume_softmax = config.model.volume_softmax
        self.volume_multiplier = config.model.volume_multiplier
        self.volume_size = config.model.volume_size

        self.cuboid_side = config.model.cuboid_side

        self.kind = config.model.kind
        self.use_gt_pelvis = config.model.use_gt_pelvis

        # heatmap
        self.heatmap_softmax = config.model.heatmap_softmax
        self.heatmap_multiplier = config.model.heatmap_multiplier

        # transfer
        self.transfer_cmu_to_human36m = config.model.transfer_cmu_to_human36m if hasattr(config.model, "transfer_cmu_to_human36m") else False

        # modules
        config.model.backbone.alg_confidences = False
        config.model.backbone.vol_confidences = False
        if self.volume_aggregation_method.startswith('conf'):
            config.model.backbone.vol_confidences = True

        #self.backbone = pose_resnet.get_pose_net(config.model.backbone, device=device)
        
        #protocol 3
        #self.backbone = pose_wholebody.get_pose_net(backbone_config, backbone_checkpoint, device=device).backbone
        self.pose_hand_model = pose_hand.get_pose_net(device=device)
        self.backbone = self.pose_hand_model.backbone
        self.keypoint_head = self.pose_hand_model.keypoint_head

        #protocol 3. with process_features layer.
        self.process_features = nn.Sequential(
            nn.Conv2d(21, 32, 1)
        )

        self.volume_net = V2VModel(32, self.num_joints)
        


    def forward(self, images, proj_matricies, batch, is_righthand=True):
        device = images.device
        batch_size, n_views = images.shape[:2]

        # reshape for backbone forward
        images = images.view(-1, *images.shape[2:]) # X, 3, 256, 256
        
        if not is_righthand:
            #FLIP for lefthand
            images = torch.flip(images, [3])

        # forward backbone
        #heatmaps, features, _, vol_confidences = self.backbone(images)
        #print('image shape is:{}'.format(images.shape)) #image shape is:torch.Size([4, 3, 256, 256])
        #protocol 3
        features = self.backbone(images) # torch.Size([batch*4, 48, 96, 72])
        #print('features shape is:{}'.format(features.shape)) #features shape is:torch.Size([4, 3, 256, 256])
        features = self.keypoint_head(features) # torch.Size([batch*4, 21, 96, 72])
        vol_confidences = None
        
        if not is_righthand:
            #FLIP for lefthand
            features = torch.flip(features, [3])

        # reshape back
        images = images.view(batch_size, n_views, *images.shape[1:])
        #heatmaps = heatmaps.view(batch_size, n_views, *heatmaps.shape[1:])
        features = features.view(batch_size, n_views, *features.shape[1:])

        if vol_confidences is not None:
            vol_confidences = vol_confidences.view(batch_size, n_views, *vol_confidences.shape[1:])

        # calcualte shapes
        #image_shape, heatmap_shape = tuple(images.shape[3:]), tuple(heatmaps.shape[3:])
        image_shape, features_shape = tuple(images.shape[3:]), tuple(features.shape[3:])
        n_joints = self.num_joints

        # norm vol confidences
        if self.volume_aggregation_method == 'conf_norm':
            vol_confidences = vol_confidences / vol_confidences.sum(dim=1, keepdim=True)

        # change camera intrinsics
        new_cameras = deepcopy(batch['cameras'])
        for view_i in range(n_views):
            for batch_i in range(batch_size):
                new_cameras[view_i][batch_i].update_after_resize(image_shape, features_shape)

        proj_matricies = torch.stack([torch.stack([torch.from_numpy(camera.projection) for camera in camera_batch], dim=0) for camera_batch in new_cameras], dim=0).transpose(1, 0)  # shape (batch_size, n_views, 3, 4)
        proj_matricies = proj_matricies.float().to(device)

        # build coord volumes
        cuboids = []
        base_points = torch.zeros(batch_size, 3, device=device)
        coord_volumes = torch.zeros(batch_size, self.volume_size, self.volume_size, self.volume_size, 3, device=device)
        for batch_i in range(batch_size):
            # if self.use_precalculated_pelvis:
            if self.use_gt_pelvis:
                keypoints_3d = batch['keypoints_3d'][batch_i]
            else:
                keypoints_3d = batch['pred_keypoints_3d'][batch_i]
                
#             if self.kind == "coco":
#                 base_point = (keypoints_3d[11, :3] + keypoints_3d[12, :3]) / 2
#             elif self.kind == "mpii":
#                 base_point = keypoints_3d[6, :3]
            
            # 10 : RIGHTWRIST
            # 15 : LEFTWRIST
            # RIGHTHAND BASEPOINT
            if is_righthand:
                base_point = keypoints_3d[10, :3]
            else:
                base_point = keypoints_3d[15, :3]

            base_points[batch_i] = torch.from_numpy(base_point).to(device)

            # build cuboid
            sides = np.array([self.cuboid_side*(3/4), self.cuboid_side*(3/4), self.cuboid_side])
            position = base_point - sides / 2
            cuboid = volumetric.Cuboid3D(position, sides)

            cuboids.append(cuboid)

            # build coord volume
            xxx, yyy, zzz = torch.meshgrid(torch.arange(self.volume_size, device=device), torch.arange(self.volume_size, device=device), torch.arange(self.volume_size, device=device))
            grid = torch.stack([xxx, yyy, zzz], dim=-1).type(torch.float)
            grid = grid.reshape((-1, 3))

            grid_coord = torch.zeros_like(grid)
            grid_coord[:, 0] = position[0] + (sides[0] / (self.volume_size - 1)) * grid[:, 0]
            grid_coord[:, 1] = position[1] + (sides[1] / (self.volume_size - 1)) * grid[:, 1]
            grid_coord[:, 2] = position[2] + (sides[2] / (self.volume_size - 1)) * grid[:, 2]

            coord_volume = grid_coord.reshape(self.volume_size, self.volume_size, self.volume_size, 3)

            # random rotation
            if self.training:
                theta = np.random.uniform(0.0, 2 * np.pi)
            else:
                theta = 0.0

            if self.kind == "coco":
                axis = [0, 1, 0]  # y axis
            elif self.kind == "mpii":
                axis = [0, 0, 1]  # z axis

            center = torch.from_numpy(base_point).type(torch.float).to(device)

            # rotate
            coord_volume = coord_volume - center
            coord_volume = volumetric.rotate_coord_volume(coord_volume, theta, axis)
            coord_volume = coord_volume + center

            # transfer
            if self.transfer_cmu_to_human36m:  # different world coordinates
                coord_volume = coord_volume.permute(0, 2, 1, 3)
                inv_idx = torch.arange(coord_volume.shape[1] - 1, -1, -1).long().to(device)
                coord_volume = coord_volume.index_select(1, inv_idx)

            coord_volumes[batch_i] = coord_volume

        # process features before unprojecting
        features = features.view(-1, *features.shape[2:])
        #protocol 2.
        #features = self.process_features(features)
        #protocol 3.
        features = self.process_features(features)
        features = features.view(batch_size, n_views, *features.shape[1:])

        # lift to volume
        volumes = op.unproject_heatmaps(features, proj_matricies, coord_volumes, volume_aggregation_method=self.volume_aggregation_method, vol_confidences=vol_confidences)

        # integral 3d
        
        #print('yk: volumes shape is : {}'.format(volumes.shape))
        #yk: volumes shape is : torch.Size([1, 32, 64, 64, 64])
        
        if not is_righthand:
            #FLIP for lefthand
            volumes = torch.flip(volumes, [2])
            
        volumes = self.volume_net(volumes)
        
        if not is_righthand:
            #FLIP for lefthand
            volumes = torch.flip(volumes, [2])
        
        vol_keypoints_3d, volumes = op.integrate_tensor_3d_with_coordinates(volumes * self.volume_multiplier, coord_volumes, softmax=self.volume_softmax)

        return vol_keypoints_3d, features, volumes, vol_confidences, cuboids, coord_volumes, base_points