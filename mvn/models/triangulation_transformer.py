from copy import deepcopy
import numpy as np
import pickle
import random

from scipy.optimize import least_squares

import torch
from torch import nn
#import torch._utils
#import torch.nn.functional as F

from mvn.utils import op, multiview, img, misc, volumetric

from mvn.models import pose_resnet
from mvn.models.v2v import V2VModel

from .bn_helper import BatchNorm2d, BatchNorm2d_class, relu_inplace

#from mvn.models import pose_wholebody

from mvn.models.transformer import ModuleHelper, SpatialGather_Module, SpatialOCR_Module
from mvn.models.transformer import OCR_2D_Module, OCR_3D_Module
#from mvn.models.transformer import SpatialGather_Module
#from mvn.models.transformer import SpatialOCR_Module


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


def weights_init_positive_uniform_rule(m):
    classname = m.__class__.__name__
    # for every Linear layer in a model..
    if classname.find('Linear') != -1:
        # apply a uniform distribution to the weights and a bias=0
        m.weight.data.uniform_(0.0, 1.0)
        m.bias.data.fill_(0)

class VolumetricTriangulationNet_TR(nn.Module):
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

        self.backbone = pose_resnet.get_pose_net(config.model.backbone, device=device)

        for p in self.backbone.final_layer.parameters():
            p.requires_grad = False

        self.process_features = nn.Sequential(
            nn.Conv2d(256, 32, 1)
        )

        self.volume_net = V2VModel(32, self.num_joints)
        
        ######## TRANSFORMER SECTION ########
        '''
        _C.MODEL.OCR = CN()
        _C.MODEL.OCR.MID_CHANNELS = 512
        _C.MODEL.OCR.KEY_CHANNELS = 256
        _C.MODEL.OCR.DROPOUT = 0.05
        _C.MODEL.OCR.SCALE = 1
        '''
        self.last_inp_channels = 256
        #self.ocr_mid_channels = 512
        self.ocr_mid_channels = 256 # this is out channel of features.
        self.ocr_key_channels = 256
        
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
        heatmaps, features, _, vol_confidences = self.backbone(images)
        #heatmaps_pre = heatmaps : torch.Size([44, 17, 96, 96])
        #features_pre = features : torch.Size([44, 256, 96, 96])
        
        # maybe 2D transformer should be here. -yk
        # transformer input : features, heatmaps
        # transformer output : attentioned features
        '''
        out_aux = self.aux_head(feats) # 17.
        # compute contrast feature
        feats = self.conv3x3_ocr(feats)

        context = self.ocr_gather_head(feats, out_aux)
        feats = self.ocr_distri_head(feats, context)

        out = self.cls_head(feats)

        out_aux_seg.append(out_aux)
        out_aux_seg.append(out)

        return out_aux_seg
        '''
        
        #features = self.conv3x3_ocr(features)
        #context = self.ocr_gather_head(features, heatmaps)
        #features = self.ocr_distri_head(features, context)
        
        # OCR module
        features = self.ocr_2d_transformer(features, heatmaps)
        #ModuleHelper, SpatialGather_Module, SpatialOCR_Module
        #print('features after transformer is : {}'.format(features.shape))
        #features after transformer is : torch.Size([44, 256, 96, 96])

        # reshape back
        images = images.view(batch_size, n_views, *images.shape[1:])
        heatmaps = heatmaps.view(batch_size, n_views, *heatmaps.shape[1:])
        features = features.view(batch_size, n_views, *features.shape[1:])
        #print(heatmaps.shape) : torch.Size([11, 4, 17, 96, 96])
        #print(features.shape) : torch.Size([11, 4, 256, 96, 96])
        

        if vol_confidences is not None:
            vol_confidences = vol_confidences.view(batch_size, n_views, *vol_confidences.shape[1:])

        # calcualte shapes
        image_shape, heatmap_shape = tuple(images.shape[3:]), tuple(heatmaps.shape[3:])
        n_joints = heatmaps.shape[2]

        # norm vol confidences
        if self.volume_aggregation_method == 'conf_norm':
            vol_confidences = vol_confidences / vol_confidences.sum(dim=1, keepdim=True)

        # change camera intrinsics
        new_cameras = deepcopy(batch['cameras'])
        for view_i in range(n_views):
            for batch_i in range(batch_size):
                new_cameras[view_i][batch_i].update_after_resize(image_shape, heatmap_shape)

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
                #keypoints_3d[6:7, :3] += 10.*(np.random.rand(1, 3)-0.5)
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
                
            #print('yk: self.training is : {}'.format(self.training))

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
        features = self.process_features(features) # nn.Conv2d(256, 32, 1) 256 -> 32
        features = features.view(batch_size, n_views, *features.shape[1:])
        
        # heatmaps : torch.Size([11, 4, 17, 96, 96])
        # features : torch.Size([11, 4, 32, 96, 96])

        # lift to volume
        volumes = op.unproject_heatmaps(features, proj_matricies, coord_volumes, volume_aggregation_method=self.volume_aggregation_method, vol_confidences=vol_confidences)

        # integral 3d
        volumes = self.volume_net(volumes)
        vol_keypoints_3d, volumes = op.integrate_tensor_3d_with_coordinates(volumes * self.volume_multiplier, coord_volumes, softmax=self.volume_softmax)

        #return vol_keypoints_3d, heatmaps, features
        return vol_keypoints_3d, features, volumes, vol_confidences, cuboids, coord_volumes, base_points
        # backup return format.
        

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
class VolumetricTriangulationNet_TR_TR(VolumetricTriangulationNet_TR):
    def __init__(self, config, device='cuda:0'):
        super().__init__(config, device)
        ######## TRANSFORMER SECTION ########
        '''
        _C.MODEL.OCR = CN()
        _C.MODEL.OCR.MID_CHANNELS = 512
        _C.MODEL.OCR.KEY_CHANNELS = 256
        _C.MODEL.OCR.DROPOUT = 0.05
        _C.MODEL.OCR.SCALE = 1
        
        self.last_inp_channels = 256
        #self.ocr_mid_channels = 512
        self.ocr_mid_channels = 256 # this is out channel of features.
        self.ocr_key_channels = 256
        
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
        '''
        self.last_inp_channels_3d = 32
        self.ocr_mid_channels_3d = 32
        self.ocr_key_channels_3d = 32

        #self.volume_net_protocol_3 = V2VModel(self.num_joints, self.num_joints)

        self.ocr_3d_transformer_protocol_1 = OCR_3D_Module(last_inp_channels=self.last_inp_channels_3d,
                                           ocr_mid_channels=self.ocr_mid_channels_3d,
                                           ocr_key_channels=self.ocr_key_channels_3d,
                                           ocr_dropout=self.ocr_dropout,
                                           num_joints=self.num_joints)

        self.ocr_3d_transformer_protocol_1.apply(weights_init_uniform_rule) # best results. 
        


    def forward(self, images, proj_matricies, batch):
        device = images.device
        batch_size, n_views = images.shape[:2]

        # reshape for backbone forward
        images = images.view(-1, *images.shape[2:])

        # forward backbone
        heatmaps, features, _, vol_confidences = self.backbone(images)
        #heatmaps_pre = heatmaps : torch.Size([44, 17, 96, 96])
        #features_pre = features : torch.Size([44, 256, 96, 96])
        
        # maybe 2D transformer should be here. -yk
        # transformer input : features, heatmaps
        # transformer output : attentioned features

        # OCR module
        features, ocr2d_context = self.ocr_2d_transformer(features, heatmaps, return_context=True)
        #print('debug point omega. ocr2d_context shape is : {}'.format(ocr2d_context.shape))#torch.Size([4, 256, 17, 1])
        #ModuleHelper, SpatialGather_Module, SpatialOCR_Module
        #print('features after transformer is : {}'.format(features.shape))
        #features after transformer is : torch.Size([44, 256, 96, 96])

        # reshape back
        images = images.view(batch_size, n_views, *images.shape[1:])
        heatmaps = heatmaps.view(batch_size, n_views, *heatmaps.shape[1:])
        features = features.view(batch_size, n_views, *features.shape[1:])
        #print(heatmaps.shape) : torch.Size([11, 4, 17, 96, 96])
        #print(features.shape) : torch.Size([11, 4, 256, 96, 96])
        

        if vol_confidences is not None:
            vol_confidences = vol_confidences.view(batch_size, n_views, *vol_confidences.shape[1:])

        # calcualte shapes
        image_shape, heatmap_shape = tuple(images.shape[3:]), tuple(heatmaps.shape[3:])
        n_joints = heatmaps.shape[2]

        # norm vol confidences
        if self.volume_aggregation_method == 'conf_norm':
            vol_confidences = vol_confidences / vol_confidences.sum(dim=1, keepdim=True)

        # change camera intrinsics
        new_cameras = deepcopy(batch['cameras'])
        for view_i in range(n_views):
            for batch_i in range(batch_size):
                new_cameras[view_i][batch_i].update_after_resize(image_shape, heatmap_shape)

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
                
            #print('yk: self.training is : {}'.format(self.training))

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
        features = self.process_features(features) # nn.Conv2d(256, 32, 1) 256 -> 32

        # middle transformer2D should be here.
        #features, ocr2d_context_middle = self.ocr_2d_transformer_middle(features, heatmaps.view(-1, *heatmaps.shape[2:]), return_context=True)
        ocr2d_context_middle = None
        #print('debug point omega. ocr2d_context_middle shape is : {}'.format(ocr2d_context_middle.shape))#torch.Size([4, 32, 17, 1])

        features = features.view(batch_size, n_views, *features.shape[1:])
        
        # heatmaps : torch.Size([11, 4, 17, 96, 96])
        # features : torch.Size([11, 4, 32, 96, 96])

        # lift to volume
        volumes = op.unproject_heatmaps(features, proj_matricies, coord_volumes, volume_aggregation_method=self.volume_aggregation_method, vol_confidences=vol_confidences)
        #print(volumes.shape) torch.Size([11, 32, 64, 64, 64])
        #print(heatmap_volumes.shape)# torch.Size([11, 17, 64, 64, 64]) 

        # protocol 1 of ocr3D had failed.
        # transformer3D should be here.
        heatmap_volumes = op.unproject_heatmaps(heatmaps, proj_matricies, coord_volumes, volume_aggregation_method=self.volume_aggregation_method, vol_confidences=vol_confidences)
        # print('ocr_3d starts.')
        volumes, ocr3d_context = self.ocr_3d_transformer_protocol_1(volumes, heatmap_volumes, return_context=True)
        #print('debug point omega. ocr3d_context shape is : {}'.format(ocr3d_context.shape)) #torch.Size([1, 32, 17, 1])
        

        # integral 3d
        volumes = self.volume_net(volumes)

        # protocol 3.
        #heatmap_volumes = self.volume_net_protocol_3(heatmap_volumes)
        #volumes, ocr3d_context = self.ocr_3d_transformer_protocol_3(volumes, heatmap_volumes, return_context=True)

        #print('debug point zeta. volumes_after_v2v shape is :{}'.format(volumes_after_v2v.shape))
        #debug point zeta. volumes_after_v2v shape is :torch.Size([2, 17, 64, 64, 64])
       # print('debug point zeta. volumes shape is :{}'.format(volumes.shape))
        #debug point zeta. volumes shape is :torch.Size([2, 32, 64, 64, 64])
        # protocol 2 of ocr3D also failed.
        '''
        volumes, ocr3d_context = self.ocr_3d_transformer(volumes, volumes_after_v2v, return_context=True)
        volumes_after_v2v = self.volume_net(volumes) # ???
        '''
        vol_keypoints_3d, volumes = op.integrate_tensor_3d_with_coordinates(volumes * self.volume_multiplier, coord_volumes, softmax=self.volume_softmax)

        # no ocr3d experiment.
        ocr3d_context = None
        
        #return vol_keypoints_3d, heatmaps, features
        return vol_keypoints_3d, features, volumes, vol_confidences, \
        cuboids, coord_volumes, base_points\
        #, ocr2d_context, ocr2d_context_middle, ocr3d_context
        # backup return format.

'''
debug point omega. ocr2d_context shape is : torch.Size([4, 256, 17, 1])
debug point omega. ocr2d_context_middle shape is : torch.Size([4, 32, 17, 1])
debug point omega. ocr3d_context shape is : torch.Size([1, 32, 17, 1])
'''























class VolumetricTriangulationNet_TR_TR_TR(VolumetricTriangulationNet_TR_TR):
    def __init__(self, config, device='cuda:0'):
        super().__init__(config, device)
        ######## TRANSFORMER SECTION ########
        
        self.ocr_2d_transformer_middle = OCR_2D_Module(last_inp_channels=self.last_inp_channels_3d,
                                           ocr_mid_channels=self.ocr_mid_channels_3d,
                                           ocr_key_channels=self.ocr_key_channels_3d,
                                           ocr_dropout=self.ocr_dropout,
                                           num_joints=self.num_joints) 
        

        #self.volume_net_protocol_3 = V2VModel(self.num_joints, self.num_joints)

        self.ocr_2d_transformer_middle.apply(weights_init_uniform_rule) # best results. 
        


    def forward(self, images, proj_matricies, batch):
        device = images.device
        batch_size, n_views = images.shape[:2]

        # reshape for backbone forward
        images = images.view(-1, *images.shape[2:])

        # forward backbone
        heatmaps, features, _, vol_confidences = self.backbone(images)
        #heatmaps_pre = heatmaps : torch.Size([44, 17, 96, 96])
        #features_pre = features : torch.Size([44, 256, 96, 96])
        
        # maybe 2D transformer should be here. -yk
        # transformer input : features, heatmaps
        # transformer output : attentioned features

        # OCR module
        features, ocr2d_context = self.ocr_2d_transformer(features, heatmaps, return_context=True)
        #print('debug point omega. ocr2d_context shape is : {}'.format(ocr2d_context.shape))#torch.Size([4, 256, 17, 1])
        #ModuleHelper, SpatialGather_Module, SpatialOCR_Module
        #print('features after transformer is : {}'.format(features.shape))
        #features after transformer is : torch.Size([44, 256, 96, 96])

        # reshape back
        images = images.view(batch_size, n_views, *images.shape[1:])
        heatmaps = heatmaps.view(batch_size, n_views, *heatmaps.shape[1:])
        features = features.view(batch_size, n_views, *features.shape[1:])
        #print(heatmaps.shape) : torch.Size([11, 4, 17, 96, 96])
        #print(features.shape) : torch.Size([11, 4, 256, 96, 96])
        

        if vol_confidences is not None:
            vol_confidences = vol_confidences.view(batch_size, n_views, *vol_confidences.shape[1:])

        # calcualte shapes
        image_shape, heatmap_shape = tuple(images.shape[3:]), tuple(heatmaps.shape[3:])
        n_joints = heatmaps.shape[2]

        # norm vol confidences
        if self.volume_aggregation_method == 'conf_norm':
            vol_confidences = vol_confidences / vol_confidences.sum(dim=1, keepdim=True)

        # change camera intrinsics
        new_cameras = deepcopy(batch['cameras'])
        for view_i in range(n_views):
            for batch_i in range(batch_size):
                new_cameras[view_i][batch_i].update_after_resize(image_shape, heatmap_shape)

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
                
            #print('yk: self.training is : {}'.format(self.training))

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
        features = self.process_features(features) # nn.Conv2d(256, 32, 1) 256 -> 32

        # middle transformer2D should be here.
        features, ocr2d_context_middle = self.ocr_2d_transformer_middle(features, heatmaps.view(-1, *heatmaps.shape[2:]), return_context=True)
        #ocr2d_context_middle = None
        #print('debug point omega. ocr2d_context_middle shape is : {}'.format(ocr2d_context_middle.shape))#torch.Size([4, 32, 17, 1])

        features = features.view(batch_size, n_views, *features.shape[1:])
        
        # heatmaps : torch.Size([11, 4, 17, 96, 96])
        # features : torch.Size([11, 4, 32, 96, 96])

        # lift to volume
        volumes = op.unproject_heatmaps(features, proj_matricies, coord_volumes, volume_aggregation_method=self.volume_aggregation_method, vol_confidences=vol_confidences)
        #print(volumes.shape) torch.Size([11, 32, 64, 64, 64])
        #print(heatmap_volumes.shape)# torch.Size([11, 17, 64, 64, 64]) 

        # protocol 1 of ocr3D had failed.
        # transformer3D should be here.
        heatmap_volumes = op.unproject_heatmaps(heatmaps, proj_matricies, coord_volumes, volume_aggregation_method=self.volume_aggregation_method, vol_confidences=vol_confidences)
        # print('ocr_3d starts.')
        volumes, ocr3d_context = self.ocr_3d_transformer_protocol_1(volumes, heatmap_volumes, return_context=True)
        #print('debug point omega. ocr3d_context shape is : {}'.format(ocr3d_context.shape)) #torch.Size([1, 32, 17, 1])
        

        # integral 3d
        volumes = self.volume_net(volumes)

        # protocol 3.
        #heatmap_volumes = self.volume_net_protocol_3(heatmap_volumes)
        #volumes, ocr3d_context = self.ocr_3d_transformer_protocol_3(volumes, heatmap_volumes, return_context=True)

        #print('debug point zeta. volumes_after_v2v shape is :{}'.format(volumes_after_v2v.shape))
        #debug point zeta. volumes_after_v2v shape is :torch.Size([2, 17, 64, 64, 64])
       # print('debug point zeta. volumes shape is :{}'.format(volumes.shape))
        #debug point zeta. volumes shape is :torch.Size([2, 32, 64, 64, 64])
        # protocol 2 of ocr3D also failed.
        '''
        volumes, ocr3d_context = self.ocr_3d_transformer(volumes, volumes_after_v2v, return_context=True)
        volumes_after_v2v = self.volume_net(volumes) # ???
        '''
        vol_keypoints_3d, volumes = op.integrate_tensor_3d_with_coordinates(volumes * self.volume_multiplier, coord_volumes, softmax=self.volume_softmax)

        #return vol_keypoints_3d, heatmaps, features
        return vol_keypoints_3d, features, volumes, vol_confidences, \
        cuboids, coord_volumes, base_points\
        , ocr2d_context, ocr2d_context_middle, ocr3d_context
        
        #backup return format. [vol_keypoints_3d, features, volumes, vol_confidences, cuboids, coord_volumes, base_points]

'''
debug point omega. ocr2d_context shape is : torch.Size([4, 256, 17, 1])
debug point omega. ocr2d_context_middle shape is : torch.Size([4, 32, 17, 1])
debug point omega. ocr3d_context shape is : torch.Size([1, 32, 17, 1])
'''