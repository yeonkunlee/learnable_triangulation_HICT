from copy import deepcopy
import numpy as np
import pickle
import random

from scipy.optimize import least_squares

import torch
from torch import nn

from mvn.utils import op, multiview, img, misc, volumetric

#from mvn.models import pose_resnet
from mvn.models.v2v import V2VModel
from mvn.models import pose_body

from mvn.models.pose_hrnet import PoseHighResolutionNet

hrnet_cfg = dict(MODEL=dict(EXTRA='',NUM_JOINTS=''))

# pretrained : "/workspace/learnable_triangulation_unification/learnable_triangulation/data/pretrained/body/hrnet_w48_coco_wholebody_384x288_dark-f5726563_20200918.pth"
#hrnet_cfg = defaultdict()
hrnet_cfg['MODEL']['EXTRA'] = dict(
            STAGE1=dict(
                NUM_MODULES=1,
                NUM_BRANCHES=1,
                BLOCK='BOTTLENECK',
                NUM_BLOCKS=(4, ),
                NUM_CHANNELS=(64, )),
            STAGE2=dict(
                NUM_MODULES=1,
                NUM_BRANCHES=2,
                BLOCK='BASIC',
                NUM_BLOCKS=(4, 4),
                NUM_CHANNELS=(48, 96),
                FUSE_METHOD=None),
            STAGE3=dict(
                NUM_MODULES=4,
                NUM_BRANCHES=3,
                BLOCK='BASIC',
                NUM_BLOCKS=(4, 4, 4),
                NUM_CHANNELS=(48, 96, 192),
                FUSE_METHOD=None),
            STAGE4=dict(
                NUM_MODULES=3,
                NUM_BRANCHES=4,
                BLOCK='BASIC',
                NUM_BLOCKS=(4, 4, 4, 4),
                NUM_CHANNELS=(48, 96, 192, 384),
                FUSE_METHOD=None),
            PRETRAINED_LAYERS = None
            )

hrnet_cfg['MODEL']['NUM_JOINTS'] = 25
hrnet_cfg['MODEL']['EXTRA']['FINAL_CONV_KERNEL'] = 1

hrnet_backbone = PoseHighResolutionNet(hrnet_cfg)
loaded_dict = torch.load('/workspace/learnable_triangulation/data/hrnet_standalone/HRNetW48_384x288_25J.pth')
hrnet_backbone.load_state_dict(loaded_dict, strict=True)


class AlgebraicTriangulationNet_384x288(nn.Module):
    def __init__(self, config, device='cuda:0'):
        super().__init__()

        self.use_confidences = config.model.use_confidences

        config.model.backbone.alg_confidences = False
        config.model.backbone.vol_confidences = False

        if self.use_confidences:
            config.model.backbone.alg_confidences = True

#         self.backbone = pose_resnet.get_pose_net(config.model.backbone, device=device)
        self.backbone = hrnet_backbone.to(device)


        self.heatmap_softmax = config.model.heatmap_softmax
        self.heatmap_multiplier = config.model.heatmap_multiplier


    def forward(self, images, proj_matricies, batch):
        device = images.device
        batch_size, n_views = images.shape[:2]

        # reshape n_views dimension to batch dimension
        images = images.view(-1, *images.shape[2:])

        # forward backbone and integral
        if self.use_confidences:
#             heatmaps, _, alg_confidences, _ = self.backbone(images)
            heatmaps = self.backbone(images)
            alg_confidences_map = heatmaps.view(batch_size*n_views, -1, heatmaps.shape[2]*heatmaps.shape[3])
            alg_confidences = torch.max(alg_confidences_map, dim=2).values
        else:
#             heatmaps, _, _, _ = self.backbone(images)
            heatmaps = self.backbone(images)
            alg_confidences = torch.ones(batch_size * n_views, heatmaps.shape[1]).type(torch.float).to(device)

        heatmaps_before_softmax = heatmaps.view(batch_size, n_views, *heatmaps.shape[1:])
        keypoints_2d, heatmaps = op.integrate_tensor_2d(heatmaps * self.heatmap_multiplier, self.heatmap_softmax)

        # reshape back
        images = images.view(batch_size, n_views, *images.shape[1:])
        heatmaps = heatmaps.view(batch_size, n_views, *heatmaps.shape[1:])
        keypoints_2d = keypoints_2d.view(batch_size, n_views, *keypoints_2d.shape[1:])
        alg_confidences = alg_confidences.view(batch_size, n_views, *alg_confidences.shape[1:])

        # norm confidences
        alg_confidences = alg_confidences / alg_confidences.sum(dim=1, keepdim=True)
        alg_confidences = alg_confidences + 1e-5  # for numerical stability

        # calcualte shapes
        image_shape = tuple(images.shape[3:])
        batch_size, n_views, n_joints, heatmap_shape = heatmaps.shape[0], heatmaps.shape[1], heatmaps.shape[2], tuple(heatmaps.shape[3:])

        # upscale keypoints_2d, because image shape != heatmap shape
        keypoints_2d_transformed = torch.zeros_like(keypoints_2d)
        keypoints_2d_transformed[:, :, :, 0] = keypoints_2d[:, :, :, 0] * (image_shape[1] / heatmap_shape[1])
        keypoints_2d_transformed[:, :, :, 1] = keypoints_2d[:, :, :, 1] * (image_shape[0] / heatmap_shape[0])
        keypoints_2d = keypoints_2d_transformed

        # triangulate
        try:
            keypoints_3d = multiview.triangulate_batch_of_points(
                proj_matricies, keypoints_2d,
                confidences_batch=alg_confidences
            )
        except RuntimeError as e:
            print("Error: ", e)

            #print("confidences =", confidences_batch_pred)
            print("proj_matricies = ", proj_matricies)
            #print("keypoints_2d_batch_pred =", keypoints_2d_batch_pred)
            exit()

        return keypoints_3d, keypoints_2d, heatmaps, alg_confidences

    

    
    
class VolumetricTriangulationNet_384x288(nn.Module):
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

            
        self.backbone = hrnet_backbone.to(device) # output channel : 25
        self.volume_net = V2VModel(25, self.num_joints) # 25 -> 25
#         self.process_features = nn.Sequential(
#             nn.Conv2d(48, 32, 1)
#         )
#         self.volume_net = V2VModel(32, self.num_joints) # 64
        


    def forward(self, images, proj_matricies, batch):
        device = images.device
        batch_size, n_views = images.shape[:2]
        # reshape for backbone forward
        images = images.view(-1, *images.shape[2:])
        # forward backbone
        #heatmaps, features, _, vol_confidences = self.backbone(images)

        features = self.backbone(images) # torch.Size([batch*4, 25, 96, 72]) ?
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
#                 theta_2 = np.random.uniform(0.0, 2 * np.pi)
#                 tilt_pi = np.random.uniform(-np.pi/9, np.pi/9)
            else:
                theta = 0.0
                theta_2 = 0.0
                tilt_pi = 0.0
                

            if self.kind == "coco":
                axis = [0, 1, 0]  # y axis
            elif self.kind == "mpii":
                axis = [0, 0, 1]  # z axis
#                 tilt_axis = [1, 0, 0]  # x axis

            center = torch.from_numpy(base_point).type(torch.float).to(device)

            # rotate
            coord_volume = coord_volume - center
            coord_volume = volumetric.rotate_coord_volume(coord_volume, theta, axis)
#             coord_volume = volumetric.rotate_coord_volume(coord_volume, tilt_pi, tilt_axis)
#             coord_volume = volumetric.rotate_coord_volume(coord_volume, theta_2, axis)
            coord_volume = coord_volume + center

            # transfer
            if self.transfer_cmu_to_human36m:  # different world coordinates
                coord_volume = coord_volume.permute(0, 2, 1, 3)
                inv_idx = torch.arange(coord_volume.shape[1] - 1, -1, -1).long().to(device)
                coord_volume = coord_volume.index_select(1, inv_idx)
                
            coord_volumes[batch_i] = coord_volume

        # process features before unprojecting
#         features = features.view(-1, *features.shape[2:])
#         features = self.process_features(features)
#         features = features.view(batch_size, n_views, *features.shape[1:])

        # lift to volume
        volumes = op.unproject_heatmaps(features, proj_matricies, coord_volumes, volume_aggregation_method=self.volume_aggregation_method, vol_confidences=vol_confidences)

        # integral 3d
        volumes = self.volume_net(volumes)
        vol_keypoints_3d, volumes = op.integrate_tensor_3d_with_coordinates(volumes * self.volume_multiplier, coord_volumes, softmax=self.volume_softmax)

        return vol_keypoints_3d, features, volumes, vol_confidences, cuboids, coord_volumes, base_points    

   