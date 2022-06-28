import os
from collections import defaultdict
import pickle

import numpy as np
import cv2

import torch
from torch.utils.data import Dataset

from mvn.utils.multiview import Camera
from mvn.utils.img import get_square_bbox, resize_image, crop_image, normalize_image, scale_bbox
from mvn.utils import volumetric

import glob
import scipy.io
from scipy.spatial.transform import Rotation as R
import copy


class NC_INTERNAL_MULTIVIEW_Dataset(Dataset):
    """
        NC_fastmotion dataset for synthetic inference by yk.
    """
    def __init__(self,
                 dataset_root='/datasets/NC_internal_multiview/',
                 action_name = 'human_data',
                 camera_idx = [],
                 image_shape=(384, 384),
                 cuboid_side=2000.0,
                 scale_bbox=1.0,
                 norm_image=True,
                 kind="human36m",
                 crop=True,
                 rectangle_bbox=False,
                 is_rawfile = True,
                 gt_dir = None,
                 option_384x288 = False
                 ):

        self.action_name = action_name
        self.dataset_root = dataset_root
        
        if gt_dir:
            self.gt_dir = gt_dir
        else:
            self.gt_dir = '/_NOT_EXIST_DIR.npy'

        self.image_shape = None if image_shape is None else tuple(image_shape)
        self.scale_bbox = scale_bbox
        self.norm_image = norm_image
        self.cuboid_side = cuboid_side
        self.kind = kind
        self.crop = crop
        self.default_camera = []
        self.bboxes = []
        self.camera_idx = camera_idx
        self.is_rawfile = is_rawfile
        self.option_384x288 = option_384x288
        
        
        if self.is_rawfile:
            self.glob_list = glob.glob(os.path.join(self.dataset_root, self.action_name,\
                                           'cam_0', '*.raw'))
        else:
            self.glob_list = glob.glob(os.path.join(self.dataset_root, self.action_name,\
                                           'cam_0', '*.jpg'))
        
        if os.path.exists(self.gt_dir):
            print(' yk : pre-calculated algebraic root motion is ready. load root motion')
            GT_mat = np.load(self.gt_dir)
        else:
            print(' yk : root motion is not exist. zero matrix dumped. this might cause error in inference results. you have to run algebraic model first.')
            GT_mat = np.zeros((len(self.glob_list),17,3))
        
        self.GT_mat = GT_mat
        #Rx_90 = R.from_euler('x', 90, degrees=True)
        #self.Rx_90_dcm = Rx_90.as_dcm()

        n_cameras = len(self.camera_idx)
        
        for camera in self.camera_idx:
            camera_name = 'Calib_Cam'+str(camera)+'.mat'
            camera_mat = scipy.io.loadmat(os.path.join(dataset_root, action_name, camera_name))
            
            R_cam = camera_mat['R_world'].astype(np.float32)
            t_cam = camera_mat['T_world'].reshape(3,1).astype(np.float32)
            
            assert R_cam.shape == (3, 3), 'R is not valid form'
            KK = camera_mat['KK'].reshape(3,3).astype(np.float32)
            kc = camera_mat['kc'].reshape(5,).astype(np.float32)
            #retval_camera = Camera(shot_camera['R'], shot_camera['t'], shot_camera['K'], shot_camera['dist'], 'camera_name')
            default_camera = Camera(R_cam, t_cam, KK, kc, 'C'+str(camera))
            self.default_camera.append(default_camera)
            
            if not rectangle_bbox:
                bbox_filename = 'bboxes_cam_'+str(camera)+'.npy'
                #bbox_filename = self.action_name+'_'+'bboxes_cam'+str(camera)+'.npy'
            else:
                bbox_filename = self.action_name+'_'+'bboxes_cam_rectangle'+str(camera)+'.npy'
            
            bbox_npy = np.load(os.path.join(dataset_root, action_name, bbox_filename))
            self.bboxes.append(bbox_npy)
        
        self.num_keypoints = 16 if kind == "mpii" else 17
        
        
    def __len__(self):
        return len(self.glob_list)
    
    def convert_bbox_384x288(self, bbox):
        #shot['bbox_by_camera_tlbr'][camera_idx] tlbr - ltrb
        bbox = bbox.astype(float)
        center = [int((bbox[1]+bbox[3])/2), int((bbox[0]+bbox[2])/2)]
        area = abs((bbox[3]-bbox[1])*(bbox[2]-bbox[0]))
        height = np.sqrt(area * (4/3))
        width = (3/4) * height

#         x_max = np.max(p_2D[:,0])
#         x_min = np.min(p_2D[:,0])
#         y_max = np.max(p_2D[:,1])
#         y_min = np.min(p_2D[:,1])

#         marginal_witdh = max(x_max-center[1], center[1]-x_min)*2
#         marginal_height = max(y_max-center[0], center[0]-y_min)*2

#         if marginal_witdh > width:
#             width = marginal_witdh
#             height *= marginal_witdh/width

#         if marginal_height > height: # this will be not happened
#             height = marginal_height
#             width *= marginal_height/height


        bbox = [center[1]-int(width/2), center[0]-int(height/2), center[1]+int(width/2), center[0]+int(height/2)]

        return bbox
        
    def __getitem__(self, idx):
        sample = defaultdict(list) # return value
        
        for camera_i in self.camera_idx:
            
            # should be updated.
            frame_idx = idx
            
            
            # load bounding box
            #bbox = shot['bbox_by_camera_tlbr'][camera_idx][[1,0,3,2]] # TLBR to LTRB
            #print('yk test')
            #print(self.bboxes[camera_i].shape) # 200,4 
            bbox = self.bboxes[camera_i][idx, :] # shape is ...4? and LTRB.
            bbox_height = bbox[2] - bbox[0]
            if bbox_height == 0:
                print('bbox height 0 occured.')
                # convention: if the bbox is empty, then this view is missing
                continue
                
            if self.option_384x288:
                # rewrite bbox
                bbox = self.convert_bbox_384x288(bbox) #LTRB

                
                
#             # load image
#             image_path = os.path.join(
#                 self.dataset_root, self.action_name,\
#                 'cam_%d'%camera_i, 'cam%d_'%camera_i + '%04d.png' % (frame_idx))

#             assert os.path.isfile(image_path), '%s doesn\'t exist' % image_path
#             image = cv2.imread(image_path)
            
            
            # load image
            
            if self.is_rawfile:
                image_path = os.path.join(
                    self.dataset_root, self.action_name,\
                    'cam_%d'%camera_i, '%04d.raw' % (frame_idx))

                assert os.path.isfile(image_path), '%s doesn\'t exist' % image_path
                image = np.fromfile(image_path, dtype='uint8').reshape(1536,2048)
                image = cv2.cvtColor(image, cv2.COLOR_BayerRG2RGB)
            else:
#                 image_path = os.path.join(
#                     self.dataset_root, self.action_name,\
#                     'cam_%d'%camera_i, 'cam%d_'%camera_i + '%04d.png' % (frame_idx))
                image_path = os.path.join(
                    self.dataset_root, self.action_name,\
                    'cam_%d'%camera_i, 'img_%04d.jpg' % (frame_idx))

                assert os.path.isfile(image_path), '%s doesn\'t exist' % image_path
                image = cv2.imread(image_path)
            

            
            if np.isnan(np.sum(bbox)):
                bbox = np.array([0., 0., image.shape[1], image.shape[0]])
                print('bbox height NaN occured.')

            # scale the bounding box
            bbox = scale_bbox(bbox, self.scale_bbox)
            
            original_image = image
            #original_image = cv2.resize(original_image, dsize=(1000,1000))
            sample['original_image'].append(original_image)

            # load camera
            retval_camera = copy.deepcopy(self.default_camera[camera_i])
            original_camera = copy.deepcopy(retval_camera)

            if self.crop:
                # crop image
                image = crop_image(image, bbox)
                retval_camera.update_after_crop(bbox)
                
            if self.image_shape is not None:
                # resize
                image_shape_before_resize = image.shape[:2]
                image = resize_image(image, self.image_shape)
                retval_camera.update_after_resize(image_shape_before_resize, self.image_shape)
                sample['image_shapes_before_resize'].append(image_shape_before_resize)

            if self.norm_image:
                image = normalize_image(image)

            sample['images'].append(image)
            sample['detections'].append(bbox + (1.0,)) # TODO add real confidences
            sample['cameras'].append(retval_camera)
            sample['proj_matrices'].append(retval_camera.projection)
            sample['original_cameras'].append(original_camera)

        sample.default_factory = None

        
        #all_3D_point = np.zeros((17, 3))
        all_3D_point = self.GT_mat[idx,:,:]
        #for joint in range(self.GT_mat['GTpose2'][0,idx].shape[0]):
        #    one_point = self.GT_mat['GTpose2'][0,idx][joint, :]
        #    rotated_point = np.matmul(self.Rx_90_dcm, one_point.reshape(3,1))
        #    all_3D_point[joint, :] = rotated_point[:,0]
        sample['keypoints_3d'] = all_3D_point
        
        return sample

