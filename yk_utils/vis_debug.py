import cv2
import numpy as np
import copy




def visualize_batch_input(batch_image):
    batch_image = copy.deepcopy(batch_image)
    batch_image = batch_image.detach().cpu().numpy()
    # batch_image shape is (1, 4, 3, 384, 288)
    visualze_batch_i = 0
    batch_image = batch_image[visualze_batch_i]

    vis_image_list = []
    for i_cam in range(4):
        image = batch_image[i_cam] # 3, 384, 288
        image = image.transpose((1, 2, 0)) # 384, 288, 3
        # denormalize image with min and max
        min = np.min(image)
        max = np.max(image)
        image = (image - min) / (max - min) * 255
        
        image = image.astype(np.uint8)
        vis_image_list.append(image)

    vertical_concated = np.concatenate(vis_image_list, axis=1)

    return vertical_concated

def make_output(batch_sample, idx_cam, keypoints_3D, joint_color = (0, 0, 255)):
    batch_image_cam = batch_sample['images'][0, :, :, :, :][idx_cam,:,:,:]
    batch_image_cam = cv2.normalize(batch_image_cam, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    batch_cameras = batch_sample['cameras']
    batch_camera = batch_cameras[idx_cam][0]
    
    P = batch_camera.projection
    #D2_keypoints = keypoints_2d.detach().cpu().numpy().astype(np.float64)[0] # (view, joint, xy)
    D3_keypoints = keypoints_3D.detach().cpu().numpy().astype(np.float64)[0]
    num_joint = keypoints_3D.shape[1]
    D3_keypoints = cv2.hconcat([D3_keypoints, np.ones((num_joint,1))]) # for # [17,4] homogenious
    
    for joint in range(num_joint):
        one_point = D3_keypoints[joint,:4]

        projected = np.matmul(P, one_point)
        projected = projected/projected[-1]
        batch_image_cam = cv2.circle(batch_image_cam, (int(projected[0]),int(projected[1])), 0, joint_color, 7)
        #batch_image_cam = cv2.circle(batch_image_cam, (int(D2_keypoints[idx_cam,joint,0]),int(D2_keypoints[idx_cam,joint,1])), 0, [255, 0, 0], 10)

    return batch_image_cam

def make_concated_output(batch, keypoints_3d_extended_gt, joint_color = (0, 0, 255)):
    batch_image_cam0 = make_output(batch, 0, keypoints_3d_extended_gt, joint_color)
    batch_image_cam1 = make_output(batch, 1, keypoints_3d_extended_gt, joint_color)
    batch_image_cam2 = make_output(batch, 2, keypoints_3d_extended_gt, joint_color)
    batch_image_cam3 = make_output(batch, 3, keypoints_3d_extended_gt, joint_color)

    concated = cv2.hconcat([batch_image_cam0, batch_image_cam1,\
                            batch_image_cam2, batch_image_cam3])

    return concated   