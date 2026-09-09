import numpy as np
import torch
from einops import rearrange

class Camera(object):
    def __init__(self, c2w):
        c2w_mat = np.array(c2w).reshape(4, 4)
        self.c2w_mat = c2w_mat
        self.w2c_mat = np.linalg.inv(c2w_mat)

def get_relative_pose(cam_params):
    abs_w2cs = [cam_param.w2c_mat for cam_param in cam_params]
    abs_c2ws = [cam_param.c2w_mat for cam_param in cam_params]
    target_cam_c2w = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    abs2rel = target_cam_c2w @ abs_w2cs[0]
    ret_poses = [target_cam_c2w, ] + [abs2rel @ abs_c2w for abs_c2w in abs_c2ws[1:]]
    ret_poses = np.array(ret_poses, dtype=np.float32)
    return ret_poses

def batch_cam2pose_embedding(batch_cam, original_coordinate='blender'):
    batch_cam = batch_cam.cpu().to(torch.float32).numpy()
    if batch_cam.ndim == 4:
        pass # B, F, 3, 4 or # B, F, 4, 4
    elif batch_cam.ndim == 3:
        B, F, L = batch_cam.shape
        batch_cam = batch_cam.reshape(B, F, -1, 4)
    B, _, _, _ = batch_cam.shape
    relative_poses = []

    for cams in batch_cam:
        c2ws = []
        for c2w in cams:
            if original_coordinate == 'blender':
                c2w = blender2opencv(c2w)
            elif original_coordinate == 'UE':
                c2w = UE2opencv(c2w)
            elif original_coordinate == 'opencv':
                c2w = c2w
            else:
                raise ValueError(f"Invalid original coordinate {original_coordinate}")
            c2ws.append(c2w)
        tgt_cam_params = [Camera(cam_param) for cam_param in c2ws]
        for i in range(len(tgt_cam_params)):
            relative_pose = get_relative_pose([tgt_cam_params[0], tgt_cam_params[i]])
            relative_poses.append(torch.as_tensor(relative_pose)[:,:3,:][1])
    pose_embedding = torch.stack(relative_poses, dim=0)  # fx3x4
    pose_embedding = rearrange(pose_embedding, '(b f) c d -> b f (c d)', b=B)
    return pose_embedding


def UE2opencv(c2w):
    c2w = c2w[:, [1, 2, 0, 3]]
    c2w[:3, 1] *= -1.
    c2w[:3, 3] /= 100 

    return c2w


def blender2opencv(c2w):
    c2w[:3, 1] *= -1.
    c2w[:3, 2] *= -1.
    c2w[:3, 3] /= 0.1 # previous: /100 /0.1

    return c2w