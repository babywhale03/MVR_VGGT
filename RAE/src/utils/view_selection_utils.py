import h5py
import numpy as np
import glob
from pathlib import Path

def load_camera_poses(cam_dir):

    with h5py.File(f"{cam_dir}/camera_keyframe_positions.hdf5", "r") as f:
        positions = f["dataset"][:]   # (N,3)

    with h5py.File(f"{cam_dir}/camera_keyframe_orientations.hdf5", "r") as f:
        orientations = f["dataset"][:]   # (N,3,3)

    poses = []

    for R, t in zip(orientations, positions):

        T = np.eye(4)
        T[:3,:3] = R
        T[:3,3] = t

        poses.append(T)

    return poses

def load_position(path):

    with h5py.File(path, "r") as f:
        pos = f["dataset"][:]   # (H,W,3)

    pos = pos.reshape(-1,3)

    return pos


cam_dir = "/mnt/dataset1/MV_Restoration/hypersim/data/ai_001_001/_detail/cam_00"
pose_dir = "/mnt/dataset1/MV_Restoration/hypersim/data/ai_001_001/images/scene_cam_00_geometry_hdf5/frame.0000.position.hdf5"
load_camera_poses(cam_dir)
load_position(pose_dir)