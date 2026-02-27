import os 
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.spatial import procrustes
from datetime import datetime
import sys
import logging
from vggt.vggt.utils.rotation import mat_to_quat
from vggt.vggt.utils.pose_enc import pose_encoding_to_extri_intri

def setup_logger(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    log_filename = datetime.now().strftime("eval_%Y%m%d_%H%M%S.log")
    log_path = os.path.join(output_dir, log_filename)

    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s',
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger()

def plot_trajectory(gt_c, pred_c, lq_c, save_path):
    save_dir = os.path.dirname(save_path)
    os.makedirs(save_dir, exist_ok=True)

    pred_c_aligned = pred_c
    lq_c_aligned = lq_c

    fig = plt.figure(figsize=(20, 10))
    
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot(gt_c[:, 0], gt_c[:, 1], gt_c[:, 2], 'g-o', label='Ground Truth', markersize=6, linewidth=3)
    ax1.plot(pred_c_aligned[:, 0], pred_c_aligned[:, 1], pred_c_aligned[:, 2], 'r-o', label='Restored (Relative)', markersize=5, linewidth=2)
    ax1.plot(lq_c_aligned[:, 0], lq_c_aligned[:, 1], lq_c_aligned[:, 2], 'b--x', label='LQ (Relative)', markersize=4, alpha=0.6)
    
    ax1.scatter(0, 0, 0, color='black', s=150, label='Start (Origin)', zorder=10)
    
    ax1.set_title("3D Camera Trajectory (Relative to First Frame)", fontsize=15)
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.legend()

    ax2 = fig.add_subplot(122)
    ax2.plot(gt_c[:, 0], gt_c[:, 2], 'g-o', label='GT', markersize=7, linewidth=3)
    ax2.plot(pred_c_aligned[:, 0], pred_c_aligned[:, 2], 'r-o', label='Restored', markersize=6)
    ax2.plot(lq_c_aligned[:, 0], lq_c_aligned[:, 2], 'b--x', label='LQ', alpha=0.5)
    ax2.scatter(0, 0, color='black', s=100) 

    ax2.set_title("Top-down View (X-Z Plane)", fontsize=15)
    ax2.set_xlabel('X (Side-to-Side)')
    ax2.set_ylabel('Z (Depth)')
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.legend()
    ax2.set_aspect('equal') 

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"[*] Trajectory visualization saved to: {save_path}")

def enc_to_se3(pose_enc, img_size):
    ext, _ = pose_encoding_to_extri_intri(pose_enc, img_size)
    ext = ext[0]
    device = ext.device
    add_row = torch.tensor([0, 0, 0, 1], dtype=torch.float64, device=device).expand(ext.size(0), 1, 4)
    return torch.cat((ext, add_row), dim=1) # [V, 4, 4]

def get_aligned_centers(se3_poses):
    first_inv = torch.inverse(se3_poses[0])
    aligned_poses = first_inv @ se3_poses 
    
    centers = aligned_poses[:, :3, 3]
    
    return centers.cpu().numpy()

def get_centers(se3_poses):
    if se3_poses.dim() == 4:
        se3_poses = se3_poses.squeeze(0)

    centers = se3_poses[:, :3, 3] 
    return centers.detach().cpu().numpy()

def build_pair_index(N, B=1):
    """
    Build indices for all possible pairs of frames.

    Args:
        N: Number of frames
        B: Batch size

    Returns:
        i1, i2: Indices for all possible pairs
    """
    i1_, i2_ = torch.combinations(torch.arange(N), 2, with_replacement=False).unbind(-1)
    i1, i2 = [(i[None] + torch.arange(B)[:, None] * N).reshape(-1) for i in [i1_, i2_]]
    return i1, i2

def closed_form_inverse_se3(se3, R=None, T=None):
    """
    Compute the inverse of each 4x4 (or 3x4) SE3 matrix in a batch.

    If `R` and `T` are provided, they must correspond to the rotation and translation
    components of `se3`. Otherwise, they will be extracted from `se3`.

    Args:
        se3: Nx4x4 or Nx3x4 array or tensor of SE3 matrices.
        R (optional): Nx3x3 array or tensor of rotation matrices.
        T (optional): Nx3x1 array or tensor of translation vectors.

    Returns:
        Inverted SE3 matrices with the same type and device as `se3`.

    Shapes:
        se3: (N, 4, 4)
        R: (N, 3, 3)
        T: (N, 3, 1)
    """
    # Check if se3 is a numpy array or a torch tensor
    is_numpy = isinstance(se3, np.ndarray)

    # Validate shapes
    if se3.shape[-2:] != (4, 4) and se3.shape[-2:] != (3, 4):
        raise ValueError(f"se3 must be of shape (N,4,4), got {se3.shape}.")

    # Extract R and T if not provided
    if R is None:
        R = se3[:, :3, :3]  # (N,3,3)
    if T is None:
        T = se3[:, :3, 3:]  # (N,3,1)

    # Transpose R
    if is_numpy:
        # Compute the transpose of the rotation for NumPy
        R_transposed = np.transpose(R, (0, 2, 1))
        # -R^T t for NumPy
        top_right = -np.matmul(R_transposed, T)
        inverted_matrix = np.tile(np.eye(4), (len(R), 1, 1))
    else:
        R_transposed = R.transpose(1, 2)  # (N,3,3)
        top_right = -torch.bmm(R_transposed, T)  # (N,3,1)
        inverted_matrix = torch.eye(4, 4)[None].repeat(len(R), 1, 1)
        inverted_matrix = inverted_matrix.to(R.dtype).to(R.device)

    inverted_matrix[:, :3, :3] = R_transposed
    inverted_matrix[:, :3, 3:] = top_right

    return inverted_matrix

def rotation_angle(rot_gt, rot_pred, batch_size=None, eps=1e-15):
    """
    Calculate rotation angle error between ground truth and predicted rotations.

    Args:
        rot_gt: Ground truth rotation matrices
        rot_pred: Predicted rotation matrices
        batch_size: Batch size for reshaping the result
        eps: Small value to avoid numerical issues

    Returns:
        Rotation angle error in degrees
    """
    q_pred = mat_to_quat(rot_pred)
    q_gt = mat_to_quat(rot_gt)

    loss_q = (1 - (q_pred * q_gt).sum(dim=1) ** 2).clamp(min=eps)
    err_q = torch.arccos(1 - 2 * loss_q)

    rel_rangle_deg = err_q * 180 / np.pi

    if batch_size is not None:
        rel_rangle_deg = rel_rangle_deg.reshape(batch_size, -1)

    return rel_rangle_deg

def compare_translation_by_angle(t_gt, t, eps=1e-15, default_err=1e6):
    """
    Normalize the translation vectors and compute the angle between them.

    Args:
        t_gt: Ground truth translation vectors
        t: Predicted translation vectors
        eps: Small value to avoid division by zero
        default_err: Default error value for invalid cases

    Returns:
        Angular error between translation vectors in radians
    """
    t_norm = torch.norm(t, dim=1, keepdim=True)
    t = t / (t_norm + eps)

    t_gt_norm = torch.norm(t_gt, dim=1, keepdim=True)
    t_gt = t_gt / (t_gt_norm + eps)

    loss_t = torch.clamp_min(1.0 - torch.sum(t * t_gt, dim=1) ** 2, eps)
    err_t = torch.acos(torch.sqrt(1 - loss_t))

    err_t[torch.isnan(err_t) | torch.isinf(err_t)] = default_err
    return err_t

def translation_angle(tvec_gt, tvec_pred, batch_size=None, ambiguity=True):
    """
    Calculate translation angle error between ground truth and predicted translations.

    Args:
        tvec_gt: Ground truth translation vectors
        tvec_pred: Predicted translation vectors
        batch_size: Batch size for reshaping the result
        ambiguity: Whether to handle direction ambiguity

    Returns:
        Translation angle error in degrees
    """
    rel_tangle_deg = compare_translation_by_angle(tvec_gt, tvec_pred)
    rel_tangle_deg = rel_tangle_deg * 180.0 / np.pi

    if ambiguity:
        rel_tangle_deg = torch.min(rel_tangle_deg, (180 - rel_tangle_deg).abs())

    if batch_size is not None:
        rel_tangle_deg = rel_tangle_deg.reshape(batch_size, -1)

    return rel_tangle_deg

def se3_to_relative_pose_error(pred_se3, gt_se3, num_frames):
    """
    Compute rotation and translation errors between predicted and ground truth poses.
    This function assumes the input poses are world-to-camera (w2c) transformations.

    Args:
        pred_se3: Predicted SE(3) transformations (w2c), shape (N, 4, 4)
        gt_se3: Ground truth SE(3) transformations (w2c), shape (N, 4, 4)
        num_frames: Number of frames (N)

    Returns:
        Rotation and translation angle errors in degrees
    """
    pair_idx_i1, pair_idx_i2 = build_pair_index(num_frames)

    relative_pose_gt = gt_se3[pair_idx_i1].bmm(
        closed_form_inverse_se3(gt_se3[pair_idx_i2])
    )
    relative_pose_pred = pred_se3[pair_idx_i1].bmm(
        closed_form_inverse_se3(pred_se3[pair_idx_i2])
    )

    rel_rangle_deg = rotation_angle(
        relative_pose_gt[:, :3, :3], relative_pose_pred[:, :3, :3]
    )
    rel_tangle_deg = translation_angle(
        relative_pose_gt[:, :3, 3], relative_pose_pred[:, :3, 3]
    )

    return rel_rangle_deg, rel_tangle_deg
