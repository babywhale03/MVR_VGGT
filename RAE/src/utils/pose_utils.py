import os 
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.spatial import procrustes
from vggt.vggt.utils.pose_enc import pose_encoding_to_extri_intri

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