import argparse
import logging
import math
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict, OrderedDict
from pathlib import Path
from tqdm import tqdm
from torchvision.utils import make_grid, save_image
from datetime import datetime

from vggt.vggt.models.vggt import VGGT
from RAE.src.stage2.models import Stage2ModelProtocol
from RAE.src.stage2.transport import create_transport, Sampler

from RAE.src.utils.model_utils import instantiate_from_config
from RAE.src.utils.train_utils_gf import parse_configs, get_autocast_scaler, ETH3DDataset
from RAE.src.utils.dist_utils import setup_distributed, cleanup_distributed
from RAE.src.utils.depth_utils import compute_depth_metrics
from RAE.src.utils.pose_utils import enc_to_se3, get_aligned_centers, se3_to_relative_pose_error
from RAE.src.utils.vis_utils import process_depth_batch, process_depth_error_batch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

def align_to_first(poses):
    if poses.dim() == 4: poses = poses.squeeze(0)
    first_inv = torch.inverse(poses[0])
    return first_inv.unsqueeze(0) @ poses

def get_centers(se3_poses):
    if se3_poses.dim() == 4: se3_poses = se3_poses.squeeze(0)
    R = se3_poses[:, :3, :3]
    t = se3_poses[:, :3, 3:4]
    centers = -torch.bmm(R.transpose(-1, -2), t).squeeze(-1)
    return centers.detach().cpu().numpy()

def extrinsic_to_cam_center(T):
    if T.shape[-2:] == (3,4):
        R = T[..., :3, :3]
        t = T[..., :3, 3]
    elif T.shape[-2:] == (4,4):
        R = T[..., :3, :3]
        t = T[..., :3, 3]
    else:
        raise ValueError

    C = -torch.matmul(R.transpose(-1,-2), t.unsqueeze(-1)).squeeze(-1)
    return C

def calculate_auc_np(r_error, t_error, max_threshold=30):
    max_errors = np.maximum(r_error, t_error)
    bins = np.arange(max_threshold + 1)
    histogram, _ = np.histogram(max_errors, bins=bins)
    num_pairs = float(len(max_errors))
    return np.mean(np.cumsum(histogram.astype(float) / num_pairs))

def to_rgb_tensor(img_batch, is_depth=False):
    if is_depth:
        vis_np = process_depth_batch(img_batch) 
        return torch.from_numpy(vis_np).permute(0, 3, 1, 2).float() / 255.0
    else:
        if img_batch.shape[1] == 1:
            img_batch = img_batch.repeat(1, 3, 1, 1)
        return img_batch.detach().cpu().float()

def error_to_tensor(vis_np):
    return torch.from_numpy(vis_np).permute(0, 3, 1, 2).float() / 255.0

def plot_cam_trajectory(gt_c, pred_c, lq_c, save_path, only_pred=False):
    save_dir = os.path.dirname(save_path)
    os.makedirs(save_dir, exist_ok=True)
    
    fig = plt.figure(figsize=(20, 10))
    
    # 3D View
    ax1 = fig.add_subplot(121, projection='3d')
    if not only_pred:
        ax1.plot(gt_c[:, 0], gt_c[:, 1], gt_c[:, 2], 'g-o', label='Ground Truth', markersize=6, linewidth=3)
    ax1.plot(pred_c[:, 0], pred_c[:, 1], pred_c[:, 2], 'r-o', label='Restored (Relative)', markersize=5, linewidth=2)
    ax1.plot(lq_c[:, 0], lq_c[:, 1], lq_c[:, 2], 'b--x', label='LQ (Relative)', markersize=4, alpha=0.6)
    ax1.scatter(0, 0, 0, color='black', s=150, label='Start (Origin)', zorder=10)
    ax1.set_title("3D Camera Trajectory (Relative to First Frame)")
    ax1.legend()

    # Top-down View
    ax2 = fig.add_subplot(122)
    if not only_pred:
        ax2.plot(gt_c[:, 0], gt_c[:, 2], 'g-o', label='GT', markersize=7, linewidth=3)
    ax2.plot(pred_c[:, 0], pred_c[:, 2], 'r-o', label='Restored', markersize=6)
    ax2.plot(lq_c[:, 0], lq_c[:, 2], 'b--x', label='LQ', alpha=0.5)
    ax2.scatter(0, 0, color='black', s=100) 
    ax2.set_title("Top-down View (X-Z Plane)")
    ax2.set_aspect('equal')
    ax2.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def create_eval_grid(batch, lq_out, res_out):
    idx = 0 
    v_clean, v_deg, v_gt_depth = batch["clean_img"][idx], batch["deg_img"][idx], batch["gt_depth"][idx]
    v_lq_depth = lq_out["depth"][idx].squeeze(-1)
    v_res_depth = res_out["depth"][idx].squeeze(-1)
    
    v_input_err = process_depth_error_batch(v_gt_depth, v_lq_depth.unsqueeze(-1))
    v_output_err = process_depth_error_batch(v_gt_depth, v_res_depth.unsqueeze(-1))
    
    V = v_clean.shape[0]
    row_elements = [
        to_rgb_tensor(v_clean), to_rgb_tensor(v_deg),
        to_rgb_tensor(v_gt_depth, True), to_rgb_tensor(v_lq_depth, True),
        error_to_tensor(v_input_err), to_rgb_tensor(v_res_depth, True),
        error_to_tensor(v_output_err)
    ]
    
    combined = []
    for i in range(V):
        for el in row_elements:
            combined.append(el[i:i+1])
    return make_grid(torch.cat(combined, dim=0), nrow=7, padding=10, pad_value=1.0)

def sim3_align(pred, gt):
    """
    Align pred trajectory to gt trajectory using Sim3
    pred, gt: (V,3) numpy arrays
    Returns aligned_pred (V,3)
    """

    pred_mean = pred.mean(axis=0)
    gt_mean = gt.mean(axis=0)

    hq_pred_centered = pred - pred_mean
    gt_centered = gt - gt_mean

    # scale
    scale = np.linalg.norm(gt_centered) / (np.linalg.norm(hq_pred_centered) + 1e-8)
    hq_pred_centered *= scale

    # rotation (Umeyama)
    H = hq_pred_centered.T @ gt_centered
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    pred_aligned = (R @ hq_pred_centered.T).T + gt_mean

    return pred_aligned

@torch.no_grad()
def run_evaluation(global_step, experiment_dir, val_loader, vggt_model, ema_model, eval_sampler, device, logger, wandb_utils=None):
    logger.info(f"--- Starting Full Evaluation at Step {global_step} ---")
    vggt_model.eval()
    ema_model.eval()
    
    vis_depth_dir = os.path.join(experiment_dir, "visualizations/val_depth")
    vis_pose_dir = os.path.join(experiment_dir, "visualizations/val_pose")
    os.makedirs(vis_depth_dir, exist_ok=True); os.makedirs(vis_pose_dir, exist_ok=True)

    metrics_accum = defaultdict(float)
    all_r, all_t, lq_r, lq_t = [], [], [], []
    total_depth_frames = 0
    num_sequences = 0

    for i, batch in enumerate(tqdm(val_loader, desc="Evaluating")):
        if i >= 10: break 
        
        deg_img = batch["deg_img"].to(device)
        gt_poses = batch["gt_poses"].to(device)
        
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            lq_out = vggt_model(deg_img, extract_layer_num=3)
            lq_latent = lq_out["extracted_latent"][:, :, :, 1024:]
            
            xt = torch.randn_like(lq_latent) + lq_latent
            restored_latent = eval_sampler(xt, ema_model.forward, img=deg_img)[-1].float()
            res_out = vggt_model(deg_img, extract_layer_num=3, vggt_result={'restored_latent': restored_latent}, change_latent=True)

        for v in range(deg_img.shape[1]):
            d_m = compute_depth_metrics(batch["gt_depth"][0, v, 0], res_out["depth"][0, v].squeeze(-1).cpu())
            if d_m:
                for k, val in d_m.items(): metrics_accum[f"depth_{k}"] += val
                total_depth_frames += 1

        v_gt_se3 = gt_poses[0]
        v_res_se3 = enc_to_se3(res_out["pose_enc"], deg_img.shape[-2:]).squeeze(0)
        v_lq_se3 = enc_to_se3(lq_out["pose_enc"], deg_img.shape[-2:]).squeeze(0)

        r_err, t_err = se3_to_relative_pose_error(v_res_se3, v_gt_se3, v_res_se3.shape[0])
        lq_r_err, lq_t_err = se3_to_relative_pose_error(v_lq_se3, v_gt_se3, v_lq_se3.shape[0])

        all_r.extend(r_err.cpu().numpy()); all_t.extend(t_err.cpu().numpy())
        lq_r.extend(lq_r_err.cpu().numpy()); lq_t.extend(lq_t_err.cpu().numpy())

        if i < 3:
            # gt_c = get_centers(align_to_first(v_gt_se3))
            # res_c = get_centers(align_to_first(v_res_se3))
            # lq_c = get_centers(align_to_first(v_lq_se3))
            gt_c = extrinsic_to_cam_center(v_gt_se3).cpu().numpy()  
            res_c = extrinsic_to_cam_center(v_res_se3).cpu().numpy()
            lq_c = extrinsic_to_cam_center(v_lq_se3).cpu().numpy()

            res_c = sim3_align(res_c, gt_c)
            lq_c = sim3_align(lq_c, gt_c)
            
            plot_cam_trajectory(gt_c, res_c, lq_c, os.path.join(vis_pose_dir, f"step_{global_step}_traj_{i:03d}.png"))
            save_image(create_eval_grid(batch, lq_out, res_out), os.path.join(vis_depth_dir, f"step_{global_step}_depth_{i:03d}.png"))
        
        num_sequences += 1

    auc30 = calculate_auc_np(np.array(all_r), np.array(all_t), 30)
    auc15 = calculate_auc_np(np.array(all_r), np.array(all_t), 15)
    lq_auc30 = calculate_auc_np(np.array(lq_r), np.array(lq_t), 30)
    lq_auc15 = calculate_auc_np(np.array(lq_r), np.array(lq_t), 15)
    
    final_stats = {
        "val/auc30": auc30, 
        "val/auc15": auc15,
        "val/lq_auc30": lq_auc30,
        "val/lq_auc15": lq_auc15
    }

    for k, v in metrics_accum.items():
        final_stats[f"val/{k}"] = v / total_depth_frames

    logger.info(f"Step {global_step} Eval: AUC30={auc30:.4f} | AUC15={auc15:.4f} | AbsRel={final_stats.get('val/depth_abs_rel', 0):.4f}")
    if wandb_utils: wandb_utils.log(final_stats, step=global_step)
    
    vggt_model.train() 