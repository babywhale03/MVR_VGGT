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

def calculate_auc_np(r_error, t_error, max_threshold=30):
    max_errors = np.maximum(r_error, t_error)
    
    bins = np.arange(max_threshold + 1)
    
    histogram, _ = np.histogram(max_errors, bins=bins)
    num_pairs = float(len(max_errors))
    
    normalized_histogram = histogram.astype(float) / num_pairs
    auc = np.mean(np.cumsum(normalized_histogram))
    return auc

def setup_eval_logger(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, f"eval_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s',
        handlers=[logging.FileHandler(log_path), logging.StreamHandler()]
    )
    return logging.getLogger()

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

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--results-dir", type=str, default="eval_eth3d_final")
    parser.add_argument("--num-sequences", type=int, default=10)
    parser.add_argument("--kernel-size", type=int, default=100)
    return parser.parse_args()

def main():
    args = parse_args()
    rank, world_size, device = setup_distributed()
    
    eval_dir = os.path.join(args.results_dir, f"{Path(args.ckpt).stem}_newv8_gt_align_fix")
    logger = setup_eval_logger(eval_dir)
    
    full_cfg = OmegaConf.load(args.config)
    configs = parse_configs(full_cfg)
    model_cfg, trans_cfg, samp_cfg, _, misc_cfg, train_cfg, data_cfg, _ = configs

    vggt_model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)
    vggt_model.eval().requires_grad_(False)
    
    model = instantiate_from_config(model_config if 'model_config' in locals() else model_cfg).to(device)
    model.eval()
    
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["ema"] if "ema" in ckpt else ckpt["model"])

    eth3d_cfg = data_cfg["val"]["library"]["eth3d"]
    val_ds = ETH3DDataset(
        clean_img_paths=eth3d_cfg["clean_img_path"],
        gt_depth_paths=eth3d_cfg["gt_depth_path"],
        mode='val', num_view=10, view_sel={'strategy': 'sequential'}, kernel_size=args.kernel_size
    )
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=4)

    vggt_patch_size = model_cfg["params"].get("vggt_patch_size", 14)
    pH, pW = 518 // vggt_patch_size, 518 // vggt_patch_size
    shift_dim = (5 + pH * pW) * 2048
    time_dist_shift = math.sqrt(shift_dim / misc_cfg.get("time_dist_shift_base", 4096))
    
    transport = create_transport(path_type="Linear", prediction="velocity", time_dist_shift=time_dist_shift)
    eval_sampler = Sampler(transport).sample_ode(sampling_method='euler', num_steps=50, atol=1e-6, rtol=1e-3)

    vis_depth_dir, vis_pose_dir = os.path.join(eval_dir, "depth_vis"), os.path.join(eval_dir, "pose_vis")
    os.makedirs(vis_depth_dir, exist_ok=True); os.makedirs(vis_pose_dir, exist_ok=True)

    metrics_accum = defaultdict(float)
    all_r_errors, all_t_errors = [], []
    lq_all_r_errors, lq_all_t_errors = [], []

    total_frames = 0

    logger.info(f"Starting Evaluation on ETH3D... Total sequences to evaluate: {args.num_sequences}")

    with torch.no_grad():
        for i, batch in enumerate(tqdm(val_loader)):
            # if i >= args.num_sequences: break

            deg_img = batch["deg_img"].to(device)
            gt_poses = batch["gt_poses"].to(device)
            
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                lq_out = vggt_model(deg_img, extract_layer_num=3)
                lq_latent = lq_out["extracted_latent"][:, :, :, 1024:]
                
                xt = torch.randn_like(lq_latent) + lq_latent
                restored_latent = eval_sampler(xt, model.forward, img=deg_img)[-1].float()
                restored_latent = torch.cat([lq_latent[:, :, :5, :], restored_latent[:, :, 5:, :]], dim=2)
                
                res_out = vggt_model(deg_img, extract_layer_num=3, vggt_result={'restored_latent': restored_latent}, change_latent=True)

            for v in range(deg_img.shape[1]):
                d_m = compute_depth_metrics(batch["gt_depth"][0, v, 0], res_out["depth"][0, v].squeeze(-1).cpu())
                if d_m:
                    for k, val in d_m.items(): metrics_accum[f"depth_{k}"] += val
                    total_frames += 1

            def align_to_first(poses):
                if poses.dim() == 4: # [1, V, 4, 4] -> [V, 4, 4]
                    poses = poses.squeeze(0)
                first_inv = torch.inverse(poses[0])
                return first_inv.unsqueeze(0) @ poses
            
            def get_centers(se3_poses):
                R = se3_poses[:, :3, :3]
                t = se3_poses[:, :3, 3:4]
                centers = -torch.bmm(R.transpose(-1, -2), t).squeeze(-1)
                return centers.detach().cpu().numpy()
                        
            v_gt_se3 = gt_poses[0] # [V, 4, 4]
            v_res_se3 = enc_to_se3(res_out["pose_enc"], deg_img.shape[-2:]) # [V, 4, 4] 
            v_lq_se3 = enc_to_se3(lq_out["pose_enc"], deg_img.shape[-2:])

            # gt_se3_rel = align_to_first(v_gt_se3)
            # res_se3_rel = align_to_first(v_res_se3)
            # lq_se3_rel = align_to_first(v_lq_se3)

            # gt_c = get_centers(gt_se3_rel) 
            # res_c = get_centers(res_se3_rel)
            # lq_c = get_centers(lq_se3_rel)
            gt_c = extrinsic_to_cam_center(v_gt_se3).cpu().numpy()  
            res_c = extrinsic_to_cam_center(v_res_se3).cpu().numpy()
            lq_c = extrinsic_to_cam_center(v_lq_se3).cpu().numpy()

            res_c = sim3_align(res_c, gt_c)
            lq_c = sim3_align(lq_c, gt_c)

            r_err, t_err = se3_to_relative_pose_error(v_res_se3, v_gt_se3, v_res_se3.shape[-3])
            lq_r_err, lq_t_err = se3_to_relative_pose_error(v_lq_se3, v_gt_se3, v_lq_se3.shape[-3])

            r_err_np = r_err.cpu().numpy()
            t_err_np = t_err.cpu().numpy()
            lq_r_err_np = lq_r_err.cpu().numpy()
            lq_t_err_np = lq_t_err.cpu().numpy()
            
            if not np.isnan(r_err_np).any() and not np.isnan(t_err_np).any():
                all_r_errors.extend(r_err_np)
                all_t_errors.extend(t_err_np)

            if not np.isnan(lq_r_err_np).any() and not np.isnan(lq_t_err_np).any():
                lq_all_r_errors.extend(lq_r_err_np)
                lq_all_t_errors.extend(lq_t_err_np)
            
            metrics_accum["pose_R_err"] += r_err.mean().item()
            metrics_accum["pose_T_err"] += t_err.mean().item()  
            metrics_accum["lq_pose_R_err"] += lq_r_err.mean().item()
            metrics_accum["lq_pose_T_err"] += lq_t_err.mean().item()

            plot_cam_trajectory(gt_c, res_c, lq_c, os.path.join(vis_pose_dir, f"traj_{i:03d}.png"))
            plot_cam_trajectory(gt_c, res_c, lq_c, os.path.join(vis_pose_dir, f"traj_only_pred_{i:03d}.png"), only_pred=True)
            save_image(create_eval_grid(batch, lq_out, res_out), os.path.join(vis_depth_dir, f"depth_{i:03d}.png"), normalize=False)

    logger.info("\n" + "="*30 + " FINAL EVALUATION RESULTS " + "="*30)
    
    if all_r_errors:
        r_err_np = np.array(all_r_errors)
        t_err_np = np.array(all_t_errors)
        auc30 = calculate_auc_np(r_err_np, t_err_np, 30)
        auc15 = calculate_auc_np(r_err_np, t_err_np, 15)
        auc05 = calculate_auc_np(r_err_np, t_err_np, 5)
        auc03 = calculate_auc_np(r_err_np, t_err_np, 3)

        lq_r_err_np = np.array(lq_all_r_errors)
        lq_t_err_np = np.array(lq_all_t_errors)
        lq_auc30 = calculate_auc_np(lq_r_err_np, lq_t_err_np, 30)
        lq_auc15 = calculate_auc_np(lq_r_err_np, lq_t_err_np, 15)
        lq_auc05 = calculate_auc_np(lq_r_err_np, lq_t_err_np, 5)
        lq_auc03 = calculate_auc_np(lq_r_err_np, lq_t_err_np, 3)
        
        logger.info(f"{'pose_AUC30':<20}: {auc30:.6f}")
        logger.info(f"{'pose_AUC15':<20}: {auc15:.6f}")
        logger.info(f"{'pose_AUC05':<20}: {auc05:.6f}")
        logger.info(f"{'pose_AUC03':<20}: {auc03:.6f}")
        logger.info(f"{'lq_pose_AUC30':<20}: {lq_auc30:.6f}")
        logger.info(f"{'lq_pose_AUC15':<20}: {lq_auc15:.6f}")
        logger.info(f"{'lq_pose_AUC05':<20}: {lq_auc05:.6f}")
        logger.info(f"{'lq_pose_AUC03':<20}: {lq_auc03:.6f}")

    if total_frames > 0:
        num_seq = min(args.num_sequences, len(val_loader))
        for k, v in sorted(metrics_accum.items()):
            final_val = v / total_frames if "depth" in k else v / num_seq
            logger.info(f"{k:<20}: {final_val:.6f}")
            
    logger.info("="*86)

    cleanup_distributed()

if __name__ == "__main__":
    main()