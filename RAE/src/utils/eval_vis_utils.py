import logging
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import make_grid, save_image
from datetime import datetime
import torch.nn.functional as F
from collections import defaultdict, OrderedDict
import os

from RAE.src.utils.vis_utils import process_depth_batch, process_depth_error_batch
from RAE.src.utils.pose_utils import enc_to_se3, se3_to_relative_pose_error
from RAE.src.utils.depth_utils import compute_depth_metrics

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

def vis_depth_all(hq_img, hq_depth, lq_img, lq_depth, res_img=None, res_depth=None):
    hq_depth = hq_depth.squeeze(-1)
    lq_depth = lq_depth.squeeze(-1)
    if res_depth is not None:
        res_depth = res_depth.squeeze(-1)

    v_input_err = process_depth_error_batch(hq_depth, lq_depth.unsqueeze(-1))
    if res_depth is not None:
        v_output_err = process_depth_error_batch(hq_depth, res_depth.unsqueeze(-1))
    
    V = hq_img.shape[0]
    hq_depth_vis = to_rgb_tensor(hq_depth, is_depth=True)
    lq_depth_vis = to_rgb_tensor(lq_depth, is_depth=True)
    if res_depth is not None:
        res_depth_vis = to_rgb_tensor(res_depth, is_depth=True)
    
    nrow = 5
    if res_depth is not None:
        nrow = 7
    if res_depth is not None and res_img is not None:
        nrow = 8

    combined = []
    for i in range(V):
        combined.append(to_rgb_tensor(hq_img[i:i+1]))
        combined.append(hq_depth_vis[i:i+1])
        combined.append(to_rgb_tensor(lq_img[i:i+1]))
        combined.append(lq_depth_vis[i:i+1])
        combined.append(error_to_tensor(v_input_err[i:i+1]))
        if res_img is not None:
            combined.append(to_rgb_tensor(res_img[i:i+1])) 
        if res_depth is not None:
            combined.append(res_depth_vis[i:i+1])
            combined.append(error_to_tensor(v_output_err[i:i+1]))

            
    return make_grid(torch.cat(combined, dim=0), nrow=nrow, padding=10, pad_value=1.0)

def vis_depth_hq(hq_img, hq_depth):
    hq_depth = hq_depth.squeeze(-1)
    hq_depth_vis = to_rgb_tensor(hq_depth, is_depth=True)
    
    V = hq_img.shape[0]
    combined = []
    for i in range(V):
        combined.append(to_rgb_tensor(hq_img[i:i+1]))
        combined.append(hq_depth_vis[i:i+1])

    return make_grid(torch.cat(combined, dim=0), nrow=2, padding=10, pad_value=1.0)

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

def visualize_token_similarity(clean_tokens, res_out_tokens, save_path="similarity_plot.png"):
    n_layers = len(clean_tokens)
    frame_sims = []
    global_sims = []

    for i in range(n_layers):
        # [B, S, P, 2048]
        c_t = clean_tokens[i].float()
        r_t = res_out_tokens[i].float()

        c_frame = c_t[..., :1024]
        r_frame = r_t[..., :1024]
        
        c_global = c_t[..., 1024:]
        r_global = r_t[..., 1024:]

        sim_frame = F.cosine_similarity(c_frame, r_frame, dim=-1)
        sim_global = F.cosine_similarity(c_global, r_global, dim=-1)

        frame_sims.append(sim_frame.mean().item())
        global_sims.append(sim_global.mean().item())

    plt.figure(figsize=(12, 6))
    layers = np.arange(n_layers)

    plt.plot(layers, frame_sims, marker='o', linestyle='-', color='royalblue', label='Frame Latent (0-1023)')
    plt.plot(layers, global_sims, marker='s', linestyle='--', color='crimson', label='Global Latent (1024-2047)')

    plt.title("Feature Cosine Similarity: Clean vs Restored Tokens", fontsize=14)
    plt.xlabel("Layer Index", fontsize=12)
    plt.ylabel("Average Cosine Similarity", fontsize=12)
    plt.xticks(layers)
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.legend()
    plt.ylim(0, 1.05)

    for i, (f, g) in enumerate(zip(frame_sims, global_sims)):
        if i % 2 == 0: 
            plt.text(i, f, f"{f:.2f}", color='royalblue', va='bottom', ha='center', fontsize=8)
            plt.text(i, g, f"{g:.2f}", color='crimson', va='top', ha='center', fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Similarity visualization saved to: {save_path}")
    plt.close()


def visualize_token_similarity(clean_tokens, res_out_tokens, save_path="similarity_plot.png"):
    n_layers = len(clean_tokens)
    frame_sims = []
    global_sims = []
    frame_camera_sims = []
    global_camera_sims = []
    frame_rest_sims = []
    global_rest_sims = []

    for i in range(n_layers):
        # [B, S, P, 2048]
        c_t = clean_tokens[i].float()
        r_t = res_out_tokens[i].float()

        c_frame = c_t[..., :1024]
        r_frame = r_t[..., :1024]
        
        c_global = c_t[..., 1024:]
        r_global = r_t[..., 1024:]

        c_frame_camera = c_frame[..., :1, :]
        r_frame_camera = r_frame[..., :1, :]
        c_frame_rest = c_frame[..., 5:, :]
        r_frame_rest = r_frame[..., 5:, :]

        c_global_camera = c_global[..., :1, :]
        r_global_camera = r_global[..., :1, :]
        c_global_rest = c_global[..., 5:, :]
        r_global_rest = r_global[..., 5:, :]

        sim_frame = F.cosine_similarity(c_frame, r_frame, dim=-1)
        sim_global = F.cosine_similarity(c_global, r_global, dim=-1)
        sim_frame_camera = F.cosine_similarity(c_frame_camera, r_frame_camera, dim=-1)
        sim_frame_rest = F.cosine_similarity(c_frame_rest, r_frame_rest, dim=-1)
        sim_global_camera = F.cosine_similarity(c_global_camera, r_global_camera, dim=-1)
        sim_global_rest = F.cosine_similarity(c_global_rest, r_global_rest, dim=-1)

        frame_sims.append(sim_frame.mean().item())
        global_sims.append(sim_global.mean().item())
        frame_camera_sims.append(sim_frame_camera.mean().item())
        frame_rest_sims.append(sim_frame_rest.mean().item())
        global_camera_sims.append(sim_global_camera.mean().item())
        global_rest_sims.append(sim_global_rest.mean().item())

    plt.figure(figsize=(12, 6))
    layers = np.arange(n_layers)

    plt.plot(layers, frame_sims, marker='o', linestyle='-', color='royalblue', label='Frame Latent (0-1023)')
    plt.plot(layers, global_sims, marker='s', linestyle='--', color='crimson', label='Global Latent (1024-2047)')
    plt.plot(layers, frame_camera_sims, marker='^', linestyle='-.', color='darkgreen', label='Frame Camera (0-5)')
    plt.plot(layers, frame_rest_sims, marker='<', linestyle='-.', color='darkorange', label='Frame Rest (5-15)')
    plt.plot(layers, global_camera_sims, marker='v', linestyle='-.', color='purple', label='Global Camera (0-5)')
    plt.plot(layers, global_rest_sims, marker='+', linestyle='-.', color='brown', label='Global Rest (5-15)')

    plt.title("Feature Cosine Similarity: Clean vs Restored Tokens", fontsize=14)
    plt.xlabel("Layer Index", fontsize=12)
    plt.ylabel("Average Cosine Similarity", fontsize=12)
    plt.xticks(layers)
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.legend()
    plt.ylim(0, 1.05)

    for i, (f, g, fc, fr, gc, gr) in enumerate(zip(frame_sims, global_sims, frame_camera_sims, frame_rest_sims, global_camera_sims, global_rest_sims)):
        if i % 2 == 0: 
            plt.text(i, f, f"{f:.2f}", color='royalblue', va='bottom', ha='center', fontsize=8)
            plt.text(i, g, f"{g:.2f}", color='crimson', va='top', ha='center', fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Similarity visualization saved to: {save_path}")
    plt.close()

def visualize_vggt_stepwise_analysis(clean_tokens, lq_tokens, res_out_tokens=None, save_path="vggt_stepwise_analysis.png"):
    """
    clean, lq, res_out: 24 layer list of [B, S, P, 2048]
    48 steps: L0_Frame, L0_Global, L1_Frame, L1_Global ...
    3 Plots: All Tokens, Camera Token (idx 0), Patch Tokens (idx 4:)
    """
    n_layers = len(clean_tokens)
    steps_labels = []
    
    metrics = {
        'all': {'c_lq': [], 'c_res': []},
        'cam': {'c_lq': [], 'c_res': []},
        'patch': {'c_lq': [], 'c_res': []}
    }

    def compute_group_similarities(base, compare):
        sim_all = F.cosine_similarity(base, compare, dim=-1).mean().item()
        sim_cam = F.cosine_similarity(base[..., 0:1, :], compare[..., 0:1, :], dim=-1).mean().item()
        sim_patch = F.cosine_similarity(base[..., 5:, :], compare[..., 5:, :], dim=-1).mean().item()
        return sim_all, sim_cam, sim_patch

    for i in range(n_layers):
        c_i = clean_tokens[i].float()
        l_i = lq_tokens[i].float()
        r_i = res_out_tokens[i].float() if res_out_tokens is not None else l_i

        for mode, start, end in [("Frame", 0, 1024), ("Global", 1024, 2048)]:
            c_part = c_i[..., start:end]
            l_part = l_i[..., start:end]
            r_part = r_i[..., start:end] if res_out_tokens is not None else l_part

            s_all_cl, s_cam_cl, s_patch_cl = compute_group_similarities(c_part, l_part)
            s_all_cr, s_cam_cr, s_patch_cr = compute_group_similarities(c_part, r_part) if res_out_tokens is not None else (s_all_cl, s_cam_cl, s_patch_cl)

            metrics['all']['c_lq'].append(s_all_cl)
            metrics['cam']['c_lq'].append(s_cam_cl)
            metrics['patch']['c_lq'].append(s_patch_cl) 
            
            if res_out_tokens is not None:
                metrics['all']['c_res'].append(s_all_cr)
                metrics['cam']['c_res'].append(s_cam_cr)
                metrics['patch']['c_res'].append(s_patch_cr)
            
            steps_labels.append(f"L{i}_{mode[0]}")

    fig, axes = plt.subplots(3, 1, figsize=(24, 18), sharex=True)
    x_axis = np.arange(len(steps_labels))
    
    plot_configs = [
        ('all', "All Tokens Similarity"),
        ('cam', "Camera Token (Index 0) Similarity"),
        ('patch', "Patch Tokens (Index 4:) Similarity")
    ]
    
    for ax, (key, title) in zip(axes, plot_configs):
        ax.plot(x_axis, metrics[key]['c_lq'], marker='o', markersize=3, 
                linestyle='-', color='royalblue', label='Clean vs LQ', alpha=0.6)
        
        if res_out_tokens is not None:
            ax.plot(x_axis, metrics[key]['c_res'], marker='s', markersize=4, 
                    linestyle='-', color='crimson', label='Clean vs Res', linewidth=2)
        
        for j in range(len(x_axis)):
            if j % 2 == 0:
                ax.axvspan(j-0.5, j+0.5, color='gray', alpha=0.07)

        ax.set_title(title, fontsize=18, fontweight='bold')
        ax.set_ylabel("Cosine Similarity", fontsize=14)
        ax.set_ylim(0, 1.05)
        ax.legend(loc='lower left', fontsize=12)
        ax.grid(True, which='both', linestyle=':', alpha=0.4)

    axes[-1].set_xticks(x_axis)
    axes[-1].set_xticklabels(steps_labels, rotation=90, fontsize=8)
    axes[-1].set_xlabel("VGGT Processing Steps (F: Frame, G: Global)", fontsize=14)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=200)
    print(f"Stepwise feature analysis saved to: {save_path}")
    plt.close()

def plot_cam_trajectory_all(gt_c, pred_c, lq_c, lq_cam_c=None, hq_cam_c=None, save_path=None, only_pred=False):
    save_dir = os.path.dirname(save_path)
    os.makedirs(save_dir, exist_ok=True)
    
    fig = plt.figure(figsize=(22, 10)) 
    
    styles = {
        'gt': {'color': 'green', 'marker': 'o', 'label': 'Ground Truth', 'linewidth': 4, 'alpha': 0.8},
        'pred': {'color': 'red', 'marker': 'o', 'label': 'Restored (All-LQ)', 'linewidth': 2},
        'lq': {'color': 'blue', 'marker': 'x', 'label': 'Input LQ', 'linewidth': 1, 'alpha': 0.4, 'linestyle': '--'},
        'lq_cam': {'color': 'orange', 'marker': '^', 'label': 'Restored (LQ-Cam/Res-Img)', 'linewidth': 2, 'linestyle': '-.'},
        'hq_cam': {'color': 'purple', 'marker': 's', 'label': 'Restored (HQ-Cam/Res-Img)', 'linewidth': 2, 'linestyle': ':'}
    }

    ax1 = fig.add_subplot(121, projection='3d')
    if not only_pred:
        ax1.plot(gt_c[:, 0], gt_c[:, 1], gt_c[:, 2], **styles['gt'])
    
    if pred_c is not None:
        ax1.plot(pred_c[:, 0], pred_c[:, 1], pred_c[:, 2], **styles['pred'])
    if lq_cam_c is not None:
        ax1.plot(lq_cam_c[:, 0], lq_cam_c[:, 1], lq_cam_c[:, 2], **styles['lq_cam'])
    if hq_cam_c is not None:
        ax1.plot(hq_cam_c[:, 0], hq_cam_c[:, 1], hq_cam_c[:, 2], **styles['hq_cam'])
    ax1.plot(lq_c[:, 0], lq_c[:, 1], lq_c[:, 2], **styles['lq'])
    
    ax1.set_title("3D Camera Trajectory Comparison", fontsize=14)
    ax1.set_xlabel("X"); ax1.set_ylabel("Y"); ax1.set_zlabel("Z")
    ax1.legend(loc='upper left')

    ax2 = fig.add_subplot(122)
    if not only_pred:
        ax2.plot(gt_c[:, 0], gt_c[:, 2], **styles['gt'])
    
    if pred_c is not None:
        ax2.plot(pred_c[:, 0], pred_c[:, 2], **styles['pred'])
    if lq_cam_c is not None:
        ax2.plot(lq_cam_c[:, 0], lq_cam_c[:, 2], **styles['lq_cam'])
    if hq_cam_c is not None:
        ax2.plot(hq_cam_c[:, 0], hq_cam_c[:, 2], **styles['hq_cam'])
    ax2.plot(lq_c[:, 0], lq_c[:, 2], **styles['lq'])
    
    # ax2.scatter(gt_c[0, 0], gt_c[0, 2], color='black', s=150, zorder=5, label='Start Point') 
    
    ax2.set_title("Top-down View (X-Z Plane)", fontsize=14)
    ax2.set_aspect('equal')
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

def save_depth_metrics(hq_depth, lq_depth, res_depth=None, save_path=None):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    v_metrics_accum = defaultdict(float)
    V = hq_depth.shape[1]
    for v in range(V):
        lq_d_m = compute_depth_metrics(hq_depth[0, v], lq_depth[0, v].squeeze(-1).cpu())
        if res_depth is not None:
            res_d_m = compute_depth_metrics(hq_depth[0, v], res_depth[0, v].squeeze(-1).cpu())

        for k, val in lq_d_m.items(): v_metrics_accum[f"lq_depth_{k}"] += val
        if res_depth is not None:
            for k, val in res_d_m.items(): v_metrics_accum[f"res_depth_{k}"] += val

    with open(save_path, "w") as f:
        for k, v in v_metrics_accum.items():
            f.write(f"{k}: {v / V:.4f}\n")

def save_pose_metrics(hq_img, hq_pose, lq_img, lq_pose, res_img=None, res_pose=None, metric_save_path=None, plot_save_path=None):       
    os.makedirs(os.path.dirname(metric_save_path), exist_ok=True)
    os.makedirs(os.path.dirname(plot_save_path), exist_ok=True)

    hq_se3 = enc_to_se3(hq_pose, hq_img.shape[-2:]) # [V, 4, 4]
    lq_se3 = enc_to_se3(lq_pose, lq_img.shape[-2:])
    if res_pose is not None:
        res_se3 = enc_to_se3(res_pose, lq_img.shape[-2:])

    hq_c = extrinsic_to_cam_center(hq_se3).cpu().numpy()  
    lq_c = extrinsic_to_cam_center(lq_se3).cpu().numpy()
    if res_pose is not None:
        res_c = extrinsic_to_cam_center(res_se3).cpu().numpy()
    
    lq_c = sim3_align(lq_c, hq_c)
    if res_pose is not None:
        res_c = sim3_align(res_c, hq_c)
    else:
        res_c = None
    
    lq_r_err, lq_t_err = se3_to_relative_pose_error(lq_se3, hq_se3, lq_se3.shape[-3])
    if res_pose is not None:
        res_r_err, res_t_err = se3_to_relative_pose_error(res_se3, hq_se3, res_se3.shape[-3])
    
    lq_r_err_np = lq_r_err.cpu().numpy()
    lq_t_err_np = lq_t_err.cpu().numpy()
    if res_pose is not None:
        res_r_err_np = res_r_err.cpu().numpy()
        res_t_err_np = res_t_err.cpu().numpy()
    
    pose_metrics = {
        "lq_auc30": calculate_auc_np(lq_r_err_np, lq_t_err_np, 30),
        "lq_auc15": calculate_auc_np(lq_r_err_np, lq_t_err_np, 15),
        "lq_auc05": calculate_auc_np(lq_r_err_np, lq_t_err_np, 5),
        "lq_auc03": calculate_auc_np(lq_r_err_np, lq_t_err_np, 3),
    }

    if res_pose is not None:
        pose_metrics.update({
            "res_auc30": calculate_auc_np(res_r_err_np, res_t_err_np, 30),
            "res_auc15": calculate_auc_np(res_r_err_np, res_t_err_np, 15),
            "res_auc05": calculate_auc_np(res_r_err_np, res_t_err_np, 5),
            "res_auc03": calculate_auc_np(res_r_err_np, res_t_err_np, 3),
        })

    with open(metric_save_path, "w") as f:
        for k, v in pose_metrics.items():
            f.write(f"{k}: {v:.6f}\n")

    plot_cam_trajectory_all(hq_c, res_c, lq_c, save_path=plot_save_path)