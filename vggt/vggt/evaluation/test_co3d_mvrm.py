import os
import sys
import csv
import torch
import numpy as np
import gzip
import json
import random
import logging
import warnings
from datetime import datetime

from vggt.vggt.models.vggt import VGGT
from vggt.vggt.utils.rotation import mat_to_quat
from vggt.vggt.utils.load_fn import *
from vggt.vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.vggt.utils.geometry import closed_form_inverse_se3
from vggt.vggt.evaluation.ba import run_vggt_with_ba
from motionblur.motionblur import Kernel 
import argparse

# Suppress DINO v2 logs
logging.getLogger("dinov2").setLevel(logging.WARNING)
warnings.filterwarnings("ignore", message="xFormers is available")
warnings.filterwarnings("ignore", message="dinov2")

# Set computation precision
torch.set_float32_matmul_precision('highest')
torch.backends.cudnn.allow_tf32 = False

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import procrustes

# def plot_trajectory(gt_poses, pred_poses, lq_poses, seq_name):
#     fig = plt.figure(figsize=(10, 8))
#     ax = fig.add_subplot(111, projection='3d')

#     def get_coords(poses):
#         if isinstance(poses, torch.Tensor):
#             poses = poses.cpu().numpy()
#         centers = []
#         for i in range(len(poses)):
#             R = poses[i, :3, :3]
#             t = poses[i, :3, 3]
#             center = -R.T @ t 
#             centers.append(center)
#         return np.array(centers)

#     gt_c = get_coords(gt_poses)
#     pred_c = get_coords(pred_poses)
#     lq_c = get_coords(lq_poses)

#     ax.plot(gt_c[:, 0], gt_c[:, 1], gt_c[:, 2], 'g-o', label='GT', markersize=4)
#     ax.plot(pred_c[:, 0], pred_c[:, 1], pred_c[:, 2], 'r-o', label='Restored (Pred)', markersize=4)
#     ax.plot(lq_c[:, 0], lq_c[:, 1], lq_c[:, 2], 'b--x', label='Degraded (LQ)', markersize=3, alpha=0.5)

#     ax.scatter(gt_c[0, 0], gt_c[0, 1], gt_c[0, 2], color='black', s=100, label='Start')

#     ax.set_title(f"Trajectory Comparison: {seq_name}")
#     ax.legend()
    
#     plt.savefig(f"traj_{seq_name}.png")
#     plt.close()

# def plot_trajectory(gt_poses, pred_poses, lq_poses, seq_name, save_dir="./vggt/vggt/evaluation/visualization/continuous_cam_vis"):
#     os.makedirs(save_dir, exist_ok=True)

#     def get_coords(poses):
#         if isinstance(poses, torch.Tensor):
#             poses = poses.detach().cpu().numpy()
#         centers = []
#         for i in range(len(poses)):
#             R = poses[i, :3, :3]
#             t = poses[i, :3, 3]
#             center = -R.T @ t 
#             centers.append(center)
#         return np.array(centers)

#     gt_c = get_coords(gt_poses)
#     pred_c = get_coords(pred_poses)
#     lq_c = get_coords(lq_poses)

#     fig = plt.figure(figsize=(20, 10))
    
#     ax1 = fig.add_subplot(121, projection='3d')
#     ax1.plot(gt_c[:, 0], gt_c[:, 1], gt_c[:, 2], 'g-o', label='Ground Truth', markersize=5, linewidth=2)
#     ax1.plot(pred_c[:, 0], pred_c[:, 1], pred_c[:, 2], 'r-o', label='Restored (Pred)', markersize=5, linewidth=2)
#     ax1.plot(lq_c[:, 0], lq_c[:, 1], lq_c[:, 2], 'b--x', label='Degraded (LQ)', markersize=4, alpha=0.4)
    
#     ax1.scatter(gt_c[0, 0], gt_c[0, 1], gt_c[0, 2], color='black', s=150, label='Start Point', zorder=10)
    
#     ax1.set_title(f"3D Trajectory: {seq_name}", fontsize=15)
#     ax1.set_xlabel('X')
#     ax1.set_ylabel('Y')
#     ax1.set_zlabel('Z')
#     ax1.legend()

#     ax2 = fig.add_subplot(122)
#     ax2.plot(gt_c[:, 0], gt_c[:, 2], 'g-o', label='GT', markersize=6)
#     ax2.plot(pred_c[:, 0], pred_c[:, 2], 'r-o', label='Restored', markersize=6)
#     ax2.plot(lq_c[:, 0], lq_c[:, 2], 'b--x', label='LQ', alpha=0.3)
#     ax2.scatter(gt_c[0, 0], gt_c[0, 2], color='black', s=100) 

#     ax2.set_title(f"Top-down View (X-Z plane)", fontsize=15)
#     ax2.set_xlabel('X (Side-to-Side)')
#     ax2.set_ylabel('Z (Depth)')
#     ax2.grid(True, linestyle='--', alpha=0.6)
#     ax2.legend()
#     ax2.set_aspect('equal') 

#     save_path = os.path.join(save_dir, f"traj_{seq_name}.png")
#     plt.tight_layout()
#     plt.savefig(save_path)
#     plt.close()
#     print(f"[*] Trajectory visualization saved to: {save_path}")

def plot_trajectory(gt_poses, pred_poses, lq_poses, seq_name, save_dir="./vggt/vggt/evaluation/visualization/continuous_cam_vis_scale_aligned"):
    os.makedirs(save_dir, exist_ok=True)

    def get_coords(poses):
        if isinstance(poses, torch.Tensor):
            poses = poses.detach().cpu().numpy()
        centers = []
        for i in range(len(poses)):
            R = poses[i, :3, :3]
            t = poses[i, :3, 3]
            center = -R.T @ t 
            centers.append(center)
        return np.array(centers)

    def align_and_scale(target_c, source_c):
        if len(target_c) < 2: return source_c
        
        mtx1, mtx2, disparity = procrustes(target_c, source_c)
        
        target_std = np.std(target_c)
        target_mean = np.mean(target_c, axis=0)
        
        return mtx2 * target_std + target_mean

    gt_c = get_coords(gt_poses)
    pred_c = get_coords(pred_poses)
    lq_c = get_coords(lq_poses)

    pred_c_aligned = align_and_scale(gt_c, pred_c)
    lq_c_aligned = align_and_scale(gt_c, lq_c)

    fig = plt.figure(figsize=(20, 10))
    
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot(gt_c[:, 0], gt_c[:, 1], gt_c[:, 2], 'g-o', label='Ground Truth', markersize=6, linewidth=3)
    ax1.plot(pred_c_aligned[:, 0], pred_c_aligned[:, 1], pred_c_aligned[:, 2], 'r-o', label='Restored (Scaled)', markersize=5, linewidth=2)
    ax1.plot(lq_c_aligned[:, 0], lq_c_aligned[:, 1], lq_c_aligned[:, 2], 'b--x', label='LQ (Scaled)', markersize=4, alpha=0.6)
    
    ax1.scatter(gt_c[0, 0], gt_c[0, 1], gt_c[0, 2], color='black', s=150, label='Start Point', zorder=10)
    
    ax1.set_title(f"3D Trajectory (Scale Aligned): {seq_name}", fontsize=15)
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.legend()

    ax2 = fig.add_subplot(122)
    ax2.plot(gt_c[:, 0], gt_c[:, 2], 'g-o', label='GT', markersize=7, linewidth=3)
    ax2.plot(pred_c_aligned[:, 0], pred_c_aligned[:, 2], 'r-o', label='Restored (Scaled)', markersize=6)
    ax2.plot(lq_c_aligned[:, 0], lq_c_aligned[:, 2], 'b--x', label='LQ (Scaled)', alpha=0.5)
    ax2.scatter(gt_c[0, 0], gt_c[0, 2], color='black', s=100) 

    ax2.set_title(f"Top-down View (Scale Aligned)", fontsize=15)
    ax2.set_xlabel('X (Side-to-Side)')
    ax2.set_ylabel('Z (Depth)')
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.legend()
    ax2.set_aspect('equal') 

    save_path = os.path.join(save_dir, f"traj_aligned_{seq_name}.png")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"[*] Aligned trajectory visualization saved to: {save_path}")

class PoseMetricTracker:
    def __init__(self):
        self.reset()

    def reset(self):
        self.r_errors = []
        self.t_errors = []
    
    def update(self, r_err, t_err):
        if isinstance(r_err, np.ndarray):
            self.r_errors.extend(r_err.tolist())
            self.t_errors.extend(t_err.tolist())
        else: 
            self.r_errors.append(r_err)
            self.t_errors.append(t_err)

    def compute_auc(self):
        if not self.r_errors:
            return None
        
        r_err = np.array(self.r_errors)
        t_err = np.array(self.t_errors)
        
        auc_30, _ = calculate_auc_np(r_err, t_err, max_threshold=30)
        auc_15, _ = calculate_auc_np(r_err, t_err, max_threshold=15)
        auc_5, _ = calculate_auc_np(r_err, t_err, max_threshold=5)
        
        return {
            "auc_30": auc_30,
            "auc_15": auc_15,
            "auc_5": auc_5,
            "mean_r_err": np.mean(r_err),
            "mean_t_err": np.mean(t_err)
        }

class PoseCSVLogger:
    def __init__(self, experiment_dir):
        self.experiment_dir = experiment_dir
        self.written_headers = set()

    def log(self, step, metrics, mode="val"):
        os.makedirs(self.experiment_dir, exist_ok=True)
        file_path = os.path.join(self.experiment_dir, f"pose_metrics_{mode}.csv")
        
        fieldnames = ["step"] + list(metrics.keys())
        
        is_new_file = not os.path.exists(file_path)
        
        with open(file_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            
            if is_new_file or file_path not in self.written_headers:
                writer.writeheader()
                self.written_headers.add(file_path)
            
            log_data = {"step": step}
            log_data.update(metrics)
            writer.writerow(log_data)

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

def convert_pt3d_RT_to_opencv(Rot, Trans):
    """
    Convert Point3D extrinsic matrices to OpenCV convention.

    Args:
        Rot: 3D rotation matrix in Point3D format
        Trans: 3D translation vector in Point3D format

    Returns:
        extri_opencv: 3x4 extrinsic matrix in OpenCV format
    """
    rot_pt3d = np.array(Rot)
    trans_pt3d = np.array(Trans)

    trans_pt3d[:2] *= -1
    rot_pt3d[:, :2] *= -1
    rot_pt3d = rot_pt3d.transpose(1, 0)
    extri_opencv = np.hstack((rot_pt3d, trans_pt3d[:, None]))
    return extri_opencv


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


def calculate_auc_np(r_error, t_error, max_threshold=30):
    """
    Calculate the Area Under the Curve (AUC) for the given error arrays using NumPy.

    Args:
        r_error: numpy array representing R error values (Degree)
        t_error: numpy array representing T error values (Degree)
        max_threshold: Maximum threshold value for binning the histogram

    Returns:
        AUC value and the normalized histogram
    """
    error_matrix = np.concatenate((r_error[:, None], t_error[:, None]), axis=1)
    max_errors = np.max(error_matrix, axis=1)
    bins = np.arange(max_threshold + 1)
    histogram, _ = np.histogram(max_errors, bins=bins)
    num_pairs = float(len(max_errors))
    normalized_histogram = histogram.astype(float) / num_pairs
    return np.mean(np.cumsum(normalized_histogram)), normalized_histogram


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


def set_random_seeds(seed):
    """
    Set random seeds for reproducibility.

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def process_sequence(model, eval_sampler,seq_name, seq_data, category, co3d_dir, min_num_images, num_frames, use_ba, device, dtype, logger):
    """
    Process a single sequence and compute pose errors.

    Args:
        model: VGGT model
        seq_name: Sequence name
        seq_data: Sequence data
        category: Category name
        co3d_dir: CO3D dataset directory
        min_num_images: Minimum number of images required
        num_frames: Number of frames to sample
        use_ba: Whether to use bundle adjustment
        device: Device to run on
        dtype: Data type for model inference
        logger: Logger instance for logging messages
    Returns:
        rError: Rotation errors
        tError: Translation errors
    """
    vggt_model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)
    vggt_model.eval()
    vggt_model.requires_grad_(False)

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    generator = torch.Generator(device=device)

    if len(seq_data) < min_num_images:
        return None, None

    metadata = []
    for data in seq_data:
        full_path = os.path.join(co3d_dir, data["filepath"])
        if not os.path.exists(full_path):
            continue

        extri_opencv = np.array(data["extri"])
        t_vec = extri_opencv[:3, 3]
        if np.sum(np.abs(t_vec)) > 1e5:
            return None, None
        # Make sure translations are not ridiculous
        # if data["T"][0] + data["T"][1] + data["T"][2] > 1e5:
        #     return None, None
        # extri_opencv = convert_pt3d_RT_to_opencv(data["R"], data["T"])
        metadata.append({
            "filepath": data["filepath"],
            "extri": extri_opencv,
        })

    if len(metadata) < num_frames:
        logger.info(f"Skipping {seq_name}: Not enough valid images found.")
        return None, None

    # Random sample num_frames images
    # ids = np.random.choice(len(metadata), num_frames, replace=False)
    # print("Image ids", ids)

    num_total = len(metadata)
    if num_total > num_frames:
        start_idx = np.random.randint(0, num_total - num_frames)
        ids = np.arange(start_idx, start_idx + num_frames)
    else:
        ids = np.arange(num_total)

    print(f"Selected consecutive image ids: {ids}")

    image_names = [os.path.join(co3d_dir, metadata[i]["filepath"]) for i in ids]
    gt_extri = [np.array(metadata[i]["extri"]) for i in ids]
    gt_extri = np.stack(gt_extri, axis=0)

    # images = load_and_preprocess_images(image_names).to(device) # [10, 3, 518, 518]
    kernel = Kernel(size=(100, 100), intensity=0.1)
    clean_images, deg_images = load_clean_deg_images(image_names, kernel)
    if len(clean_images.shape) == 4:
        clean_images = clean_images.unsqueeze(0).to(device) # [1, 10, 3, 518, 518]
        deg_images = deg_images.unsqueeze(0).to(device) # [1, 10, 3, 518, 518]
    # breakpoint()
    if use_ba:
        try:
            pred_extrinsic = run_vggt_with_ba(model, clean_images, image_names=image_names, dtype=dtype)
        except Exception as e:
            print(f"BA failed with error: {e}. Falling back to standard VGGT inference.")
            with torch.no_grad():
                with torch.cuda.amp.autocast(dtype=dtype):
                    predictions = model(clean_images)
            with torch.cuda.amp.autocast(dtype=torch.float64):
                extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], clean_images.shape[-2:])
                pred_extrinsic = extrinsic[0]
    else:
        sample_model_kwargs = {}
        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=dtype):
                val_lq_predictions = vggt_model(deg_images.to(device), extract_layer_num=3)
                val_lq_deg_latent = val_lq_predictions["extracted_latent"][:, :, :, 1024:] # [B, 1, 1041, 1024]
                generator.manual_seed(42)
                zs = torch.randn(*val_lq_deg_latent.shape, generator=generator, device=device, dtype=torch.float32)
                val_xt = zs + val_lq_deg_latent 
                sample_model_kwargs["img"] = deg_images.to(device)

                restored_latent = eval_sampler(val_xt, model.forward, **sample_model_kwargs)[-1].float()

                vggt_result = {}
                vggt_result['restored_latent'] = restored_latent

                val_predictions = vggt_model(deg_images.to(device), extract_layer_num=3, vggt_result=vggt_result, change_latent=True)

        with torch.cuda.amp.autocast(dtype=torch.float64):
            extrinsic, intrinsic = pose_encoding_to_extri_intri(val_predictions["pose_enc"], deg_images.shape[-2:]) # [10, 1, 3, 4], [10, 1, 3, 3]
            lq_extrinsic, lq_intrinsic = pose_encoding_to_extri_intri(val_lq_predictions["pose_enc"], deg_images.shape[-2:]) # [10, 1, 3, 4], [10, 1, 3, 3] 
            pred_extrinsic = extrinsic[0]
            lq_extrinsic = lq_extrinsic[0]
            # pred_extrinsic = extrinsic[:, 0, :, :] # [10, 3, 4] [B, 3, 4]
            
    with torch.cuda.amp.autocast(dtype=torch.float64):
        gt_extrinsic = torch.from_numpy(gt_extri).to(device) # [10, 3, 4]
        add_row = torch.tensor([0, 0, 0, 1], device=device).expand(pred_extrinsic.size(0), 1, 4) # [1, 1, 4]

        pred_se3 = torch.cat((pred_extrinsic, add_row), dim=1)
        lq_pred_se3 = torch.cat((lq_extrinsic, add_row), dim=1)
        gt_se3 = torch.cat((gt_extrinsic, add_row), dim=1)

        rel_rangle_deg, rel_tangle_deg = se3_to_relative_pose_error(pred_se3, gt_se3, num_frames)
        lq_rel_rangle_deg, lq_rel_tangle_deg = se3_to_relative_pose_error(lq_pred_se3, gt_se3, num_frames)

        Racc_5 = (rel_rangle_deg < 5).float().mean().item()
        Tacc_5 = (rel_tangle_deg < 5).float().mean().item()
        lq_Racc_5 = (lq_rel_rangle_deg < 5).float().mean().item()
        lq_Tacc_5 = (lq_rel_tangle_deg < 5).float().mean().item()

        print(f"{category} sequence {seq_name} R_ACC@5: {Racc_5:.4f}")
        print(f"{category} sequence {seq_name} T_ACC@5: {Tacc_5:.4f}")
        print(f"{category} sequence {seq_name} LQ_R_ACC@5: {lq_Racc_5:.4f}")
        print(f"{category} sequence {seq_name} LQ_T_ACC@5: {lq_Tacc_5:.4f}")

        def align_to_first(poses):
            first_inv = torch.inverse(poses[0])
            return first_inv @ poses

        pred_se3_aligned = align_to_first(pred_se3)
        lq_se3_aligned = align_to_first(lq_pred_se3)
        gt_se3_aligned = align_to_first(gt_se3)

        plot_trajectory(gt_se3_aligned, pred_se3_aligned, lq_se3_aligned, seq_name)

        return rel_rangle_deg.cpu().numpy(), rel_tangle_deg.cpu().numpy(), lq_rel_rangle_deg.cpu().numpy(), lq_rel_tangle_deg.cpu().numpy()


def evaluate_co3d(model, eval_sampler, log_dir, co3d_dir, co3d_anno_dir, min_num_images=50, num_frames=10, use_ba=False):
    # Setup device and data type
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    model = model.eval()

    log_dir = os.path.join(log_dir, "co3d_eval")
    os.makedirs(log_dir, exist_ok=True)
    logger = setup_logger(log_dir)

    # Set random seeds
    set_random_seeds(42)

    # Categories to evaluate
    SEEN_CATEGORIES = [
        "apple", "backpack", "banana", "baseballbat", "baseballglove",
        "bench", "bicycle", "bottle", "bowl", "broccoli",
        "cake", "car", "carrot", "cellphone", "chair",
        "cup", "donut", "hairdryer", "handbag", "hydrant",
        "keyboard", "laptop", "microwave", "motorcycle", "mouse",
        "orange", "parkingmeter", "pizza", "plant", "stopsign",
        "teddybear", "toaster", "toilet", "toybus", "toyplane",
        "toytrain", "toytruck", "tv", "umbrella", "vase", "wineglass",
    ]
    
    categories = ["apple", "bench", "bowl"]

    per_category_results = {}

    for category in categories:
        print(f"Loading annotation for {category} test set")
        annotation_file = os.path.join(co3d_anno_dir, f"{category}_test.jgz")

        category_dir = os.path.join(co3d_dir, category)

        try:
            with gzip.open(annotation_file, "r") as fin:
                annotation = json.loads(fin.read())
        except FileNotFoundError:
            print(f"Annotation file not found for {category}, skipping")
            continue

        rError = []
        tError = []
        lq_rError = []
        lq_tError = []
        # breakpoint()
        seq_names = sorted(list(annotation.keys()))
        seq_names = sorted(seq_names)


        print("Testing Sequences: ")
        print(seq_names)

        for seq_name in seq_names:
            seq_dir = os.path.join(category_dir, seq_name)
            if not os.path.exists(seq_dir):
                continue

            seq_data = annotation[seq_name]
            print("-" * 50)
            print(f"Processing {seq_name} for {category} test set")
            if not os.path.exists(os.path.join(co3d_dir, category, seq_name)):
                print(f"Skipping {seq_name} (not found)")
                continue

            seq_rError, seq_tError, lq_seq_rError, lq_seq_tError = process_sequence(
                model, eval_sampler, seq_name, seq_data, category, co3d_dir,
                min_num_images, num_frames, use_ba, device, dtype, logger
            )

            print("-" * 50)

            if seq_rError is not None and seq_tError is not None:
                rError.extend(seq_rError)
                tError.extend(seq_tError)
            if lq_seq_rError is not None and lq_seq_tError is not None:
                lq_rError.extend(lq_seq_rError)
                lq_tError.extend(lq_seq_tError)
        
        if not rError:
            print(f"No valid sequences found for {category}, skipping")
            continue

        rError = np.array(rError)
        tError = np.array(tError)
        lq_rError = np.array(lq_rError)
        lq_tError = np.array(lq_tError)

        Auc_30, _ = calculate_auc_np(rError, tError, max_threshold=30)
        Auc_15, _ = calculate_auc_np(rError, tError, max_threshold=15)
        Auc_5, _ = calculate_auc_np(rError, tError, max_threshold=5)
        Auc_3, _ = calculate_auc_np(rError, tError, max_threshold=3)

        lq_Auc_30, _ = calculate_auc_np(lq_rError, lq_tError, max_threshold=30)
        lq_Auc_15, _ = calculate_auc_np(lq_rError, lq_tError, max_threshold=15)
        lq_Auc_5, _ = calculate_auc_np(lq_rError, lq_tError, max_threshold=5)
        lq_Auc_3, _ = calculate_auc_np(lq_rError, lq_tError, max_threshold=3)

        per_category_results[category] = {
            "rError": rError,
            "tError": tError,
            "lq_rError": lq_rError,
            "lq_tError": lq_tError,
            "Auc_30": Auc_30,
            "Auc_15": Auc_15,
            "Auc_5": Auc_5,
            "Auc_3": Auc_3,
            "lq_Auc_30": lq_Auc_30,
            "lq_Auc_15": lq_Auc_15,
            "lq_Auc_5": lq_Auc_5,
            "lq_Auc_3": lq_Auc_3,
        }

        logger.info("="*80)
        logger.info(f"AUC of {category} test set: "
                    f"{Auc_30:.4f} (AUC@30), {Auc_15:.4f} (AUC@15), "
                    f"{Auc_5:.4f} (AUC@5), {Auc_3:.4f} (AUC@3)")
        logger.info(f"Low-quality AUC of {category} test set: "
                    f"{lq_Auc_30:.4f} (AUC@30), {lq_Auc_15:.4f} (AUC@15), "
                    f"{lq_Auc_5:.4f} (AUC@5), {lq_Auc_3:.4f} (AUC@3)")

        mean_AUC_30_by_now = np.mean([v["Auc_30"] for v in per_category_results.values()])
        mean_AUC_15_by_now = np.mean([v["Auc_15"] for v in per_category_results.values()])
        mean_AUC_5_by_now = np.mean([v["Auc_5"] for v in per_category_results.values()])
        mean_AUC_3_by_now = np.mean([v["Auc_3"] for v in per_category_results.values()])
        mean_lq_AUC_30_by_now = np.mean([v["lq_Auc_30"] for v in per_category_results.values()])
        mean_lq_AUC_15_by_now = np.mean([v["lq_Auc_15"] for v in per_category_results.values()])
        mean_lq_AUC_5_by_now = np.mean([v["lq_Auc_5"] for v in per_category_results.values()])
        mean_lq_AUC_3_by_now = np.mean([v["lq_Auc_3"] for v in per_category_results.values()])
        
        logger.info(f"Mean AUC of categories by now: {mean_AUC_30_by_now:.4f} (AUC@30), {mean_AUC_15_by_now:.4f} (AUC@15), {mean_AUC_5_by_now:.4f} (AUC@5), {mean_AUC_3_by_now:.4f} (AUC@3)")
        logger.info(f"Mean Low-quality AUC of categories by now: {mean_lq_AUC_30_by_now:.4f} (AUC@30), {mean_lq_AUC_15_by_now:.4f} (AUC@15), {mean_lq_AUC_5_by_now:.4f} (AUC@5), {mean_lq_AUC_3_by_now:.4f} (AUC@3)")
        logger.info("="*80)

    logger.info("\nSummary of AUC results:")
    logger.info("-" * 50)
    for cat, res in sorted(per_category_results.items()):
        logger.info(f"{cat:<15}: {res['Auc_30']:.4f} (AUC@30), {res['Auc_15']:.4f} (AUC@15), {res['Auc_5']:.4f} (AUC@5), {res['Auc_3']:.4f} (AUC@3)")
        logger.info(f"{cat:<15}: {res['lq_Auc_30']:.4f} (Low-quality AUC@30), {res['lq_Auc_15']:.4f} (Low-quality AUC@15), {res['lq_Auc_5']:.4f} (Low-quality AUC@5), {res['lq_Auc_3']:.4f} (Low-quality AUC@3)")

    if per_category_results:
        final_mean_30 = np.mean([v["Auc_30"] for v in per_category_results.values()])
        final_mean_15 = np.mean([v["Auc_15"] for v in per_category_results.values()])
        final_mean_5 = np.mean([v["Auc_5"] for v in per_category_results.values()])
        final_mean_3 = np.mean([v["Auc_3"] for v in per_category_results.values()])
        final_lq_mean_30 = np.mean([v["lq_Auc_30"] for v in per_category_results.values()])
        final_lq_mean_15 = np.mean([v["lq_Auc_15"] for v in per_category_results.values()])
        final_lq_mean_5 = np.mean([v["lq_Auc_5"] for v in per_category_results.values()])
        final_lq_mean_3 = np.mean([v["lq_Auc_3"] for v in per_category_results.values()])
        logger.info("-" * 50)
        logger.info(f"Final Mean AUC@30: {final_mean_30:.4f}, Final Mean AUC@15: {final_mean_15:.4f}, Final Mean AUC@5: {final_mean_5:.4f}, Final Mean AUC@3: {final_mean_3:.4f}")
        logger.info(f"Final Mean Low-quality AUC@30: {final_lq_mean_30:.4f}, Final Mean Low-quality AUC@15: {final_lq_mean_15:.4f}, Final Mean Low-quality AUC@5: {final_lq_mean_5:.4f}, Final Mean Low-quality AUC@3: {final_lq_mean_3:.4f}")