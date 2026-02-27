import os
import torch
import numpy as np
import logging
from tqdm import tqdm
from torchvision.utils import make_grid, save_image

from vggt.vggt.models.vggt import VGGT
from stage2.transport import create_transport, Sampler
from RAE.src.utils.train_utils_gf import ETH3DDataset
from RAE.src.utils.pose_utils import *

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

def run_evaluation():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_path = "/mnt/dataset1/MV_Restoration/vggt/global_step_0014000.pt"
    save_dir = "/mnt/dataset1/jaeeun/MVR_vggt/eval_results"
    os.makedirs(save_dir, exist_ok=True)
    
    logger = setup_logger(save_dir)
    
    vggt_model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)
    # 여기에 본인이 학습한 Diffusion/Flow 모델(예: DiT) 정의 및 로드
    model = YourFlowModel().to(device) 
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    vggt_model.eval()

    # 3. Setup Sampler (ODE)
    # 제공해주신 YAML 설정을 바탕으로 생성
    transport = create_transport(time_dist_shift=16.0) # shift는 모델 세팅에 맞춰 조정
    transport_sampler = Sampler(transport)
    eval_sampler = transport_sampler.sample_ode(
        sampling_method='euler', num_steps=50, atol=1e-6, rtol=1e-3
    )

    dataset = ETH3DDataset(
        clean_img_paths="path/to/eth3d/clean",
        gt_depth_paths="path/to/eth3d/gt",
        mode='val', 
        num_view=10, 
        view_sel={'strategy': 'sequential'}
    )
    
    tracker = PoseMetricTracker()
    
    # 5. Inference Loop (Sliding Window)
    # ETH3D는 시퀀스별로 나뉘어 있으므로 scene_to_indices를 활용
    for scene, indices in dataset.scene_to_indices.items():
        logger.info(f"Processing Scene: {scene}")
        scene_save_dir = os.path.join(save_dir, scene)
        os.makedirs(scene_save_dir, exist_ok=True)
        
        # 10프레임씩 슬라이딩 (예: stride 5)
        stride = 5
        num_frames = 10
        
        for start_idx in tqdm(range(0, len(indices) - num_frames + 1, stride)):
            window_indices = indices[start_idx : start_idx + num_frames]
            
            # 데이터 수동 로드 (Dataset.__getitem__ 로직 활용)
            batch = load_custom_window(dataset, window_indices, device)
            
            clean_img = batch["clean_img"].unsqueeze(0) # [1, V, 3, H, W]
            deg_img = batch["deg_img"].unsqueeze(0)
            gt_poses = batch["gt_poses"] # [V, 4, 4]
            
            with torch.no_grad():
                # Step A: LQ Prediction (Original VGGT)
                lq_out = vggt_model(deg_img, extract_layer_num=3)
                lq_latent = lq_out["extracted_latent"][:, :, :, 1024:]
                
                # Step B: Diffusion Sampling (Restoration)
                generator = torch.Generator(device=device).manual_seed(42)
                zs = torch.randn_like(lq_latent, generator=generator)
                xt = zs + lq_latent
                restored_latent = eval_sampler(xt, model.forward, img=deg_img)[-1].float()
                
                # Step C: Restored Prediction
                vggt_res = {'restored_latent': restored_latent}
                res_out = vggt_model(deg_img, extract_layer_num=3, vggt_result=vggt_res, change_latent=True)

            # --- Pose Processing ---
            # 6DoF Pose Extraction (Camera-to-World or World-to-Camera 기준 확인 필요)
            pred_se3 = enc_to_se3(res_out["pose_enc"], deg_img.shape[-2:])
            lq_se3 = enc_to_se3(lq_out["pose_enc"], deg_img.shape[-2:])
            
            # Align to First Frame of the Window
            def align(poses):
                first_inv = torch.inverse(poses[0])
                return first_inv @ poses

            gt_aligned = align(gt_poses)
            pred_aligned = align(pred_se3[0])
            lq_aligned = align(lq_se3[0])

            # Error calculation for metrics
            r_err, t_err = se3_to_relative_pose_error(pred_se3[0], gt_poses, num_frames)
            tracker.update(r_err.cpu().numpy(), t_err.cpu().numpy())

            # --- Visualization ---
            if start_idx % (stride * 2) == 0: # 너무 많으면 일부만 저장
                # 1. Trajectory Plot
                traj_path = os.path.join(scene_save_dir, f"traj_{start_idx:04d}.png")
                plot_trajectory(gt_aligned, pred_aligned, lq_aligned, scene, save_dir=scene_save_dir)
                
                # 2. Depth Map Grid
                v_gt_depth = batch["gt_depth"] # [V, 1, H, W]
                v_lq_depth = lq_out["depth"][0].squeeze(-1)
                v_res_depth = res_out["depth"][0].squeeze(-1)
                
                grid = build_depth_grid(
                    batch["clean_img"], batch["deg_img"], 
                    v_gt_depth, v_lq_depth, v_res_depth
                )
                save_image(grid, os.path.join(scene_save_dir, f"depth_{start_idx:04d}.png"))

    # 6. Final Metrics
    final_auc = tracker.compute_auc()
    logger.info(f"Final Results for ETH3D: {final_auc}")

def build_depth_grid(clean, deg, gt_d, lq_d, res_d):
    V = clean.shape[0]
    rows = []
    for i in range(V):
        row = [
            to_rgb_tensor(clean[i:i+1]),
            to_rgb_tensor(deg[i:i+1]),
            to_rgb_tensor(gt_d[i:i+1], is_depth=True),
            to_rgb_tensor(lq_d[i:i+1], is_depth=True),
            to_rgb_tensor(res_d[i:i+1], is_depth=True)
        ]
        rows.append(torch.cat(row, dim=0))
    return make_grid(torch.cat(rows, dim=0), nrow=5, padding=5)

def load_custom_window(dataset, indices, device):
    deg_list, clean_list, depth_list, pose_list = [], [], [], []
    for i in indices:
        c_t, deg_t = process_clean_deg_tensors_from_image(dataset.clean_img_paths[i], kernel=dataset.kernel)
        d_t = process_eth3d_depth_bin(dataset.gt_depth_paths[i])
        
        scene = dataset.scene_names[i]
        img_name = os.path.basename(dataset.clean_img_paths[i])
        pose_data = dataset.cam_dict[f"{scene}/{img_name}"]
        se3 = np.eye(4)
        se3[:3, :3], se3[:3, 3] = pose_data["R"], pose_data["t"]
        
        clean_list.append(c_t)
        deg_list.append(deg_t)
        depth_list.append(d_t)
        pose_list.append(torch.from_numpy(se3).float())
        
    return {
        "clean_img": torch.stack(clean_list).to(device),
        "deg_img": torch.stack(deg_list).to(device),
        "gt_depth": torch.stack(depth_list).to(device),
        "gt_poses": torch.stack(pose_list).to(device)
    }