import argparse
import logging
import math
import os
import torch
import numpy as np
from collections import defaultdict, OrderedDict
from copy import deepcopy
from pathlib import Path
from tqdm import tqdm
from torchvision.utils import make_grid, save_image

from vggt.vggt.models.vggt import VGGT
from RAE.src.stage2.models import Stage2ModelProtocol
from RAE.src.stage2.transport import create_transport, Sampler

from RAE.src.utils.model_utils import instantiate_from_config
from RAE.src.utils.train_utils_gf import parse_configs, get_autocast_scaler
from RAE.src.utils.dist_utils import setup_distributed, cleanup_distributed
from RAE.src.utils.depth_utils import compute_depth_metrics
from RAE.src.utils.pose_utils import enc_to_se3, get_aligned_centers, plot_trajectory
from RAE.src.utils.vis_utils import process_depth_batch, process_depth_error_batch
from omegaconf import OmegaConf

def to_rgb_tensor(img_batch, is_depth=False):
    if is_depth:
        vis_np = process_depth_batch(img_batch) # [B, H, W, 3]
        return torch.from_numpy(vis_np).permute(0, 3, 1, 2).float() / 255.0
    else:
        if img_batch.shape[1] == 1:
            img_batch = img_batch.repeat(1, 3, 1, 1)
        return img_batch.detach().cpu().float()

def error_to_tensor(vis_np):
    return torch.from_numpy(vis_np).permute(0, 3, 1, 2).float() / 255.0

def create_eval_grid(batch, lq_out, res_out):
    idx = 0 
    v_clean = batch["clean_img"][idx]
    v_deg = batch["deg_img"][idx]
    v_gt_depth = batch["gt_depth"][idx]
    
    v_lq_depth = lq_out["depth"][idx].squeeze(-1)
    v_res_depth = res_out["depth"][idx].squeeze(-1)
    
    v_input_err = process_depth_error_batch(v_gt_depth, v_lq_depth.unsqueeze(-1))
    v_output_err = process_depth_error_batch(v_gt_depth, v_res_depth.unsqueeze(-1))
    
    V = v_clean.shape[0]
    row_elements = [
        to_rgb_tensor(v_clean),                       # 1. Clean RGB
        to_rgb_tensor(v_deg),                         # 2. Deg RGB
        to_rgb_tensor(v_gt_depth, True),              # 3. GT Depth
        to_rgb_tensor(v_lq_depth, True),              # 4. LQ Depth
        error_to_tensor(v_input_err),                 # 5. LQ Error
        to_rgb_tensor(v_res_depth, True),             # 6. Restored Depth
        error_to_tensor(v_output_err)                 # 7. Restored Error
    ]
    
    combined = []
    for i in range(V):
        for el in row_elements:
            combined.append(el[i:i+1])
    
    return make_grid(torch.cat(combined, dim=0), nrow=7, padding=10, pad_value=1.0)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--ckpt", type=str, required=True, help="Trained model checkpoint path")
    parser.add_argument("--results-dir", type=str, default="eval_eth3d")
    parser.add_argument("--precision", type=str, choices=["fp32", "fp16", "bf16"], default="bf16")
    parser.add_argument("--num-sequences", type=int, default=10, help="Number of sequences to evaluate and visualize from ETH3D dataset.")
    parser.add_argument("--kernel-size", type=int, default=100)
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    rank, world_size, device = setup_distributed()
    
    full_cfg = OmegaConf.load(args.config) # encoder_input: 224
    (
        model_config,
        transport_config,
        sampler_config,
        guidance_config,
        misc_config,
        training_config,
        data_config,
        eval_config,
    ) = parse_configs(full_cfg)

    if model_config is None:
        raise ValueError("Config must provide stage_2 section.")

    def to_dict(cfg_section):
        if cfg_section is None:
            return {}
        return OmegaConf.to_container(cfg_section, resolve=True)

    misc = to_dict(misc_config) # {'latent_size': [768, 16, 16], 'num_classes': 1000, 'time_dist_shift_dim': 196608, 'time_dist_shift_base': 4096}
    transport_cfg = to_dict(transport_config) # {'params': {'path_type': 'Linear', 'prediction': 'velocity', 'loss_weight': None, 'time_dist_type': 'logit-normal_0_1'}}
    sampler_cfg = to_dict(sampler_config) # {'mode': 'ODE', 'params': {'sampling_method': 'euler', 'num_steps': 50, 'atol': 1e-06, 'rtol': 0.001, 'reverse': False}}
    guidance_cfg = to_dict(guidance_config) # {'method': 'cfg', 'scale': 1.0, 't_min': 0.0, 't_max': 1.0}
    training_cfg = to_dict(training_config) 
    data_cfg = to_dict(data_config)

    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    extract_layer = training_cfg.get("extract_layer", 3)

    vggt_model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)
    vggt_model.eval().requires_grad_(False)

    model: Stage2ModelProtocol = instantiate_from_config(model_config).to(device)
    model.eval()

    logging.info(f"Loading weights from {args.ckpt}")
    checkpoint = torch.load(args.ckpt, map_location=device)
    state_dict = checkpoint["ema"] if "ema" in checkpoint else checkpoint["model"]
    
    base_res = 518
    vggt_patch_size = model_config["params"].get("vggt_patch_size", 14)
    pH, pW = base_res // vggt_patch_size, base_res // vggt_patch_size
    vec_size = 5 + pH * pW 
    shift_dim = vec_size * 2048
    time_dist_shift = math.sqrt(shift_dim / misc.get("time_dist_shift_base", 4096))

    transport = create_transport(path_type="Linear", prediction="velocity", time_dist_shift=time_dist_shift)
    transport_sampler = Sampler(transport)
    eval_sampler = transport_sampler.sample_ode(sampling_method='euler', num_steps=50, atol=1e-6, rtol=1e-3)

    from RAE.src.utils.train_utils_gf import ETH3DDataset
    from torch.utils.data import DataLoader
    try:
        eth3d_cfg = data_cfg["val"]["library"]["eth3d"]
        
        logging.info(f"Loading ETH3D Dataset from: {eth3d_cfg['clean_img_path']}")
        
        val_ds = ETH3DDataset(
            clean_img_paths=eth3d_cfg["clean_img_path"],
            gt_depth_paths=eth3d_cfg["gt_depth_path"],
            mode='val',
            num_view=10,
            view_sel={'strategy': 'sequential'},
            kernel_size=args.kernel_size
        )
    except KeyError as e:
        logging.error(f"ETH3D dataset configuration error: {e}")
        raise e

    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    logging.info(f"Dataset loaded: {len(val_ds)} samples found.")

    eval_dir = os.path.join(args.results_dir, Path(args.ckpt).stem)
    vis_depth_dir = os.path.join(eval_dir, "depth_vis")
    vis_pose_dir = os.path.join(eval_dir, "pose_vis")
    os.makedirs(vis_depth_dir, exist_ok=True)
    os.makedirs(vis_pose_dir, exist_ok=True)

    eval_metrics = defaultdict(float)
    count = 0

    print(f"Starting ETH3D Evaluation (rank {rank})...")

    with torch.no_grad():
        for i, batch in enumerate(tqdm(val_loader)):
            if i >= args.num_sequences: break

            clean_img = batch["clean_img"].to(device) # [1, V, 3, H, W]
            deg_img = batch["deg_img"].to(device)
            gt_depth = batch["gt_depth"].to(device)
            gt_poses = batch["gt_poses"].to(device) # [1, V, 4, 4]
            
            with torch.cuda.amp.autocast(dtype=dtype):
                lq_predictions = vggt_model(deg_img, extract_layer_num=extract_layer)
                lq_latent = lq_predictions["extracted_latent"][:, :, :, 1024:]

                generator = torch.Generator(device=device).manual_seed(42 + i)
                zs = torch.randn(*lq_latent.shape, generator=generator, device=device)
                xt = zs + lq_latent
                sample_model_kwargs = {"img": deg_img}
                restored_latent = eval_sampler(xt, model.forward, **sample_model_kwargs)[-1].float()
                
                vggt_result = {'restored_latent': restored_latent}
                res_predictions = vggt_model(deg_img, extract_layer_num=extract_layer, 
                                           vggt_result=vggt_result, change_latent=True)

            for v in range(clean_img.shape[1]):
                m = compute_depth_metrics(gt_depth[0, v, 0], res_predictions["depth"][0, v].squeeze(-1))
                if m:
                    for k, val in m.items(): eval_metrics[k] += val
                    count += 1

            v_lq_se3 = enc_to_se3(lq_predictions["pose_enc"], deg_img.shape[-2:])
            v_res_se3 = enc_to_se3(res_predictions["pose_enc"], deg_img.shape[-2:])

            gt_centers = get_aligned_centers(gt_poses[0])
            lq_centers = get_aligned_centers(v_lq_se3)
            res_centers = get_aligned_centers(v_res_se3)

            traj_path = os.path.join(vis_pose_dir, f"traj_{i:03d}.png")
            plot_trajectory(gt_centers, res_centers, lq_centers, save_path=traj_path)

            grid = create_eval_grid(batch, lq_predictions, res_predictions)
            grid_path = os.path.join(vis_depth_dir, f"depth_{i:03d}.png")
            save_image(grid, grid_path, normalize=False)

    if count > 0:
        print(f"\n--- Final Results ({count} frames) ---")
        for k, v in eval_metrics.items():
            print(f"{k}: {v/count:.4f}")
    
    cleanup_distributed()
    print("Evaluation Done!")

if __name__ == "__main__":
    main()