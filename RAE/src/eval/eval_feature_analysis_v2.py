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
from RAE.src.utils.pca_utils import *
from RAE.src.utils.cka_utils import * 

from omegaconf import OmegaConf
from torch.utils.data import DataLoader

import torch.nn.functional as F
from sklearn.decomposition import PCA

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--results-dir", type=str, default="eval_eth3d_final")
    parser.add_argument("--num-sequences", type=int, default=10)
    parser.add_argument("--num-views", type=int, default=10)
    parser.add_argument("--kernel-size", type=int, default=100)
    parser.add_argument("--extract-layer", type=int, default=12)
    # parser.add_argument("--visualize-layers", type=int, nargs="+", default=[3,6,9,12,15,18,21])
    # parser.add_argument("--visualize-layers", type=int, nargs="+", default=[3,4,7,11,13,15,17,19,21,23])
    parser.add_argument("--visualize-layers", type=int, nargs="+", default=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23])
    return parser.parse_args()

def main():
    args = parse_args()
    breakpoint()
    rank, world_size, device = setup_distributed()
    
    eval_dir = os.path.join(args.results_dir, f"{Path(args.ckpt).stem}_full_kernel{args.kernel_size}_view{args.num_views}_cka")

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
        mode='val', num_view=args.num_views, view_sel={'strategy': 'sequential'}, kernel_size=args.kernel_size
    )
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)

    vggt_patch_size = model_cfg["params"].get("vggt_patch_size", 14)
    pH, pW = 518 // vggt_patch_size, 518 // vggt_patch_size
    shift_dim = (5 + pH * pW) * 2048
    time_dist_shift = math.sqrt(shift_dim / misc_cfg.get("time_dist_shift_base", 4096))
    
    transport = create_transport(path_type="Linear", prediction="velocity", time_dist_shift=time_dist_shift)
    eval_sampler = Sampler(transport).sample_ode(sampling_method='euler', num_steps=50, atol=1e-6, rtol=1e-3)

    with torch.no_grad():
        for i, batch in enumerate(tqdm(val_loader)):
            if i >= args.num_sequences: break
            clean_img = batch["clean_img"].to(device)
            deg_img = batch["deg_img"].to(device)
            gt_poses = batch["gt_poses"].to(device)
            
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                clean_out = vggt_model(clean_img, extract_layer_num=args.extract_layer)
                clean_latent = clean_out["extracted_latent"][:, :, :, 1024:]
                clean_tokens = clean_out["aggregated_tokens_list"]
                lq_out = vggt_model(deg_img, extract_layer_num=args.extract_layer)
                lq_latent = lq_out["extracted_latent"][:, :, :, 1024:]
                lq_tokens = lq_out["aggregated_tokens_list"]

                xt = torch.randn_like(lq_latent) + lq_latent
                restored_latent = eval_sampler(xt, model.forward, img=deg_img).float()

                res_out = vggt_model(deg_img, extract_layer_num=args.extract_layer, vggt_result={'restored_latent': restored_latent}, change_latent=True)

                res_out_tokens = res_out["aggregated_tokens_list"]

                analysis_dir = os.path.join(eval_dir, "layer_analysis")
                os.makedirs(analysis_dir, exist_ok=True)

                # vis_save_path = os.path.join(
                #     analysis_dir,
                #     f"scene_{i}_pca_layers_.png"
                # )

                # save_layer_grid(
                #     clean_img=clean_img,
                #     clean_tokens=clean_tokens,
                #     lq_tokens=lq_tokens,
                #     res_tokens=res_out_tokens,
                #     visualize_layers=args.visualize_layers,
                #     save_path=vis_save_path,
                #     patch_size=vggt_patch_size
                # )

                metrics = compute_layerwise_metrics(
                    clean_tokens,
                    lq_tokens,
                    res_out_tokens,
                    args.visualize_layers
                )

                # save numpy
                np.save(
                    os.path.join(analysis_dir, f"seq_{i}_metrics.npy"),
                    metrics
                )

                # plots
                save_metric_plot(
                    metrics["cka_clean_res"],
                    metrics["cka_clean_lq"],
                    args.visualize_layers,
                    "Layerwise CKA Similarity",
                    os.path.join(analysis_dir, f"seq_{i}_cka.png")
                )

                save_metric_plot(
                    metrics["dist_clean_res"],
                    metrics["dist_clean_lq"],
                    args.visualize_layers,
                    "Pairwise Distance Correlation",
                    os.path.join(analysis_dir, f"seq_{i}_distance_corr.png")
                )

                save_metric_plot(
                    metrics["cos_clean_res"],
                    metrics["cos_clean_lq"],
                    args.visualize_layers,
                    "Layerwise Cosine Similarity",
                    os.path.join(analysis_dir, f"seq_{i}_cosine.png")
                )
                                    
if __name__ == "__main__":
    main()