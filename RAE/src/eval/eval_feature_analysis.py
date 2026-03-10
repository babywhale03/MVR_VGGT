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

import os
import math
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

# ============================================================
# 1. 48 Layer Extractor
# ============================================================

def extract_48_layers(token_list):
    layers = []
    for i in range(24):
        t = token_list[i]  # [B,S,P,2048]
        t1 = t[:, :, :, :1024]
        t2 = t[:, :, :, 1024:]
        layers.append(t1)
        layers.append(t2)
    return layers  # len = 48


# ============================================================
# 2. Flatten helper
# ============================================================

def flatten_tokens(t):
    B,S,P,C = t.shape
    return t.reshape(B*S*P, C)


# ============================================================
# 3. Single Layer Analysis
# ============================================================

def analyze_single_layer(clean, lq, res, layer_id, save_dir):

    clean_np = flatten_tokens(clean).detach().cpu().numpy()
    lq_np    = flatten_tokens(lq).detach().cpu().numpy()
    res_np   = flatten_tokens(res).detach().cpu().numpy()

    # subsample for speed
    max_points = 3000
    if clean_np.shape[0] > max_points:
        idx = np.random.choice(clean_np.shape[0], max_points, replace=False)
        clean_np = clean_np[idx]
        lq_np    = lq_np[idx]
        res_np   = res_np[idx]

    pca = PCA(n_components=64)
    pca.fit(clean_np)  

    clean_np = pca.transform(clean_np)
    lq_np    = pca.transform(lq_np)
    res_np   = pca.transform(res_np)

    all_data = np.concatenate([clean_np, lq_np, res_np], axis=0)
    all_data = all_data / (np.linalg.norm(all_data, axis=1, keepdims=True) + 1e-8)

    all_data = pca.fit_transform(all_data)

    labels = (
        ["clean"]*len(clean_np) +
        ["lq"]*len(lq_np) +
        ["restored"]*len(res_np)
    )
    labels = np.array(labels)

    # ======================
    # 1. t-SNE
    # ======================
    tsne = TSNE(n_components=2, perplexity=30, init='pca', random_state=0)
    reduced = tsne.fit_transform(all_data)

    plt.figure(figsize=(6,6))
    for name, color in zip(["clean","lq","restored"], ["blue","red","green"]):
        mask = labels == name
        plt.scatter(reduced[mask,0], reduced[mask,1], s=5, alpha=0.5, label=name)

    plt.legend()
    plt.title(f"Layer {layer_id} t-SNE")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"layer_{layer_id}_tsne.png"))
    plt.close()

    # ======================
    # 2. KMeans
    # ======================
    k = 8
    kmeans = KMeans(n_clusters=k, random_state=0).fit(all_data)
    cluster_labels = kmeans.labels_

    sil = silhouette_score(all_data, cluster_labels)

    dist = {}
    start = 0
    for name, arr in zip(["clean","lq","restored"], [clean_np, lq_np, res_np]):
        end = start + len(arr)
        hist = np.bincount(cluster_labels[start:end], minlength=k)
        dist[name] = hist / hist.sum()
        start = end

    # ======================
    # 3. Distance Metrics
    # ======================
    centroid_clean = clean_np.mean(axis=0)
    centroid_lq    = lq_np.mean(axis=0)
    centroid_res   = res_np.mean(axis=0)

    def cos(a,b):
        return np.dot(a,b) / (np.linalg.norm(a)*np.linalg.norm(b)+1e-8)

    metrics = {
        "silhouette": sil,
        "centroid_dist_lq": np.linalg.norm(centroid_clean-centroid_lq),
        "centroid_dist_res": np.linalg.norm(centroid_clean-centroid_res),
        "cos_clean_lq": cos(centroid_clean, centroid_lq),
        "cos_clean_res": cos(centroid_clean, centroid_res),
    }

    return metrics


# ============================================================
# 4. Full 48-Layer Analysis
# ============================================================

def analyze_all_layers(clean_tokens, lq_tokens, res_tokens, save_root):

    os.makedirs(save_root, exist_ok=True)

    clean_layers = extract_48_layers(clean_tokens)
    lq_layers    = extract_48_layers(lq_tokens)
    res_layers   = extract_48_layers(res_tokens)

    all_metrics = []

    for i in range(48):
        print(f"Analyzing Layer {i}...")
        metrics = analyze_single_layer(
            clean_layers[i],
            lq_layers[i],
            res_layers[i],
            i,
            save_root
        )
        all_metrics.append(metrics)

    import json

    metrics_path = os.path.join(save_root, "all_layer_metrics.json")

    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=4)

    return all_metrics


# ============================================================
# 5. Layer-wise Curve Plot
# ============================================================

def plot_layer_curves(all_metrics, save_root):

    centroid_lq  = [m["centroid_dist_lq"] for m in all_metrics]
    centroid_res = [m["centroid_dist_res"] for m in all_metrics]

    cos_lq  = [m["cos_clean_lq"] for m in all_metrics]
    cos_res = [m["cos_clean_res"] for m in all_metrics]

    sil = [m["silhouette"] for m in all_metrics]

    # centroid distance
    plt.figure()
    plt.plot(centroid_lq, label="LQ")
    plt.plot(centroid_res, label="Restored")
    plt.legend()
    plt.title("Centroid Distance to Clean")
    plt.xlabel("Layer")
    plt.ylabel("L2 Distance")
    plt.savefig(os.path.join(save_root, "centroid_distance_curve.png"))
    plt.close()

    # cosine similarity
    plt.figure()
    plt.plot(cos_lq, label="LQ")
    plt.plot(cos_res, label="Restored")
    plt.legend()
    plt.title("Cosine Similarity to Clean")
    plt.xlabel("Layer")
    plt.ylabel("Cosine")
    plt.savefig(os.path.join(save_root, "cosine_similarity_curve.png"))
    plt.close()

    # silhouette
    plt.figure()
    plt.plot(sil)
    plt.title("Silhouette Score")
    plt.xlabel("Layer")
    plt.ylabel("Score")
    plt.savefig(os.path.join(save_root, "silhouette_curve.png"))
    plt.close()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--results-dir", type=str, default="eval_eth3d_final")
    parser.add_argument("--num-sequences", type=int, default=10)
    parser.add_argument("--num-views", type=int, default=10)
    parser.add_argument("--kernel-size", type=int, default=100)
    parser.add_argument("--extract-layer", type=int, default=12)
    return parser.parse_args()

def main():
    args = parse_args()
    breakpoint()
    rank, world_size, device = setup_distributed()
    
    eval_dir = os.path.join(args.results_dir, f"{Path(args.ckpt).stem}_full_kernel{args.kernel_size}_view{args.num_views}_analysis_new")

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

    vis_depth_dir, vis_depth_metric, vis_pose_dir, vis_feature_sim, vis_pose_metric = os.path.join(eval_dir, "depth_vis"), os.path.join(eval_dir, "depth_metrics"), os.path.join(eval_dir, "pose_vis"), os.path.join(eval_dir, "feature_sim"), os.path.join(eval_dir, "pose_metric")
    os.makedirs(vis_depth_dir, exist_ok=True); os.makedirs(vis_depth_metric, exist_ok=True); os.makedirs(vis_pose_dir, exist_ok=True); os.makedirs(vis_feature_sim, exist_ok=True); os.makedirs(vis_pose_metric, exist_ok=True)

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

                analysis_dir = os.path.join(eval_dir, f"scene_{i}", "layer_analysis")
                os.makedirs(analysis_dir, exist_ok=True)

                all_metrics = analyze_all_layers(
                    clean_tokens,
                    lq_tokens,
                    res_out_tokens,
                    analysis_dir
                )

                plot_layer_curves(all_metrics, analysis_dir)

if __name__ == "__main__":
    main()