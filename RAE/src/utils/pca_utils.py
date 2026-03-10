import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import os

NUM_SPECIAL_TOKENS = 5


def compute_spatial_dims(img, patch_size=14):
    H, W = img.shape[-2:]
    pH = H // patch_size
    pW = W // patch_size
    return H, W, pH, pW


def tokens_to_spatial(feat, H, W, patch_size=14):
    """
    feat: [B, S, P, C]
    Remove special tokens and reshape to spatial
    """
    B, S, P, C = feat.shape
    pH = H // patch_size
    pW = W // patch_size

    feat = feat[:, :, NUM_SPECIAL_TOKENS:, :]
    feat = feat.view(B, S, pH, pW, C)

    return feat


def pca_project(feat_spatial):
    """
    feat_spatial: [B, S, pH, pW, C]
    returns: [B, S, 3, pH, pW]
    """

    B, S, pH, pW, C = feat_spatial.shape
    feat_flat = feat_spatial.reshape(-1, C)

    # Disable autocast for PCA stability
    with torch.cuda.amp.autocast(enabled=False):
        feat_flat = feat_flat.float()
        mean = feat_flat.mean(0, keepdim=True)
        feat_centered = feat_flat - mean
        U, Sval, V = torch.pca_lowrank(feat_centered, q=3)
        proj = feat_centered @ V[:, :3]

    proj = proj.view(B, S, pH, pW, 3)

    # Normalize per-view
    proj_min = proj.amin(dim=(2,3), keepdim=True)
    proj_max = proj.amax(dim=(2,3), keepdim=True)
    proj = (proj - proj_min) / (proj_max - proj_min + 1e-6)

    proj = proj.permute(0,1,4,2,3)  # B,S,3,pH,pW

    return proj


def upsample_to_image(pca_map, H, W):
    B, S, C, pH, pW = pca_map.shape
    pca_map = F.interpolate(
        pca_map.reshape(B*S, C, pH, pW),
        size=(H, W),
        mode='bilinear',
        align_corners=False
    )
    return pca_map.view(B, S, C, H, W)


def overlay_image(img, pca_map, alpha=0.7):
    return (1 - alpha) * img + alpha * pca_map


def save_layer_grid(clean_img, clean_tokens, lq_tokens, res_tokens,
                    visualize_layers, save_path,
                    patch_size=14):

    B, S = clean_img.shape[:2]
    H, W, pH, pW = compute_spatial_dims(clean_img, patch_size)

    clean_img_vis = clean_img.clamp(0,1)

    num_layers = len(visualize_layers)
    num_cols = S * 3  # Clean | LQ | Res per view

    def generate_and_save(overlay_flag, suffix):

        fig, axes = plt.subplots(
            num_layers,
            num_cols,
            figsize=(num_cols * 2, num_layers * 2)
        )

        if num_layers == 1:
            axes = axes[None, :]

        for row_idx, layer in enumerate(visualize_layers):

            clean_feat = clean_tokens[layer]
            lq_feat = lq_tokens[layer]
            res_feat = res_tokens[layer]

            clean_sp = tokens_to_spatial(clean_feat, H, W, patch_size)
            lq_sp = tokens_to_spatial(lq_feat, H, W, patch_size)
            res_sp = tokens_to_spatial(res_feat, H, W, patch_size)

            clean_pca = upsample_to_image(pca_project(clean_sp), H, W)
            lq_pca = upsample_to_image(pca_project(lq_sp), H, W)
            res_pca = upsample_to_image(pca_project(res_sp), H, W)

            if overlay_flag:
                clean_vis = overlay_image(clean_img_vis, clean_pca)
                lq_vis = overlay_image(clean_img_vis, lq_pca)
                res_vis = overlay_image(clean_img_vis, res_pca)
            else:
                clean_vis = clean_pca
                lq_vis = lq_pca
                res_vis = res_pca

            for view in range(S):

                col_base = view * 3

                axes[row_idx, col_base + 0].imshow(
                    clean_vis[0, view].permute(1,2,0).cpu().numpy()
                )
                axes[row_idx, col_base + 1].imshow(
                    lq_vis[0, view].permute(1,2,0).cpu().numpy()
                )
                axes[row_idx, col_base + 2].imshow(
                    res_vis[0, view].permute(1,2,0).cpu().numpy()
                )

                ax = axes[row_idx, col_base + 0]
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_visible(False)
                axes[row_idx, col_base + 1].axis("off")
                axes[row_idx, col_base + 2].axis("off")

            # ---- Row label (Layer number) ----
            axes[row_idx, 0].set_ylabel(
                f"Layer {layer}",
                fontsize=14,
                rotation=90,
                labelpad=20
            )

        for view in range(S):
            col_base = view * 3
            axes[0, col_base + 0].set_title(f"View {view} - Clean", fontsize=10)
            axes[0, col_base + 1].set_title(f"View {view} - LQ", fontsize=10)
            axes[0, col_base + 2].set_title(f"View {view} - Res", fontsize=10)

        plt.tight_layout()
        final_path = save_path.replace(".png", f"_{suffix}.png")
        plt.savefig(final_path, dpi=200)
        plt.close()

    generate_and_save(overlay_flag=False, suffix="pca")
    generate_and_save(overlay_flag=True, suffix="overlay")