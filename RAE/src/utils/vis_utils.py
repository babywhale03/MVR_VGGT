import cv2 
import torch 
import numpy as np

def load_depth(depth_path, H, W):
    depth = np.fromfile(depth_path, dtype=np.float32)
    assert depth.size == H * W, f"Size mismatch: {depth_path}"
    depth = depth.reshape(H, W)
    depth[np.isinf(depth)] = np.nan
    return depth


@torch.no_grad()
def align_scale_median(gt, pred, eps=1e-6):
    valid = torch.isfinite(gt) & torch.isfinite(pred) & (gt > eps) & (pred > eps)
    if valid.sum() == 0:
        return pred
    scale = torch.median(gt[valid]) / (torch.median(pred[valid]) + eps)
    return pred * scale


@torch.no_grad()
def compute_depth_metrics(gt, pred, eps=1e-6):
    valid = torch.isfinite(gt) & torch.isfinite(pred) & (gt > eps) & (pred > eps)
    if valid.sum() == 0:
        return [float("nan")] * 7

    gt = torch.clamp(gt[valid], min=eps)
    pred = torch.clamp(pred[valid], min=eps)

    thresh = torch.maximum(gt / pred, pred / gt)

    d1 = (thresh < 1.25).float().mean()
    d2 = (thresh < 1.25 ** 2).float().mean()
    d3 = (thresh < 1.25 ** 3).float().mean()

    abs_rel = torch.mean(torch.abs(gt - pred) / gt)
    sq_rel = torch.mean((gt - pred) ** 2 / gt)
    rmse = torch.sqrt(torch.mean((gt - pred) ** 2))
    rmse_log = torch.sqrt(torch.mean((torch.log(gt) - torch.log(pred)) ** 2))

    return [
        abs_rel.item(),
        sq_rel.item(),
        rmse.item(),
        rmse_log.item(),
        d1.item(),
        d2.item(),
        d3.item(),
    ]

def process_depth_batch(val_depths, resize=None):
    if torch.is_tensor(val_depths):
        val_depths = val_depths.detach().cpu().numpy()
    
    val_depths = np.squeeze(val_depths) 
    if val_depths.ndim == 2: 
        val_depths = val_depths[np.newaxis, ...]
        
    B = val_depths.shape[0]
    color_results = []

    for i in range(B):
        # color_map = depth_to_colormap(val_depths[i], resize=resize)
        color_map = depth_to_colormap(val_depths[i])
        color_results.append(color_map)
    
    return np.array(color_results) # [B, H, W, 3]

def process_depth_error_batch(gt_depths, pred_depths, resize=None):
    # gt_depths, pred_depths: [V, 1, H, W]
    if torch.is_tensor(gt_depths):
        gt_depths = gt_depths.detach().cpu().float().numpy()
    if torch.is_tensor(pred_depths):
        pred_depths = pred_depths.detach().cpu().float().numpy()
    
    gt_depths = np.squeeze(gt_depths)
    pred_depths = np.squeeze(pred_depths)
    
    if gt_depths.ndim == 2:
        gt_depths = gt_depths[np.newaxis, ...]
    if pred_depths.ndim == 2:
        pred_depths = pred_depths[np.newaxis, ...]
        
    if gt_depths.shape != pred_depths.shape:
        raise ValueError(f"Shape mismatch: GT {gt_depths.shape} vs Pred {pred_depths.shape}")

    B = gt_depths.shape[0]
    color_results = []

    for i in range(B):
        # color_map = depth_error_to_colormap(gt_depths[i], pred_depths[i], resize=resize)
        color_map = depth_error_to_colormap_thresholded(gt_depths[i], pred_depths[i], thr=0.1)
        color_results.append(color_map)
    
    return np.array(color_results) # [B, H, W, 3]

# def process_depth_batch(val_depths, resize=None):
#     B = val_depths.shape[0]
#     color_results = []

#     for i in range(B):
#         depth_2d = np.squeeze(val_depths[i]) 
        
#         color_map = depth_to_colormap(depth_2d, resize=resize)
        
#         color_results.append(color_map)
    
#     return np.array(color_results)

def depth_to_colormap(depth, invalid_color=(0, 0, 0), resize=None):
    valid = np.isfinite(depth) & (depth > 0)
    if valid.sum() == 0:
        return np.zeros((*depth.shape, 3), dtype=np.uint8)

    vis = depth.copy()
    vmin, vmax = np.percentile(vis[valid], [2, 98])
    vis = np.clip(vis, vmin, vmax)
    vis = (vis - vmin) / (vmax - vmin + 1e-8)
    vis = (vis * 255).astype(np.uint8)

    color = cv2.applyColorMap(vis, cv2.COLORMAP_VIRIDIS)
    
    color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
    
    color[~valid] = invalid_color
    
    if resize is not None:
        h, w = resize
        color = cv2.resize(color, (w, h), interpolation=cv2.INTER_LINEAR)
    
    return color

def depth_error_to_colormap(gt, pred, invalid_color=(0, 0, 0), resize=None):
    """
    gt, pred: (H, W) numpy arrays
    resize: (h, w) or None
    """
    if torch.is_tensor(gt): gt = gt.detach().cpu().numpy().squeeze()
    if torch.is_tensor(pred): pred = pred.detach().cpu().numpy().squeeze()

    valid = np.isfinite(gt) & np.isfinite(pred) & (gt > 0) & (pred > 0)
    
    if valid.sum() == 0:
        h, w = gt.shape
        vis = np.zeros((h, w, 3), dtype=np.uint8)
        if resize is not None:
            vis = cv2.resize(vis, (resize[1], resize[0]), interpolation=cv2.INTER_NEAREST)
        return vis

    error = np.zeros_like(gt, dtype=np.float32)
    rel_error = np.abs(pred[valid] - gt[valid]) / (gt[valid] + 1e-8)
    
    error[valid] = np.log10(rel_error + 1e-4)

    vmin, vmax = np.percentile(error[valid], [2, 98])
    error_clipped = np.clip(error, vmin, vmax)
    error_norm = (error_clipped - vmin) / (vmax - vmin + 1e-8)
    error_uint8 = (error_norm * 255).astype(np.uint8)

    vis = cv2.applyColorMap(error_uint8, cv2.COLORMAP_TURBO)
    
    vis = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
    
    vis[~valid] = invalid_color

    if resize is not None:
        h, w = resize
        vis = cv2.resize(vis, (w, h), interpolation=cv2.INTER_LINEAR)

    return vis


def depth_error_to_colormap_thresholded(
    gt,
    pred,
    thr=0.5,                 # AbsRel threshold
    invalid_color=(0, 0, 0),
    resize=None,
):
    """
    Visualize ONLY high-error regions using TURBO.
    Low-error pixels are blacked out.

    gt, pred: (H, W) numpy arrays
    thr: AbsRel threshold
    """

    valid = np.isfinite(gt) & np.isfinite(pred) & (gt > 0) & (pred > 0)
    if valid.sum() == 0:
        h, w = gt.shape
        vis = np.zeros((h, w, 3), dtype=np.uint8)
        if resize is not None:
            vis = cv2.resize(vis, (resize[1], resize[0]), interpolation=cv2.INTER_NEAREST)
        return vis

    # AbsRel
    absrel = np.zeros_like(gt, dtype=np.float32)
    absrel[valid] = np.abs(pred[valid] - gt[valid]) / gt[valid]

    # High-error mask
    high = valid & (absrel >= thr)
    if high.sum() == 0:
        return np.zeros((*gt.shape, 3), dtype=np.uint8)

    # Optional log compression (recommended)
    err = np.zeros_like(gt, dtype=np.float32)
    err[high] = np.log10(absrel[high] + 1e-4)

    # Normalize only on high-error pixels
    vmin, vmax = np.percentile(err[high], [5, 95])
    err = np.clip(err, vmin, vmax)
    err = (err - vmin) / (vmax - vmin + 1e-8)
    err = (err * 255).astype(np.uint8)

    vis = cv2.applyColorMap(err, cv2.COLORMAP_TURBO)

    # Mask everything else
    vis[~high] = invalid_color

    if resize is not None:
        h, w = resize
        vis = cv2.resize(vis, (w, h), interpolation=cv2.INTER_LINEAR)

    return vis


# ============================================================
# Formatting helpers
# ============================================================
def fmt(val, width=8, prec=4):
    return f"{val:{width}.{prec}f}"

def fmt_int(val, width=12):
    return f"{val:{width},}"

def write_scene_header(f, scene):
    f.write("\n" + "=" * 90 + "\n")
    f.write(f"Scene: {scene}\n")
    f.write("-" * 90 + "\n")
    f.write(
        f"{'Image':<12}"
        f"{'AbsRel':>8}"
        f"{'SqRel':>8}"
        f"{'RMSE':>9}"
        f"{'RMSElog':>10}"
        f"{'δ1':>8}"
        f"{'δ2':>8}"
        f"{'δ3':>8}"
        f"{'Pixels':>12}\n"
    )
    f.write("-" * 90 + "\n")