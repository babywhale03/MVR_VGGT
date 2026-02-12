import torch

# def compute_depth_metrics(gt, pred):
#     with torch.no_grad():
#         device = pred.device
#         gt = gt.to(device)

#         if pred.shape[-1] == 1: pred = pred.squeeze(-1)
#         if gt.shape[-1] == 1: gt = gt.squeeze(-1)

#         pred = pred.squeeze() 
#         gt = gt.squeeze()

#         if pred.ndim == 4:
#             pred = pred.reshape(-1, pred.shape[-2], pred.shape[-1])
#         if gt.ndim == 4:
#             gt = gt.reshape(-1, gt.shape[-2], gt.shape[-1])

#         mask = (gt > 0) & (~torch.isnan(gt)) & (~torch.isinf(gt))
        
#         if not mask.any():
#             return None

#         pred_v = pred[mask].clamp(min=1e-7)
#         gt_v = gt[mask]

#         ratio = torch.median(gt_v) / (torch.median(pred_v) + 1e-8)
#         pred_v = pred_v * ratio

#         abs_rel = torch.mean(torch.abs(gt_v - pred_v) / gt_v)
#         sq_rel = torch.mean(((gt_v - pred_v) ** 2) / gt_v)
#         rmse = torch.sqrt(torch.mean((gt_v - pred_v) ** 2))
        
#         log_err = torch.log(gt_v) - torch.log(pred_v)
#         rmse_log = torch.sqrt(torch.mean(log_err ** 2))

#         thresh = torch.max((gt_v / pred_v), (pred_v / gt_v))
#         d1 = (thresh < 1.25).float().mean()
#         d2 = (thresh < 1.25**2).float().mean()
#         d3 = (thresh < 1.25**3).float().mean()

#         all_metrics = torch.stack([abs_rel, sq_rel, rmse, rmse_log, d1, d2, d3])
#         metrics = all_metrics.cpu().tolist()

#     return {
#         "abs_rel": metrics[0],
#         "sq_rel": metrics[1],
#         "rmse": metrics[2],
#         "rmse_log": metrics[3],
#         "d1": metrics[4],
#         "d2": metrics[5],
#         "d3": metrics[6]
#     }


def compute_depth_metrics(gt, pred):
    with torch.no_grad():
        device = pred.device
        gt = gt.to(device)

        if pred.shape[-1] == 1: pred = pred.squeeze(-1)
        if gt.shape[-1] == 1: gt = gt.squeeze(-1)

        pred = pred.squeeze() 
        gt = gt.squeeze()

        if pred.ndim == 4:
            pred = pred.reshape(-1, pred.shape[-2], pred.shape[-1])
        if gt.ndim == 4:
            gt = gt.reshape(-1, gt.shape[-2], gt.shape[-1])

        mask = (gt > 0.1) & (gt < 100.0) & (~torch.isnan(gt)) & (~torch.isinf(gt))
        
        if not mask.any():
            return None

        pred_v = pred[mask].clamp(min=1e-7)
        gt_v = gt[mask]
        ratio = torch.median(gt_v) / (torch.median(pred_v) + 1e-8)
        pred_v = pred_v * ratio

        abs_rel = torch.mean(torch.abs(gt_v - pred_v) / (gt_v + 1e-7))
        sq_rel = torch.mean(((gt_v - pred_v) ** 2) / (gt_v + 1e-7))
        rmse = torch.sqrt(torch.mean((gt_v - pred_v) ** 2))
        
        log_err = torch.log(gt_v + 1e-7) - torch.log(pred_v + 1e-7)
        rmse_log = torch.sqrt(torch.mean(log_err ** 2))

        thresh = torch.max((gt_v / (pred_v + 1e-7)), (pred_v / (gt_v + 1e-7)))
        d1 = (thresh < 1.25).float().mean()
        d2 = (thresh < 1.25**2).float().mean()
        d3 = (thresh < 1.25**3).float().mean()

        all_metrics = torch.stack([abs_rel, sq_rel, rmse, rmse_log, d1, d2, d3])
        metrics = all_metrics.cpu().tolist()

    return {
        "abs_rel": metrics[0],
        "sq_rel": metrics[1],
        "rmse": metrics[2],
        "rmse_log": metrics[3],
        "d1": metrics[4],
        "d2": metrics[5],
        "d3": metrics[6]
    }