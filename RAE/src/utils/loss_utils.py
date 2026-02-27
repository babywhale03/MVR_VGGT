from vggt.vggt.utils.pose_enc import extri_intri_to_pose_encoding
from vggt.training.train_utils.general import check_and_fix_inf_nan

def compute_camera_loss(
    pred_pose_encodings,              # predictions dict, contains pose encodings
    batch_data,             # ground truth and mask batch dict
    loss_type="l1",         # "l1" or "l2" loss
    gamma=0.6,              # temporal decay weight for multi-stage training
    pose_encoding_type="absT_quaR_FoV",
    weight_trans=1.0,       # weight for translation loss
    weight_rot=1.0,         # weight for rotation loss
    weight_focal=0.5,       # weight for focal length loss
    **kwargs
):
    # Binary mask for valid points per frame (B, N, H, W)
    point_masks = batch_data['point_masks']
    # Only consider frames with enough valid points (>100)
    valid_frame_mask = point_masks[:, 0].sum(dim=[-1, -2]) > 100
    # Number of prediction stages
    n_stages = len(pred_pose_encodings)

    # Get ground truth camera extrinsics and intrinsics
    gt_extrinsics = batch_data['extrinsics']
    gt_intrinsics = batch_data['intrinsics']
    image_hw = batch_data['images'].shape[-2:]

    # Encode ground truth pose to match predicted encoding format
    gt_pose_encoding = extri_intri_to_pose_encoding(
        gt_extrinsics, gt_intrinsics, image_hw, pose_encoding_type=pose_encoding_type
    )

    # Initialize loss accumulators for translation, rotation, focal length
    total_loss_T = total_loss_R = total_loss_FL = 0

    # Compute loss for each prediction stage with temporal weighting
    for stage_idx in range(n_stages):
        # Later stages get higher weight (gamma^0 = 1.0 for final stage)
        stage_weight = gamma ** (n_stages - stage_idx - 1)
        pred_pose_stage = pred_pose_encodings[stage_idx]

        if valid_frame_mask.sum() == 0:
            # If no valid frames, set losses to zero to avoid gradient issues
            loss_T_stage = (pred_pose_stage * 0).mean()
            loss_R_stage = (pred_pose_stage * 0).mean()
            loss_FL_stage = (pred_pose_stage * 0).mean()
        else:
            # Only consider valid frames for loss computation
            loss_T_stage, loss_R_stage, loss_FL_stage = camera_loss_single(
                pred_pose_stage[valid_frame_mask].clone(),
                gt_pose_encoding[valid_frame_mask].clone(),
                loss_type=loss_type
            )
        # Accumulate weighted losses across stages
        total_loss_T += loss_T_stage * stage_weight
        total_loss_R += loss_R_stage * stage_weight
        total_loss_FL += loss_FL_stage * stage_weight

    # Average over all stages
    avg_loss_T = total_loss_T / n_stages
    avg_loss_R = total_loss_R / n_stages
    avg_loss_FL = total_loss_FL / n_stages

    # Compute total weighted camera loss
    total_camera_loss = (
        avg_loss_T * weight_trans +
        avg_loss_R * weight_rot +
        avg_loss_FL * weight_focal
    )

    # Return loss dictionary with individual components
    return {
        "loss_camera": total_camera_loss,
        "loss_T": avg_loss_T,
        "loss_R": avg_loss_R,
        "loss_FL": avg_loss_FL
    }

def camera_loss_single(pred_pose_enc, gt_pose_enc, loss_type="l1"):
    """
    Computes translation, rotation, and focal loss for a batch of pose encodings.
    
    Args:
        pred_pose_enc: (N, D) predicted pose encoding
        gt_pose_enc: (N, D) ground truth pose encoding
        loss_type: "l1" (abs error) or "l2" (euclidean error)
    Returns:
        loss_T: translation loss (mean)
        loss_R: rotation loss (mean)
        loss_FL: focal length/intrinsics loss (mean)
    
    NOTE: The paper uses smooth l1 loss, but we found l1 loss is more stable than smooth l1 and l2 loss.
        So here we use l1 loss.
    """
    if loss_type == "l1":
        # Translation: first 3 dims; Rotation: next 4 (quaternion); Focal/Intrinsics: last dims
        loss_T = (pred_pose_enc[..., :3] - gt_pose_enc[..., :3]).abs()
        loss_R = (pred_pose_enc[..., 3:7] - gt_pose_enc[..., 3:7]).abs()
        loss_FL = (pred_pose_enc[..., 7:] - gt_pose_enc[..., 7:]).abs()
    elif loss_type == "l2":
        # L2 norm for each component
        loss_T = (pred_pose_enc[..., :3] - gt_pose_enc[..., :3]).norm(dim=-1, keepdim=True)
        loss_R = (pred_pose_enc[..., 3:7] - gt_pose_enc[..., 3:7]).norm(dim=-1)
        loss_FL = (pred_pose_enc[..., 7:] - gt_pose_enc[..., 7:]).norm(dim=-1)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")

    # Check/fix numerical issues (nan/inf) for each loss component
    loss_T = check_and_fix_inf_nan(loss_T, "loss_T")
    loss_R = check_and_fix_inf_nan(loss_R, "loss_R")
    loss_FL = check_and_fix_inf_nan(loss_FL, "loss_FL")

    # Clamp outlier translation loss to prevent instability, then average
    loss_T = loss_T.clamp(max=100).mean()
    loss_R = loss_R.mean()
    loss_FL = loss_FL.mean()

    return loss_T, loss_R, loss_FL