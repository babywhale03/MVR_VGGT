# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
A minimal training script for SiT using PyTorch DDP.
"""
import argparse
import gc
import logging
import math
import os
from collections import defaultdict, OrderedDict
import re
import torch
from datetime import datetime

# the first flag below was False when we tested this script but True makes A100 training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import ImageFolder
from torchvision import transforms
import numpy as np
from PIL import Image
from copy import deepcopy
from glob import glob
from time import time
import argparse
import logging
from pathlib import Path
import math
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import LambdaLR
from omegaconf import OmegaConf
from einops import rearrange
import sys
from tqdm import tqdm
import torchvision.transforms as T 
import torch.nn.functional as F
from torchvision.utils import make_grid, save_image

from vggt.vggt.models.vggt import VGGT
from vggt.vggt.evaluation.test_co3d_mvrm_gf import evaluate_co3d

##### model imports
# from stage1 import RAE
from stage2.models import Stage2ModelProtocol
from stage2.transport import create_transport, Sampler

##### general utils
from utils import wandb_utils
from utils.model_utils import instantiate_from_config
from utils.train_utils import *
from utils.optim_utils import build_optimizer, build_scheduler
from utils.resume_utils import *
from utils.wandb_utils import *
from utils.dist_utils import *
from utils.depth_utils import *

##### Eval utils
from eval import evaluate_generation_distributed

##### Visualization
from torchvision.utils import save_image
from utils.vis_utils import process_depth_batch, process_depth_error_batch

import csv

class MetricLogger:
    def __init__(self, save_path):
        self.save_path = save_path
        self.header_saved = os.path.exists(save_path)

    def log(self, step, train_metrics, val_metrics):
        data = {"step": step}
        
        for k, v in train_metrics.items():
            data[f"train_{k}"] = v
            
        for k, v in val_metrics.items():
            data[f"val_{k}"] = v

        with open(self.save_path, mode='a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=data.keys())
            if not self.header_saved:
                writer.writeheader()
                self.header_saved = True
            writer.writerow(data)

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

def save_checkpoint(
    path: str,
    step: int,
    epoch: int,
    model: DDP,
    ema_model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[LambdaLR],
) -> None:
    state = {
        "step": step,
        "epoch": epoch,
        "model": model.module.state_dict(),
        "ema": ema_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)


def load_checkpoint(
    path: str,
    model: DDP,
    ema_model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[LambdaLR],
) -> Tuple[int, int]:
    checkpoint = torch.load(path, map_location="cpu")
    model.module.load_state_dict(checkpoint["model"])
    ema_model.load_state_dict(checkpoint["ema"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler is not None and checkpoint.get("scheduler") is not None:
        scheduler.load_state_dict(checkpoint["scheduler"])
    return checkpoint.get("epoch", 0), checkpoint.get("step", 0)

def visualize_latent_error_map(deg_img, clean_latent, restored_latent):
    if clean_latent.ndim == 4:
        clean_latent = clean_latent.squeeze(1) # [B, 1041, 1024]
    
    _, _, H, W = deg_img.shape
    pH, pW = H // 14, W // 14

    c_lat = clean_latent[:, 5:, :] # [B, 1036, 1024]
    r_lat = restored_latent[:, 5:, :] # [B, 1036, 1024]

    error = torch.norm(c_lat - r_lat, p=2, dim=-1) # [B, 1036]

    error_map = error.view(-1, 1, pH, pW) # [B, 1, 28, 37]

    error_map_rescaled = F.interpolate(error_map, size=(H, W), mode='bilinear', align_corners=False)

    b, c, h, w = error_map_rescaled.shape
    min_v = error_map_rescaled.view(b, -1).min(dim=-1)[0].view(b, 1, 1, 1)
    max_v = error_map_rescaled.view(b, -1).max(dim=-1)[0].view(b, 1, 1, 1)
    error_map_norm = (error_map_rescaled - min_v) / (max_v - min_v + 1e-8)

    return error_map_norm # [B, 1, 392, 518]

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
                        
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Stage-2 transport model on RAE latents.")
    parser.add_argument("--config", type=str, required=True, help="YAML config containing stage_1 and stage_2 sections.")
    parser.add_argument("--results-dir", type=str, default="ckpts", help="Directory to store training outputs.")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256, help="Input image resolution.")
    parser.add_argument("--precision", type=str, choices=["fp32", "fp16", "bf16"], default="fp32", help="Compute precision for training.")
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging.")
    parser.add_argument("--compile", action="store_true", help="Use torch compile (for rae.encode and model.forward).")
    parser.add_argument("--ckpt", type=str, default=None, help="Optional checkpoint path to resume training.")
    parser.add_argument("--global-seed", type=int, default=None, help="Override training.global_seed from the config.")
    parser.add_argument("--max-views", type=int, default=4, help="Maximum number of views to use during training.")
    parser.add_argument("--kernel-size", type=int, default=100, help="Number of pixels in the square kernel for cost volume.")
    args = parser.parse_args()
    return args

def main():
    torch.autograd.set_detect_anomaly(True)

    """Trains a new SiT model using config-driven hyperparameters."""
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Training currently requires at least one GPU.")
    # world_size: number of processes (GPUs), rank: process index
    rank, world_size, device = setup_distributed()
    # breakpoint()
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

    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    misc = to_dict(misc_config) # {'latent_size': [768, 16, 16], 'num_classes': 1000, 'time_dist_shift_dim': 196608, 'time_dist_shift_base': 4096}
    transport_cfg = to_dict(transport_config) # {'params': {'path_type': 'Linear', 'prediction': 'velocity', 'loss_weight': None, 'time_dist_type': 'logit-normal_0_1'}}
    sampler_cfg = to_dict(sampler_config) # {'mode': 'ODE', 'params': {'sampling_method': 'euler', 'num_steps': 50, 'atol': 1e-06, 'rtol': 0.001, 'reverse': False}}
    guidance_cfg = to_dict(guidance_config) # {'method': 'cfg', 'scale': 1.0, 't_min': 0.0, 't_max': 1.0}
    training_cfg = to_dict(training_config) 
    data_cfg = to_dict(data_config)
    # eval_cfg = to_dict(eval_config)

    # latent_size = tuple(int(dim) for dim in misc.get("latent_size", (768, 16, 16)))
    # shift_dim = misc.get("time_dist_shift_dim", math.prod(latent_size))

    grad_accum_steps = int(training_cfg.get("grad_accum_steps", 1))
    if grad_accum_steps < 1:
        raise ValueError("Gradient accumulation steps must be >= 1.")
    clip_grad_val = training_cfg.get("clip_grad", 1.0)
    clip_grad = float(clip_grad_val) if clip_grad_val is not None else None
    if clip_grad is not None and clip_grad <= 0:
        clip_grad = None
    ema_decay = float(training_cfg.get("ema_decay", 0.9995))
    num_epochs = int(training_cfg.get("epochs", 1400))
    global_batch_size = training_cfg.get("global_batch_size", None) # optional global batch size for override
    if global_batch_size is not None:
        global_batch_size = int(global_batch_size)
        assert global_batch_size % world_size == 0, "global_batch_size must be divisible by world_size"
    else:
        batch_size = int(training_cfg.get("batch_size", 16))
        global_batch_size = batch_size * world_size * grad_accum_steps
    num_workers = int(training_cfg.get("num_workers", 4))
    log_interval = int(training_cfg.get("log_interval", 100))
    sample_every = int(training_cfg.get("sample_every", 5000))
    checkpoint_interval = int(training_cfg.get("checkpoint_interval", 4)) # ckpt interval is epoch based
    cfg_scale_override = training_cfg.get("cfg_scale", None)
    default_seed = int(training_cfg.get("global_seed", 0))

    vggt_model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)
    vggt_model.eval()
    vggt_model.requires_grad_(False)

    # prepare clean img data paths
    # search_pattern = os.path.join(args.input_img_path, "ai_*", "images", "scene_cam_*_final_hdf5", "frame.*.color.hdf5")
    # all_image_paths = sorted(glob(search_pattern))

    # image_paths = []
    # for p in all_image_paths:
    #     match = re.search(r'ai_(\d{3})', p)
    #     if match:
    #         scene_num = int(match.group(1)) 
    #         if scene_num <= 10: 
    #             image_paths.append(p)

    # with torch.no_grad():
    #     with torch.cuda.amp.autocast(enabled=True, dtype=dtype):
    #         clean_input_images = load_and_preprocess_images_hdf5(image_paths).to(device).to(dtype)
    
    if eval_config:
        """
        FID online evaluation setup
        """
        do_eval = True
        eval_interval = int(eval_config.get("eval_interval", 5000))
        eval_model = eval_config.get("eval_model", False) # by default eval ema. This decides whether to **additionally** eval the non-ema model.
        eval_data = eval_config.get("data_path", None)
        reference_npz_path = eval_config.get("reference_npz_path", None)
        assert eval_data, "eval.data_path must be specified to enable evaluation."
        assert reference_npz_path, "eval.reference_npz_path must be specified to enable evaluation."
    else:
        do_eval = False
    global_seed = args.global_seed if args.global_seed is not None else default_seed
    seed = global_seed * world_size + rank
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    generator = torch.Generator(device=device)
    micro_batch_size = global_batch_size // (world_size * grad_accum_steps)
    use_fp16 = args.precision == "fp16"
    use_bf16 = args.precision == "bf16"
    if use_bf16 and not torch.cuda.is_bf16_supported():
        raise ValueError("Requested bf16 precision, but the current CUDA device does not support bfloat16.")
    autocast_dtype = torch.float16 if use_fp16 else torch.bfloat16
    autocast_enabled = use_fp16 or use_bf16
    autocast_kwargs = dict(dtype=autocast_dtype, enabled=autocast_enabled)
    scaler = GradScaler(enabled=use_fp16)

    transport_params = dict(transport_cfg.get("params", {})) # {'path_type': 'Linear', 'prediction': 'velocity', 'loss_weight': None, 'time_dist_type': 'logit-normal_0_1'}
    path_type = transport_params.get("path_type", "Linear")
    prediction = transport_params.get("prediction", "velocity")
    loss_weight = transport_params.get("loss_weight")
    transport_params.pop("time_dist_shift", None)
    extract_layer = training_cfg.get("extract_layer", 3)

    sampler_mode = sampler_cfg.get("mode", "ODE").upper()
    sampler_params = dict(sampler_cfg.get("params", {}))

    guidance_scale = float(guidance_cfg.get("scale", 1.0))
    if cfg_scale_override is not None:
        guidance_scale = float(cfg_scale_override)
    guidance_method = guidance_cfg.get("method", "cfg")

    def guidance_value(key: str, default: float) -> float:
        if key in guidance_cfg:
            return guidance_cfg[key]
        dashed_key = key.replace("_", "-")
        return guidance_cfg.get(dashed_key, default)

    t_min = float(guidance_value("t_min", 0.0))
    t_max = float(guidance_value("t_max", 1.0))
    
    experiment_dir, checkpoint_dir, logger = configure_experiment_dirs(args, full_cfg, rank)
    # breakpoint()
    #### Model init
    model: Stage2ModelProtocol = instantiate_from_config(model_config).to(device) 
    if args.compile:
        try:
            model.forward = torch.compile(model.forward)
        except:
            print('MODEL FORWARD compile meets error, falling back to no compile')
    else:
        pass
        # raise NotImplementedError('ARGS>COMPILE')
    ema_model = deepcopy(model).to(device)
    ema_model.requires_grad_(False)
    ema_model.eval()
    model.requires_grad_(True) # train stage2 model
    ddp_model = DDP(model, device_ids=[device.index], broadcast_buffers=False, find_unused_parameters=False)
    # ddp_model = torch.compile(ddp_model) # fix shape compile, see if it works
    model = ddp_model.module
    ddp_model.train()
    # no need to put RAE into DDP since it's frozen
    model_param_count = sum(p.numel() for p in model.parameters())
    logger.info(f"Model Parameters: {model_param_count/1e6:.2f}M")
    
    #### Opt, Schedl init
    optimizer, optim_msg = build_optimizer([p for p in model.parameters() if p.requires_grad], training_cfg)

    ### AMP init
    scaler, autocast_kwargs = get_autocast_scaler(args)
    
    if rank == 0:
        csv_path = os.path.join(experiment_dir, "depth_metrics.csv")
        metric_csv_logger = MetricLogger(csv_path)
    
    ### Data init
    # stage2_transform = transforms.Compose([
    #     transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.ToTensor(),
    # ])
    
    train_loader, train_sampler = load_dataloader(
        data_cfg, micro_batch_size, num_workers, rank, world_size, mode='train', max_views=args.max_views, kernel_size=args.kernel_size
    )

    # val_loader, val_sampler = load_dataloader(
    #     data_cfg, micro_batch_size, num_workers, rank, world_size, mode='val', max_views=args.max_views, kernel_size=args.kernel_size
    # )

    # loader, sampler = prepare_latent_dataloader(
    #     args.clean_img_path, args.deg_img_path, args.gt_depth_path, micro_batch_size, num_workers, rank, world_size
    # ) # loader: ([B, 3, 392, 518], [B, 3, 392, 518], [B, 1, 392, 518], [B,])
    # breakpoint()

    # if do_eval:
    #     eval_dataset = ImageFolder(
    #         str(eval_data),
    #         transform=transforms.Compose([
    #             transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
    #             transforms.ToTensor(),
    #         ])
    #     )
    #     logger.info(f"Evaluation dataset loaded from {eval_data}, containing {len(eval_dataset)} images.")
        
    loader_batches = len(train_loader) # 469 (15008 / 32)
    if loader_batches % grad_accum_steps != 0:
        raise ValueError("Number of loader batches must be divisible by grad_accum_steps when drop_last=True.")
    steps_per_epoch = loader_batches // grad_accum_steps # 469
    if steps_per_epoch <= 0:
        raise ValueError("Gradient accumulation configuration results in zero optimizer steps per epoch.")
    
    if training_cfg.get("scheduler"):
        scheduler, sched_msg = build_scheduler(optimizer, steps_per_epoch, training_cfg) # LambdaLR
     
    if len(data_cfg["train"]["dataset"]) == 1:
        sample_img = train_loader.dataset[0]["deg_img"].to(device) # [V, 3, H, W]
        vggt_patch_size = model_config["params"].get("vggt_patch_size", 14)
        pH, pW = sample_img.shape[2] // vggt_patch_size, sample_img.shape[3] // vggt_patch_size
        vec_size = 5 + pH * pW # 5 special tokens + num_patches
        latent_size = [vec_size, 1024]
        shift_dim = latent_size[0] * latent_size[1]
        shift_base = misc.get("time_dist_shift_base", 4096)
        time_dist_shift = math.sqrt(shift_dim / shift_base)

        # zs = torch.randn(micro_batch_size, *latent_size, device=device, dtype=torch.float32) # [B, 1041, 1024]
    else:
        base_res = 518
        vggt_patch_size = model_config["params"].get("vggt_patch_size", 14)
        pH, pW = base_res // vggt_patch_size, base_res // vggt_patch_size
        vec_size = 5 + pH * pW # 5 special tokens + num_patches
        latent_dim = 1024
        shift_dim = vec_size * latent_dim
        shift_base = misc.get("time_dist_shift_base", 4096)
        time_dist_shift = math.sqrt(shift_dim / shift_base)
        
    # breakpoint()
    n = micro_batch_size 
    #### Transport init
    transport = create_transport(
        **transport_params,
        time_dist_shift=time_dist_shift,
    )
    transport_sampler = Sampler(transport)

    if sampler_mode == "ODE":
        eval_sampler = transport_sampler.sample_ode(**sampler_params)
    elif sampler_mode == "SDE":
        eval_sampler = transport_sampler.sample_sde(**sampler_params)
    else:
        raise NotImplementedError(f"Invalid sampling mode {sampler_mode}.")
    
    
    ### Guidance Init
    guid_model_forward = None
    if guidance_scale > 1.0 and guidance_method == "autoguidance":
        guidance_model_cfg = guidance_cfg.get("guidance_model")
        if guidance_model_cfg is None:
            raise ValueError("Please provide a guidance model config when using autoguidance.")
        guid_model: Stage2ModelProtocol = instantiate_from_config(guidance_model_cfg).to(device)
        guid_model.eval()
        guid_model_forward = guid_model.forward
            
    log_steps = 0
    running_loss = 0.0
    use_guidance = guidance_scale > 1.0

    sample_model_kwargs = dict()
    ema_model_fn = ema_model.forward
    model_fn = model.forward

    ### Resuming and checkpointing
    start_epoch = 0
    global_step = 0
    # maybe_resume_ckpt_path = find_resume_checkpoint(experiment_dir)
    maybe_resume_ckpt_path = full_cfg.stage_2.ckpt
    if maybe_resume_ckpt_path is not None:
        logger.info(f"Experiment resume checkpoint found at {maybe_resume_ckpt_path}, automatically resuming...")
        ckpt_path = Path(maybe_resume_ckpt_path)
        if ckpt_path.is_file():
            start_epoch, global_step = load_checkpoint(
                ckpt_path,
                ddp_model,
                ema_model,
                optimizer,
                scheduler,
            )
            logger.info(f"[Rank {rank}] Resumed from {ckpt_path} (epoch={start_epoch}, step={global_step}).")
        else:
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    elif args.ckpt is not None:
        logger.info(f"Resuming from specified checkpoint {args.ckpt}...")
        checkpoint = torch.load(args.ckpt, map_location="cpu")
        ddp_model.module.load_state_dict(checkpoint["model"])
        ema_model.load_state_dict(checkpoint["ema"])
        logger.info(f"[Rank {rank}] Resumed model weights from {args.ckpt}.")
    else:
        # starting from fresh, save worktree and configs
        if rank == 0:
            save_worktree(experiment_dir, full_cfg)
            logger.info(f"Saved training worktree and config to {experiment_dir}.")
    ### Logging experiment details
    if rank == 0:
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"DDT Model parameters: {num_params/1e6:.2f}M")
        if clip_grad is not None:
            logger.info(f"Clipping gradients to max norm {clip_grad}.")
        else:
            logger.info("Not clipping gradients.")
        # print optim and schel
        logger.info(optim_msg)
        print(sched_msg if sched_msg else "No LR scheduler.")
        logger.info(f"Training for {num_epochs} epochs, batch size {micro_batch_size} per GPU.")
        logger.info(f"Dataset contains {len(train_loader.dataset)} samples, {steps_per_epoch} steps per epoch.")
        logger.info(f"Running with world size {world_size}, starting from epoch {start_epoch} to {num_epochs}.")

    dist.barrier() 
   
    # pose_logger = PoseCSVLogger(experiment_dir)
    # val_pose_tracker = PoseMetricTracker()

    for epoch in range(start_epoch, num_epochs): # epoch: 1400
        model.train()
        train_sampler.set_epoch(epoch) # shuffle data order
        epoch_metrics = defaultdict(lambda: torch.zeros(1, device=device))
        num_batches = 0
        accum_counter = 0
        step_loss_accum = 0.0
        optimizer.zero_grad(set_to_none=True)
        
        for step, batch in enumerate(train_loader):
            # print(f"View number in batch: {batch['deg_img'].shape[1]}")
            clean_img = batch["clean_img"] # [B, V, 3, 392, 518]
            deg_img = batch["deg_img"]     # [B, V, 3, 392, 518]
            gt_depth = batch["gt_depth"]   # [B, V, 1, 392, 518]
            batch_size = clean_img.shape[0]
            ### visualization and save clean and deg img examples
            if step == 0 and epoch % 10 == 0 and rank == 0:
                save_image(clean_img[0], f"{experiment_dir}/clean_input_example.png", normalize=True)
                save_image(deg_img[0], f"{experiment_dir}/deg_input_example.png", normalize=True)
            # z: clean latent path
            # z = z.to(device) # [B, S, P, 2C] = [B, 1, 1041, 2048]
            # z = z[:, :, 5:, :] # remove special tokens -> [B, 1, 1036, 2048]
            # z = rearrange(z, 'b s (pH pW) d -> b s d pH pW', pH=pH, pW=pW) # [B, 1, 2048, 28, 37]  
            with torch.no_grad():
                # try:
                #     deg_img = load_and_preprocess_images(deg_img_path).to(device).to(dtype) # [B, 3, 392, 518]
                #     if step == 0:
                #         save_image(deg_img[0], f"{experiment_dir}/deg_input_example.png", normalize=True)
                # except Exception as e:
                #     print(f"\nError loading degraded images at step {step} of epoch {epoch}: {e}")
                #     continue
                with torch.cuda.amp.autocast(dtype=dtype):
                    clean_predictions = vggt_model(clean_img.to(device), extract_layer_num=extract_layer) # [B, 1, 1041, 2048]
                    clean_img_latent = clean_predictions["extracted_latent"] # [B, 1, 1041, 2048]
                    clean_latent_depths = clean_predictions['depth'] # [B, 1, 392, 518, 1]

                    # frame + global latent 
                    # imagenet normalization to deg_img
                    predictions = vggt_model(deg_img.to(device), extract_layer_num=extract_layer) # [B, 1, 1041, 2048]
                    train_depths = predictions['depth'] # [B, 1, 392, 518, 1]
                    deg_latent = predictions["extracted_latent"] # [B, 1, 1041, 2048]
                assert clean_img_latent.shape == deg_latent.shape, f"Latent shape mismatch: {clean_img_latent.shape} vs {deg_latent.shape}" # [B, 1, 1041, 2048]

                # only use global latent, remove special tokens
                # clean_img_latent = clean_img_latent[:, :, 5:, :] # [B, 1, 1036, 2048]
                # deg_latent = deg_latent[:, :, 5:, 1024:] # [B, 1, 1036, 1024]
                # clean_img_latent = rearrange(clean_img_latent, 'b s (pH pW) d -> (b s) d pH pW', pH=pH, pW=pW) # [B, 2048, 28, 37]
                # deg_latent_patch = rearrange(deg_latent, 'b s (pH pW) d -> (b s) d pH pW', pH=pH, pW=pW) # [B, 1024, 28, 37]
                clean_img_latent = clean_img_latent[:, :, :, 1024:] # [B, 1, 1041, 2048]
                deg_latent = deg_latent[:, :, :, 1024:] # [B, 1, 1041, 1024]

                if step == 0:
                    gt_depth_sample = gt_depth[0, 0, 0]
                    deg_depth_sample = train_depths[0, 0].squeeze(-1)
                    deg_depth_metric = compute_depth_metrics(gt_depth_sample, deg_depth_sample)
                    print(f"Deg input depth metrics at epoch {epoch}, step {step}: {deg_depth_metric}")

            model_kwargs = {"deg_latent": deg_latent}
            # model_kwargs = {}

            with autocast(**autocast_kwargs):
                # terms = transport.training_losses(ddp_model, clean_img_latent, model_kwargs)
                terms = transport.training_losses_sequence(ddp_model, clean_img, clean_img_latent, step, experiment_dir, model_kwargs)
                loss = terms["loss"].mean()

            # if scaler:
            #     scaler.scale(loss / grad_accum_steps).backward()
            # else:
            #     (loss / grad_accum_steps).backward()
            if scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
            else:
                loss.backward()
                # total_norm = 0
                # for p in model.parameters():
                #     if p.grad is not None:
                #         param_norm = p.grad.data.norm(2)
                #         total_norm += param_norm.item() ** 2
                # total_norm = total_norm ** 0.5 
                # print(f"Gradient Norm: {total_norm}")
            if clip_grad:
                # if scaler:
                #     scaler.unscale_(optimizer) 
                # torch.nn.utils.clip_grad_norm_(ddp_model.parameters(), clip_grad)
                total_norm = torch.nn.utils.clip_grad_norm_(ddp_model.parameters(), clip_grad)
            else:
                total_norm = 0.0
                for p in model.parameters():
                    if p.grad is not None:
                        total_norm += p.grad.data.norm(2).item() ** 2
                total_norm = total_norm ** 0.5

            if torch.isnan(loss) or torch.isinf(loss):
                if rank == 0:
                    logger.error(f"NaN Loss detected at Step {global_step}!")
                    logger.info(f"clean_img_latent stats: mean={clean_img_latent.mean():.4f}, max={clean_img_latent.max():.4f}, min={clean_img_latent.min():.4f}")
                    logger.info(f"deg_latent stats: mean={deg_latent.mean():.4f}, max={deg_latent.max():.4f}")
                    logger.info(f"Grad Norm: {total_norm:.4f}")

            if scaler:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            optimizer.zero_grad(set_to_none=True)

            if scheduler is not None:
                scheduler.step()
            update_ema(ema_model, ddp_model.module, decay=ema_decay)
            
            # if global_step % grad_accum_steps == 0:
            #     if scaler:
            #         scaler.step(optimizer)
            #         scaler.update()
            #     else:
            #         optimizer.step()

            #     optimizer.zero_grad(set_to_none=True)

            #     if scheduler is not None:
            #         scheduler.step()
            #     update_ema(ema_model, ddp_model.module, decay=ema_decay)

            running_loss += loss.item()
            epoch_metrics['loss'] += loss.detach()

            if checkpoint_interval > 0 and global_step > 0 and global_step % checkpoint_interval == 0 and rank == 0:
                logger.info(f"Saving checkpoint at epoch {epoch}...")
                ckpt_path = f"{checkpoint_dir}/global_step_{global_step:07d}.pt" 
                save_checkpoint(
                    ckpt_path,
                    global_step,
                    epoch,
                    ddp_model,
                    ema_model,
                    optimizer,
                    scheduler,
                )
            
            if log_interval > 0 and global_step % log_interval == 0 and rank == 0:
                avg_loss = running_loss / log_interval # flow loss often has large variance so we record avg loss
                steps = torch.tensor(log_interval, device=device)
                stats = {
                    "train/loss": avg_loss,
                    "train/lr": optimizer.param_groups[0]["lr"],
                    "train/grad_norm": total_norm,
                }
                logger.info(
                    f"[Epoch {epoch} | Step {global_step}] "
                    + ", ".join(f"{k}: {v:.6f}" for k, v in stats.items())
                )
                if args.wandb:
                    wandb_utils.log(
                        stats,
                        step=global_step,
                    )
                running_loss = 0.0

            if global_step % sample_every == 0 and rank == 0:
                logger.info("Starting Sampling...")
                co3d_args = data_cfg["val"]["library"]["co3d"]
                save_dir = f"{experiment_dir}/visualization/step_{global_step:07d}"
                os.makedirs(save_dir, exist_ok=True)
                
                log_dir = os.path.join(experiment_dir, "co3d_eval")
                os.makedirs(log_dir, exist_ok=True)
                logger = setup_logger(log_dir)
                evaluate_co3d(model, eval_sampler, logger=logger, co3d_dir=co3d_args["clean_img_path"], co3d_anno_dir=co3d_args["annotation_path"], save_dir=save_dir)

                logger.info("Resuming training...")
                model.train()

            global_step += 1
            num_batches += 1
        # breakpoint()
        if rank == 0 and num_batches > 0:
            avg_loss = epoch_metrics['loss'].item() / num_batches 
            epoch_stats = {
                "epoch/loss": avg_loss,
            }
            logger.info(
                f"[Epoch {epoch}] "
                + ", ".join(f"{k}: {v:.4f}" for k, v in epoch_stats.items())
            )
            if args.wandb:
                wandb_utils.log(epoch_stats, step=global_step)

    # save the final ckpt
    if rank == 0:
        logger.info(f"Saving final checkpoint at epoch {num_epochs}...")
        ckpt_path = f"{checkpoint_dir}/ep-last.pt" 
        save_checkpoint(
            ckpt_path,
            global_step,
            num_epochs,
            ddp_model,
            ema_model,
            optimizer,
            scheduler,
        )
    dist.barrier()
    logger.info("Done!")
    cleanup_distributed()



if __name__ == "__main__":
    main()
