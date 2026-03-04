import os
import time
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from typing import Optional, Sequence, List, Dict, Any
import sys
from addict import Dict as AddictDict
from torch.cuda.amp import autocast, GradScaler
from torchvision.utils import save_image

from RAE.src.utils.eval_vis_utils import *
from vggt.vggt.models.vggt import VGGT
    
from depth_anything_3.cfg import create_object, load_config
from depth_anything_3.registry import MODEL_REGISTRY
from depth_anything_3.specs import Prediction
from depth_anything_3.utils.export import export
from depth_anything_3.utils.geometry import affine_inverse
from depth_anything_3.utils.io.input_processor_vggt import InputProcessor
from depth_anything_3.utils.io.output_processor_vggt import OutputProcessor
from depth_anything_3.utils.logger import logger
from depth_anything_3.utils.pose_align import align_poses_umeyama

class VGGTInference(nn.Module):
    def __init__(self, model=None,preset="vitl", **kwargs):
        super().__init__()
        if model is not None:
            self.model = model 
        else:
            vggt_core = VGGT(
                img_size=518, 
                patch_size=14, 
                embed_dim=1024, 
                enable_camera=True, 
                enable_point=True, 
                enable_depth=True, 
                enable_track=False
            )
            self.model = vggt_core
        self.model.eval()

        self.input_processor = InputProcessor()
        self.output_processor = OutputProcessor()
        self.device = None

    @torch.inference_mode()
    def forward(self, image: torch.Tensor, extract_layer_num: int = 0, change_latent: bool = False, query_points: torch.Tensor | None = None, **kwargs) -> dict:
        model_kwargs = {
            k: v for k, v in kwargs.items() 
            if k not in ['export_dir', 'export_format', 'process_res', 'process_res_method', 'ref_view_strategy']
        }
        autocast_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        with torch.autocast(device_type=image.device.type, dtype=autocast_dtype):
            return self.model(
                images=image, 
                extract_layer_num=extract_layer_num,
                change_latent=change_latent,
                query_points=query_points,
                **model_kwargs
            )

    def inference(
        self,
        image: list[np.ndarray | Image.Image | str],
        extrinsics: np.ndarray | None = None,
        intrinsics: np.ndarray | None = None,
        align_to_input_ext_scale: bool = True,
        infer_gs: bool = False,
        use_ray_pose: bool = False,
        ref_view_strategy: str = "saddle_balanced",
        render_exts: np.ndarray | None = None,
        render_ixts: np.ndarray | None = None,
        render_hw: tuple[int, int] | None = None,
        process_res: int = 518,
        process_res_method: str = "upper_bound_resize",
        export_dir: str | None = None,
        export_format: str = "mini_npz",
        export_feat_layers: Sequence[int] | None = None,
        # GLB export parameters
        conf_thresh_percentile: float = 40.0,
        num_max_points: int = 1_000_000,
        show_cameras: bool = True,
        # Feat_vis export parameters
        feat_vis_fps: int = 15,
        # Other export parameters, e.g., gs_ply, gs_video
        export_kwargs: Optional[dict] = {},
        stage2_model=None,
        eval_sampler=None,
        generator=None,
        config=None,
        work_dir=None,
        scene_info=None,
        use_pose=None
    ) -> Prediction:
        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        autocast_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        autocast_kwargs = dict(dtype=autocast_dtype)

        data, scene = scene_info 
        pose_setting = 'pose' if use_pose else 'unposed'
        
        if "gs" in export_format:
            assert infer_gs, "must set `infer_gs=True` to perform gs-related export."

        if "colmap" in export_format:
            assert isinstance(image[0], str), "`image` must be image paths for COLMAP export."

        if 'lq_image_files' in image.keys():
            lq_imgs_cpu, _, _ = self._preprocess_inputs(
                image.lq_image_files, None, None, process_res, process_res_method
            )
            lq_imgs, _, _ = self._prepare_model_inputs(lq_imgs_cpu, None, None)

        if 'res_image_files' in image.keys():
            res_imgs_cpu, _, _ = self._preprocess_inputs(
                image.res_image_files, None, None, process_res, process_res_method
            )
            res_imgs, _, _ = self._prepare_model_inputs(res_imgs_cpu, None, None)

        # Preprocess images
        print("Preprocessing Inputs...")
        imgs_cpu, extrinsics, intrinsics = self._preprocess_inputs(
            image.image_files, extrinsics, intrinsics, process_res, process_res_method
        )

        # Prepare tensors for model
        print("Preparing Model Inputs...")
        imgs, ex_t, in_t = self._prepare_model_inputs(imgs_cpu, extrinsics, intrinsics)

        # Normalize extrinsics
        print("Normalizing Extrinsics...")
        ex_t_norm = self._normalize_extrinsics(ex_t.clone() if ex_t is not None else None)
        
        vis_depth_dir, vis_depth_metric, vis_pose_dir, vis_feature_sim, vis_pose_metric = os.path.join(work_dir, "depth_vis", data, pose_setting), os.path.join(work_dir, "depth_metrics", data, pose_setting), os.path.join(work_dir, "pose_vis", data, pose_setting), os.path.join(work_dir, "feature_sim", data, pose_setting), os.path.join(work_dir, "pose_metric", data, pose_setting)
        os.makedirs(vis_depth_dir, exist_ok=True); os.makedirs(vis_depth_metric, exist_ok=True); os.makedirs(vis_pose_dir, exist_ok=True); os.makedirs(vis_feature_sim, exist_ok=True); os.makedirs(vis_pose_metric, exist_ok=True)

        if config["stage_2"]["model"] == "VGGTMVRM":
            sample_model_kwargs = dict()
            
            with torch.no_grad():
                with torch.cuda.amp.autocast(dtype=dtype):
                    print("Using VGGTMVRM inference...")
                    breakpoint()
                    clean_predictions = self._run_model_forward(imgs)
                    clean_tokens = clean_predictions["aggregated_tokens_list"]
                    clean_depth = clean_predictions["depth"]
                    val_lq_predictions = self._run_model_forward(lq_imgs, extract_layer_num=config["stage_2"]["extract_layer"])
                    val_lq_deg_latent = val_lq_predictions["extracted_latent"][:, :, :, 1024:]
                    val_lq_tokens = val_lq_predictions["aggregated_tokens_list"]
                    lq_depth = val_lq_predictions["depth"]

                    generator.manual_seed(42)
                    zs = torch.randn(*val_lq_deg_latent.shape, generator=generator, device=imgs.device)
                    val_xt = zs + val_lq_deg_latent
                    sample_model_kwargs["img"] = imgs.to(val_lq_deg_latent.device)

                with autocast(**autocast_kwargs):
                    torch.cuda.empty_cache()
                    restored_latent = eval_sampler(val_xt, stage2_model.forward, **sample_model_kwargs).float()
                    # restored_latent[:, :, :5, :] = val_lq_deg_latent[:, :, :5, :].float()

                vggt_result = {}
                vggt_result['restored_latent'] = restored_latent     

                raw_output = self._run_model_forward(lq_imgs, extract_layer_num=config["stage_2"]["extract_layer"], vggt_result=vggt_result, change_latent=True)
                raw_tokens = raw_output["aggregated_tokens_list"]
                raw_depth = raw_output["depth"]

                visualize_vggt_stepwise_analysis(clean_tokens, val_lq_tokens, raw_tokens, save_path=os.path.join(vis_feature_sim, f"{scene}.png"))
                depth_vis_path = os.path.join(vis_depth_dir, f"{scene}.png")
                os.makedirs(os.path.dirname(depth_vis_path), exist_ok=True)
                save_image(vis_depth_all(imgs[0], clean_depth[0], lq_imgs[0], lq_depth[0], res_depth=raw_depth[0]), depth_vis_path, normalize=False)
                save_depth_metrics(clean_depth, lq_depth, res_depth=raw_depth, save_path=os.path.join(vis_depth_metric, f"{scene}.txt"))
                save_pose_metrics(
                    imgs[0], clean_predictions["pose_enc"], 
                    lq_imgs[0], val_lq_predictions["pose_enc"], 
                    res_pose=raw_output["pose_enc"], 
                    metric_save_path=os.path.join(vis_pose_metric, f"{scene}.txt"),
                    plot_save_path=os.path.join(vis_pose_dir, f"{scene}.png")
                )  

        elif config["stage_2"]["model"] == "VGGT":
            with torch.no_grad():
                with torch.cuda.amp.autocast(dtype=dtype):
                    print("Using VGGT inference...")
                    clean_predictions = self._run_model_forward(imgs)
                    clean_tokens = clean_predictions["aggregated_tokens_list"]
                    clean_depth = clean_predictions["depth"]
                    lq_predictions = self._run_model_forward(lq_imgs)
                    lq_tokens = lq_predictions["aggregated_tokens_list"]
                    lq_depth = lq_predictions["depth"]
                    raw_output = self._run_model_forward(res_imgs)
                    raw_tokens = raw_output["aggregated_tokens_list"]
                    raw_depth = raw_output["depth"]

                    visualize_vggt_stepwise_analysis(clean_tokens, lq_tokens, raw_tokens, save_path=os.path.join(vis_feature_sim, f"{scene}.png"))
                    depth_vis_path = os.path.join(vis_depth_dir, f"{scene}.png")
                    os.makedirs(os.path.dirname(depth_vis_path), exist_ok=True)
                    save_image(vis_depth_all(imgs[0], clean_depth[0], lq_imgs[0], lq_depth[0], res_imgs[0], raw_depth[0]), depth_vis_path, normalize=False)
                    save_depth_metrics(clean_depth, lq_depth, res_depth=raw_depth, save_path=os.path.join(vis_depth_metric, f"{scene}.txt"))
                    save_pose_metrics(
                        imgs[0], clean_predictions["pose_enc"], 
                        lq_imgs[0], lq_predictions["pose_enc"], 
                        res_img=res_imgs[0], res_pose=raw_output["pose_enc"], 
                        metric_save_path=os.path.join(vis_pose_metric, f"{scene}.txt"),
                        plot_save_path=os.path.join(vis_pose_dir, f"{scene}.png")
                    )

        elif config["stage_2"]["model"] == "VGGT_HQ":
            with torch.no_grad():
                with torch.cuda.amp.autocast(dtype=dtype):
                    print("Using VGGT_HQ inference...")
                    clean_predictions = self._run_model_forward(imgs)
                    clean_tokens = clean_predictions["aggregated_tokens_list"]
                    clean_depth = clean_predictions["depth"]

                    raw_output = clean_predictions

                    depth_vis_path = os.path.join(vis_depth_dir, f"{scene}.png")
                    os.makedirs(os.path.dirname(depth_vis_path), exist_ok=True)
                    save_image(vis_depth_hq(imgs[0], clean_depth[0]), depth_vis_path, normalize=False)
        else:
            with torch.no_grad():
                with torch.cuda.amp.autocast(dtype=dtype):
                    print("Using VGGT_LQ inference...")
                    clean_predictions = self._run_model_forward(imgs)
                    clean_tokens = clean_predictions["aggregated_tokens_list"]
                    clean_depth = clean_predictions["depth"]
                    lq_predictions = self._run_model_forward(lq_imgs)
                    lq_tokens = lq_predictions["aggregated_tokens_list"]
                    lq_depth = lq_predictions["depth"]

                    raw_output = lq_predictions

                    visualize_vggt_stepwise_analysis(clean_tokens, lq_tokens, save_path=os.path.join(vis_feature_sim, f"{scene}.png"))
                    depth_vis_path = os.path.join(vis_depth_dir, f"{scene}.png")
                    os.makedirs(os.path.dirname(depth_vis_path), exist_ok=True)
                    save_image(vis_depth_all(imgs[0], clean_depth[0], lq_imgs[0], lq_depth[0]), depth_vis_path, normalize=False)
                    save_depth_metrics(clean_depth, lq_depth, save_path=os.path.join(vis_depth_metric, f"{scene}.txt"))
                    save_pose_metrics(
                        imgs[0], clean_predictions["pose_enc"], 
                        lq_imgs[0], lq_predictions["pose_enc"],
                        metric_save_path=os.path.join(vis_pose_metric, f"{scene}.txt"),
                        plot_save_path=os.path.join(vis_pose_dir, f"{scene}.png")
                    )

        # Convert raw output to prediction
        print("Converting to Prediction...")
        prediction = self._convert_to_prediction(raw_output, imgs)

        # Align prediction to extrinsincs
        print("Aligning to Input Extrinsics/Intrinsics...")
        prediction = self._align_to_input_extrinsics_intrinsics(
            extrinsics, intrinsics, prediction, align_to_input_ext_scale
        )

        # Add processed images for visualization
        print("Adding Processed Images...")
        prediction = self._add_processed_images(prediction, imgs_cpu)

        # Export if requested
        if export_dir is not None:
            if "gs" in export_format:
                if infer_gs and "gs_video" not in export_format:
                    export_format = f"{export_format}-gs_video"
                if "gs_video" in export_format:
                    if "gs_video" not in export_kwargs:
                        export_kwargs["gs_video"] = {}
                    export_kwargs["gs_video"].update(
                        {
                            "extrinsics": render_exts,
                            "intrinsics": render_ixts,
                            "out_image_hw": render_hw,
                        }
                    )
            # Add GLB export parameters
            if "glb" in export_format:
                if "glb" not in export_kwargs:
                    export_kwargs["glb"] = {}
                export_kwargs["glb"].update(
                    {
                        "conf_thresh_percentile": conf_thresh_percentile,
                        "num_max_points": num_max_points,
                        "show_cameras": show_cameras,
                    }
                )
            # Add Feat_vis export parameters
            if "feat_vis" in export_format:
                if "feat_vis" not in export_kwargs:
                    export_kwargs["feat_vis"] = {}
                export_kwargs["feat_vis"].update(
                    {
                        "fps": feat_vis_fps,
                    }
                )
            # Add COLMAP export parameters
            if "colmap" in export_format:
                if "colmap" not in export_kwargs:
                    export_kwargs["colmap"] = {}
                export_kwargs["colmap"].update(
                    {
                        "image_paths": image,
                        "conf_thresh_percentile": conf_thresh_percentile,
                        "process_res_method": process_res_method,
                    }
                )
            self._export_results(prediction, export_format, export_dir, **export_kwargs)

        return prediction

    def _preprocess_inputs(
        self,
        image: list[np.ndarray | Image.Image | str],
        extrinsics: np.ndarray | None = None,
        intrinsics: np.ndarray | None = None,
        process_res: int = 504,
        process_res_method: str = "upper_bound_resize",
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """Preprocess input images using input processor."""
        start_time = time.time()
        # imgs_cpu: [38, 3, 336, 504]
        imgs_cpu, extrinsics, intrinsics = self.input_processor(
            image,
            extrinsics.copy() if extrinsics is not None else None,
            intrinsics.copy() if intrinsics is not None else None,
            process_res,
            process_res_method,
        )
        end_time = time.time()
        logger.info(
            "Processed Images Done taking",
            end_time - start_time,
            "seconds. Shape: ",
            imgs_cpu.shape,
        )
        return imgs_cpu, extrinsics, intrinsics
    
    def _prepare_model_inputs(
        self,
        imgs_cpu: torch.Tensor,
        extrinsics: torch.Tensor | None,
        intrinsics: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """Prepare tensors for model input."""
        device = self._get_model_device()

        # Move images to model device
        imgs = imgs_cpu.to(device, non_blocking=True)[None].float()

        # Convert camera parameters to tensors
        ex_t = (
            extrinsics.to(device, non_blocking=True)[None].float()
            if extrinsics is not None
            else None
        )
        in_t = (
            intrinsics.to(device, non_blocking=True)[None].float()
            if intrinsics is not None
            else None
        )

        return imgs, ex_t, in_t

    def _normalize_extrinsics(self, ex_t: torch.Tensor | None) -> torch.Tensor | None:
        """Normalize extrinsics"""
        if ex_t is None:
            return None
        transform = affine_inverse(ex_t[:, :1])
        ex_t_norm = ex_t @ transform
        c2ws = affine_inverse(ex_t_norm)
        translations = c2ws[..., :3, 3]
        dists = translations.norm(dim=-1)
        median_dist = torch.median(dists)
        median_dist = torch.clamp(median_dist, min=1e-1)
        ex_t_norm[..., :3, 3] = ex_t_norm[..., :3, 3] / median_dist
        return ex_t_norm

    def _run_model_forward(
        self,
        imgs: torch.Tensor,
        extract_layer_num: int = 0,
        change_latent: bool = False,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        """Run model forward pass."""
        device = imgs.device
        need_sync = device.type == "cuda"
        if need_sync:
            torch.cuda.synchronize(device)
        start_time = time.time()
        # depth, depth_conf, extrinsics, intrinsics, aux
        # depth, depth_conf: [1, 38, 336, 504]
        # extrinsics: [1, 38, 3, 4], intrinsics: [1, 38, 3, 3]
        output = self.forward(imgs, extract_layer_num=extract_layer_num, change_latent=change_latent, **kwargs)
        if need_sync:
            torch.cuda.synchronize(device)
        end_time = time.time()
        logger.info(f"Model Forward Pass Done. Time: {end_time - start_time} seconds")
        return output
    
    def _convert_to_prediction(self, raw_output: dict[str, torch.Tensor], imgs: torch.Tensor) -> Prediction:
        """Convert raw model output to Prediction object."""
        start_time = time.time()
        output = self.output_processor(raw_output, imgs)
        end_time = time.time()
        logger.info(f"Conversion to Prediction Done. Time: {end_time - start_time} seconds")
        return output

    def _align_to_input_extrinsics_intrinsics(
        self,
        extrinsics: torch.Tensor | None,
        intrinsics: torch.Tensor | None,
        prediction: Prediction,
        align_to_input_ext_scale: bool = True,
        ransac_view_thresh: int = 10,
    ) -> Prediction:
        """Align depth map to input extrinsics"""
        if extrinsics is None:
            return prediction
        prediction.intrinsics = intrinsics.numpy()
        _, _, scale, aligned_extrinsics = align_poses_umeyama(
            prediction.extrinsics,
            extrinsics.numpy(),
            ransac=len(extrinsics) >= ransac_view_thresh,
            return_aligned=True,
            random_state=42,
        )
        if align_to_input_ext_scale:
            prediction.extrinsics = extrinsics[..., :3, :].numpy()
            prediction.depth /= scale
        else:
            prediction.extrinsics = aligned_extrinsics
        return prediction
    
    def _add_processed_images(self, prediction: Prediction, imgs_cpu: torch.Tensor) -> Prediction:
        """Add processed images to prediction for visualization."""
        # Convert from (N, 3, H, W) to (N, H, W, 3) and denormalize
        processed_imgs = imgs_cpu.permute(0, 2, 3, 1).cpu().numpy()  # (N, H, W, 3)

        processed_imgs = (np.clip(processed_imgs, 0, 1) * 255).astype(np.uint8)

        prediction.processed_images = processed_imgs
        return prediction

        # Denormalize from ImageNet normalization
        # mean = np.array([0.485, 0.456, 0.406])
        # std = np.array([0.229, 0.224, 0.225])
        # processed_imgs = processed_imgs * std + mean
        # processed_imgs = np.clip(processed_imgs, 0, 1)
        # processed_imgs = (processed_imgs * 255).astype(np.uint8)

        # prediction.processed_images = processed_imgs
        # return prediction

    def _export_results(
        self, prediction: Prediction, export_format: str, export_dir: str, **kwargs
    ) -> None:
        """Export results to specified format and directory."""
        start_time = time.time()
        export(prediction, export_format, export_dir, **kwargs)
        end_time = time.time()
        logger.info(f"Export Results Done. Time: {end_time - start_time} seconds")

    def _get_model_device(self) -> torch.device:
        """
        Get the device where the model is located.

        Returns:
            Device where the model parameters are located

        Raises:
            ValueError: If no tensors are found in the model
        """
        if self.device is not None:
            return self.device

        # Find device from parameters
        for param in self.parameters():
            self.device = param.device
            return param.device

        # Find device from buffers
        for buffer in self.buffers():
            self.device = buffer.device
            return buffer.device

        raise ValueError("No tensor found in model")