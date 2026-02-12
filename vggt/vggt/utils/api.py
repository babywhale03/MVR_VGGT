import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from typing import Optional, Sequence, List, Dict, Any
import sys
from addict import Dict as AddictDict  # 추가 필요

from vggt.models.vggt import VGGT
from vggt.utils.pose_enc import pose_encoding_to_extri_intri

da3_path = "/mnt/dataset1/jaeeun/MVR/Depth-Anything-3"
if da3_path not in sys.path:
    sys.path.append(da3_path)
    
from depth_anything_3.specs import Prediction
from depth_anything_3.utils.io.input_processor import InputProcessor
from depth_anything_3.utils.io.output_processor import OutputProcessor

class VGGTInference(nn.Module):
    def __init__(self, preset="vitl", **kwargs):
        super().__init__()
        self.model = VGGT(
            img_size=518, 
            patch_size=14, 
            embed_dim=1024, 
            enable_camera=True, 
            enable_point=True, 
            enable_depth=True, 
            enable_track=True
        )
        self.model.eval()

        self.input_processor = InputProcessor()
        self.output_processor = OutputProcessor()
        self.device = None

    @torch.inference_mode()
    def forward(self, image: torch.Tensor, query_points: torch.Tensor | None = None, extract_layer_num: int = 3, **kwargs) -> dict:
        model_kwargs = {
            k: v for k, v in kwargs.items() 
            if k not in ['export_dir', 'export_format', 'process_res', 'process_res_method', 'ref_view_strategy']
        }
        autocast_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        with torch.autocast(device_type=image.device.type, dtype=autocast_dtype):
            return self.model(
                images=image, 
                query_points=query_points,
                **model_kwargs
            )

    def inference(self, image: list, query_points: np.ndarray = None, process_res: int = 518, 
                  process_res_method: str = "upper_bound_resize", **kwargs) -> Prediction:
        imgs_cpu, _, _ = self.input_processor(image, None, None, process_res, process_res_method)
        
        device = self._get_model_device()
        imgs = imgs_cpu.to(device)[None].float()
        q_pts = torch.from_numpy(query_points).to(device).float() if query_points is not None else None

        raw_output = self.forward(imgs, query_points=q_pts, **kwargs)

        prediction = self._convert_vggt_to_prediction(raw_output, process_res_hw=imgs_cpu.shape[2:4])

        prediction.processed_images = self._denormalize_images(imgs_cpu)

        return prediction

    def _convert_vggt_to_prediction(self, raw_output: dict, process_res_hw: tuple) -> Prediction:
        processed_output = {}
        B, S = raw_output["depth"].shape[:2]

        if "depth" in raw_output:
            depth = raw_output["depth"]
            if depth.dim() == 5 and depth.shape[-1] == 1:
                depth = depth.permute(0, 1, 4, 2, 3) # [B, S, 1, H, W]
            processed_output["depth"] = depth

        if "depth_conf" in raw_output:
            processed_output["depth_conf"] = raw_output["depth_conf"]

        if "pose_enc" in raw_output:
            extri, intri = pose_encoding_to_extri_intri(
                raw_output["pose_enc"],
                image_size_hw=process_res_hw,
                pose_encoding_type="absT_quaR_FoV",
                build_intrinsics=True
            )
            pad = torch.tensor([0, 0, 0, 1], device=extri.device).view(1, 1, 1, 4).expand(B, S, 1, 4)
            extri_4x4 = torch.cat([extri, pad], dim=2)
            
            processed_output["extrinsics"] = extri_4x4
            processed_output["intrinsics"] = intri

        prediction = self.output_processor(processed_output)

        aux_data = AddictDict()
        for key in ["world_points", "track", "vis", "conf"]:
            if key in raw_output:
                aux_data[key] = raw_output[key].squeeze(0).float().cpu().numpy()

        prediction.aux = aux_data
        prediction.is_metric = raw_output.get("is_metric", False)

        return prediction

    def _denormalize_images(self, imgs_cpu: torch.Tensor) -> np.ndarray:
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = imgs_cpu.permute(0, 2, 3, 1).numpy()
        img = (img * std + mean).clip(0, 1)
        return (img * 255).astype(np.uint8)

    def _get_model_device(self):
        if self.device is None:
            self.device = next(self.model.parameters()).device
        return self.device