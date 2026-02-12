# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Output processor for Depth Anything 3.

This module handles model output processing, including tensor-to-numpy conversion,
batch dimension removal, and Prediction object creation.
"""

from __future__ import annotations

import sys
import numpy as np
import torch
from addict import Dict as AddictDict

from depth_anything_3.specs import Prediction

vggt_path = "/mnt/dataset1/jaeeun/test/vggt"
if vggt_path in sys.path:
    sys.path.remove(vggt_path) 
sys.path.insert(0, vggt_path)
from vggt.utils.pose_enc import pose_encoding_to_extri_intri


class OutputProcessor:
    """
    Output processor for converting model outputs to Prediction objects.

    Handles tensor-to-numpy conversion, batch dimension removal,
    and creates structured Prediction objects with proper data types.
    """

    def __init__(self) -> None:
        """Initialize the output processor."""

    def __call__(self, model_output: dict[str, torch.Tensor], imgs: torch.Tensor) -> Prediction:
        """
        Convert model output to Prediction object.

        Args:
            model_output: Model output dictionary containing depth, conf, extrinsics, intrinsics
                         Expected shapes: depth (B, N, 1, H, W), conf (B, N, 1, H, W),
                         extrinsics (B, N, 4, 4), intrinsics (B, N, 3, 3)

        Returns:
            Prediction: Object containing depth estimation results with shapes:
                       depth (N, H, W), conf (N, H, W), extrinsics (N, 4, 4), intrinsics (N, 3, 3)
        """
        # Extract data from batch dimension (B=1, N=number of images)
        depth = self._extract_depth(model_output) # [38, 336, 504]
        conf = self._extract_conf(model_output) # [38, 336, 504] 
        extrinsics, intrinsics = self._extract_pose(model_output, imgs.shape[-2:])
        # extrinsics = self._extract_extrinsics(model_output)
        # intrinsics = self._extract_intrinsics(model_output)
        sky = self._extract_sky(model_output)
        aux = self._extract_aux(model_output)
        gaussians = model_output.get("gaussians", None)
        scale_factor = model_output.get("scale_factor", None)

        return Prediction(
            depth=depth,
            sky=sky,
            conf=conf,
            extrinsics=extrinsics,
            intrinsics=intrinsics,
            is_metric=getattr(model_output, "is_metric", 0),
            gaussians=gaussians,
            aux=aux,
            scale_factor=scale_factor,
        )

    def _extract_depth(self, model_output: dict[str, torch.Tensor]) -> np.ndarray:
        """
        Extract depth tensor from model output and convert to numpy.

        Args:
            model_output: Model output dictionary

        Returns:
            Depth array with shape (N, H, W)
        """
        # vggt depth: [B, N, H, W, 1]
        depth = model_output["depth"].squeeze(0).squeeze(-1).float().cpu().numpy()  # (N, H, W)
        return depth

    def _extract_conf(self, model_output: dict[str, torch.Tensor]) -> np.ndarray | None:
        """
        Extract confidence tensor from model output and convert to numpy.

        Args:
            model_output: Model output dictionary

        Returns:
            Confidence array with shape (N, H, W) or None
        """
        # vggt conf: [B, N, H, W]
        conf = model_output.get("depth_conf", None)
        if conf is not None:
            conf = conf.squeeze(0).float().cpu().numpy()  # (N, H, W)
        return conf
    
    def _extract_pose(self, model_output: dict[str, torch.Tensor], process_res_hw: tuple[int, int]) -> tuple[np.ndarray | None, np.ndarray | None]:
        """
        Extract extrinsics and intrinsics tensors from model output and convert to numpy.

        Args:
            model_output: Model output dictionary
        Returns:
            Tuple of extrinsics array with shape (N, 4, 4) or None
            and intrinsics array with shape (N, 3, 3) or None
        """
        print(f"Extracting pose with process_res_hw: {process_res_hw}")
        extrinsics, intrinsics = pose_encoding_to_extri_intri(
            model_output["pose_enc"],
            image_size_hw=process_res_hw,
            pose_encoding_type="absT_quaR_FoV",
            build_intrinsics=True
        )
        if extrinsics is not None:
            extrinsics = extrinsics.squeeze(0).float().cpu().numpy()  # (N, 3, 4)
        if intrinsics is not None:
            intrinsics = intrinsics.squeeze(0).float().cpu().numpy()  # (N, 3, 3)
        return extrinsics, intrinsics

    def _extract_sky(self, model_output: dict[str, torch.Tensor]) -> np.ndarray | None:
        """
        Extract sky tensor from model output and convert to numpy.

        Args:
            model_output: Model output dictionary

        Returns:
            Sky mask array with shape (N, H, W) or None
        """
        sky = model_output.get("sky", None)
        if sky is not None:
            sky = sky.squeeze(0).float().cpu().numpy() >= 0.5  # (N, H, W)
        return sky

    def _extract_aux(self, model_output: dict[str, torch.Tensor]) -> AddictDict:
        """
        Extract auxiliary data from model output and convert to numpy.

        Args:
            model_output: Model output dictionary

        Returns:
            Dictionary containing auxiliary data
        """
        aux = model_output.get("aux", None)
        ret = AddictDict()
        if aux is not None:
            for k in aux.keys():
                if isinstance(aux[k], torch.Tensor):
                    ret[k] = aux[k].squeeze(0).float().cpu().numpy()
                else:
                    ret[k] = aux[k]
        return ret


# Backward compatibility alias
OutputAdapter = OutputProcessor
