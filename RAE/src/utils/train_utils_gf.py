import bisect
import csv
from glob import glob
import random
import sys
from turtle import mode
from omegaconf import OmegaConf, DictConfig
from typing import List, Tuple, Union
from PIL import Image
import numpy as np
from collections import OrderedDict
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
from torchvision.datasets import ImageFolder
from pathlib import Path
from copy import deepcopy
from .dist_utils import setup_distributed
import os 
import re
import h5py
from .view_sel_near_camera import compute_ranking
from scipy.spatial.transform import Rotation

from motionblur.motionblur import Kernel 
from vggt.vggt.utils.load_fn import *

def parse_configs(config: Union[DictConfig, str]) -> Tuple[DictConfig, DictConfig, DictConfig, DictConfig, DictConfig, DictConfig, DictConfig]:
    """Load a config file and return component sections as DictConfigs."""
    if isinstance(config, str):
        config = OmegaConf.load(config)
    stage2_config = config.get("stage_2", None)
    transport_config = config.get("transport", None)
    sampler_config = config.get("sampler", None)
    guidance_config = config.get("guidance", None)
    misc = config.get("misc", None)
    training_config = config.get("training", None)
    data_config = config.get("data", None)
    eval_config = config.get("eval", None)
    return stage2_config, transport_config, sampler_config, guidance_config, misc, training_config, data_config, eval_config

def none_or_str(value):
    if value == 'None':
        return None
    return value

def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])

#################################################################################
#                             Training Helper Functions                         #
#################################################################################

def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        if not param.requires_grad:
            continue
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)

def prepare_dataloader(
    data_path: Path,
    batch_size: int,
    workers: int,
    rank: int,
    world_size: int,
    transform: List= None,
) -> Tuple[DataLoader, DistributedSampler]:
    dataset = ImageFolder(str(data_path), transform=transform)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=workers,
        pin_memory=True,
        drop_last=True,
    )
    return loader, sampler

class HypersimDataset(Dataset):
    def __init__(self, clean_img_paths: str, gt_depth_paths: str, annotation_path: str = None, mode: str = 'train', num_eval_img: int = 100, num_view: int = 1, view_sel: dict = None, kernel_size: int = 100):
        self.mode = mode
        self.num_eval_img = num_eval_img
        self.num_view = num_view
        self.view_sel = view_sel
        self.curr_batch_num_view = num_view
        self.kernel_size = kernel_size

        def generate_paths(anns, root, sub_folder, suffix):
            return sorted([
                os.path.join(root, a['scene_name'], 'images', 
                             f"scene_{a['camera_name']}_{sub_folder}", 
                             f"frame.{int(a['frame_id']):04d}.{suffix}.hdf5") 
                for a in anns
            ])
        
        with open(annotation_path, 'r') as f:
            annotations = list(csv.DictReader(f))

        target_annotations = [line for line in annotations if line['split_partition_name'] == mode]
        all_hq = generate_paths(target_annotations, clean_img_paths, "final_hdf5", "color")
        all_depth = generate_paths(target_annotations, gt_depth_paths, "geometry_hdf5", "depth_meters")

        # camera annotations (for view_sel)
        target_scenes = set([a['scene_name'] for a in target_annotations])
        cam_ori_paths = sorted(glob(f'{clean_img_paths}/*/_detail/cam_*/camera_keyframe_orientations.hdf5')) # camera rotation
        cam_pos_paths = sorted(glob(f'{clean_img_paths}/*/_detail/cam_*/camera_keyframe_positions.hdf5'))    # camera translation
        cam_ori_paths = sorted([path for path in cam_ori_paths if path.split('/')[-4] in target_scenes])
        cam_pos_paths = sorted([path for path in cam_pos_paths if path.split('/')[-4] in target_scenes])
        assert len(cam_ori_paths) == len(cam_pos_paths)

        def cam_key(p):
            return (p.split('/')[-4], p.split('/')[-2])  # (scene_id, cam_xx)
        ori_map = {cam_key(p): p for p in cam_ori_paths}
        pos_map = {cam_key(p): p for p in cam_pos_paths}

        self.camera_rankings = {}
        self.camera_num_frames = {}
        common_keys = ori_map.keys() & pos_map.keys()

        for scene_id, camera_id in sorted(common_keys):
            # parse identifiers
            cam_ori_path = ori_map[(scene_id, camera_id)]
            cam_pos_path = pos_map[(scene_id, camera_id)]

            cache_key = (scene_id, camera_id)
            if cache_key in self.camera_rankings:
                continue  # already computed

            # load camera data
            with h5py.File(cam_ori_path, 'r') as f_ori:
                ext_r = f_ori['dataset'][:]   # (N,3,3)

            with h5py.File(cam_pos_path, 'r') as f_pos:
                ext_t = f_pos['dataset'][:]   # (N,3)

            assert ext_r.shape[0] == ext_t.shape[0]
            N = ext_r.shape[0]

            # build extrinsics
            extrinsics = np.zeros((N, 4, 4), dtype=np.float32)
            extrinsics[:, :3, :3] = ext_r
            extrinsics[:, :3, 3]  = ext_t
            extrinsics[:, 3, 3]   = 1.0

            # compute ranking
            ranking, _ = compute_ranking(
                extrinsics,
                lambda_t=1.0,
                normalize=True,
                batched=True
            )

            # cache
            self.camera_rankings[cache_key] = ranking
            self.camera_num_frames[cache_key] = N  

        valid_all_hq = []
        valid_all_depth = []
        valid_scene_names = []
        
        for h, d, ann in zip(all_hq, all_depth, target_annotations):
            if os.path.exists(h) and os.path.exists(d):
                valid_all_hq.append(h)
                valid_all_depth.append(d)
                valid_scene_names.append(ann['scene_name'])

        self.clean_img_paths = valid_all_hq
        self.gt_depth_paths = valid_all_depth
        self.scene_names = valid_scene_names

        self.global_to_camera_idx = {} 
        self.camera_to_global_idx = {}

        for global_idx, hq_path in enumerate(self.clean_img_paths):
            scene_id  = hq_path.split('/')[-4]
            camera_id = hq_path.split('/')[-2]
            frame_id  = int(hq_path.split('.')[-3])

            key = (scene_id, camera_id)
            if key not in self.camera_to_global_idx:
                self.camera_to_global_idx[key] = {}

            self.camera_to_global_idx[key][frame_id] = global_idx
            self.global_to_camera_idx[global_idx] = (scene_id, camera_id, frame_id)
        
        self.scene_to_indices = {}
        for i, scene in enumerate(self.scene_names):
            if scene not in self.scene_to_indices:
                self.scene_to_indices[scene] = []
            self.scene_to_indices[scene].append(i)

        if mode == 'val':
            random.seed(0)
            self.val_indices = list(range(len(self.clean_img_paths)))
            random.shuffle(self.val_indices)
            self.val_indices = self.val_indices[:num_eval_img]

        self.kernel = Kernel(size=(self.kernel_size, self.kernel_size), intensity=0.1)

        print(f"--- Hypersim Dataset ({mode}) ---")
        print(f"Number of clean images: {len(self.clean_img_paths)}")
        print(f"Number of gt depth maps: {len(self.gt_depth_paths)}")
        
    def get_sequential_ids(self, anchor, num_frames):
        indices = []
        current_scene = self.scene_names[anchor]
        for i in range(num_frames):
            target_idx = anchor + i
            if target_idx < len(self.clean_img_paths) and self.scene_names[target_idx] == current_scene:
                indices.append(target_idx)
            else: break
        backward_idx = anchor - 1
        while len(indices) < num_frames:
            if backward_idx >= 0 and self.scene_names[backward_idx] == current_scene:
                indices.append(backward_idx)
                backward_idx -= 1
            else: indices.append(indices[-1])
        return indices

    def get_random_ids(self, anchor, num_frames):
        current_scene = self.scene_names[anchor]
        candidates = self.scene_to_indices[current_scene].copy()
        if anchor in candidates:
            candidates.remove(anchor)
        
        K = num_frames - 1
        if len(candidates) >= K and K > 0:
            sampled = random.sample(candidates, K)
        elif K > 0:
            sampled = random.choices(candidates, k=K) if candidates else [anchor] * K
        else:
            sampled = []

        return [anchor] + sampled

    def get_nearby_ids_random(self, anchor, num_frames, expand_ratio=2.0):
        current_scene = self.scene_names[anchor]
        all_scene_indices = self.scene_to_indices[current_scene]
        
        local_idx = all_scene_indices.index(anchor)
        total_len = len(all_scene_indices)

        desired_span = int(num_frames * expand_ratio) * 2 
        
        half_span = desired_span // 2
        low = local_idx - half_span
        high = local_idx + half_span + 1
        
        if low < 0:
            high = min(total_len, high + abs(low))
            low = 0
            
        if high > total_len:
            low = max(0, low - (high - total_len))
            high = total_len
        
        candidates = all_scene_indices[low:high].copy()
        if anchor in candidates:
            candidates.remove(anchor)
            
        K = num_frames - 1
        if len(candidates) >= K:
            sampled = random.sample(candidates, K)
        else:
            sampled = random.choices(candidates, k=K) if candidates else [anchor] * K
            
        return [anchor] + sampled
    
    def get_nearby_ids_camera(self, anchor, num_frames, expand_ratio=2.0):
        try:
            if num_frames == 1:
                return np.array([anchor], dtype=np.int64)
            
            if anchor not in self.global_to_camera_idx:
                print(f"Anchor {anchor} not found in global_to_camera_idx mapping.")
                return self.get_sequential_ids(anchor, num_frames)

            scene_id, camera_id, cam_frame_idx = self.global_to_camera_idx[anchor]
            tmp_camera_id = f"cam_{camera_id.split('_')[-3]}" 
            
            ranking = self.camera_rankings.get((scene_id, tmp_camera_id))
            if ranking is None:
                print(f"No ranking found for scene: {scene_id}, camera: {tmp_camera_id}")
                return self.get_sequential_ids(anchor, num_frames)

            if cam_frame_idx >= len(ranking):
                print(f"cam_frame_idx {cam_frame_idx} out of ranking bounds (len: {len(ranking)})")
                return self.get_sequential_ids(anchor, num_frames)

            cam_neighbors = ranking[cam_frame_idx][1:] 
            
            valid_global_indices = []
            cam_mapping = self.camera_to_global_idx.get((scene_id, camera_id), {})
            
            for cam_i in cam_neighbors:
                global_i = cam_mapping.get(cam_i)
                if global_i is not None:
                    if 0 <= global_i < len(self.clean_img_paths):
                        valid_global_indices.append(global_i)
                    else:
                        print(f"Global index {global_i} (from cam_i {cam_i}) is out of clean_img_paths range (len: {len(self.clean_img_paths)})")
                        
            K = num_frames - 1
            max_candidates = int(num_frames * expand_ratio)
            valid_global_indices = valid_global_indices[:max_candidates]

            if len(valid_global_indices) == 0:
                print(f"No valid neighbors found for anchor {anchor}. Falling back to sequential.")
                return self.get_sequential_ids(anchor, num_frames)

            replace = len(valid_global_indices) < K
            selected_indices = np.random.choice(valid_global_indices, size=K, replace=replace)
            
            return np.array([anchor] + selected_indices.tolist(), dtype=np.int64)

        except Exception as e:
            print(f"Unexpected error in get_nearby_ids_camera: {str(e)}")
            print(f"Context: anchor={anchor}, num_frames={num_frames}, scene_id={scene_id if 'scene_id' in locals() else 'N/A'}")
            return self.get_nearby_ids_random(anchor, num_frames)

    """
    def get_nearby_ids_camera(self, anchor, num_frames, expand_ratio=2.0):
        if num_frames == 1:
            return np.array([anchor], dtype=np.int64)
        # ----------------------------
        # map global idx -> camera frame
        # ----------------------------
        scene_id, camera_id, cam_frame_idx = self.global_to_camera_idx[anchor]
        tmp_camera_id = f"cam_{camera_id.split('_')[-3]}"
        # get ranking
        ranking = self.camera_rankings[(scene_id, tmp_camera_id)]
        cam_neighbors = ranking[cam_frame_idx]
        # skip itself
        cam_neighbors = cam_neighbors[1:]
        # only keep frames that exist in dataset
        valid_frames = self.camera_to_global_idx[(scene_id, camera_id)]
        cam_neighbors = [i for i in cam_neighbors if i in valid_frames]
        # optionally expand candidate pool
        max_candidates = int(num_frames * expand_ratio)
        cam_neighbors = cam_neighbors[:max_candidates]
        # pick K-1 neighbors
        K = num_frames - 1
        
        # no randomness and only sample the closest views
        # selected_cam_idxs = cam_neighbors[:K]
        
        # add randomness when sampling from neighboring views 
        selected_cam_idxs = np.random.choice(cam_neighbors, size=K, replace=len(cam_neighbors) < K)
        # ----------------------------
        # map camera frame idx -> global idx
        # ----------------------------
        global_indices = [anchor]
        for cam_i in selected_cam_idxs:
            global_i = self.camera_to_global_idx[(scene_id, camera_id)][cam_i]
            global_indices.append(global_i)
        return np.array(global_indices, dtype=np.int64)
    """

    def __len__(self):
        if self.mode == 'val':
            return len(self.val_indices)
        return len(self.clean_img_paths)
    
    def __getitem__(self, item):
        if isinstance(item, (tuple, list)):
            idx, curr_num_view = item
        else:
            idx = item
            curr_num_view = self.num_view 

        if self.mode == 'val':
            random.seed(idx)
            np.random.seed(idx)
            torch.manual_seed(idx)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(idx)

        actual_idx = self.val_indices[idx] if self.mode == 'val' else idx
        
        strategy = self.view_sel.get('strategy', 'sequential')
        if strategy == 'random':
            indices = self.get_random_ids(actual_idx, curr_num_view)
        elif strategy == 'near_random':
            indices = self.get_nearby_ids_random(actual_idx, curr_num_view, self.view_sel.get('expand_ratio', 2.0))
        elif strategy == 'sequential':
            indices = self.get_sequential_ids(actual_idx, curr_num_view)
        elif strategy == 'near_camera':
            indices = self.get_nearby_ids_camera(actual_idx, num_frames=curr_num_view, expand_ratio=self.view_sel.get('expand_ratio', 2.0))

        deg_imgs, clean_imgs, gt_depths = [], [], []
        valid_indices = []
        
        for i in indices:
            cp = self.clean_img_paths[i]
            dp = self.gt_depth_paths[i]
            
            try:
                clean_t, deg_t = process_clean_deg_tensors_from_hdf5(
                    cp, 
                    kernel=self.kernel, 
                    target_size=518, 
                    mode="crop"
                )
                
                depth_t = process_single_depth_hdf5(dp, target_size=518, mode="crop")
                
            except Exception as e:
                print(f"[Warning] Error processing frame {cp}: {e}")
                continue

            if clean_t is None or deg_t is None or depth_t is None:
                print(f"[Warning] Failed to load/process frame {cp} or {dp}. Skipping.")
                continue 
        
            clean_imgs.append(clean_t)
            deg_imgs.append(deg_t)    
            gt_depths.append(depth_t)
            valid_indices.append(i)

        if len(clean_imgs) == 0:
            raise RuntimeError(f"Failed to load any vaild frames")
        
        return {
            "deg_img": torch.stack(deg_imgs),     # [curr_num_view, 3, 518, 518]
            "clean_img": torch.stack(clean_imgs), # [curr_num_view, 3, 518, 518]
            "gt_depth": torch.stack(gt_depths),   # [curr_num_view, 1, 518, 518]
            "gt_poses": torch.zeros((len(valid_indices), 4, 4), dtype=torch.float32), 
            "indices": torch.tensor(valid_indices),
        }
    
class TartanAirDataset(Dataset):
    def __init__(self, clean_img_paths: str, gt_depth_paths: str, annotation_path: str = None, mode: str = 'train', num_eval_img: int = 100, num_view: int = 1, view_sel: dict = None, kernel_size: int = 100):
        self.mode = mode
        self.num_view = num_view
        self.view_sel = view_sel
        self.curr_batch_num_view = num_view
        self.kernel_size = kernel_size

        self.clean_img_paths = sorted(glob(os.path.join(clean_img_paths, "*", "Easy", "*", "image_left", "*.png")))
        self.gt_depth_paths = sorted(glob(os.path.join(gt_depth_paths, "*", "Easy", "*", "depth_left", "*.npy")))

        self.scene_names = [os.path.dirname(os.path.dirname(p)) for p in self.clean_img_paths]

        self.scene_to_indices = {}
        for i, scene in enumerate(self.scene_names):
            if scene not in self.scene_to_indices:
                self.scene_to_indices[scene] = []
            self.scene_to_indices[scene].append(i)

        if mode == 'val':
            self.clean_img_paths = self.clean_img_paths[:num_eval_img]
            self.gt_depth_paths = self.gt_depth_paths[:num_eval_img]
            self.scene_names = self.scene_names[:num_eval_img]
        self.kernel = Kernel(size=(self.kernel_size, self.kernel_size), intensity=0.1)

        print(f"--- TartanAir Dataset ({mode}) ---")
        print(f"Number of clean images: {len(self.clean_img_paths)}")
        print(f"Number of gt depth maps: {len(self.gt_depth_paths)}")

    def get_sequential_ids(self, anchor, num_frames):
        indices = []
        current_scene = self.scene_names[anchor]
        for i in range(num_frames):
            target_idx = anchor + i
            if target_idx < len(self.clean_img_paths) and self.scene_names[target_idx] == current_scene:
                indices.append(target_idx)
            else: break
        backward_idx = anchor - 1
        while len(indices) < num_frames:
            if backward_idx >= 0 and self.scene_names[backward_idx] == current_scene:
                indices.append(backward_idx)
                backward_idx -= 1
            else: indices.append(indices[-1])
        return indices

    def get_random_ids(self, anchor, num_frames):
        current_scene = self.scene_names[anchor]
        candidates = self.scene_to_indices[current_scene].copy()
        if anchor in candidates:
            candidates.remove(anchor)
        
        K = num_frames - 1
        if len(candidates) >= K and K > 0:
            sampled = random.sample(candidates, K)
        elif K > 0:
            sampled = random.choices(candidates, k=K) if candidates else [anchor] * K
        else:
            sampled = []

        return [anchor] + sampled

    def get_nearby_ids_random(self, anchor, num_frames, expand_ratio=2.0):
        current_scene = self.scene_names[anchor]
        all_scene_indices = self.scene_to_indices[current_scene]
        
        local_idx = all_scene_indices.index(anchor)
        total_len = len(all_scene_indices)

        desired_span = int(num_frames * expand_ratio) * 2 
        
        half_span = desired_span // 2
        low = local_idx - half_span
        high = local_idx + half_span + 1
        
        if low < 0:
            high = min(total_len, high + abs(low))
            low = 0
            
        if high > total_len:
            low = max(0, low - (high - total_len))
            high = total_len
        
        candidates = all_scene_indices[low:high].copy()
        if anchor in candidates:
            candidates.remove(anchor)
            
        K = num_frames - 1
        if len(candidates) >= K:
            sampled = random.sample(candidates, K)
        else:
            sampled = random.choices(candidates, k=K) if candidates else [anchor] * K
            
        return [anchor] + sampled
    
    def __len__(self):
        return len(self.clean_img_paths)
    
    def __getitem__(self, item):
        if isinstance(item, (tuple, list)):
            idx, curr_num_view = item
        else:
            idx = item
            curr_num_view = self.num_view 

        if self.mode == 'val':
            random.seed(idx)
            np.random.seed(idx)
            torch.manual_seed(idx)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(idx)

        actual_idx = self.val_indices[idx] if self.mode == 'val' else idx
        
        strategy = self.view_sel.get('strategy', 'sequential')
        if strategy == 'random':
            indices = self.get_random_ids(actual_idx, curr_num_view)
        elif strategy == 'near_random':
            indices = self.get_nearby_ids_random(actual_idx, curr_num_view, self.view_sel.get('expand_ratio', 2.0))
        elif strategy == 'sequential':
            indices = self.get_sequential_ids(actual_idx, curr_num_view)

        deg_list, clean_list, depth_list = [], [], []
        last_valid_clean = None
        last_valid_deg = None
        last_valid_depth = None

        for i in indices:
            try:
                c_t, deg_t = process_clean_deg_tensors_from_image(
                    self.clean_img_paths[i], 
                    kernel=self.kernel, 
                    target_size=518, 
                    mode="crop"
                )
                d_t = process_single_depth_npy(self.gt_depth_paths[i], target_size=518, mode="crop")
                
                if c_t is None or deg_t is None or d_t is None:
                    raise ValueError("Loaded data is None")
                
                last_valid_clean, last_valid_deg, last_valid_depth = c_t, deg_t, d_t
                
            except Exception as e:
                print(f"[Warning] Failed to load frame {self.clean_img_paths[i]}: {e}")
                
                if last_valid_clean is not None:
                    c_t, deg_t, d_t = last_valid_clean.clone(), last_valid_deg.clone(), last_valid_depth.clone()
                else:
                    c_t, deg_t = process_clean_deg_tensors_from_image(
                        self.clean_img_paths[indices[0]], 
                        kernel=self.kernel, 
                        target_size=518, 
                        mode="crop"
                    )
                    d_t = process_single_depth_npy(self.gt_depth_paths[indices[0]], target_size=518, mode="crop")
                    
                    last_valid_clean, last_valid_deg, last_valid_depth = c_t, deg_t, d_t

            clean_list.append(c_t)
            deg_list.append(deg_t)
            depth_list.append(d_t)

        return {
            "deg_img": torch.stack(deg_list),
            "clean_img": torch.stack(clean_list),
            "gt_depth": torch.stack(depth_list),
            "gt_poses": torch.zeros((len(indices), 4, 4), dtype=torch.float32), 
            "indices": torch.tensor(indices)
        }
    
class ETH3DDataset(Dataset):
    def __init__(self, clean_img_paths: str, gt_depth_paths: str, annotation_path: str = None, mode: str = 'train', num_eval_img: int = 100, num_view: int = 1, view_sel: dict = None, kernel_size: int = 100):
        self.mode = mode
        self.num_eval_img = num_eval_img
        self.num_view = num_view
        self.view_sel = view_sel
        self.curr_batch_num_view = num_view
        self.kernel_size = kernel_size
        self.cam_dict = self._parse_colmap_images(clean_img_paths)

        self.clean_img_paths = sorted(glob(os.path.join(clean_img_paths, "*", "images", "dslr_images", "*.JPG")))
        self.gt_depth_paths = sorted(glob(os.path.join(gt_depth_paths, "*", "ground_truth_depth", "dslr_images", "*.JPG")))
        assert len(self.clean_img_paths) == len(self.gt_depth_paths)

        self.scene_names = [p.split('/')[-4] for p in self.clean_img_paths]

        self.scene_to_indices = {}
        for i, scene in enumerate(self.scene_names):
            if scene not in self.scene_to_indices:
                self.scene_to_indices[scene] = []
            self.scene_to_indices[scene].append(i)

        if mode == 'val':
            random.seed(0)
            self.val_indices = list(range(len(self.clean_img_paths)))
            random.shuffle(self.val_indices)
            self.val_indices = self.val_indices[:num_eval_img]

        self.kernel = Kernel(size=(self.kernel_size, self.kernel_size), intensity=0.1)

        print(f"--- ETH3D Dataset ({mode}) ---")
        print(f"Number of clean images: {len(self.clean_img_paths)}")
        print(f"Number of gt depth maps: {len(self.gt_depth_paths)}")

    def _parse_colmap_images(self, clean_img_paths):
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(clean_img_paths)))
        cam_dict = {}
        
        for scene in os.listdir(base_dir):
            img_txt_path = os.path.join(base_dir, scene, "dslr_calibration_jpg", "images.txt")
            if not os.path.exists(img_txt_path): continue
            
            with open(img_txt_path, "r") as f:
                lines = f.readlines()
                for i in range(4, len(lines), 2): 
                    parts = lines[i].strip().split()
                    if len(parts) < 10: continue
                    img_name = parts[-1]
                    q = np.array(parts[1:5], dtype=float)
                    t = np.array(parts[5:8], dtype=float).reshape(3, 1)
                    
                    
                    r = Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
                    
                    cam_dict[f"{scene}/{img_name}"] = {"R": r, "t": t.flatten()}
        return cam_dict

    def get_sequential_ids(self, anchor, num_frames):
        indices = []
        current_scene = self.scene_names[anchor]
        for i in range(num_frames):
            target_idx = anchor + i
            if target_idx < len(self.clean_img_paths) and self.scene_names[target_idx] == current_scene:
                indices.append(target_idx)
            else: break
        backward_idx = anchor - 1
        while len(indices) < num_frames:
            if backward_idx >= 0 and self.scene_names[backward_idx] == current_scene:
                indices.append(backward_idx)
                backward_idx -= 1
            else: indices.append(indices[-1])
        return indices

    def get_random_ids(self, anchor, num_frames):
        current_scene = self.scene_names[anchor]
        candidates = self.scene_to_indices[current_scene].copy()
        if anchor in candidates:
            candidates.remove(anchor)
        
        K = num_frames - 1
        if len(candidates) >= K and K > 0:
            sampled = random.sample(candidates, K)
        elif K > 0:
            sampled = random.choices(candidates, k=K) if candidates else [anchor] * K
        else:
            sampled = []

        return [anchor] + sampled

    def get_nearby_ids_random(self, anchor, num_frames, expand_ratio=2.0):
        current_scene = self.scene_names[anchor]
        all_scene_indices = self.scene_to_indices[current_scene]
        
        local_idx = all_scene_indices.index(anchor)
        total_len = len(all_scene_indices)

        desired_span = int(num_frames * expand_ratio) * 2 
        
        half_span = desired_span // 2
        low = local_idx - half_span
        high = local_idx + half_span + 1
        
        if low < 0:
            high = min(total_len, high + abs(low))
            low = 0
            
        if high > total_len:
            low = max(0, low - (high - total_len))
            high = total_len
        
        candidates = all_scene_indices[low:high].copy()
        if anchor in candidates:
            candidates.remove(anchor)
            
        K = num_frames - 1
        if len(candidates) >= K:
            sampled = random.sample(candidates, K)
        else:
            sampled = random.choices(candidates, k=K) if candidates else [anchor] * K
            
        return [anchor] + sampled
    
    def __len__(self):
        if self.mode == 'val':
            return len(self.val_indices)
        return len(self.clean_img_paths)
    
    def __getitem__(self, item):
        if isinstance(item, (tuple, list)):
            idx, curr_num_view = item
        else:
            idx = item
            curr_num_view = self.num_view

        if self.mode == 'val':
            random.seed(idx)
            np.random.seed(idx)
            torch.manual_seed(idx)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(idx)

        actual_idx = self.val_indices[idx] if self.val_indices is not None else idx
        
        strategy = self.view_sel.get('strategy', 'sequential')
        if strategy == 'random':
            indices = self.get_random_ids(actual_idx, curr_num_view)
        elif strategy == 'near_random':
            indices = self.get_nearby_ids_random(actual_idx, curr_num_view, self.view_sel.get('expand_ratio', 2.0))
        elif strategy == 'sequential':
            indices = self.get_sequential_ids(actual_idx, curr_num_view)

        deg_list, clean_list, depth_list, gt_poses = [], [], [], []
        for i in indices:
            c_t, deg_t = process_clean_deg_tensors_from_image(
                self.clean_img_paths[i], 
                kernel=self.kernel, 
                target_size=518, 
                mode="crop"
            )
            
            d_t = process_eth3d_depth_bin(self.gt_depth_paths[i], target_size=518, mode="crop")
            
            if d_t is not None:
                invalid_mask = (d_t <= 0) | (d_t > 100.0) | torch.isnan(d_t) | torch.isinf(d_t)
                d_t[invalid_mask] = 0.0
            
            clean_list.append(c_t)
            deg_list.append(deg_t)
            depth_list.append(d_t)

            scene = self.scene_names[i]
            img_name = os.path.basename(self.clean_img_paths[i])
            key = f"{scene}/{img_name}"
            pose = self.cam_dict.get(key, {"R": np.eye(3), "t": np.zeros(3)})
            
            se3 = np.eye(4)
            se3[:3, :3] = pose["R"]
            se3[:3, 3] = pose["t"]
            gt_poses.append(torch.from_numpy(se3).float())

        return {
            "deg_img": torch.stack(deg_list),
            "clean_img": torch.stack(clean_list),
            "gt_depth": torch.stack(depth_list),
            "gt_poses": torch.stack(gt_poses),
            "indices": torch.tensor(indices)
        }

class RandomViewBatchSampler:
    def __init__(self, sampler, batch_size, dataset, min_views=1, is_train=True):
        self.sampler = sampler
        self.batch_size = batch_size
        self.dataset = dataset
        self.min_views = min_views
        self.max_views = getattr(dataset, 'num_view', 1)
        self.is_train = is_train
        
    def __iter__(self):
        batch = [] 
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:
                if self.is_train:
                    num_views = random.randint(self.min_views, self.max_views)
                else:
                    num_views = self.max_views
                
                yield [(idx, num_views) for idx in batch]
                batch = []

        if len(batch) > 0 and not getattr(self, 'drop_last', False):
            num_views = random.randint(self.min_views, self.max_views) if self.is_train else self.max_views
            yield [(idx, num_views) for idx in batch]
                
    def __len__(self):
        return len(self.sampler) // self.batch_size

class ConcatDatasetWrapper(ConcatDataset):
    def __init__(self, datasets):
        super().__init__(datasets)
        self.num_view = datasets[0].num_view if datasets else 1
    
    def __getitem__(self, item):
        if isinstance(item, (tuple, list)):
            idx, curr_num_view = item
        else:
            idx, curr_num_view = item, self.num_view

        if idx < 0:
            if -idx > len(self):
                raise ValueError("Absolute value of index should not exceed dataset length")
            idx = len(self) + idx
        
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
            
        return self.datasets[dataset_idx][(sample_idx, curr_num_view)]

def create_datasets_from_config(
    cfg: DictConfig,
    mode: str = 'train',
    max_views: int = 8,
    kernel_size: int = 100,
) -> Tuple[Dataset, Dataset]:
    active_dataset = cfg.get('dataset', [])
    library = cfg.get('library', {})
    num_view = max_views

    all_datasets = []
    
    for name in active_dataset:
        if name not in library:
            print(f"Warning: Dataset '{name}' not found in library.")
            continue
            
        info = library[name]
        print(f"Loading {name}...")
        
        clean_img_paths=info.get('clean_img_path')
        gt_depth_paths=info.get('gt_depth_path')
        annotation_path=info.get('annotation_path')
        num_eval_img=info.get('num_eval_img', 100)
        view_sel=info.get('view_sel', None)

        if name == "hypersim":
            ds = HypersimDataset(
                clean_img_paths=clean_img_paths,
                gt_depth_paths=gt_depth_paths,
                annotation_path=annotation_path,
                mode=mode,
                num_eval_img=num_eval_img,
                num_view=num_view,
                view_sel=view_sel,
                kernel_size=kernel_size
            )
        elif name == "tartanair":
            ds = TartanAirDataset(
                clean_img_paths=clean_img_paths,
                gt_depth_paths=gt_depth_paths,
                annotation_path=annotation_path,
                mode=mode,
                num_eval_img=num_eval_img,
                num_view=num_view,
                view_sel=view_sel,
                kernel_size=kernel_size
            )
        elif name == "eth3d":
            ds = ETH3DDataset(
                clean_img_paths=clean_img_paths,
                gt_depth_paths=gt_depth_paths,
                annotation_path=annotation_path,
                mode=mode,
                num_eval_img=num_eval_img,
                num_view=num_view,
                view_sel=view_sel,
                kernel_size=kernel_size
            )
        all_datasets.append(ds)
    
    if not all_datasets:
        return None
        
    if len(all_datasets) == 1:
        return all_datasets[0]
    else:
        return ConcatDatasetWrapper(all_datasets)

def load_dataloader(
    cfg: DictConfig,
    batch_size: int,
    workers: int,
    rank: int,
    world_size: int,
    mode: str='train',
    max_views: int=8,
    kernel_size: int=100,
) -> Tuple[DataLoader, DistributedSampler]:
    # breakpoint()
    ds = create_datasets_from_config(cfg[mode], mode=mode, max_views=max_views, kernel_size=kernel_size)
    sampler = DistributedSampler(ds, num_replicas=world_size, rank=rank, shuffle=True)

    batch_sampler = RandomViewBatchSampler(sampler, batch_size, ds, min_views=1, is_train=(mode=='train'))
    loader = DataLoader(
        ds,
        batch_sampler=batch_sampler,
        num_workers=workers,
        pin_memory=True,
    )
    return loader, sampler

def get_autocast_scaler(args) -> Tuple[dict, torch.cuda.amp.GradScaler | None]:
    if args.precision == "fp16":
        scaler = GradScaler()
        autocast_kwargs = dict(enabled=True, dtype=torch.float16)
    elif args.precision == "bf16":
        scaler = None
        autocast_kwargs = dict(enabled=True, dtype=torch.bfloat16)
    else:
        scaler = None
        autocast_kwargs = dict(enabled=False)
    
    return scaler, autocast_kwargs