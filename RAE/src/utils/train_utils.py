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

motionblur_path = "/mnt/dataset1/jaeeun/MVR"
if motionblur_path not in sys.path:
    sys.path.append(motionblur_path)
from motionblur.motionblur import Kernel 

vggt_path = "/mnt/dataset1/jaeeun/MVR/vggt"
if vggt_path not in sys.path:
    sys.path.append(vggt_path)
from vggt.utils.load_fn import *

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


# class LatentDataset(Dataset):
#     def __init__(self, deg_img_paths: List[str], data_path: str):
#         # breakpoint()
#         self.data_path = Path(data_path)
#         self.latent_files = sorted(list(self.data_path.rglob("*.pt")))

#         self.deg_img_paths = deg_img_paths
        
#         if not self.latent_files:
#             raise FileNotFoundError(f"No .pt files found in {data_path}")

#     def __len__(self):
#         print(f"len latent files: {len(self.latent_files)}, len deg img paths: {len(self.deg_img_paths)}")
#         return min(len(self.latent_files), len(self.deg_img_paths)) # 15008

#     def __getitem__(self, idx):
#         latent_path = self.latent_files[idx]
#         clean_latent = torch.load(latent_path, map_location="cpu")

#         deg_img_path = self.deg_img_paths[idx]
        
#         if clean_latent.ndim == 4:
#             clean_latent = clean_latent.squeeze(0)
            
#         return deg_img_path, clean_latent # (S, P, 2C) = [1, 1041, 2048]

# class LatentDataset(Dataset):
#     def __init__(self, deg_img_paths: List[str], data_path: str):
#         # breakpoint()
#         self.data_path = Path(data_path)
        
#         self.deg_img_dict = {}
#         for p in deg_img_paths:
#             path_obj = Path(p)
#             parts = path_obj.parts
            
#             scene = parts[-4]               # ai_001_002
#             cam = parts[-3].split('_final')[0] # scene_cam_00
#             frame = path_obj.stem.replace('_color', '') # frame_0000
            
#             key = f"{scene}_{cam}_{frame}"
#             self.deg_img_dict[key] = p

#         # breakpoint()
#         all_latent_files = list(self.data_path.rglob("*.pt"))
        
#         self.valid_pairs = []
#         for lp in all_latent_files:
#             path_obj = Path(lp)
#             parts = path_obj.parts
            
#             scene = parts[-4]               # ai_004_003
#             cam = parts[-3].split('_final')[0] # scene_cam_00
#             frame = parts[-2]               # frame_0000
            
#             key = f"{scene}_{cam}_{frame}"
            
#             if key in self.deg_img_dict:
#                 self.valid_pairs.append({
#                     "latent_path": lp,
#                     "deg_img_path": self.deg_img_dict[key]
#                 })

#         self.valid_pairs.sort(key=lambda x: x["deg_img_path"])

#         if not self.valid_pairs:
#             raise FileNotFoundError(f"Can't find any matching latent and degraded image pairs in {data_path}")
        
#         print(f"{len(self.valid_pairs)} pairs of latent and degraded images found.")

#     def __len__(self):
#         return len(self.valid_pairs)

#     def __getitem__(self, idx):
#         pair = self.valid_pairs[idx]
        
#         clean_latent = torch.load(pair["latent_path"], map_location="cpu")
#         if clean_latent.ndim == 4:
#             clean_latent = clean_latent.squeeze(0)
            
#         deg_img_path = pair["deg_img_path"]
        
#         return deg_img_path, clean_latent

### 전체 데이터셋 불러오기 
# class LatentDataset(Dataset):
#     def __init__(self, clean_img_paths: str, deg_img_paths: str, gt_depth_paths: str):
#         self.clean_img_paths = sorted(glob(os.path.join(clean_img_paths, "ai_*", "images", "scene_cam_*_final_hdf5", "frame.*.color.hdf5")))
#         self.gt_depth_paths = sorted(glob(os.path.join(gt_depth_paths, "ai_*", "images", "scene_cam_*_geometry_hdf5", "frame.*.depth_meters.hdf5")))
        
#         if deg_img_paths is not None:
#             self.deg_img_paths = sorted(glob(os.path.join(deg_img_paths, "ai_*", "scene_cam_*_final_hdf5", "images", "frame_*_color.png")))
#         else:
#             self.deg_img_paths = None
#             self.blur_intensity = 0.1
#             self.kernel_size = 50
#             self.kernel = Kernel(size=(self.kernel_size, self.kernel_size), intensity=self.blur_intensity)
        
#         print(f"Number of clean images: {len(self.clean_img_paths)}")
#         print(f"Number of degraded images: {len(self.deg_img_paths) if self.deg_img_paths is not None else 0}")
#         print(f"Number of gt depth maps: {len(self.gt_depth_paths)}")

#     def __len__(self):
#         return len(self.deg_img_paths) if self.deg_img_paths is not None else len(self.clean_img_paths)
    
#     def __getitem__(self, idx):
#         clean_p = self.clean_img_paths[idx]
#         if self.deg_img_paths is not None:
#             deg_p = self.deg_img_paths[idx]
#         else:
#             deg_p = None
#         depth_p = self.gt_depth_paths[idx]

#         clean_tensor = process_single_hdf5(clean_p, target_size=518, mode="crop")  # [3, 518, 518]
#         depth_tensor = process_single_depth_hdf5(depth_p, target_size=518, mode="crop")  # [1, 518, 518]

#         if self.deg_img_paths is not None:
#             deg_tensor = process_single_image(deg_p, target_size=518, mode="crop")      # [3, 518, 518]
#         else:
#             deg_tensor = process_degraded_from_tensor(clean_tensor, self.kernel)

#         return {
#             "deg_img": deg_tensor,     # [3, H, W]
#             "clean_img": clean_tensor, # [3, H, W]
#             "gt_depth": depth_tensor,  # [1, H, W]
#         }


# clean input images
# class LatentDataset(Dataset):
#     def __init__(self, data_path: str):
#         self.data_path = Path(data_path)
#         self.latent_files = sorted(list(self.data_path.rglob("*.pt")))
        
#         if not self.latent_files:
#             raise FileNotFoundError(f"No .pt files found in {data_path}")
    
#     def __len__(self):
#         return len(self.latent_files)

#     def __getitem__(self, idx):
#         latent_path = self.latent_files[idx]
#         clean_latent = torch.load(latent_path, map_location="cpu")

        
#         if clean_latent.ndim == 4:
#             clean_latent = clean_latent.squeeze(0)
            
#         return clean_latent
    
### ai_010까지 필터링
class LatentDataset(Dataset):
    def __init__(self, clean_img_paths: str, deg_img_paths: str, gt_depth_paths: str):
        all_clean = sorted(glob(os.path.join(clean_img_paths, "ai_*", "images", "scene_cam_*_final_hdf5", "frame.*.color.hdf5")))
        all_depth = sorted(glob(os.path.join(gt_depth_paths, "ai_*", "images", "scene_cam_*_geometry_hdf5", "frame.*.depth_meters.hdf5")))
        
        exclude_patterns = [
            r"ai_003_.*_cam_00",
            r"ai_004_.*_cam_01"
        ]

        def is_valid(path):
            match = re.search(r'ai_(\d{3})', path)
            if not match:
                return False
            
            scene_idx = int(match.group(1))
            if not (1 <= scene_idx <= 10):
                return False
            
            for pattern in exclude_patterns:
                if re.search(pattern, path):
                    return False
            
            return True

        self.clean_img_paths = [p for p in all_clean if is_valid(p)]
        self.gt_depth_paths = [p for p in all_depth if is_valid(p)]

        if deg_img_paths is not None:
            all_deg = sorted(glob(os.path.join(deg_img_paths, "ai_*", "scene_cam_*_final_hdf5", "images", "frame_*_color.png")))
            self.deg_img_paths = [p for p in all_deg if is_valid(p)]
        else:
            self.deg_img_paths = None
            self.blur_intensity = 0.1
            self.kernel_size = 50
            self.kernel = Kernel(size=(self.kernel_size, self.kernel_size), intensity=self.blur_intensity)

        print(f"--- Dataset Filtered ---")
        print(f"Number of clean images: {len(self.clean_img_paths)}")
        print(f"Number of degraded images: {len(self.deg_img_paths) if self.deg_img_paths is not None else 0}")
        print(f"Number of gt depth maps: {len(self.gt_depth_paths)}")

    def __len__(self):
        return len(self.deg_img_paths) if self.deg_img_paths is not None else len(self.clean_img_paths)
    
    def __getitem__(self, idx):
        clean_p = self.clean_img_paths[idx]
        if self.deg_img_paths is not None:
            deg_p = self.deg_img_paths[idx]
        else:
            deg_p = None
        depth_p = self.gt_depth_paths[idx]

        clean_tensor = process_single_hdf5(clean_p, target_size=518, mode="crop")  # [3, 518, 518]
        depth_tensor = process_single_depth_hdf5(depth_p, target_size=518, mode="crop")  # [1, 518, 518]

        if self.deg_img_paths is not None:
            deg_tensor = process_single_image(deg_p, target_size=518, mode="crop")      # [3, 518, 518]
        else:
            deg_tensor = process_degraded_from_tensor(clean_tensor, self.kernel)

        return {
            "deg_img": deg_tensor,     # [3, H, W]
            "clean_img": clean_tensor, # [3, H, W]
            "gt_depth": depth_tensor,  # [1, H, W]
        }

class HypersimDataset(Dataset):
    def __init__(self, clean_img_paths: str, gt_depth_paths: str, annotation_path: str = None, mode: str = 'train', num_eval_img: int = 100, num_view: int = 1, view_sel: dict = None):
        self.mode = mode
        self.num_eval_img = num_eval_img
        self.num_view = num_view
        self.view_sel = view_sel
        self.curr_batch_num_view = num_view
        self.kernel_size = 100

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
    
    def set_batch_num_view(self, num_view):
        self.curr_batch_num_view = num_view

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
    
    def __getitem__(self, idx):
        if self.mode == 'val':
            random.seed(idx)
            np.random.seed(idx)
            torch.manual_seed(idx)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(idx)

        actual_idx = self.val_indices[idx] if self.mode == 'val' else idx
        
        strategy = self.view_sel.get('strategy', 'sequential')
        if strategy == 'random':
            indices = self.get_random_ids(actual_idx, self.curr_batch_num_view)
        elif strategy == 'near_random':
            indices = self.get_nearby_ids_random(actual_idx, self.curr_batch_num_view, self.view_sel.get('expand_ratio', 2.0))
        elif strategy == 'sequential':
            indices = self.get_sequential_ids(actual_idx, self.curr_batch_num_view)
        elif strategy == 'near_camera':
            indices = self.get_nearby_ids_camera(actual_idx, num_frames=self.curr_batch_num_view, expand_ratio=self.view_sel.get('expand_ratio', 2.0))

        deg_imgs, clean_imgs, gt_depths = [], [], []
        
        for i in indices:
            cp = self.clean_img_paths[i]
            dp = self.gt_depth_paths[i]
            
            clean_t = process_single_hdf5(cp, target_size=518, mode="crop")
            depth_t = process_single_depth_hdf5(dp, target_size=518, mode="crop")
            
            deg_t = process_degraded_from_tensor(clean_t, self.kernel)
            
            clean_imgs.append(clean_t)
            gt_depths.append(depth_t)
            deg_imgs.append(deg_t)

        return {
            "deg_img": torch.stack(deg_imgs),     # [V, 3, 518, 518]
            "clean_img": torch.stack(clean_imgs), # [V, 3, 518, 518]
            "gt_depth": torch.stack(gt_depths),   # [V, 1, 518, 518]
            "indices": torch.tensor(indices)
        }
    
class TartanAirDataset(Dataset):
    def __init__(self, clean_img_paths: str, gt_depth_paths: str, annotation_path: str = None, mode: str = 'train', num_eval_img: int = 100, num_view: int = 1, view_sel: dict = None):
        self.mode = mode
        self.num_view = num_view
        self.view_sel = view_sel
        self.curr_batch_num_view = num_view
        self.kernel_size = 100

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

    def set_batch_num_view(self, num_view):
        self.curr_batch_num_view = num_view
    
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
    
    def __getitem__(self, idx):
        actual_idx = self.val_indices[idx] if self.mode == 'val' else idx
        
        strategy = self.view_sel.get('strategy', 'sequential')
        if strategy == 'random':
            indices = self.get_random_ids(actual_idx, self.curr_batch_num_view)
        elif strategy == 'near_random':
            indices = self.get_nearby_ids_random(actual_idx, self.curr_batch_num_view, self.view_sel.get('expand_ratio', 2.0))
        elif strategy == 'sequential':
            indices = self.get_sequential_ids(actual_idx, self.curr_batch_num_view)

        deg_list, clean_list, depth_list = [], [], []
        
        last_valid_clean = None
        last_valid_depth = None

        for i in indices:
            try:
                c_t = process_single_image(self.clean_img_paths[i], target_size=518, mode="crop")
                d_t = process_single_depth_npy(self.gt_depth_paths[i], target_size=518, mode="crop")
                
                if c_t is None or d_t is None:
                    raise ValueError("Loaded data is None")
                    
                last_valid_clean, last_valid_depth = c_t, d_t
                
            except Exception as e:
                print(f"[Warning] Failed to load frame {self.clean_img_paths[i]}: {e}")
                if last_valid_clean is not None:
                    c_t, d_t = last_valid_clean, last_valid_depth
                else:
                    c_t = process_single_image(self.clean_img_paths[indices[0]], target_size=518, mode="crop")
                    d_t = process_single_depth_npy(self.gt_depth_paths[indices[0]], target_size=518, mode="crop")

            deg_t = process_degraded_from_tensor(c_t, self.kernel)
            clean_list.append(c_t)
            depth_list.append(d_t)
            deg_list.append(deg_t)

        return {
            "deg_img": torch.stack(deg_list),
            "clean_img": torch.stack(clean_list),
            "gt_depth": torch.stack(depth_list),
            "indices": torch.tensor(indices)
        }
    
class ETH3DDataset(Dataset):
    def __init__(self, clean_img_paths: str, gt_depth_paths: str, annotation_path: str = None, mode: str = 'train', num_eval_img: int = 100, num_view: int = 1, view_sel: dict = None):
        self.mode = mode
        self.num_eval_img = num_eval_img
        self.num_view = num_view
        self.view_sel = view_sel
        self.curr_batch_num_view = num_view
        self.kernel_size = 100

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
        
    def set_batch_num_view(self, num_view):
        self.curr_batch_num_view = num_view

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
    
    def __getitem__(self, idx):
        if self.mode == 'val':
            random.seed(idx)
            np.random.seed(idx)
            torch.manual_seed(idx)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(idx)

        actual_idx = self.val_indices[idx] if self.val_indices is not None else idx
        
        strategy = self.view_sel.get('strategy', 'sequential')
        if strategy == 'random':
            indices = self.get_random_ids(actual_idx, self.num_view)
        elif strategy == 'near_random':
            indices = self.get_nearby_ids_random(actual_idx, self.num_view, self.view_sel.get('expand_ratio', 2.0))
        elif strategy == 'sequential':
            indices = self.get_sequential_ids(actual_idx, self.num_view)

        deg_list, clean_list, depth_list = [], [], []
        for i in indices:
            c_t = process_single_image(self.clean_img_paths[i], target_size=518, mode="crop")
            d_t = process_eth3d_depth_bin(self.gt_depth_paths[i], target_size=518, mode="crop")
            invalid_mask = (d_t <= 0) | (d_t > 100.0) | torch.isnan(d_t) | torch.isinf(d_t)
            d_t[invalid_mask] = 0.0
            deg_t = process_degraded_from_tensor(c_t, self.kernel)
            
            clean_list.append(c_t)
            depth_list.append(d_t)
            deg_list.append(deg_t)

        return {
            "deg_img": torch.stack(deg_list),
            "clean_img": torch.stack(clean_list),
            "gt_depth": torch.stack(depth_list),
            "indices": torch.tensor(indices)
        }

class RandomViewBatchSampler:
    def __init__(self, sampler, batch_size, dataset, min_views=1):
        self.sampler = sampler
        self.batch_size = batch_size
        self.dataset = dataset
        self.min_views = min_views
        self.max_views = dataset.num_view
        
    def __iter__(self):
        batch = []
        batch_count = 0 
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:
                num_views = random.randint(self.min_views, self.max_views)
                self.dataset.set_batch_num_view(num_views)
                
                # print(f"[Batch {batch_count}] Set num_views to: {num_views}")
                yield batch
                batch = []
                batch_count += 1
                
    def __len__(self):
        return len(self.sampler) // self.batch_size

def prepare_latent_dataloader(
    clean_img_paths: Path,
    deg_img_paths: Path,
    gt_depth_paths: Path,
    batch_size: int,
    workers: int,
    rank: int,
    world_size: int,
) -> Tuple[DataLoader, DistributedSampler]:
    latent_dataset = LatentDataset(clean_img_paths, deg_img_paths, gt_depth_paths)
    sampler = DistributedSampler(latent_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    loader = DataLoader(
        latent_dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=workers,
        pin_memory=True,
        drop_last=True,
    )
    return loader, sampler

class ConcatDatasetWrapper(ConcatDataset):
    def __init__(self, datasets):
        super().__init__(datasets)
        self.num_view = datasets[0].num_view if datasets else 1
        self.curr_batch_num_view = self.num_view
    
    def set_batch_num_view(self, num_view):
        self.curr_batch_num_view = num_view
        for dataset in self.datasets:
            if hasattr(dataset, 'set_batch_num_view'):
                dataset.set_batch_num_view(num_view)

def create_datasets_from_config(
    cfg: DictConfig,
    mode: str = 'train'
) -> Tuple[Dataset, Dataset]:
    active_dataset = cfg.get('dataset', [])
    library = cfg.get('library', {})
    num_view = cfg.get('num_view', 1)

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
                view_sel=view_sel
            )
        elif name == "tartanair":
            ds = TartanAirDataset(
                clean_img_paths=clean_img_paths,
                gt_depth_paths=gt_depth_paths,
                annotation_path=annotation_path,
                mode=mode,
                num_eval_img=num_eval_img,
                num_view=num_view,
                view_sel=view_sel
            )
        elif name == "eth3d":
            ds = ETH3DDataset(
                clean_img_paths=clean_img_paths,
                gt_depth_paths=gt_depth_paths,
                annotation_path=annotation_path,
                mode=mode,
                num_eval_img=num_eval_img,
                num_view=num_view,
                view_sel=view_sel
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
    mode: str='train'
) -> Tuple[DataLoader, DistributedSampler]:
    # breakpoint()
    ds = create_datasets_from_config(cfg[mode], mode=mode)
    sampler = DistributedSampler(ds, num_replicas=world_size, rank=rank, shuffle=True)

    batch_sampler = RandomViewBatchSampler(sampler, batch_size, ds, min_views=1)
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