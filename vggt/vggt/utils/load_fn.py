# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from PIL import Image
import numpy as np
from tqdm import tqdm
import h5py
import os
import torch.nn.functional as F
from torchvision import transforms as TF
import torchvision.transforms as T 

def load_and_preprocess_images_square(image_path_list, target_size=1024):
    """
    Load and preprocess images by center padding to square and resizing to target size.
    Also returns the position information of original pixels after transformation.

    Args:
        image_path_list (list): List of paths to image files
        target_size (int, optional): Target size for both width and height. Defaults to 518.

    Returns:
        tuple: (
            torch.Tensor: Batched tensor of preprocessed images with shape (N, 3, target_size, target_size),
            torch.Tensor: Array of shape (N, 5) containing [x1, y1, x2, y2, width, height] for each image
        )

    Raises:
        ValueError: If the input list is empty
    """
    # Check for empty list
    if len(image_path_list) == 0:
        raise ValueError("At least 1 image is required")

    images = []
    original_coords = []  # Renamed from position_info to be more descriptive
    to_tensor = T.functional.ToTensor()

    for image_path in image_path_list:
        # Open image
        img = Image.open(image_path)

        # If there's an alpha channel, blend onto white background
        if img.mode == "RGBA":
            background = Image.new("RGBA", img.size, (255, 255, 255, 255))
            img = Image.alpha_composite(background, img)

        # Convert to RGB
        img = img.convert("RGB")

        # Get original dimensions
        width, height = img.size

        # Make the image square by padding the shorter dimension
        max_dim = max(width, height)

        # Calculate padding
        left = (max_dim - width) // 2
        top = (max_dim - height) // 2

        # Calculate scale factor for resizing
        scale = target_size / max_dim

        # Calculate final coordinates of original image in target space
        x1 = left * scale
        y1 = top * scale
        x2 = (left + width) * scale
        y2 = (top + height) * scale

        # Store original image coordinates and scale
        original_coords.append(np.array([x1, y1, x2, y2, width, height]))

        # Create a new black square image and paste original
        square_img = Image.new("RGB", (max_dim, max_dim), (0, 0, 0))
        square_img.paste(img, (left, top))

        # Resize to target size
        square_img = square_img.resize((target_size, target_size), Image.Resampling.BICUBIC)

        # Convert to tensor
        img_tensor = to_tensor(square_img)
        images.append(img_tensor)

    # Stack all images
    images = torch.stack(images)
    original_coords = torch.from_numpy(np.array(original_coords)).float()

    # Add additional dimension if single image to ensure correct shape
    if len(image_path_list) == 1:
        if images.dim() == 3:
            images = images.unsqueeze(0)
            original_coords = original_coords.unsqueeze(0)

    return images, original_coords


def load_and_preprocess_images(image_path_list, mode="crop"):
    """
    A quick start function to load and preprocess images for model input.
    This assumes the images should have the same shape for easier batching, but our model can also work well with different shapes.

    Args:
        image_path_list (list): List of paths to image files
        mode (str, optional): Preprocessing mode, either "crop" or "pad".
                             - "crop" (default): Sets width to 518px and center crops height if needed.
                             - "pad": Preserves all pixels by making the largest dimension 518px
                               and padding the smaller dimension to reach a square shape.

    Returns:
        torch.Tensor: Batched tensor of preprocessed images with shape (N, 3, H, W)

    Raises:
        ValueError: If the input list is empty or if mode is invalid

    Notes:
        - Images with different dimensions will be padded with white (value=1.0)
        - A warning is printed when images have different shapes
        - When mode="crop": The function ensures width=518px while maintaining aspect ratio
          and height is center-cropped if larger than 518px
        - When mode="pad": The function ensures the largest dimension is 518px while maintaining aspect ratio
          and the smaller dimension is padded to reach a square shape (518x518)
        - Dimensions are adjusted to be divisible by 14 for compatibility with model requirements
    """
    # RGBA processing -> RGB conversion -> resize -> crop/pad -> tensor conversion -> batch padding

    # Check for empty list
    if len(image_path_list) == 0:
        raise ValueError("At least 1 image is required")

    # Validate mode
    if mode not in ["crop", "pad"]:
        raise ValueError("Mode must be either 'crop' or 'pad'")

    images = []
    shapes = set()
    to_tensor = T.functional.to_tensor
    target_size = 518

    # First process all images and collect their shapes
    for image_path in image_path_list:
    # for image_path in tqdm(image_path_list, desc="Loading Images", leave=True):
        # Open image
        img = Image.open(image_path)

        # If there's an alpha channel, blend onto white background:
        if img.mode == "RGBA":
            # Create white background
            background = Image.new("RGBA", img.size, (255, 255, 255, 255))
            # Alpha composite onto the white background
            img = Image.alpha_composite(background, img)

        # Now convert to "RGB" (this step assigns white for transparent areas)
        img = img.convert("RGB")

        width, height = img.size

        if mode == "pad":
            # Make the largest dimension 518px while maintaining aspect ratio
            if width >= height:
                new_width = target_size
                new_height = round(height * (new_width / width) / 14) * 14  # Make divisible by 14
            else:
                new_height = target_size
                new_width = round(width * (new_height / height) / 14) * 14  # Make divisible by 14
        else:  # mode == "crop"
            # Original behavior: set width to 518px
            new_width = target_size
            # Calculate height maintaining aspect ratio, divisible by 14
            new_height = round(height * (new_width / width) / 14) * 14

        IMAGENET_NORMALIZE = T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

        # Resize with new dimensions (width, height)
        img = img.resize((new_width, new_height), Image.Resampling.BICUBIC)
        img = to_tensor(img)  # Convert to tensor (0, 1)
        
        img = IMAGENET_NORMALIZE(img)  # Normalize

        # Center crop height if it's larger than 518 (only in crop mode)
        if mode == "crop" and new_height > target_size:
            start_y = (new_height - target_size) // 2
            img = img[:, start_y : start_y + target_size, :]

        # For pad mode, pad to make a square of target_size x target_size
        if mode == "pad":
            h_padding = target_size - img.shape[1]
            w_padding = target_size - img.shape[2]

            if h_padding > 0 or w_padding > 0:
                pad_top = h_padding // 10
                pad_bottom = h_padding - pad_top
                pad_left = w_padding // 10
                pad_right = w_padding - pad_left

                # Pad with white (value=1.0)
                img = torch.nn.functional.pad(
                    img, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=1.0
                )

        shapes.add((img.shape[1], img.shape[2]))
        images.append(img)

    # Check if we have different shapes
    # In theory our model can also work well with different shapes
    if len(shapes) > 1:
        print(f"Warning: Found images with different shapes: {shapes}")
        # Find maximum dimensions
        max_height = max(shape[0] for shape in shapes)
        max_width = max(shape[1] for shape in shapes)

        # Pad images if necessary
        padded_images = []
        for img in images:
            h_padding = max_height - img.shape[1]
            w_padding = max_width - img.shape[2]

            if h_padding > 0 or w_padding > 0:
                pad_top = h_padding // 2
                pad_bottom = h_padding - pad_top
                pad_left = w_padding // 2
                pad_right = w_padding - pad_left

                img = torch.nn.functional.pad(
                    img, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=1.0
                )
            padded_images.append(img)
        images = padded_images

    images = torch.stack(images)  # concatenate images

    # Ensure correct shape when single image
    if len(image_path_list) == 1:
        # Verify shape is (1, C, H, W)
        if images.dim() == 3:
            images = images.unsqueeze(0)

    return images

def process_and_save_hdf5_images(image_paths, save_deg_dir, kernel=None, crop=False):
    gamma = 1.0 / 2.2
    inv_gamma = 1.0 / gamma
    percentile = 90
    brightness_nth_percentile_desired = 0.8
    eps = 0.0001

    for idx, image_path in enumerate(tqdm(image_paths, desc="Processing & Saving")):
        try:    
            path_parts = image_path.split(os.sep)
            scene_id = path_parts[-4]    
            cam_id = path_parts[-2]      
            frame_token = path_parts[-1].split('.')
            frame_name = f"{frame_token[0]}_{frame_token[1]}" 

            target_dir = os.path.join(save_deg_dir, scene_id, cam_id)
            os.makedirs(target_dir, exist_ok=True)
            save_path = os.path.join(target_dir, f"{frame_name}.png")

            with h5py.File(image_path, "r") as f:
                rgb_color = f["dataset"][:].astype(np.float32)

            entity_path = image_path.replace("final_hdf5", "geometry_hdf5").replace(".color.hdf5", ".render_entity_id.hdf5")
            if os.path.exists(entity_path):
                with h5py.File(entity_path, "r") as f:
                    render_entity_id = f["dataset"][:].astype(np.int32)
                valid_mask = render_entity_id != -1
            else:
                valid_mask = np.ones(rgb_color.shape[:2], dtype=bool)

            if np.count_nonzero(valid_mask) == 0:
                scale = 1.0
            else:
                brightness = 0.3 * rgb_color[:,:,0] + 0.59 * rgb_color[:,:,1] + 0.11 * rgb_color[:,:,2]
                brightness_valid = brightness[valid_mask]
                curr_p = np.percentile(brightness_valid, percentile)
                scale = np.power(brightness_nth_percentile_desired, inv_gamma) / curr_p if curr_p >= eps else 0.0

            rgb_tm = np.power(np.maximum(scale * rgb_color, 0), gamma)
            rgb_tm = np.clip(rgb_tm, 0, 1)
            
            if kernel is not None:
                img_pil = Image.fromarray((rgb_tm * 255).astype(np.uint8))
                img_pil = kernel.applyTo(img_pil, keep_image_dim=True)
                
                rgb_tm = np.array(img_pil)

                if crop:
                    k_h, k_w = kernel.SIZE
                    margin_h, margin_w = k_h // 2, k_w // 2
                    h, w, _ = rgb_tm.shape
                    rgb_tm = rgb_tm[margin_h:h-margin_h, margin_w:w-margin_w, :]
                
                final_img = Image.fromarray(rgb_tm)
            else:
                final_img = Image.fromarray((rgb_tm * 255).astype(np.uint8))

            final_img.save(save_path, format='PNG')

            if idx == 0:
                preview_path = f"./preview_{frame_name}.png"
                final_img.save(preview_path, format='PNG')
                print(f"\n[Preview] First image saved to: {preview_path}")

        except Exception as e:
            print(f"\n[Error] Failed to process {image_path}: {e}")
            continue

    print(f"\n[Finished] All images saved to: {save_deg_dir}")

def load_and_preprocess_images_hdf5(image_path_list, batch_size=8, mode="crop"):
    if len(image_path_list) == 0:
        raise ValueError("At least 1 image is required")
    if mode not in ["crop", "pad"]:
        raise ValueError("Mode must be either 'crop' or 'pad'")

    images = []
    shapes = set()
    target_size = 518

    gamma = 1.0 / 2.2
    inv_gamma = 1.0 / gamma
    percentile = 90
    brightness_nth_percentile_desired = 0.8
    eps = 0.0001

    for image_path in tqdm(image_path_list, desc="Loading HDF5 Images"):
        try:
            with h5py.File(image_path, "r") as f:
                rgb_color = f["dataset"][:].astype(np.float32) # [H, W, 3]
            
            entity_path = image_path.replace("final_hdf5", "geometry_hdf5").replace(".color.hdf5", ".render_entity_id.hdf5")
            if os.path.exists(entity_path):
                with h5py.File(entity_path, "r") as f:
                    render_entity_id = f["dataset"][:].astype(np.int32)
                valid_mask = render_entity_id != -1
            else:
                valid_mask = np.ones(rgb_color.shape[:2], dtype=bool)
        except Exception as e:
            print(f"Error loading {image_path}: {e}")
            continue

        brightness = 0.3 * rgb_color[:, :, 0] + 0.59 * rgb_color[:, :, 1] + 0.11 * rgb_color[:, :, 2]
        brightness_valid = brightness[valid_mask]
        
        if len(brightness_valid) == 0 or np.percentile(brightness_valid, percentile) < eps:
            scale = 1.0
        else:
            current_p = np.percentile(brightness_valid, percentile)
            scale = np.power(brightness_nth_percentile_desired, inv_gamma) / current_p

        rgb_tm = np.power(np.maximum(scale * rgb_color, 0), gamma)
        rgb_tm = np.clip(rgb_tm, 0, 1) # [H, W, 3] (0~1 float)

        img_tensor = torch.from_numpy(rgb_tm).permute(2, 0, 1)
        # breakpoint()

        _, height, width = img_tensor.shape
        
        if mode == "pad":
            if width >= height:
                new_width = target_size
                new_height = round(height * (new_width / width) / 14) * 14
            else:
                new_height = target_size
                new_width = round(width * (new_height / height) / 14) * 14
        else: # crop
            new_width = target_size
            new_height = round(height * (new_width / width) / 14) * 14

        # Resize (Bilinear/Bicubic)
        img_tensor = TF.functional.resize(img_tensor, [new_height, new_width], interpolation=T.InterpolationMode.BICUBIC)

        # Center Crop
        if mode == "crop" and new_height > target_size:
            start_y = (new_height - target_size) // 2
            img_tensor = img_tensor[:, start_y : start_y + target_size, :]

        # Pad to Square
        if mode == "pad":
            h_padding = target_size - img_tensor.shape[1]
            w_padding = target_size - img_tensor.shape[2]
            if h_padding > 0 or w_padding > 0:
                img_tensor = torch.nn.functional.pad(
                    img_tensor, (w_padding//2, w_padding - w_padding//2, h_padding//2, h_padding - h_padding//2), 
                    mode="constant", value=1.0
                )

        shapes.add((img_tensor.shape[1], img_tensor.shape[2]))
        # breakpoint()
        images.append(img_tensor)

    if len(shapes) > 1:
        max_h = max(s[0] for s in shapes)
        max_w = max(s[1] for s in shapes)
        images = [torch.nn.functional.pad(img, (0, max_w - img.shape[2], 0, max_h - img.shape[1]), value=1.0) for img in images]

    stacked_images = torch.stack(images)
    num_sequence = len(images) // batch_size
    final_tensor = stacked_images.view(batch_size, num_sequence, 3, stacked_images.shape[-2], stacked_images.shape[-1])

    return final_tensor

def preprocess_image_tensors(image_tensor_list, mode="crop"):
    if len(image_tensor_list) == 0:
        raise ValueError("At least 1 image tensor is required")

    if mode not in ["crop", "pad"]:
        raise ValueError("Mode must be either 'crop' or 'pad'")

    processed_images = []
    shapes = set()
    target_size = 518

    for img in tqdm(image_tensor_list, desc="Preprocessing Images", leave=True):
        # img shape: (C, H, W)
        C, height, width = img.shape

        if mode == "pad":
            if width >= height:
                new_width = target_size
                new_height = round(height * (new_width / width) / 14) * 14
            else:
                new_height = target_size
                new_width = round(width * (new_height / height) / 14) * 14
        else:  # crop
            new_width = target_size
            new_height = round(height * (new_width / width) / 14) * 14

        img = torch.nn.functional.interpolate(img.unsqueeze(0), size=(new_height, new_width), 
                            mode='bicubic', align_corners=False).squeeze(0)

        if mode == "crop" and new_height > target_size:
            start_y = (new_height - target_size) // 2
            img = img[:, start_y : start_y + target_size, :]

        if mode == "pad":
            h_padding = target_size - img.shape[1]
            w_padding = target_size - img.shape[2]
            if h_padding > 0 or w_padding > 0:
                pad_t, pad_l = h_padding // 2, w_padding // 2
                pad_b, pad_r = h_padding - pad_t, w_padding - pad_l
                img = F.pad(img, (pad_l, pad_r, pad_t, pad_b), mode="constant", value=1.0)

        shapes.add((img.shape[1], img.shape[2]))
        processed_images.append(img)

    if len(shapes) > 1:
        max_h = max(s[0] for s in shapes)
        max_w = max(s[1] for s in shapes)
        final_images = []
        for img in processed_images:
            h_pad = max_h - img.shape[1]
            w_pad = max_w - img.shape[2]
            if h_pad > 0 or w_pad > 0:
                img = F.pad(img, (w_pad//2, w_pad-w_pad//2, h_pad//2, h_pad-h_pad//2), value=1.0)
            final_images.append(img)
        processed_images = final_images

    return torch.stack(processed_images)

def load_clean_deg_images(image_path_list, kernel=None, mode="crop"):
    if len(image_path_list) == 0:
        raise ValueError("At least 1 image is required")

    clean_images = []
    deg_images = []
    target_size = 518

    for image_path in image_path_list:
        img = Image.open(image_path)
        
        if img.mode == "RGBA":
            background = Image.new("RGBA", img.size, (255, 255, 255, 255))
            img = Image.alpha_composite(background, img)
        img = img.convert("RGB")

        width, height = img.size
        new_width = target_size
        new_height = round(height * (new_width / width) / 14) * 14

        img_resized = img.resize((new_width, new_height), Image.Resampling.BICUBIC)

        img_deg = img_resized.copy()
        if kernel is not None:
            img_deg = kernel.applyTo(img_deg, keep_image_dim=True)

        def finalize_img(target_img):
            t = T.functional.to_tensor(target_img)
            
            if mode == "crop" and new_height > target_size:
                start_y = (new_height - target_size) // 2
                t = t[:, start_y : start_y + target_size, :]
            
            if mode == "pad":
                h_pad = target_size - t.shape[1]
                w_pad = target_size - t.shape[2]
                if h_pad > 0 or w_pad > 0:
                    t = torch.nn.functional.pad(
                        t, (w_pad//2, w_pad - w_pad//2, h_pad//2, h_pad - h_pad//2), 
                        mode="constant", value=1.0
                    )
            return t

        clean_images.append(finalize_img(img_resized))
        deg_images.append(finalize_img(img_deg))

    clean_batch = torch.stack(clean_images)
    deg_batch = torch.stack(deg_images)

    return clean_batch, deg_batch

### Single Image Processing Functions ###
def process_single_hdf5(image_path, target_size=518, mode="crop"):
    gamma, inv_gamma = 1.0 / 2.2, 2.2
    percentile = 90
    brightness_desired = 0.8
    eps = 0.0001

    with h5py.File(image_path, "r") as f:
        rgb_color = f["dataset"][:].astype(np.float32) # [H, W, 3]

    entity_path = image_path.replace("final_hdf5", "geometry_hdf5").replace(".color.hdf5", ".render_entity_id.hdf5")
    if os.path.exists(entity_path):
        with h5py.File(entity_path, "r") as f:
            valid_mask = f["dataset"][:] != -1
    else:
        valid_mask = np.ones(rgb_color.shape[:2], dtype=bool)

    brightness = 0.3 * rgb_color[..., 0] + 0.59 * rgb_color[..., 1] + 0.11 * rgb_color[..., 2]
    brightness_valid = brightness[valid_mask]
    
    if len(brightness_valid) == 0 or np.percentile(brightness_valid, percentile) < eps:
        scale = 1.0
    else:
        current_p = np.percentile(brightness_valid, percentile)
        scale = np.power(brightness_desired, inv_gamma) / current_p

    rgb_tm = np.power(np.maximum(scale * rgb_color, 0), gamma)
    rgb_tm = np.clip(rgb_tm, 0, 1)

    img_tensor = torch.from_numpy(rgb_tm).permute(2, 0, 1) # [3, H, W]
    
    _, h, w = img_tensor.shape
    new_width = target_size
    new_height = round(h * (new_width / w) / 14) * 14
    
    img_tensor = TF.functional.resize(img_tensor, [new_height, new_width], 
                                      interpolation=T.InterpolationMode.BICUBIC)

    if mode == "crop" and new_height > target_size:
        start_y = (new_height - target_size) // 2
        img_tensor = img_tensor[:, start_y : start_y + target_size, :]
    elif mode == "pad" and (img_tensor.shape[1] < target_size or img_tensor.shape[2] < target_size):
        h_pad = target_size - img_tensor.shape[1]
        w_pad = target_size - img_tensor.shape[2]
        img_tensor = F.pad(img_tensor, (w_pad//2, w_pad-w_pad//2, h_pad//2, h_pad-h_pad//2), value=1.0)

    return img_tensor # [3, target_size, target_size]

def process_clean_deg_tensors_from_hdf5(image_path, kernel=None, target_size=518, mode="crop"):
    gamma, inv_gamma = 1.0 / 2.2, 2.2
    percentile = 90
    brightness_desired = 0.8
    eps = 0.0001

    with h5py.File(image_path, "r") as f:
        rgb_color = f["dataset"][:].astype(np.float32) # [H, W, 3]

    entity_path = image_path.replace("final_hdf5", "geometry_hdf5").replace(".color.hdf5", ".render_entity_id.hdf5")
    if os.path.exists(entity_path):
        with h5py.File(entity_path, "r") as f:
            valid_mask = f["dataset"][:] != -1
    else:
        valid_mask = np.ones(rgb_color.shape[:2], dtype=bool)

    brightness = 0.3 * rgb_color[..., 0] + 0.59 * rgb_color[..., 1] + 0.11 * rgb_color[..., 2]
    brightness_valid = brightness[valid_mask]
    
    if len(brightness_valid) == 0 or np.percentile(brightness_valid, percentile) < eps:
        scale = 1.0
    else:
        current_p = np.percentile(brightness_valid, percentile)
        scale = np.power(brightness_desired, inv_gamma) / current_p

    rgb_tm = np.power(np.maximum(scale * rgb_color, 0), gamma)
    rgb_tm = np.clip(rgb_tm, 0, 1)

    clean_tensor = torch.from_numpy(rgb_tm).permute(2, 0, 1) 

    if kernel is not None:
        img_pil = T.functional.to_pil_image(clean_tensor)
        img_pil_deg = kernel.applyTo(img_pil, keep_image_dim=True)
        deg_tensor = T.functional.to_tensor(img_pil_deg)
    else:
        deg_tensor = clean_tensor.clone()

    _, h, w = clean_tensor.shape
    new_width = target_size
    new_height = round(h * (new_width / w) / 14) * 14
    
    combined = torch.stack([clean_tensor, deg_tensor]) # [2, 3, H, W]
    combined = T.functional.resize(combined, [new_height, new_width], 
                         interpolation=T.InterpolationMode.BICUBIC, antialias=True)

    if mode == "crop" and new_height > target_size:
        start_y = (new_height - target_size) // 2
        combined = combined[:, :, start_y : start_y + target_size, :]
        
    elif mode == "pad" and (combined.shape[2] < target_size or combined.shape[3] < target_size):
        h_pad = target_size - combined.shape[2]
        w_pad = target_size - combined.shape[3]
        combined = F.pad(combined, (w_pad//2, w_pad-w_pad//2, h_pad//2, h_pad-h_pad//2), value=1.0)

    final_clean = combined[0]
    final_deg = combined[1]

    return final_clean, final_deg

def process_single_hdf5_dit4sr(image_path, target_size=512, mode="crop"):
    gamma, inv_gamma = 1.0 / 2.2, 2.2
    percentile = 90
    brightness_desired = 0.8
    eps = 0.0001

    with h5py.File(image_path, "r") as f:
        rgb_color = f["dataset"][:].astype(np.float32)

    entity_path = image_path.replace("final_hdf5", "geometry_hdf5").replace(".color.hdf5", ".render_entity_id.hdf5")
    if os.path.exists(entity_path):
        with h5py.File(entity_path, "r") as f:
            valid_mask = f["dataset"][:] != -1
    else:
        valid_mask = np.ones(rgb_color.shape[:2], dtype=bool)

    brightness = 0.3 * rgb_color[..., 0] + 0.59 * rgb_color[..., 1] + 0.11 * rgb_color[..., 2]
    brightness_valid = brightness[valid_mask]
    
    if len(brightness_valid) == 0 or np.percentile(brightness_valid, percentile) < eps:
        scale = 1.0
    else:
        current_p = np.percentile(brightness_valid, percentile)
        scale = np.power(brightness_desired, inv_gamma) / current_p

    rgb_tm = np.power(np.maximum(scale * rgb_color, 0), gamma)
    rgb_tm = np.clip(rgb_tm, 0, 1)

    img_tensor = torch.from_numpy(rgb_tm).permute(2, 0, 1)
    
    _, h, w = img_tensor.shape

    scale_factor = target_size / max(h, w)
    new_h, new_w = int(h * scale_factor), int(w * scale_factor)
    
    new_h = (new_h // 8) * 8
    new_w = (new_w // 8) * 8
    
    img_tensor = TF.functional.resize(img_tensor, [new_h, new_w], 
                           interpolation=T.InterpolationMode.BICUBIC, antialias=True)

    if mode == "crop":
        h_pad = target_size - new_h
        w_pad = target_size - new_w
        
        img_tensor = F.pad(img_tensor, (w_pad//2, w_pad - w_pad//2, h_pad//2, h_pad - h_pad//2), value=0.0)

    return img_tensor

def process_single_depth_hdf5(depth_path, target_size=518, mode="crop"):
    W_orig, H_orig, FOCAL = 1024, 768, 886.81

    with h5py.File(depth_path, "r") as f:
        dist = f["dataset"][:].astype(np.float32)
    
    x = np.linspace(-W_orig / 2 + 0.5, W_orig / 2 - 0.5, W_orig)
    y = np.linspace(-H_orig / 2 + 0.5, H_orig / 2 - 0.5, H_orig)
    X, Y = np.meshgrid(x, y)
    Z = np.full_like(X, FOCAL)
    plane_norm = np.sqrt(X**2 + Y**2 + Z**2)
    
    depth_np = (dist / plane_norm * FOCAL)
    depth_np = np.nan_to_num(depth_np, nan=0.0, posinf=0.0, neginf=0.0)
    
    depth_tensor = torch.from_numpy(depth_np).unsqueeze(0)

    h, w = depth_tensor.shape[1], depth_tensor.shape[2]
    new_width = target_size
    new_height = round(h * (new_width / w) / 14) * 14

    depth_tensor = TF.functional.resize(depth_tensor, [new_height, new_width], 
                                        interpolation=T.InterpolationMode.NEAREST)

    if mode == "crop" and new_height > target_size:
        start_y = (new_height - target_size) // 2
        depth_tensor = depth_tensor[:, start_y : start_y + target_size, :]
    elif mode == "pad":
        h_pad = target_size - depth_tensor.shape[1]
        w_pad = target_size - depth_tensor.shape[2]
        depth_tensor = F.pad(depth_tensor, (w_pad//2, w_pad-w_pad//2, h_pad//2, h_pad-h_pad//2), value=0.0)

    return depth_tensor # [1, target_size, target_size]

def process_single_depth_npy(depth_path, target_size=518, mode="crop"):
    FOCAL = 886.81

    try:
        dist = np.load(depth_path).astype(np.float32)
    except Exception as e:
        print(f"Error loading {depth_path}: {e}")
        return None
    
    H_orig, W_orig = dist.shape

    x = np.linspace(-W_orig / 2 + 0.5, W_orig / 2 - 0.5, W_orig)
    y = np.linspace(-H_orig / 2 + 0.5, H_orig / 2 - 0.5, H_orig)
    X, Y = np.meshgrid(x, y)
    Z = np.full_like(X, FOCAL)
    plane_norm = np.sqrt(X**2 + Y**2 + Z**2)
    
    depth_np = (dist / plane_norm * FOCAL)
    depth_np = np.nan_to_num(depth_np, nan=0.0, posinf=0.0, neginf=0.0)
    
    depth_tensor = torch.from_numpy(depth_np).unsqueeze(0) # [1, H, W]

    h, w = depth_tensor.shape[1], depth_tensor.shape[2]
    new_width = target_size
    new_height = round(h * (new_width / w) / 14) * 14
    
    depth_tensor = TF.functional.resize(depth_tensor, [new_height, new_width], 
                             interpolation=T.InterpolationMode.NEAREST)

    if mode == "crop" and new_height > target_size:
        start_y = (new_height - target_size) // 2
        depth_tensor = depth_tensor[:, start_y : start_y + target_size, :]
    elif mode == "pad":
        h_curr, w_curr = depth_tensor.shape[1], depth_tensor.shape[2]
        h_pad = max(0, target_size - h_curr)
        w_pad = max(0, target_size - w_curr)
        depth_tensor = F.pad(depth_tensor, (w_pad//2, w_pad - w_pad//2, 
                                            h_pad//2, h_pad - h_pad//2), value=0.0)

    return depth_tensor # [1, target_size, target_size]

def process_eth3d_depth_bin(depth_path, target_size=518, mode="crop"):
    H_orig, W_orig = 4032, 6048
    try:
        depth = np.fromfile(depth_path, dtype=np.float32)
        if depth.size != H_orig * W_orig:
            print(f"Size mismatch at {depth_path}: expected {H_orig*W_orig}, got {depth.size}")
            return None
            
        depth = depth.reshape(H_orig, W_orig)
        depth[np.isinf(depth)] = 0.0 
        depth = np.nan_to_num(depth, nan=0.0)
    except Exception as e:
        print(f"Error loading ETH3D depth {depth_path}: {e}")
        return None

    depth_tensor = torch.from_numpy(depth).unsqueeze(0)

    h, w = depth_tensor.shape[1], depth_tensor.shape[2]
    new_width = target_size
    new_height = round(h * (new_width / w) / 14) * 14
    
    depth_tensor = TF.functional.resize(depth_tensor, [new_height, new_width], 
                             interpolation=T.InterpolationMode.NEAREST)

    if mode == "crop" and new_height > target_size:
        start_y = (new_height - target_size) // 2
        depth_tensor = depth_tensor[:, start_y : start_y + target_size, :]
    elif mode == "pad":
        h_curr, w_curr = depth_tensor.shape[1], depth_tensor.shape[2]
        h_pad = max(0, target_size - h_curr)
        w_pad = max(0, target_size - w_curr)
        depth_tensor = F.pad(depth_tensor, (w_pad//2, w_pad - w_pad//2, 
                                            h_pad//2, h_pad - h_pad//2), value=0.0)

    return depth_tensor # [1, target_size, target_size]

def process_single_image(image_path, target_size=518, mode="crop"):
    img = Image.open(image_path)

    if img.mode == "RGBA":
        background = Image.new("RGBA", img.size, (255, 255, 255, 255))
        img = Image.alpha_composite(background, img).convert("RGB")
    else:
        img = img.convert("RGB")

    width, height = img.size

    new_width = target_size
    new_height = round(height * (new_width / width) / 14) * 14

    img = img.resize((new_width, new_height), Image.Resampling.BICUBIC)
    img_tensor = T.functional.to_tensor(img)  # [3, H, W]

    if mode == "crop" and new_height > target_size:
        start_y = (new_height - target_size) // 2
        img_tensor = img_tensor[:, start_y : start_y + target_size, :]
    elif mode == "pad" and (img_tensor.shape[1] < target_size or img_tensor.shape[2] < target_size):
        h_pad = target_size - img_tensor.shape[1]
        w_pad = target_size - img_tensor.shape[2]
        img_tensor = torch.nn.functional.pad(
            img_tensor, (w_pad//2, w_pad-w_pad//2, h_pad//2, h_pad-h_pad//2), value=0.0
        )

    return img_tensor

def process_clean_deg_tensors_from_image(image_path, kernel=None, target_size=518, mode="crop"):
    img = Image.open(image_path)

    if img.mode == "RGBA":
        background = Image.new("RGBA", img.size, (255, 255, 255, 255))
        img = Image.alpha_composite(background, img).convert("RGB")
    else:
        img = img.convert("RGB")

    clean_img = img
    if kernel is not None:
        deg_img = kernel.applyTo(clean_img, keep_image_dim=True)
    else:
        deg_img = clean_img.copy()

    clean_tensor = T.functional.to_tensor(clean_img)
    deg_tensor = T.functional.to_tensor(deg_img)

    _, h, w = clean_tensor.shape
    new_width = target_size
    new_height = round(h * (new_width / w) / 14) * 14
    
    combined = torch.stack([clean_tensor, deg_tensor]) # [2, 3, H, W]
    combined = T.functional.resize(combined, [new_height, new_width], 
                         interpolation=T.InterpolationMode.BICUBIC, antialias=True)

    if mode == "crop" and new_height > target_size:
        start_y = (new_height - target_size) // 2
        combined = combined[:, :, start_y : start_y + target_size, :]
        
    elif mode == "pad" and (combined.shape[2] < target_size or combined.shape[3] < target_size):
        h_pad = target_size - combined.shape[2]
        w_pad = target_size - combined.shape[3]
        combined = F.pad(combined, (w_pad//2, w_pad-w_pad//2, h_pad//2, h_pad-h_pad//2), value=0.0)

    final_clean = combined[0]
    final_deg = combined[1]

    return final_clean, final_deg

def process_degraded_from_tensor(clean_tensor, kernel):
    # clean_tensor shape: (3, H, W) with values in [0, 1] to be converted to PIL Image
    img_pil = T.functional.to_pil_image(clean_tensor)

    if kernel is not None:
        img_pil = kernel.applyTo(img_pil, keep_image_dim=True)

    deg_tensor = T.functional.to_tensor(img_pil)
    
    return deg_tensor