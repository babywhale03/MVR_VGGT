import torch
import glob
import os
import sys
import re
from tqdm import tqdm
# from torchvision.transforms.functional import to_tensor, to_pil_image

motionblur_path = "/mnt/dataset1/jaeeun/MVR"
if motionblur_path not in sys.path:
    sys.path.append(motionblur_path)

from motionblur.motionblur import Kernel 
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images, process_and_save_hdf5_images

KERNEL_SIZE = 50
BATCH_SIZE = 8
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)
model.eval()

deg_img_dir = "/mnt/dataset1/MV_Restoration/hypersim/deg_blur/kernel50_intensity01"
clean_latent_dir = "/mnt/dataset1/MV_Restoration/hypersim/vggt_clean_latent/input_singleview"

search_pattern = os.path.join(deg_img_dir, "ai_*", "scene_cam_*_final_hdf5", "frame_*_color.png")
all_deg_paths = sorted(glob.glob(search_pattern))

with torch.no_grad():
    with torch.cuda.amp.autocast(enabled=True, dtype=dtype):
        for i in tqdm(range(0, len(all_deg_paths), BATCH_SIZE), desc="VGGT Inferencing"):
            batch_input_paths = all_deg_paths[i : i + BATCH_SIZE]
            try:
                images = load_and_preprocess_images(batch_input_paths).to(device).to(dtype)
            except Exception as e:
                print(f"\nError loading images at index {i}: {e}")
                continue

            predictions = model(images, image_names=batch_input_paths)

            del images, predictions
            if i % 100 == 0:
                torch.cuda.empty_cache()

print("All Hypersim processes (up to ai_010) are finished.")