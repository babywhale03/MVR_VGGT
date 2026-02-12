import torch
import glob
import os
import sys
import re
from tqdm import tqdm
from torchvision.transforms.functional import to_tensor, to_pil_image

motionblur_path = "/mnt/dataset1/jaeeun/MVR"
if motionblur_path not in sys.path:
    sys.path.append(motionblur_path)

from motionblur.motionblur import Kernel 
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images, process_and_save_hdf5_images

BLUR_INTENSITY = 0.1
BATCH_SIZE = 8
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)
model.eval()

dataset_dir = "/mnt/dataset1/MV_Restoration/hypersim/data/"
search_pattern = os.path.join(dataset_dir, "ai_*", "images", "scene_cam_*_final_hdf5", "frame.*.color.hdf5")
all_paths = sorted(glob.glob(search_pattern))

image_paths = []
for p in all_paths:
    match = re.search(r'ai_(\d{3})', p)
    if match and int(match.group(1)) <= 10:
        image_paths.append(p)

print(f"Total Filtered Hypersim Image Paths (up to ai_010): {len(image_paths)}")

for KERNEL_SIZE in [50, 100, 200]:
    print(f'\n--- Starting Hypersim Process with Kernel Size: {KERNEL_SIZE} ---')
    
    save_dir = f"/mnt/dataset1/MV_Restoration/vggt_outputs/hypersim/deg_image/kernel_{KERNEL_SIZE}"
    latent_dir = f"/mnt/dataset1/MV_Restoration/hypersim/vggt_deg_latent/input_singleview/kernel_{KERNEL_SIZE}"
    save_deg_dir = f"/mnt/dataset1/MV_Restoration/hypersim/deg_data/kernel_{KERNEL_SIZE}"
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(latent_dir, exist_ok=True)
    os.makedirs(save_deg_dir, exist_ok=True)
    
    first_image_saved = False
    kernel = Kernel(size=(KERNEL_SIZE, KERNEL_SIZE), intensity=BLUR_INTENSITY)

    # convert hdf5 images to degraded png images & save
    process_and_save_hdf5_images(image_paths, save_deg_dir, kernel, crop=False)
    # breakpoint()
    search_deg_pattern = os.path.join(save_deg_dir, "**", "*.png")
    all_deg_paths = sorted(glob.glob(search_deg_pattern, recursive=True))

    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=True, dtype=dtype):
            for i in tqdm(range(0, len(all_deg_paths), BATCH_SIZE), desc="VGGT Inferencing"):
                batch_input_paths = all_deg_paths[i : i + BATCH_SIZE]
                try:
                    images = load_and_preprocess_images(batch_input_paths).to(device).to(dtype)
                except Exception as e:
                    print(f"\nError loading images at index {i}: {e}")
                    continue

                predictions = model(images, image_names=batch_input_paths, latent_save_dir=latent_dir)

                if not first_image_saved:
                    import torchvision.utils as vutils
                    save_test_path = "./first_frame_check.png"
                    vutils.save_image(images[0].float().cpu(), save_test_path) 
                    print(f"\nExample Image saved at: {save_test_path}")
                    first_image_saved = True

                # for j, path in enumerate(batch_input_paths):
                #     path_parts = path.split('/')
                #     # breakpoint()
                #     scene_id = path_parts[-4] # ai_XXX_XXX
                #     cam_id = path_parts[-2]   # scene_cam_XX_final_hdf5
                #     frame_token = path_parts[-1].split('.')
                #     frame_name = f"{frame_token[0]}_{frame_token[1]}" # frame_0000
                    
                #     output_filename = f"{scene_id}_{cam_id}_{frame_name}_deg.pt"
                #     save_path = os.path.join(save_dir, output_filename)
                    
                #     torch.save({
                #         'pose_enc': predictions['pose_enc'][0, j].cpu(),
                #         'depth': predictions['depth'][0, j].cpu(),
                #         'world_points': predictions['world_points'][0, j].cpu(),
                #         'depth_conf': predictions['depth_conf'][0, j].cpu()
                #     }, save_path)

                del images, predictions
                if i % 100 == 0:
                    torch.cuda.empty_cache()

print("All Hypersim processes (up to ai_010) are finished.")