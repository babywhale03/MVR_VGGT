import torch
import glob
import os
import re 
from tqdm import tqdm
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images_hdf5

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)
model.eval()

dataset_dir = "/mnt/dataset1/MV_Restoration/hypersim/data/"
save_dir = "/mnt/dataset1/MV_Restoration/vggt_outputs/hypersim/clean_image"
latent_save_dir = "/mnt/dataset1/MV_Restoration/hypersim/vggt_clean_latent/input_singleview"
os.makedirs(save_dir, exist_ok=True)

search_pattern = os.path.join(dataset_dir, "ai_*", "images", "scene_cam_*_final_hdf5", "frame.*.color.hdf5")
all_image_paths = sorted(glob.glob(search_pattern))

image_paths = []
for p in all_image_paths:
    match = re.search(r'ai_(\d{3})', p)
    if match:
        scene_num = int(match.group(1)) 
        if scene_num <= 10: 
            image_paths.append(p)

print(f"Filtered Image Paths (up to ai_010): {len(image_paths)}")

batch_size = 8
first_image_saved = False 

with torch.no_grad():
    with torch.cuda.amp.autocast(enabled=True, dtype=dtype):
        for i in tqdm(range(0, len(image_paths), batch_size), desc="VGGT Inferencing"):
            batch_input_paths = image_paths[i : i + batch_size]
            try:
                images = load_and_preprocess_images_hdf5(batch_input_paths).to(device).to(dtype)
            except Exception as e:
                print(f"\nError loading images at index {i}: {e}")
                continue

            predictions = model(images, image_names=batch_input_paths, latent_save_dir=latent_save_dir)

            if not first_image_saved:
                import torchvision.utils as vutils
                save_test_path = "./first_frame_check.png"
                vutils.save_image(images[0].float().cpu(), save_test_path) 
                print(f"\nExample Image saved at: {save_test_path}")
                first_image_saved = True

            for j, path in enumerate(batch_input_paths):
                path_parts = path.split('/')
                # breakpoint()
                # scene_id = path_parts[-4] # ai_XXX_XXX
                # cam_id = path_parts[-2]   # scene_cam_XX_final_hdf5
                # frame_token = path_parts[-1].split('.')
                # frame_name = f"{frame_token[0]}_{frame_token[1]}" # frame_0000

                scene_id = path_parts[-3] # ai_XXX_XXX
                cam_id = path_parts[-2]   # scene_cam_XX_final_hdf5
                frame_name = path_parts[-1].split('.')[0]
                
                output_filename = f"{scene_id}_{cam_id}_{frame_name}.pt"
                save_path = os.path.join(save_dir, output_filename)
                
                # torch.save({
                #     'pose_enc': predictions['pose_enc'][0, j].cpu(),
                #     'depth': predictions['depth'][0, j].cpu(),
                #     'world_points': predictions['world_points'][0, j].cpu(),
                #     'depth_conf': predictions['depth_conf'][0, j].cpu()
                # }, save_path)

            del images, predictions
            if i % 100 == 0:
                torch.cuda.empty_cache()

print(f"\nAll processing (up to ai_010) completed. Results saved to: {save_dir}")