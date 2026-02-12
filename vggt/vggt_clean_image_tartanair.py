import torch
import glob
import os
from tqdm import tqdm
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)
model.eval() 

dataset_dir = "/mnt/dataset1/MV_Restoration/tartanair/data/"
save_dir = "/mnt/dataset1/MV_Restoration/vggt_outputs/tartanair/clean_image"
os.makedirs(save_dir, exist_ok=True)

search_pattern = os.path.join(dataset_dir, "*", "Easy", "*", "image_left", "*.png")
image_paths = sorted(glob.glob(search_pattern))

print(f"Total TartanAir Image Paths: {len(image_paths)}")

batch_size = 4 
with torch.no_grad():
    with torch.cuda.amp.autocast(dtype=dtype):
        for i in tqdm(range(0, len(image_paths), batch_size), desc="VGGT Inferencing (TartanAir)"):
            batch_input_paths = image_paths[i : i + batch_size]
            
            try:
                images = load_and_preprocess_images(batch_input_paths).to(device) 
            except Exception as e:
                print(f"\nError loading images at index {i}: {e}")
                continue

            predictions = model(images, image_names=batch_input_paths, latent_save_dir="/mnt/dataset1/MV_Restoration/tartanair/clean_latent")

            for j, path in enumerate(batch_input_paths):
                path_parts = path.split('/')
                
                env_name = path_parts[-5]      # abandonedfactory
                difficulty = path_parts[-4]    # Easy
                seq_id = path_parts[-3]        # P000
                file_name = path_parts[-1].replace('.png', '') # 000000_left
                
                output_filename = f"{env_name}_{difficulty}_{seq_id}_{file_name}.pt"
                save_path = os.path.join(save_dir, output_filename)
                
                torch.save({
                    'pose_enc': predictions['pose_enc'][0, j].cpu(),
                    'depth': predictions['depth'][0, j].cpu(),
                    'world_points': predictions['world_points'][0, j].cpu(),
                    'depth_conf': predictions['depth_conf'][0, j].cpu()
                }, save_path)

            del images, predictions
            if i % 100 == 0: 
                torch.cuda.empty_cache()

print(f"\nAll processing completed. Results saved to: {save_dir}")