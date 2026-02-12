import torch
import glob
import os
import sys
from tqdm import tqdm
from torchvision.transforms.functional import to_tensor

motionblur_path = "/mnt/dataset1/jaeeun/MVR"
if motionblur_path not in sys.path:
    sys.path.append(motionblur_path)

from motionblur.motionblur import Kernel 
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import preprocess_image_tensors

BLUR_INTENSITY = 0.1
BATCH_SIZE = 4 
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)
model.eval()

dataset_dir = "/mnt/dataset1/MV_Restoration/tartanair/data/"
search_pattern = os.path.join(dataset_dir, "*", "*", "*", "image_left", "*.png")
image_paths = sorted(glob.glob(search_pattern))

print(f"Total TartanAir Image Paths: {len(image_paths)}")

for KERNEL_SIZE in [400, 600, 800]:
    print(f'\n--- Starting TartanAir Process with Kernel Size: {KERNEL_SIZE} ---')
    
    save_dir = f"/mnt/dataset1/MV_Restoration/vggt_outputs/tartanair/deg_image/kernel_{KERNEL_SIZE}"
    os.makedirs(save_dir, exist_ok=True)
    
    for i in tqdm(range(0, len(image_paths), BATCH_SIZE), desc=f"Kernel {KERNEL_SIZE}"):
        batch_paths = image_paths[i : i + BATCH_SIZE]
        current_blur_tensors = []

        kernel = Kernel(size=(KERNEL_SIZE, KERNEL_SIZE), intensity=BLUR_INTENSITY)
        for path in batch_paths:
            blurred_pil = kernel.applyTo(path, keep_image_dim=True)
            current_blur_tensors.append(to_tensor(blurred_pil))
        
        input_tensor = preprocess_image_tensors(current_blur_tensors).to(device)

        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=dtype):
                predictions = model(input_tensor, image_names=batch_paths, latent_save_dir=f"/mnt/dataset1/MV_Restoration/tartanair/deg_image/kernel_{KERNEL_SIZE}")

        for j, path in enumerate(batch_paths):
            parts = path.split('/')
            env_name = parts[-5]      # abandonedfactory
            difficulty = parts[-4]    # Easy
            seq_id = parts[-3]        # P000
            file_name = parts[-1].replace('.png', '') # 000000_left
            
            save_name = f"{env_name}_{difficulty}_{seq_id}_{file_name}.pt"
            save_path = os.path.join(save_dir, save_name)
            
            torch.save({
                'pose_enc': predictions['pose_enc'][0, j].cpu(),
                'depth': predictions['depth'][0, j].cpu(),
                'world_points': predictions['world_points'][0, j].cpu(),
                'depth_conf': predictions['depth_conf'][0, j].cpu()
            }, save_path)

        del current_blur_tensors, input_tensor, predictions
        if i % 100 == 0:
            torch.cuda.empty_cache()

print("All TartanAir processes are finished.")