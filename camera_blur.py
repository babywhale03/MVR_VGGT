import os 
import sys
sys.path.append(os.getcwd())
from motionblur.motionblur import Kernel 
import glob 
from torchvision.transforms.functional import to_tensor
from torchvision.utils import save_image 
from tqdm import tqdm


# KERNEL_SIZE=100
BLUR_INTENSITY=0.1

for KERNEL_SIZE in [400, 600, 800]:

    deg_save_dir = f'/mnt/dataset1/MV_Restoration/ECCV26_RESULTS/eth3d/deg_blur/kernel{KERNEL_SIZE}_intensity01'
    os.makedirs(deg_save_dir, exist_ok=True)

    scenes = ['courtyard', 'delivery_area', 'electro', 'facade', 'kicker','meadow', 'office', 'pipes', 'playground', 'relief','relief_2', 'terrace', 'terrains']
    for scene in tqdm(scenes, desc='Scenes'):
        print(f'Applying blur, Processing scene: {scene}')
        os.makedirs(f'{deg_save_dir}/blur_kernel/{scene}', exist_ok=True)
        os.makedirs(f'{deg_save_dir}/blur_img/{scene}', exist_ok=True)

        rgb_paths = sorted(glob.glob(f'/mnt/dataset1/MV_Restoration/eth3d/eth3d_jisang_train/rgb/{scene}/images/dslr_images/*'))
        
        for rgb_path in rgb_paths:
            
            img_id = rgb_path.split('/')[-1].split('.')[0]
        
            # init kernel
            kernel = Kernel(size=(KERNEL_SIZE, KERNEL_SIZE), intensity=BLUR_INTENSITY)

            # Save kernel as image. (Do not show kernel, just save.)
            kernel.displayKernel(save_to=f"{deg_save_dir}/blur_kernel/{scene}/{img_id}.png", show=False)
            
            blurred = kernel.applyTo(rgb_path, keep_image_dim=True)
            blurred.save(f'{deg_save_dir}/blur_img/{scene}/{img_id}.png')
            