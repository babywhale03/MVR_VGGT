import time
import torch
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images

device = "cuda" if torch.cuda.is_available() else "cpu"
# bfloat16 is supported on Ampere GPUs (Compute Capability 8.0+) 
dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

# Initialize the model and load the pretrained weights.
# This will automatically download the model weights the first time it's run, which may take a while.
model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)

# Load and preprocess example images (replace with your own image paths)
image_names = ["/mnt/dataset1/jaeeun/MVR/vggt/examples/kitchen/images/00.png", "/mnt/dataset1/jaeeun/MVR/vggt/examples/kitchen/images/01.png", "/mnt/dataset1/jaeeun/MVR/vggt/examples/kitchen/images/02.png"]  
images = load_and_preprocess_images(image_names).to(device)

with torch.no_grad():
    with torch.cuda.amp.autocast(dtype=dtype):
        # Predict attributes including cameras, depth maps, and point maps.
        start_time = time.time()
        predictions = model(images)
        end_time = time.time()
        print(f"Inference time: {end_time - start_time} seconds")
        print("Predicted keys:", predictions.keys())
        print("Predicted depth shape:", predictions["depth"].shape)
        print("Predicted world points shape:", predictions["world_points"].shape)
        print("Predicted camera pose encoding shape:", predictions["pose_enc"].shape)
        print("Predicted depth confidence shape:", predictions["depth_conf"].shape)
        print("Predicted world points confidence shape:", predictions["world_points_conf"].shape)
        # print("Predicted track shape:", predictions["track"].shape)
        # print("Predicted visibility shape:", predictions["vis"].shape)
        # print("Predicted confidence shape:", predictions["conf"].shape)