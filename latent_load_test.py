import torch
import os

# file_path = "/mnt/dataset1/MV_Restoration/hypersim/vggt_deg_latent/kernel_200/ai_001_001/scene_cam_00_final_hdf5/frame_0000/latent_block_04_deg.pt"
file_path = "/mnt/dataset1/MV_Restoration/hypersim/vggt_clean_latent/input_singleview/ai_001_001/scene_cam_00_final_hdf5/frame_0000/latent_block_04.pt" 
# file_path = "/mnt/dataset1/MV_Restoration/hypersim/da3_clean_latent/input_singleview/ai_001_001/scene_cam_00/0000.pt"

def check_latent_tensor(path):
    if not os.path.exists(path):
        print(f"Error: 파일을 찾을 수 없습니다 -> {path}")
        return

    try:
        data = torch.load(path, map_location="cpu")
        
        print("-" * 50)
        print(f"파일 이름: {os.path.basename(path)}")
        
        # 1. 데이터 타입 확인
        if isinstance(data, torch.Tensor):
            print(f"데이터 타입: torch.Tensor")
            # 2. 차원(Shape) 확인
            print(f"텐서 크기 (Shape): {list(data.shape)}") # [1, 1041, 2048]
            # 3. 상세 정보
            print(f"데이터 형식 (dtype): {data.dtype}")
            print(f"장치 (Device): {data.device}")
            print(f"최댓값: {data.max().item():.4f}")
            print(f"최솟값: {data.min().item():.4f}")
            print(f"평균값: {data.mean().item():.4f}")
            
        elif isinstance(data, dict):
            print(f"데이터 타입: Dictionary")
            print(f"포함된 Keys: {list(data.keys())}")
            for k, v in data.items():
                if isinstance(v, torch.Tensor):
                    print(f"  - Key '{k}': Tensor Shape {list(v.shape)}")
        else:
            print(f"알 수 없는 데이터 타입: {type(data)}")
            
        print("-" * 50)

    except Exception as e:
        print(f"로드 중 에러 발생: {e}")

if __name__ == "__main__":
    check_latent_tensor(file_path)