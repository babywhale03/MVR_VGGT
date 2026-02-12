import h5py
import numpy as np
import cv2
import os

def save_hypersim_official_tonemap_png(hdf5_path, save_path):
    if not os.path.exists(hdf5_path):
        print(f"파일을 찾을 수 없습니다: {hdf5_path}")
        return

    try:
        with h5py.File(hdf5_path, 'r') as f:
            rgb_color = f['dataset'][:].astype(np.float32)
        
        entity_path = hdf5_path.replace("final_hdf5", "geometry_hdf5").replace(".color.hdf5", ".render_entity_id.hdf5")
        
        if os.path.exists(entity_path):
            with h5py.File(entity_path, 'r') as f:
                render_entity_id = f['dataset'][:].astype(np.int32)
            valid_mask = render_entity_id != -1
        else:
            print("공식 가이드: Entity ID 파일을 찾을 수 없어 전체 영역을 기준으로 계산합니다.")
            valid_mask = np.ones(rgb_color.shape[:2], dtype=bool)
            
    except Exception as e:
        print(f"로드 중 에러 발생: {e}")
        return

    gamma = 1.0 / 2.2
    inv_gamma = 1.0 / gamma
    percentile = 90  
    brightness_nth_percentile_desired = 0.8  
    eps = 0.0001

    brightness = 0.3 * rgb_color[:, :, 0] + 0.59 * rgb_color[:, :, 1] + 0.11 * rgb_color[:, :, 2]
    brightness_valid = brightness[valid_mask]

    if len(brightness_valid) == 0 or np.percentile(brightness_valid, percentile) < eps:
        scale = 1.0
    else:
        current_p = np.percentile(brightness_valid, percentile)
        scale = np.power(brightness_nth_percentile_desired, inv_gamma) / current_p

    rgb_tm = np.power(np.maximum(scale * rgb_color, 0), gamma)
    rgb_tm = np.clip(rgb_tm, 0, 1)

    img_8bit = (rgb_tm * 255).astype(np.uint8)
    img_bgr = cv2.cvtColor(img_8bit, cv2.COLOR_RGB2BGR)
    
    cv2.imwrite(save_path, img_bgr)
    print(f"이미지 저장 완료! (적용된 Scale: {scale:.4f})")
    print(f"저장 경로: {os.path.abspath(save_path)}")

hdf5_file_path = "/mnt/dataset1/MV_Restoration/hypersim/data/ai_001_001/images/scene_cam_00_final_hdf5/frame.0001.color.hdf5"
output_sample_path = "./frame_0001_official_tm.png"

save_hypersim_official_tonemap_png(hdf5_file_path, output_sample_path)