## 📦 Installation 

### 1. Conda Environment

```bash
conda create -n mvrvggt python=3.10 -y
conda activate mvrvggt
```

### 2. Library Installation
```bash
cd MVR_VGGT
pip install torch==2.8.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu129 
pip install -r requirements.txt
```

## 🔥 Training 

### 1. Training bash script
```bash
# CUDA=0
bash RAE/scripts/JIHYE/JIHYE_run_train_multiview_vggt_ddt_g3_lqkernel100_maxview1.sh

# CUDA=1
bash RAE/scripts/JIHYE/JIHYE_run_train_multiview_vggt_ddt_g3_lqkernel100_maxview2.sh

# CUDA=2,3
bash RAE/scripts/JIHYE/JIHYE_run_train_multiview_vggt_ddt_g3_lqkernel100_maxview8.sh

# CUDA=4,5
bash RAE/scripts/JIHYE/JIHYE_run_train_multiview_vggt_ddt_g3_lqkernel100_maxview16.sh

# CUDA=6
bash RAE/scripts/JIHYE/JIHYE_run_train_multiview_vggt_ddt_g3_lqkernel200_maxview1.sh

# CUDA=7
bash RAE/scripts/JIHYE/JIHYE_run_train_multiview_vggt_ddt_g3_lqkernel200_maxview2.sh

# CUDA=8,9
bash RAE/scripts/JIHYE/JIHYE_run_train_multiview_vggt_ddt_g3_lqkernel200_maxview8.sh

# CUDA=10,11
bash RAE/scripts/JIHYE/JIHYE_run_train_multiview_vggt_ddt_g3_lqkernel200_maxview16.sh
```

### 2. Training config yaml file : [MVR_VGGT](RAE/configs/JIHYE/JIHYE_run_train_multiview_vggt_ddt_g3.yaml)