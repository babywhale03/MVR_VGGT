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
# CUDA=0,1,2,3,4,5,6,7
bash RAE/scripts/JIHYE/JIHYE_run_train_multiview_vggt_ddt_g12_lqkernel100_maxview8.sh
```

### 2. Training config yaml file : [MVR_VGGT (g12)](RAE/configs/JIHYE/JIHYE_run_train_multiview_vggt_ddt_g12.yaml)