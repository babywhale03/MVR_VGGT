NUM_GPUS=1
CUDA=1

export CUDA=${CUDA}
export PYTHONPATH=$PWD
export PATH="$CONDA_PREFIX/bin:$PATH"
export PYTHONBREAKPOINT=0

CUDA_VISIBLE_DEVICES=${CUDA} python -m torch.distributed.run --standalone --nproc_per_node=${NUM_GPUS} \
    RAE/src/je_mvrm_JIHYE.py --config /mnt/dataset1/jaeeun/MVR_vggt/RAE/configs/JIHYE/JIHYE_run_train_multiview_vggt_ddt_g3.yaml \
    --image-size 256 \
    --precision fp32 \
    --max-view 2 \
    --kernel-size 100
