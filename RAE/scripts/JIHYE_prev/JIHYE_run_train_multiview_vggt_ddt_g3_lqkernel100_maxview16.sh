NUM_GPUS=2
CUDA=4,5

export CUDA=${CUDA}
export PYTHONPATH=$PWD
export PATH="$CONDA_PREFIX/bin:$PATH"
export PYTHONBREAKPOINT=0

CUDA_VISIBLE_DEVICES=${CUDA} python -m torch.distributed.run --standalone --nproc_per_node=${NUM_GPUS} \
    RAE/src/je_mvrm_JIHYE.py --config RAE/configs/JIHYE/JIHYE_run_train_multiview_vggt_ddt_g3.yaml \
    --image-size 256 \
    --precision fp32 \
    --max-view 16 \
    --kernel-size 100 \
    --global-batch-size 4
