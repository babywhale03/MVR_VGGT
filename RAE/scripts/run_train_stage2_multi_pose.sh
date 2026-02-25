export WANDB_KEY="a44758ca07858594065872c389df0962658144f6"
export ENTITY="whale03-org"
export PROJECT="mv_restoration"
export NUM_GPUS=1
export CUDA=1
export PYTHONPATH=$PWD
export PATH="$CONDA_PREFIX/bin:$PATH"
# export PYTHONBREAKPOINT=0

CUDA_VISIBLE_DEVICES=${CUDA} python -m torch.distributed.run --standalone --nproc_per_node=${NUM_GPUS} \
    RAE/src/je_mvrm_val_pose.py --config /mnt/dataset1/jaeeun/MVR_vggt/RAE/configs/mvrm/DiTDH-XL_DINOv2-B_Multi_co3d.yaml \
    --results-dir RAE/ckpts/ \
    --image-size 256 \
    --max-views 4 \
    --kernel-size 200 \
    --precision fp32 --wandb