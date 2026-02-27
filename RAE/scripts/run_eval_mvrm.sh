export CUDA=6
export PYTHONPATH=$PWD
export PATH="$CONDA_PREFIX/bin:$PATH"

CUDA_VISIBLE_DEVICES=${CUDA} python RAE/src/eval/eval_eth3d_script.py \
    --config /mnt/dataset1/jaeeun/MVR_vggt/RAE/configs/eval/JIHYE_run_train_multiview_vggt_ddt_g3.yaml \
    --ckpt /media/data1/MV_Restoration/JIHYE2_CKPT/kernel100_14000.pt \
    --results-dir /mnt/dataset1/jaeeun/MVR_vggt/eval_results \
    --num-sequences 10