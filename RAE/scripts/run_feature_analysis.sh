export CUDA=0
export PYTHONPATH=$PWD
export PATH="$CONDA_PREFIX/bin:$PATH"
export PYTHONBREAKPOINT=0

CUDA_VISIBLE_DEVICES=${CUDA} python RAE/src/eval/eval_feature_analysis.py \
    --config /mnt/dataset1/jaeeun/MVR_vggt/RAE/configs/eval/JIHYE_run_train_multiview_vggt_ddt_g3.yaml \
    --ckpt /media/data1/MV_Restoration/JIHYE2_CKPT/kernel100_14000.pt \
    --results-dir /mnt/dataset1/jaeeun/MVR_vggt/eval_results/final \
    --num-sequences 10 \
    --num-views 10 \
    --kernel-size 500 \
    --extract-layer 3