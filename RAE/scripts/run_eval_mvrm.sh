export CUDA=1
export PYTHONPATH=$PWD
export PATH="$CONDA_PREFIX/bin:$PATH"
export PYTHONBREAKPOINT=0

CUDA_VISIBLE_DEVICES=${CUDA} python RAE/src/eval/eval_feature_similarity_final.py \
    --config /mnt/dataset1/jaeeun/MVR_vggt/RAE/configs/eval/JIHYE_run_train_multiview_vggt_ddt_g12.yaml \
    --ckpt /mnt/dataset1/jaeeun/MVR_vggt/RAE/ckpts/nvidia/step-0134000.pt \
    --results-dir /mnt/dataset1/jaeeun/MVR_vggt/eval_results/final \
    --num-sequences 10 \
    --num-views 10 \
    --kernel-size 500 \
    --eval-mode full \
    --extract-layer 12