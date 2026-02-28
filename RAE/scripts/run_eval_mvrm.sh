export CUDA=5
export PYTHONPATH=$PWD
export PATH="$CONDA_PREFIX/bin:$PATH"

CUDA_VISIBLE_DEVICES=${CUDA} python RAE/src/eval/eval_feature_similarity_final.py \
    --config /mnt/dataset1/jaeeun/MVR_vggt/RAE/configs/eval/JIHYE_run_train_multiview_vggt_ddt_g3.yaml \
    --ckpt /media/data1/MV_Restoration/JIHYE2_CKPT/kernel150_14000.pt \
    --results-dir /mnt/dataset1/jaeeun/MVR_vggt/eval_results/final \
    --num-sequences 10 \
    --num-views 5 \
    --kernel-size 500 \
    --eval-mode full