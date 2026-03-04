export CUDA=6
export PYTHONPATH=$PWD
export PATH="$CONDA_PREFIX/bin:$PATH"

CUDA_VISIBLE_DEVICES=${CUDA} python RAE/src/eval/eval_feature_similarity_final.py \
    --config RAE/configs/mvrm/DiTDH-XL_DINOv2-B_Multi_nvidia.yaml \
    --ckpt RAE/ckpts/train_hypersim_tartanair_nearcam__val_eth3d__nearrandom/VGGTMVRM__nvidia__ddp2__test_gf0__deg_first__val20_seed__1152,2880__g0__kernel100__view4__fp32__ep10__bs-8__lr-2e-04__ema0.9995/checkpoints/global_step_0036000.pt \
    --results-dir eval_results/final \
    --num-sequences 10 \
    --num-views 5 \
    --kernel-size 500 \
    --eval-mode full
    