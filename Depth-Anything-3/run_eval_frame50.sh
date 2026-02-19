#!/bin/bash
export PYTHONPATH=$PWD
export PATH="$CONDA_PREFIX/bin:$PATH"
export PYTHONBREAKPOINT=0

DEVICE=1
DATASETS=("cam_blur_50" "cam_blur_100" "cam_blur_300" "cam_blur_500")
EXP_NAME="VGGTMVRM__ddp2__deg_first__val30_seed__change_learning_rate_schedule__g0__kernel100__view4__fp32__ep10__bs-8__lr-2e-04__ema0.9995"

########################################
# MAX_FRAMES = 50
########################################
MAX_FRAMES=50

for data in "${DATASETS[@]}"
do
    echo "========================================="
    echo "Running dataset: $data (frames=${MAX_FRAMES})"
    echo "========================================="

    CUDA_VISIBLE_DEVICES=$DEVICE python -m depth_anything_3.bench.evaluator_vggt \
        --config "/mnt/dataset1/jaeeun/MVR_vggt/Depth-Anything-3/src/depth_anything_3/bench/configs/eval_bench_50.yaml" \
        --work_dir "/mnt/dataset1/MV_Restoration/ECCV26_RESULTS/vggt_eval/clean_womvrm_${EXP_NAME}/${MAX_FRAMES}/${data}/" \
        --clean_root_path "/mnt/dataset1/MV_Restoration/da3_benchmark_dataset/clean" \
        --deg_root_path "/mnt/dataset1/MV_Restoration/da3_benchmark_dataset/clean"
        # --deg_root_path "/mnt/dataset1/MV_Restoration/da3_benchmark_dataset/${data}"
done