#!/bin/bash

DEVICE=0
DATASETS=("cam_blur_50" "cam_blur_100" "cam_blur_300" "cam_blur_500")

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
        --config "/mnt/dataset1/jaeeun/MVR/Depth-Anything-3/src/depth_anything_3/bench/configs/eval_bench_50.yaml" \
        --work_dir "/mnt/dataset1/MV_Restoration/ECCV26_RESULTS/VGGT/wo_mvrm/${MAX_FRAMES}/${data}/" \
        --clean_root_path "/mnt/dataset1/MV_Restoration/da3_benchmark_dataset/clean" \
        --deg_root_path "/mnt/dataset1/MV_Restoration/da3_benchmark_dataset/${data}"
done