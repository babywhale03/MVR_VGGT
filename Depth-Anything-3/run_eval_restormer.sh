#!/bin/bash
export PYTHONPATH=$PWD
export PATH="$CONDA_PREFIX/bin:$PATH"
export PYTHONBREAKPOINT=0
DEVICE=0
# DATASETS=("cam_blur_50" "cam_blur_100" "cam_blur_300" "cam_blur_500")
DATASETS=("filtered_cam_blur_50" "filtered_cam_blur_100")
EXP_NAME="restored_restormer"

########################################
# MAX_FRAMES = 100
########################################
MAX_FRAMES=100

for data in "${DATASETS[@]}"
do
    echo "========================================="
    echo "Running dataset: $data (frames=${MAX_FRAMES})"
    echo "========================================="

    CUDA_VISIBLE_DEVICES=$DEVICE python -m depth_anything_3.bench.evaluator_vggt \
        --config "/mnt/dataset1/jaeeun/MVR_vggt/Depth-Anything-3/src/depth_anything_3/bench/configs/eval_bench_womvrm.yaml" \
        --work_dir "/mnt/dataset1/MV_Restoration/ECCV26_RESULTS/vggt_eval/${EXP_NAME}/${MAX_FRAMES}/${data}/" \
        --clean_root_path "/mnt/dataset1/MV_Restoration/da3_benchmark_dataset/clean" \
        --deg_root_path "/mnt/dataset1/MV_Restoration/da3_benchmark_dataset/${EXP_NAME}/${data}"
done
