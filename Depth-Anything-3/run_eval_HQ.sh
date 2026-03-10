#!/bin/bash
export PYTHONPATH=$PWD
export PATH="$CONDA_PREFIX/bin:$PATH"
export PYTHONBREAKPOINT=0
DEVICE=5
EXP_NAME="restored_restormer"

########################################
# MAX_FRAMES = 100
########################################
MAX_FRAMES=50

echo "========================================="
echo "Running dataset: $data (frames=${MAX_FRAMES})"
echo "========================================="

CUDA_VISIBLE_DEVICES=$DEVICE python -m depth_anything_3.bench.evaluator_vggt_new \
    --config "/mnt/dataset1/jaeeun/MVR_vggt/Depth-Anything-3/src/depth_anything_3/bench/configs/eval_bench_womvrm_HQ_depth.yaml" \
    --work_dir "/mnt/dataset1/MV_Restoration/ECCV26_RESULTS/vggt_eval_final/clean_single/${MAX_FRAMES}" \
    --clean_root_path "/mnt/dataset1/MV_Restoration/da3_benchmark_dataset/clean" \
    --lq_root_path "/mnt/dataset1/MV_Restoration/da3_benchmark_dataset/filtered_cam_blur_50" \
    --res_root_path "/mnt/dataset1/MV_Restoration/da3_benchmark_dataset/${EXP_NAME}/filtered_cam_blur_50" \
    --max_frames $MAX_FRAMES \
    --process_single true
