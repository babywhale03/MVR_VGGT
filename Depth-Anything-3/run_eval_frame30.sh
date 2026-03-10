#!/bin/bash
export PYTHONPATH=$PWD
export PATH="$CONDA_PREFIX/bin:$PATH"
export PYTHONBREAKPOINT=0
DEVICE=2
DATASETS=("filtered_cam_blur_50" "filtered_cam_blur_100" "filtered_cam_blur_300")
EXP_NAME="JIHYE_kernel100_step123025"

########################################
# MAX_FRAMES = 10
########################################
MAX_FRAMES=30

for data in "${DATASETS[@]}"
do
    echo "========================================="
    echo "Running dataset: $data (frames=${MAX_FRAMES})"
    echo "========================================="

    CUDA_VISIBLE_DEVICES=$DEVICE python -m depth_anything_3.bench.evaluator_vggt_new \
        --config "/mnt/dataset1/jaeeun/MVR_vggt/Depth-Anything-3/src/depth_anything_3/bench/configs/eval_bench_30.yaml" \
        --work_dir "/mnt/dataset1/MV_Restoration/ECCV26_RESULTS/vggt_eval_final/${EXP_NAME}/${MAX_FRAMES}/${data}/" \
        --clean_root_path "/mnt/dataset1/MV_Restoration/da3_benchmark_dataset/clean" \
        --lq_root_path "/mnt/dataset1/MV_Restoration/da3_benchmark_dataset/${data}" \
        --res_root_path "/mnt/dataset1/MV_Restoration/da3_benchmark_dataset/${EXP_NAME}/${data}" \
        --max_frames $MAX_FRAMES \
        --process_single true
done
