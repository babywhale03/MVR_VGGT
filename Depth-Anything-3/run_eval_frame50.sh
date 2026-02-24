#!/bin/bash
export PYTHONPATH=$PWD
export PATH="$CONDA_PREFIX/bin:$PATH"
export PYTHONBREAKPOINT=0

DEVICE=0
DATASETS=("cam_blur_50" "cam_blur_100" "cam_blur_300")
EXP_NAME="JIHYE__kernel50__step123025"

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
        --work_dir "/mnt/dataset1/MV_Restoration/ECCV26_RESULTS/vggt_eval/${EXP_NAME}/${MAX_FRAMES}/${data}/" \
        --clean_root_path "/mnt/dataset1/MV_Restoration/da3_benchmark_dataset/clean" \
        --deg_root_path "/mnt/dataset1/MV_Restoration/da3_benchmark_dataset/${data}"
done

# python -m depth_anything_3.bench.evaluator model.path=$MODEL

# # Quick test on HiRoom only
# python -m depth_anything_3.bench.evaluator \
#     model.path=$MODEL \
#     eval.datasets=[hiroom] \
#     eval.modes=[pose]

# # Pose-only evaluation (all 5 pose datasets)
# python -m depth_anything_3.bench.evaluator \
#     model.path=$MODEL \
#     eval.datasets=[eth3d,7scenes,scannetpp,hiroom,dtu64] \
#     eval.modes=[pose]

# # Recon-only evaluation (all 5 recon datasets)
# python -m depth_anything_3.bench.evaluator \
#     model.path=$MODEL \
#     eval.datasets=[eth3d,7scenes,scannetpp,hiroom,dtu] \
#     eval.modes=[recon_unposed,recon_posed]

# # Debug specific scenes
# python -m depth_anything_3.bench.evaluator \
#     model.path=$MODEL \
#     eval.datasets=[eth3d] \
#     eval.scenes=[courtyard] \
#     inference.debug=true

# # Re-evaluate without re-running inference
# python -m depth_anything_3.bench.evaluator eval.eval_only=true

# # Just view results
# python -m depth_anything_3.bench.evaluator eval.print_only=true