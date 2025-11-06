#!/bin/bash
# 번호판 detection 필터링 스크립트
# 사용법: ./build_filtering.sh <runs_dir> <output_dir> <lp_model>

RUNS_DIR=${1:-"datasets/car/runs/runs_1104"}
OUTPUT_DIR=${2:-"datasets/car/runs/runs_1104_filtered_by_lp"}
LP_MODEL=${3:-"checkpoints/detection/lp_detection.pt"}

python -m datasets.car.filtering.filtering_by_lp \
    --runs_dir "$RUNS_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --lp_detection_model "$LP_MODEL"
