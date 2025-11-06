#!/bin/bash
# 자동차 추적 및 크롭 스크립트
# 사용법: ./build_tracking.sh <video_dir> <output_dir>

VIDEO_DIR=${1:-"datasets/car/video"}
OUTPUT_DIR=${2:-"datasets/car/runs"}

python -m datasets.car.tracking.pipeline \
    --source "$VIDEO_DIR" \
    --output_dir "$OUTPUT_DIR"
