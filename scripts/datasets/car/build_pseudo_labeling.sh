#!/bin/bash
# 번호판 recognition pseudo labeling 스크립트
# 사용법: ./build_pseudo_labeling.sh <filtered_dir>

FILTERED_DIR=${1:-"datasets/car/runs/runs_1104_filtered_by_lp"}

python -m datasets.car.pseudo_labeling.vllm.vllm_server_simple_example \
    "$FILTERED_DIR" \
    --prompt "차량 번호판의 문자를 추출해주세요."
