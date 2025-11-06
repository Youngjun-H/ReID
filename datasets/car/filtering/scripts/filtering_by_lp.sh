#!/bin/bash

# 날짜 변수 설정 (명령줄 인자로 받거나 기본값 사용)
DATE="1031"
LP_MODEL="/data/reid/reid_master/lp_detection.pt"

python filtering_by_lp.py \
    --lp_detection_model ${LP_MODEL} \
    --runs_dir runs_${DATE} \
    --output_dir runs_${DATE}_filtered_by_lp \
    --output_lp_dir runs_${DATE}_lp_cropped