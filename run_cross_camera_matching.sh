#!/bin/bash

# 카메라 간 사람 매칭 실행 스크립트

echo "=== 카메라 간 사람 매칭 시스템 ==="
echo "시작 시간: $(date)"

# 기본 설정
TRACKLETS_DIR="/data/reid/reid_master/tracklets"
MODEL_PATH="/data/reid/reid_master/reid_embedding_extractor/checkpoints/swin_base_msmt17.pth"
CONFIG_PATH="/data/reid/reid_master/reid_embedding_extractor/models/solider/configs/msmt17/swin_base.yml"
OUTPUT_DIR="/data/reid/reid_master/cross_camera_results_0.85"
SIMILARITY_THRESHOLD=0.85
DEVICE="cuda"

# 출력 디렉토리 생성
mkdir -p "$OUTPUT_DIR"

echo "설정 정보:"
echo "  - Tracklets 디렉토리: $TRACKLETS_DIR"
echo "  - 모델 경로: $MODEL_PATH"
echo "  - 설정 파일: $CONFIG_PATH"
echo "  - 출력 디렉토리: $OUTPUT_DIR"
echo "  - 유사도 임계값: $SIMILARITY_THRESHOLD"
echo "  - 디바이스: $DEVICE"
echo ""

# Python 스크립트 실행
python cross_camera_matching.py \
    --tracklets_dir "$TRACKLETS_DIR" \
    --model_path "$MODEL_PATH" \
    --config_path "$CONFIG_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --similarity_threshold "$SIMILARITY_THRESHOLD" \
    --device "$DEVICE"

# 실행 결과 확인
if [ $? -eq 0 ]; then
    echo ""
    echo "=== 매칭 완료 ==="
    echo "완료 시간: $(date)"
    echo ""
    echo "결과 파일들:"
    ls -la "$OUTPUT_DIR"
    echo ""
    echo "통계 정보:"
    if [ -f "$OUTPUT_DIR/statistics.json" ]; then
        cat "$OUTPUT_DIR/statistics.json"
    fi
else
    echo ""
    echo "=== 오류 발생 ==="
    echo "실행 중 오류가 발생했습니다."
    exit 1
fi
