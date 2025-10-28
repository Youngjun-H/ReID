#!/bin/bash

# 카메라 간 사람 매칭 전체 파이프라인 실행 스크립트

echo "=========================================="
echo "    카메라 간 사람 매칭 전체 파이프라인"
echo "=========================================="
echo "시작 시간: $(date)"
echo ""

# 설정
BASE_DIR="/data/reid/reid_master"
TRACKLETS_DIR="$BASE_DIR/tracklets"
MODEL_PATH="$BASE_DIR/reid_embedding_extractor/checkpoints/swin_base_msmt17.pth"
CONFIG_PATH="$BASE_DIR/reid_embedding_extractor/models/solider/configs/msmt17/swin_base.yml"
MATCHING_OUTPUT_DIR="$BASE_DIR/cross_camera_results_0.8_new"
ANALYSIS_OUTPUT_DIR="$BASE_DIR/analysis_results"
SIMILARITY_THRESHOLD=0.8
DEVICE="cuda"

echo "=== 설정 정보 ==="
echo "기본 디렉토리: $BASE_DIR"
echo "Tracklets 디렉토리: $TRACKLETS_DIR"
echo "모델 경로: $MODEL_PATH"
echo "설정 파일: $CONFIG_PATH"
echo "매칭 결과 디렉토리: $MATCHING_OUTPUT_DIR"
echo "분석 결과 디렉토리: $ANALYSIS_OUTPUT_DIR"
echo "유사도 임계값: $SIMILARITY_THRESHOLD"
echo "디바이스: $DEVICE"
echo ""

# 1단계: 환경 확인
echo "=== 1단계: 환경 확인 ==="
if [ ! -d "$TRACKLETS_DIR" ]; then
    echo "오류: Tracklets 디렉토리가 존재하지 않습니다: $TRACKLETS_DIR"
    exit 1
fi

if [ ! -f "$MODEL_PATH" ]; then
    echo "오류: 모델 파일이 존재하지 않습니다: $MODEL_PATH"
    exit 1
fi

if [ ! -f "$CONFIG_PATH" ]; then
    echo "오류: 설정 파일이 존재하지 않습니다: $CONFIG_PATH"
    exit 1
fi

echo "✓ 환경 확인 완료"
echo ""

# 2단계: 카메라 간 매칭 실행
echo "=== 2단계: 카메라 간 매칭 실행 ==="
echo "시작 시간: $(date)"

# 출력 디렉토리 생성
mkdir -p "$MATCHING_OUTPUT_DIR"

# 매칭 실행
python cross_camera_matching.py \
    --tracklets_dir "$TRACKLETS_DIR" \
    --model_path "$MODEL_PATH" \
    --config_path "$CONFIG_PATH" \
    --output_dir "$MATCHING_OUTPUT_DIR" \
    --similarity_threshold "$SIMILARITY_THRESHOLD" \
    --device "$DEVICE"

if [ $? -ne 0 ]; then
    echo "오류: 카메라 간 매칭 실행 실패"
    exit 1
fi

echo "✓ 카메라 간 매칭 완료"
echo "완료 시간: $(date)"
echo ""

# 3단계: 결과 분석
echo "=== 3단계: 결과 분석 ==="
echo "시작 시간: $(date)"

# 분석 결과 디렉토리 생성
mkdir -p "$ANALYSIS_OUTPUT_DIR"

# 분석 실행
python analyze_matching_results.py \
    --results_dir "$MATCHING_OUTPUT_DIR" \
    --output_dir "$ANALYSIS_OUTPUT_DIR"

if [ $? -ne 0 ]; then
    echo "오류: 결과 분석 실행 실패"
    exit 1
fi

echo "✓ 결과 분석 완료"
echo "완료 시간: $(date)"
echo ""

# 4단계: 결과 요약
echo "=== 4단계: 결과 요약 ==="

echo "매칭 결과 파일들:"
ls -la "$MATCHING_OUTPUT_DIR"

echo ""
echo "분석 결과 파일들:"
ls -la "$ANALYSIS_OUTPUT_DIR"

echo ""
echo "=== 통계 정보 ==="
if [ -f "$MATCHING_OUTPUT_DIR/statistics.json" ]; then
    echo "매칭 통계:"
    cat "$MATCHING_OUTPUT_DIR/statistics.json" | python -m json.tool
fi

echo ""
echo "=== 시각화 파일들 ==="
echo "매칭 그룹 시각화:"
ls -la "$MATCHING_OUTPUT_DIR"/group_*.png 2>/dev/null || echo "시각화 파일이 없습니다."

echo ""
echo "분석 시각화:"
ls -la "$ANALYSIS_OUTPUT_DIR"/*.png 2>/dev/null || echo "분석 시각화 파일이 없습니다."

echo ""
echo "유사도 행렬 시각화:"
ls -la "$MATCHING_OUTPUT_DIR"/*similarity*.png 2>/dev/null || echo "유사도 시각화 파일이 없습니다."

echo ""
echo "=========================================="
echo "    전체 파이프라인 완료!"
echo "=========================================="
echo "완료 시간: $(date)"
echo ""
echo "주요 결과 파일들:"
echo "  - 통합 ID 매핑: $MATCHING_OUTPUT_DIR/unified_id_mapping.json"
echo "  - 매칭 결과: $MATCHING_OUTPUT_DIR/matches.json"
echo "  - 통계 정보: $MATCHING_OUTPUT_DIR/statistics.json"
echo "  - 통합 이미지 폴더: $MATCHING_OUTPUT_DIR/unified_images/"
echo "  - 상세 보고서: $ANALYSIS_OUTPUT_DIR/detailed_report.json"
echo "  - ID 매핑 CSV: $ANALYSIS_OUTPUT_DIR/id_mapping.csv"
echo "  - 그룹 정보 CSV: $ANALYSIS_OUTPUT_DIR/group_info.csv"
echo ""
echo "통합 이미지 폴더 구조:"
if [ -d "$MATCHING_OUTPUT_DIR/unified_images" ]; then
    echo "  - unified_id_001/, unified_id_002/, ... (각 통합 ID별 폴더)"
    echo "  - 각 폴더에는 해당 ID의 모든 이미지들이 저장됨"
    echo "  - 파일명 형식: {층}_{track_id}_{frame_index}_{원본파일명}"
    echo "  - group_info.json: 각 그룹의 상세 정보"
    echo "  - summary.json: 전체 통계 정보"
fi

echo ""
echo "매칭되지 않은 track 폴더 구조:"
if [ -d "$MATCHING_OUTPUT_DIR/unmatched_images" ]; then
    echo "  - 1F_track_XXX/, 2F_track_XXX/, 3F_track_XXX/ (개별 track 폴더)"
    echo "  - 각 폴더에는 매칭되지 않은 track의 모든 이미지들이 저장됨"
    echo "  - 파일명 형식: {frame_index}_{원본파일명}"
    echo "  - track_info.json: 각 track의 상세 정보"
    echo "  - unmatched_summary.json: 전체 통계 정보"
fi
echo ""
