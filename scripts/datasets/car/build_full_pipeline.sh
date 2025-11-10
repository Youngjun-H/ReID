#!/bin/bash
# 차량 데이터셋 구축 통합 파이프라인 실행 스크립트
# 사용법: ./build_full_pipeline.sh <video_dir> <output_dir> [옵션]

# 인자 파싱
VIDEO_DIR=${1:-""}
OUTPUT_DIR=${2:-""}

if [ -z "$VIDEO_DIR" ] || [ -z "$OUTPUT_DIR" ]; then
    echo "사용법: $0 <video_dir> <output_dir> [옵션]"
    echo ""
    echo "예제:"
    echo "  $0 /data/reid/data/raw/videos/car /data/reid/data/datasets/car/output"
    echo ""
    echo "옵션:"
    echo "  --skip_tracking      Step 1 (Tracking) 건너뛰기"
    echo "  --skip_filtering     Step 2 (Filtering) 건너뛰기"
    echo "  --skip_labeling      Step 3 (Pseudo Labeling) 건너뛰기"
    exit 1
fi

# 프로젝트 루트 찾기
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
CACHE_DIR="$PROJECT_ROOT/cache"

# 캐시 디렉토리 생성
mkdir -p "$CACHE_DIR"

# 모든 캐시 환경 변수 설정
# HuggingFace 캐시
export HUGGINGFACE_HUB_CACHE="$CACHE_DIR"
export HF_HOME="$CACHE_DIR"
export TRANSFORMERS_CACHE="$CACHE_DIR"
export HF_DATASETS_CACHE="$CACHE_DIR"

# Triton 캐시 (VLLM에서 사용)
export TRITON_CACHE_DIR="$CACHE_DIR/triton"
mkdir -p "$TRITON_CACHE_DIR"

# 기타 캐시
export TORCH_HOME="$CACHE_DIR/torch"
mkdir -p "$TORCH_HOME"

# 임시 파일 디렉토리 설정 (HuggingFace 다운로드 임시 파일용)
export TMPDIR="$CACHE_DIR/tmp"
export TEMP="$CACHE_DIR/tmp"
export TMP="$CACHE_DIR/tmp"
mkdir -p "$TMPDIR"

# HuggingFace 임시 파일 디렉토리 설정
export HF_HUB_TEMP="$CACHE_DIR/tmp"
export HF_HUB_CACHE="$CACHE_DIR"

# VLLM 캐시 디렉토리 설정 (중요!)
export VLLM_CACHE_ROOT="$CACHE_DIR/vllm"
mkdir -p "$VLLM_CACHE_ROOT"

# VLLM Usage Stats 파일 경로 설정
export VLLM_USAGE_STATS_PATH="$CACHE_DIR/vllm/usage_stats.json"
mkdir -p "$(dirname "$VLLM_USAGE_STATS_PATH")"

echo "=========================================="
echo "캐시 디렉토리 설정:"
echo "  프로젝트 루트: $PROJECT_ROOT"
echo "  캐시 디렉토리: $CACHE_DIR"
echo "  - HuggingFace: $CACHE_DIR"
echo "  - Triton: $TRITON_CACHE_DIR"
echo "  - PyTorch: $TORCH_HOME"
echo "  - 임시 파일: $TMPDIR"
echo "  - VLLM 캐시: $VLLM_CACHE_ROOT"
echo "  - VLLM Usage Stats: $VLLM_USAGE_STATS_PATH"
echo "=========================================="

# Python 모듈로 실행
cd "$PROJECT_ROOT"

python -m datasets.car.pipeline \
    --video_dir "$VIDEO_DIR" \
    --output_dir "$OUTPUT_DIR" \
    "${@:3}"  # 나머지 인자들 전달

