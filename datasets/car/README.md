# 차량 데이터셋 구축 파이프라인

CCTV 영상에서 차량을 추적하고, 번호판을 detection하여 필터링하고, OCR로 라벨링하는 통합 파이프라인입니다.

## 파이프라인 구조

```
CCTV 영상 디렉토리
    ↓
[Step 1: Tracking]
    → 차량 추적 및 crop
    → 01_tracking/
    ↓
[Step 2: Filtering]
    → 번호판 detection으로 필터링
    → 02_filtered_vehicles/ (필터링된 차량 이미지)
    → 03_license_plates/ (번호판 crop 이미지)
    ↓
[Step 3: Pseudo Labeling]
    → VLLM OCR로 번호판 라벨링
    → 03_license_plates/*/label.txt
```

## 사용 방법

### 1. 전체 파이프라인 실행

```bash
python -m datasets.car.pipeline \
    --video_dir /path/to/cctv/videos \
    --output_dir /path/to/output
```

### 2. 스크립트 사용

```bash
./scripts/datasets/car/build_full_pipeline.sh \
    /path/to/cctv/videos \
    /path/to/output
```

### 3. 특정 단계만 실행

```bash
# Tracking만 실행
python -m datasets.car.pipeline \
    --video_dir /path/to/videos \
    --output_dir /path/to/output \
    --skip_filtering \
    --skip_labeling

# Filtering만 실행 (이미 Tracking 결과가 있는 경우)
python -m datasets.car.pipeline \
    --video_dir /path/to/videos \
    --output_dir /path/to/output \
    --skip_tracking \
    --skip_labeling

# Labeling만 실행 (이미 Filtering 결과가 있는 경우)
python -m datasets.car.pipeline \
    --video_dir /path/to/videos \
    --output_dir /path/to/output \
    --skip_tracking \
    --skip_filtering
```

## 주요 파라미터

### Tracking 파라미터

- `--tracking_model`: YOLO 모델 경로 (기본값: `checkpoints/detection/yolo11x.pt`)
- `--tracking_conf`: Confidence threshold (기본값: `0.7`)
- `--tracking_iou`: IoU threshold (기본값: `0.5`)
- `--max_frames_per_track`: 트랙별 최대 저장 프레임 수 (기본값: `200`)
- `--roi_file`: ROI 파일 경로 (선택사항)
- `--roi_min_iou`: ROI 필터링 최소 IoU 임계값 (기본값: `0.7`)
- `--num_workers`: 병렬 처리 프로세스 수 (기본값: 자동)

### Filtering 파라미터

- `--lp_detection_model`: 번호판 detection 모델 경로 (기본값: `checkpoints/detection/lp_detection.pt`)

### Pseudo Labeling 파라미터

- `--vllm_model`: VLLM 모델 이름 (기본값: `Qwen/Qwen3-VL-4B-Instruct`)
- `--vllm_gpu_util`: GPU 사용률 (기본값: `0.8`)
- `--vllm_prompt`: OCR 프롬프트 (기본값: `차량 번호판의 문자를 추출해주세요.`)
- `--vllm_max_tokens`: 최대 토큰 수 (기본값: `300`)
- `--label_filename`: 생성할 라벨 파일 이름 (기본값: `label.txt`)

## 출력 디렉토리 구조

```
output_dir/
├── 01_tracking/              # Step 1: 차량 추적 결과
│   └── {video_name}/
│       └── track_*/
│           └── *.jpg
│
├── 02_filtered_vehicles/     # Step 2: 필터링된 차량 이미지
│   └── {video_name}/
│       └── track_*/
│           └── *_conf*.jpg
│
└── 03_license_plates/        # Step 2: 번호판 crop 이미지 + Step 3: 라벨
    └── {video_name}/
        └── track_*/
            ├── *_lp_conf*.jpg
            └── label.txt      # Step 3에서 생성
```

## 예제

### 기본 실행

```bash
python -m datasets.car.pipeline \
    --video_dir /data/reid/data/raw/videos/car \
    --output_dir /data/reid/data/datasets/car/output_20241106
```

### ROI 필터링 사용

```bash
python -m datasets.car.pipeline \
    --video_dir /data/reid/data/raw/videos/car \
    --output_dir /data/reid/data/datasets/car/output_20241106 \
    --roi_file /data/reid/reid_master/roi.txt \
    --roi_min_iou 0.7
```

### 커스텀 모델 사용

```bash
python -m datasets.car.pipeline \
    --video_dir /data/reid/data/raw/videos/car \
    --output_dir /data/reid/data/datasets/car/output_20241106 \
    --tracking_model checkpoints/detection/yolo11n.pt \
    --lp_detection_model checkpoints/detection/lp_detection.pt \
    --vllm_model Qwen/Qwen3-VL-8B-Instruct
```

## 개별 모듈 사용

각 단계를 개별적으로 실행할 수도 있습니다:

### Step 1: Tracking

```bash
python -m datasets.car.tracking.pipeline_roi \
    --source /path/to/videos \
    --output /path/to/output \
    --weights checkpoints/detection/yolo11x.pt \
    --roi-file /path/to/roi.txt
```

### Step 2: Filtering

```bash
python -m datasets.car.filtering.filtering_by_lp \
    --runs_dir /path/to/tracking/output \
    --output_dir /path/to/filtered/output \
    --output_lp_dir /path/to/lp/crop/output \
    --lp_detection_model checkpoints/detection/lp_detection.pt
```

### Step 3: Pseudo Labeling

```bash
python -m datasets.car.pseudo_labeling.vllm.vllm_server_simple_example \
    /path/to/lp/crop/output \
    --prompt "차량 번호판의 문자를 추출해주세요."
```

## 주의사항

1. **모델 파일**: Tracking과 Filtering에 사용되는 모델 파일이 `checkpoints/detection/` 디렉토리에 있어야 합니다.
2. **GPU 메모리**: VLLM 서버는 GPU 메모리를 많이 사용하므로, 다른 프로세스와 충돌하지 않도록 주의하세요.
3. **디스크 공간**: 전체 파이프라인은 많은 디스크 공간을 사용할 수 있습니다. 충분한 공간을 확보하세요.

## 문제 해결

### Import 오류

모듈로 실행할 때는 프로젝트 루트에서 실행하세요:

```bash
cd /data/reid/reid_master
python -m datasets.car.pipeline --video_dir ... --output_dir ...
```

### VLLM 서버 오류

VLLM 서버가 이미 실행 중이면 포트를 변경하세요:

```bash
python -m datasets.car.pipeline \
    --video_dir ... \
    --output_dir ... \
    --vllm_port 8001
```

