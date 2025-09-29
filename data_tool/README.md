# Person Tracker

YOLO11을 사용하여 영상에서 사람을 detection하고 track_id별로 저장하는 도구입니다.

## 🚀 성능 최적화 버전

최신 버전에서는 다음과 같은 최적화가 적용되었습니다:

- **배치 처리**: 여러 프레임을 동시에 처리하여 GPU 활용도 향상
- **비동기 I/O**: 이미지 저장을 비동기로 처리하여 I/O 대기 시간 최소화
- **메모리 최적화**: 스트리밍 처리 및 가비지 컬렉션 최적화
- **GPU 최적화**: CUDA 설정 및 메모리 관리 최적화
- **병렬 처리**: 멀티스레딩을 통한 I/O 작업 병렬화

## 주요 기능

- **사람만 detection**: COCO 데이터셋의 person 클래스(0)만 감지
- **Track ID별 저장**: 각 사람의 track_id에 따라 별도 디렉토리에 저장
- **Confidence threshold 설정**: detection 결과의 confidence 값 필터링
- **Frame interval 설정**: 몇 프레임마다 한번씩 처리할지 설정 가능
- **메타데이터 저장**: JSON 형태로 tracking 정보 저장

## 설치 요구사항

```bash
pip install ultralytics opencv-python torch
```

## 사용법

### 1. 최적화된 버전 사용 (권장)

```python
from person_tracker_optimized import OptimizedPersonTracker

# 최적화된 PersonTracker 인스턴스 생성
tracker = OptimizedPersonTracker(model_path="yolo11x.pt", max_workers=4)

# 사람 tracking 실행
tracked_persons, summary = tracker.track_persons_optimized(
    video_path="../cctv_dataset/0926_cctv0.avi",
    conf_threshold=0.85,  # confidence threshold
    frame_interval=1,     # 프레임 간격
    save_dir="./tracked_persons",
    batch_size=8,         # 배치 크기
    enable_async=True     # 비동기 I/O 사용
)
```

### 2. 기본 버전 사용

```python
from person_tracker import PersonTracker

# PersonTracker 인스턴스 생성
tracker = PersonTracker(model_path="yolo11x.pt")

# 사람 tracking 실행
tracked_persons, summary = tracker.track_persons(
    video_path="../cctv_dataset/0926_cctv0.avi",
    conf_threshold=0.6,  # confidence threshold
    frame_interval=5,    # 5프레임마다 한번씩 처리
    save_dir="./tracked_persons"
)
```

### 3. 명령행 사용법

```bash
# 최적화된 버전 (권장)
python person_tracker_optimized.py --video ../cctv_dataset/0926_cctv0.avi --conf 0.85 --batch_size 8

# 기본 버전
python person_tracker.py --video ../cctv_dataset/0926_cctv0.avi --conf 0.6 --interval 5
```

### 4. 성능 벤치마크

```bash
# 두 버전 성능 비교
python benchmark.py --video ../cctv_dataset/0926_cctv0.avi --conf 0.85

# 최적화된 버전만 테스트
python benchmark.py --video ../cctv_dataset/0926_cctv0.avi --optimized_only
```

### 5. 예제 스크립트 실행

```bash
python example_usage.py
```

## 매개변수 설명

### 공통 매개변수
- `video_path`: 입력 영상 파일 경로
- `conf_threshold`: confidence threshold (0.0 ~ 1.0, 기본값: 0.5)
- `frame_interval`: 몇 프레임마다 한번씩 처리할지 (기본값: 1)
- `save_dir`: 결과를 저장할 디렉토리 (기본값: "./tracked_persons")
- `classes`: detection할 클래스 ID (기본값: [0] - person)

### 최적화된 버전 추가 매개변수
- `batch_size`: 배치 크기 (기본값: 8, GPU 메모리에 따라 조정)
- `max_workers`: I/O 워커 수 (기본값: 4)
- `enable_async`: 비동기 I/O 사용 여부 (기본값: True)

## 출력 구조

```
tracked_persons/
├── person_0001/
│   ├── frame_000001_conf_0.750.jpg
│   ├── frame_000006_conf_0.820.jpg
│   └── ...
├── person_0002/
│   ├── frame_000003_conf_0.680.jpg
│   └── ...
└── tracking_summary.json
```

## tracking_summary.json 구조

```json
{
  "video_path": "../cctv_dataset/0926_cctv0.avi",
  "total_frames": 361175,
  "processed_frames": 72235,
  "conf_threshold": 0.6,
  "frame_interval": 5,
  "tracked_persons_count": 4,
  "tracked_persons": {
    "1": [
      {
        "frame_number": 1,
        "track_id": 1,
        "confidence": 0.750,
        "bbox": [100, 200, 300, 400],
        "timestamp": 0.033,
        "image_path": "./tracked_persons/person_0001/frame_000001_conf_0.750.jpg"
      }
    ]
  }
}
```

## 성능 최적화 팁

### 기본 버전
1. **빠른 처리**: `frame_interval`을 높게 설정 (예: 10)
2. **높은 정확도**: `conf_threshold`를 높게 설정 (예: 0.8)
3. **메모리 절약**: `stream=True` 옵션 사용 (기본값)

### 최적화된 버전
1. **GPU 메모리 최적화**: `batch_size`를 GPU 메모리에 맞게 조정
2. **I/O 최적화**: `max_workers`를 CPU 코어 수에 맞게 설정
3. **비동기 처리**: `enable_async=True`로 I/O 대기 시간 최소화
4. **메모리 관리**: 자동 가비지 컬렉션 및 GPU 메모리 정리

## 예제 시나리오

### 시나리오 1: 기본 사용
- confidence threshold: 0.6
- frame interval: 5
- 모든 프레임을 처리하되 confidence가 0.6 이상인 결과만 저장

### 시나리오 2: 높은 정확도
- confidence threshold: 0.8
- frame interval: 1
- 모든 프레임을 처리하되 높은 confidence 결과만 저장

### 시나리오 3: 빠른 처리
- confidence threshold: 0.4
- frame interval: 10
- 10프레임마다 처리하되 낮은 confidence도 허용

## 문제 해결

1. **GPU 메모리 부족**: `frame_interval`을 높게 설정
2. **낮은 detection 정확도**: `conf_threshold`를 낮게 설정
3. **처리 속도가 느림**: `frame_interval`을 높게 설정
