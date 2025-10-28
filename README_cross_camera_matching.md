# 카메라 간 사람 매칭 시스템

층별로 추적된 사람 ID들을 ReID 임베딩을 사용하여 통합 ID로 매핑하는 시스템입니다.

## 🎯 주요 기능

- **ReID 기반 매칭**: SOLIDER 모델을 사용한 고정밀 사람 매칭
- **다층 지원**: 1F, 2F, 3F 카메라 간 사람 매칭
- **자동 ID 통합**: 층별 ID를 통합된 ID로 자동 매핑
- **시각화**: 매칭 결과 시각화 및 분석
- **통계 분석**: 매칭 품질 분석 및 보고서 생성

## 📁 파일 구조

```
/data/reid/reid_master/
├── cross_camera_matching.py          # 핵심 매칭 시스템
├── analyze_matching_results.py       # 결과 분석 도구
├── test_matching_system.py           # 시스템 테스트
├── run_cross_camera_matching.sh      # 매칭 실행 스크립트
├── run_full_pipeline.sh              # 전체 파이프라인 실행
├── tracklets/                        # 입력 데이터
│   ├── 1F/                          # 1층 tracklets
│   ├── 2F/                          # 2층 tracklets
│   └── 3F/                          # 3층 tracklets
├── cross_camera_results/             # 매칭 결과
└── analysis_results/                 # 분석 결과
```

## 🚀 사용법

### 1. 시스템 테스트

먼저 시스템이 정상적으로 작동하는지 테스트합니다:

```bash
cd /data/reid/reid_master
python test_matching_system.py
```

### 2. 전체 파이프라인 실행

테스트가 성공하면 전체 파이프라인을 실행합니다:

```bash
./run_full_pipeline.sh
```

### 3. 개별 실행

필요에 따라 개별 단계를 실행할 수 있습니다:

#### 매칭만 실행:
```bash
python cross_camera_matching.py \
    --tracklets_dir /data/reid/reid_master/tracklets \
    --model_path /data/reid/reid_master/reid_embedding_extractor/checkpoints/swin_base_msmt17.pth \
    --config_path /data/reid/reid_master/reid_embedding_extractor/models/solider/configs/msmt17/swin_base.yml \
    --output_dir /data/reid/reid_master/cross_camera_results \
    --similarity_threshold 0.7 \
    --device cuda
```

#### 결과 분석만 실행:
```bash
python analyze_matching_results.py \
    --results_dir /data/reid/reid_master/cross_camera_results \
    --output_dir /data/reid/reid_master/analysis_results
```

## 📊 결과 파일

### 매칭 결과 (`cross_camera_results/`)
- `unified_id_mapping.json`: 층별 ID → 통합 ID 매핑
- `matches.json`: 매칭된 그룹 정보
- `statistics.json`: 기본 통계 정보
- `group_*.png`: 매칭 그룹 시각화 이미지
- `similarity_matrix_heatmap.png`: 전체 유사도 행렬 히트맵
- `similarity_matrix_by_floor.png`: 층별 유사도 행렬
- `similarity_distribution.png`: 유사도 분포 히스토그램
- `matching_results_visualization.png`: 매칭 결과 시각화
- `unified_images/`: 통합 ID별 이미지 폴더
  - `unified_id_001/`, `unified_id_002/`, ... (각 통합 ID별 폴더)
  - 각 폴더에는 해당 ID의 모든 이미지들이 저장됨
  - 파일명 형식: `{층}_{track_id}_{frame_index}_{원본파일명}`
  - `group_info.json`: 각 그룹의 상세 정보
  - `summary.json`: 전체 통계 정보
- `unmatched_images/`: 매칭되지 않은 track 이미지 폴더
  - `1F_track_XXX/`, `2F_track_XXX/`, `3F_track_XXX/` (개별 track 폴더)
  - 각 폴더에는 매칭되지 않은 track의 모든 이미지들이 저장됨
  - 파일명 형식: `{frame_index}_{원본파일명}`
  - `track_info.json`: 각 track의 상세 정보
  - `unmatched_summary.json`: 전체 통계 정보

### 분석 결과 (`analysis_results/`)
- `detailed_report.json`: 상세 분석 보고서
- `id_mapping.csv`: ID 매핑 테이블 (CSV)
- `group_info.csv`: 그룹 정보 테이블 (CSV)
- `matching_distribution.png`: 매칭 분포 시각화

## ⚙️ 설정 옵션

### 주요 파라미터

- `--similarity_threshold`: 유사도 임계값 (기본값: 0.7)
- `--device`: 사용할 디바이스 (cpu/cuda)
- `--model_path`: ReID 모델 경로
- `--config_path`: 모델 설정 파일 경로

### 유사도 임계값 조정

- **높은 값 (0.8-0.9)**: 더 엄격한 매칭, 적은 오매칭
- **낮은 값 (0.5-0.6)**: 더 관대한 매칭, 더 많은 매칭

## 📈 결과 해석

### 통계 정보
- **총 Track 수**: 모든 층의 track 수
- **통합 ID 수**: 매칭 후 고유 ID 수
- **압축률**: 통합 ID 수 / 총 Track 수

### 매칭 품질
- **다층 매칭 그룹**: 여러 층에 걸친 매칭 그룹 수
- **그룹 크기 분포**: 매칭 그룹의 크기 분포
- **층 간 매칭 히트맵**: 층 간 매칭 빈도

## 🔧 문제 해결

### 일반적인 문제들

1. **CUDA 메모리 부족**
   ```bash
   # CPU 사용으로 변경
   --device cpu
   ```

2. **모델 파일 없음**
   ```bash
   # 모델 파일 경로 확인
   ls -la /data/reid/reid_master/reid_embedding_extractor/checkpoints/
   ```

3. **Tracklets 데이터 없음**
   ```bash
   # 데이터 구조 확인
   ls -la /data/reid/reid_master/tracklets/
   ```

### 로그 확인

실행 중 문제가 발생하면 로그를 확인하세요:

```bash
# 상세 로그와 함께 실행
python cross_camera_matching.py --tracklets_dir /path/to/tracklets --model_path /path/to/model 2>&1 | tee matching.log
```

## 📝 예제 결과

### 성공적인 매칭 예제
```json
{
  "group_1": [
    ["1F", 1],
    ["2F", 1],
    ["3F", 1]
  ],
  "group_2": [
    ["1F", 4],
    ["2F", 3]
  ]
}
```

### 통계 예제
```json
{
  "total_tracks": 45,
  "total_unified_ids": 23,
  "compression_ratio": 0.51,
  "multi_floor_groups": 8
}
```

## 🎨 시각화

### 매칭 그룹 시각화
- 각 매칭 그룹의 대표 이미지들을 나란히 표시
- 그룹 크기에 따라 이미지 개수 조정

### 분포 시각화
- 층별 Track 수 분포
- 그룹 크기 분포 히스토그램
- 층 간 매칭 히트맵

## 📞 지원

문제가 발생하거나 질문이 있으시면:
1. 로그 파일 확인
2. 테스트 스크립트 실행
3. 설정 파일 경로 확인
4. 디스크 공간 및 메모리 확인
