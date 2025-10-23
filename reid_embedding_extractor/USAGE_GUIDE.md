# ReID Embedding Extractor 사용 가이드

## 🚀 YAML 설정 파일 지원

이제 다양한 모델별 YAML 설정 파일을 사용할 수 있습니다!

### 📁 설정 파일 구조

```
reid_embedding_extractor/
├── configs/
│   ├── msmt17/
│   │   ├── swin_base.yml      # Swin Transformer Base (MSMT17)
│   │   ├── swin_tiny.yml      # Swin Transformer Tiny (MSMT17)
│   │   └── swin_small.yml     # Swin Transformer Small (MSMT17)
│   └── market1501/
│       ├── swin_base.yml      # Swin Transformer Base (Market1501)
│       ├── swin_tiny.yml      # Swin Transformer Tiny (Market1501)
│       └── swin_small.yml     # Swin Transformer Small (Market1501)
├── models/
│   └── swin_transformer.py
├── config.py
├── model_factory.py
├── embedding_extractor.py
└── ...
```

## 🎯 사용법

### 1. 자동 설정 파일 로드

모델 파일명에 모델 타입이 포함되어 있으면 자동으로 해당 설정 파일을 로드합니다.

```python
from embedding_extractor import ReIDEmbeddingExtractor

# 모델 파일명에 'swin_base'가 포함되어 있으면 swin_base.yml 자동 로드
extractor = ReIDEmbeddingExtractor(
    model_path="swin_base_model.pth",  # 자동으로 swin_base.yml 로드
    device="cuda",
    semantic_weight=0.2,
    image_size=(384, 128),
    normalize_features=True
)
```

### 2. 수동 설정 파일 지정

특정 설정 파일을 직접 지정할 수 있습니다.

```python
from embedding_extractor import ReIDEmbeddingExtractor

# 특정 설정 파일 지정
extractor = ReIDEmbeddingExtractor(
    model_path="your_model.pth",
    config_path="configs/msmt17/swin_tiny.yml",  # 수동 지정
    device="cuda",
    semantic_weight=0.2,
    image_size=(384, 128),
    normalize_features=True
)
```

### 3. 사용 가능한 설정 파일 확인

```python
from config import get_available_configs, get_available_datasets, find_config_file

# 사용 가능한 데이터셋 확인
datasets = get_available_datasets()
print(f"사용 가능한 데이터셋: {datasets}")
# 출력: ['market1501', 'msmt17']

# 전체 설정 파일 목록
all_configs = get_available_configs('all')
print(f"전체 설정 파일: {all_configs}")
# 출력: ['swin_base (market1501)', 'swin_base (msmt17)', ...]

# 특정 데이터셋의 설정 파일 목록
msmt17_configs = get_available_configs('msmt17')
print(f"MSMT17 설정 파일: {msmt17_configs}")
# 출력: ['swin_base', 'swin_tiny', 'swin_small']

# 특정 모델의 설정 파일 경로 찾기
config_path = find_config_file('swin_base')
print(f"swin_base 설정 파일: {config_path}")
# 출력: /path/to/configs/msmt17/swin_base.yml
```

## 🔧 설정 파일 예제

### Swin Transformer Base 설정 (swin_base.yml)

```yaml
MODEL:
  TRANSFORMER_TYPE: 'swin_base_patch4_window7_224'
  STRIDE_SIZE: [16, 16]
  SEMANTIC_WEIGHT: 0.2
  METRIC_LOSS_TYPE: 'triplet'
  IF_LABELSMOOTH: 'off'
  IF_WITH_CENTER: 'no'
  NO_MARGIN: True

INPUT:
  SIZE_TRAIN: [384, 128]
  SIZE_TEST: [384, 128]
  PIXEL_MEAN: [0.5, 0.5, 0.5]
  PIXEL_STD: [0.5, 0.5, 0.5]
  PROB: 0.5
  RE_PROB: 0.5
  PADDING: 10

TEST:
  EVAL: True
  IMS_PER_BATCH: 256
  RE_RANKING: False
  NECK_FEAT: 'before'
  FEAT_NORM: 'yes'
```

### Vision Transformer Base 설정 (vit_base.yml)

```yaml
MODEL:
  TRANSFORMER_TYPE: 'vit_base_patch16_224_TransReID'
  STRIDE_SIZE: [16, 16]
  SEMANTIC_WEIGHT: 0.2
  # ... 기타 설정들
```

## 🎨 새로운 모델 추가하기

### 1. 설정 파일 생성

새로운 모델을 추가하려면 `configs/msmt17/` 디렉토리에 YAML 파일을 생성하세요.

```yaml
# configs/msmt17/new_model.yml
MODEL:
  TRANSFORMER_TYPE: 'your_model_type'
  STRIDE_SIZE: [16, 16]
  SEMANTIC_WEIGHT: 0.2
  # ... 기타 설정들

INPUT:
  SIZE_TRAIN: [384, 128]
  SIZE_TEST: [384, 128]
  # ... 기타 설정들

TEST:
  EVAL: True
  IMS_PER_BATCH: 256
  # ... 기타 설정들
```

### 2. 모델 팩토리에 추가

`model_factory.py`의 `transformer_factory`에 새로운 모델을 추가하세요.

```python
transformer_factory = {
    'swin_base_patch4_window7_224': swin_base_patch4_window7_224,
    'swin_small_patch4_window7_224': swin_small_patch4_window7_224,
    'swin_tiny_patch4_window7_224': swin_tiny_patch4_window7_224,
    'vit_base_patch16_224_TransReID': vit_base_patch16_224_TransReID,
    'vit_small_patch16_224_TransReID': vit_small_patch16_224_TransReID,
    'your_new_model': your_new_model_function,  # 새 모델 추가
}
```

### 3. 자동 감지에 추가

`embedding_extractor.py`의 `_load_config` 메서드에서 모델 이름을 추가하세요.

```python
for name in ['swin_base', 'swin_tiny', 'swin_small', 'vit_base', 'vit_small', 'your_new_model']:
    if name in model_path.stem:
        model_name = name
        break
```

## 📋 지원되는 모델들

### MSMT17 데이터셋
| 모델 | 설정 파일 | Transformer Type |
|------|-----------|------------------|
| Swin Transformer Base | `msmt17/swin_base.yml` | `swin_base_patch4_window7_224` |
| Swin Transformer Tiny | `msmt17/swin_tiny.yml` | `swin_tiny_patch4_window7_224` |
| Swin Transformer Small | `msmt17/swin_small.yml` | `swin_small_patch4_window7_224` |

### Market1501 데이터셋
| 모델 | 설정 파일 | Transformer Type |
|------|-----------|------------------|
| Swin Transformer Base | `market1501/swin_base.yml` | `swin_base_patch4_window7_224` |
| Swin Transformer Tiny | `market1501/swin_tiny.yml` | `swin_tiny_patch4_window7_224` |
| Swin Transformer Small | `market1501/swin_small.yml` | `swin_small_patch4_window7_224` |

## 🔍 자동 감지 규칙

모델 파일명에 다음 키워드가 포함되어 있으면 해당 설정 파일을 자동으로 로드합니다:

- `swin_base` → `swin_base.yml`
- `swin_tiny` → `swin_tiny.yml`
- `swin_small` → `swin_small.yml`

## 🚀 예제 실행

```bash
# YAML 설정 파일 사용 예제
cd /data/reid/reid_master/reid_embedding_extractor
python yaml_config_example.py

# 다중 데이터셋 설정 파일 사용 예제
python multi_dataset_example.py

# 독립적인 사용 예제
python independent_example.py
```

## ✨ 주요 장점

1. **자동 설정 로드**: 모델 파일명으로 설정 파일 자동 감지
2. **다중 데이터셋 지원**: MSMT17, Market1501 등 다양한 데이터셋
3. **모델별 맞춤 설정**: 각 모델에 최적화된 설정
4. **확장성**: 새로운 모델과 데이터셋 쉽게 추가 가능
5. **독립성**: SOLIDER_REID 의존성 완전 제거
6. **유연성**: 수동 설정 파일 지정도 가능

이제 다양한 모델과 데이터셋 설정을 쉽게 사용할 수 있습니다! 🎉
