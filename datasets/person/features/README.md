# SOLIDER ReID Embedding Extractor

SOLIDER_REID 모델을 사용한 ReID 임베딩 추출 도구

## 🚀 Quick Start

```python
from models import SOLIDEREmbeddingExtractor

# 임베딩 추출기 초기화
extractor = SOLIDEREmbeddingExtractor(
    model_path="checkpoints/swin_base_market.pth",
    device="cuda"
)

# 임베딩 추출
embedding = extractor.extract_embedding("path/to/image.jpg")
```

## 📁 Structure

```
reid_embedding_extractor/
├── models/                    # SOLIDER 모델 구현
│   └── solider/              # SOLIDER 모델들
├── checkpoints/              # 사전 훈련된 모델들
├── simple_example.py         # 간단한 사용 예제
├── reid_visualization.py     # ReID 결과 시각화
└── requirements.txt          # 의존성 목록
```

## 📦 Installation

```bash
pip install -r requirements.txt
```

## 🎯 Features

- **SOLIDER 모델 지원**: Swin Transformer 기반
- **자동 클래스 감지**: 체크포인트에서 클래스 수 자동 감지
- **배치 처리**: 여러 이미지 동시 처리
- **시각화 도구**: ReID 결과 시각화
- **CPU/CUDA 지원**: 유연한 디바이스 선택

## 📖 Examples

### 기본 사용법
```bash
python simple_example.py
```

### ReID 시각화
```bash
python reid_visualization.py \
    --query_dir query_images/ \
    --gallery_dir gallery_images/ \
    --model_path checkpoints/swin_base_market.pth
```