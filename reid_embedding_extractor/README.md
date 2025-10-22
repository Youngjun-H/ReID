# ReID Embedding Extractor - PyTorch 최신 버전 호환

SOLIDER_REID 모델을 사용하여 사람 이미지에서 ReID(Re-identification) 임베딩을 추출하는 최적화된 도구입니다. PyTorch 2.0+ 버전과 완전히 호환됩니다.

## 🚀 주요 기능

- **PyTorch 최신 버전 호환**: PyTorch 2.0+ 완전 지원
- **단일 이미지 임베딩 추출**: 개별 이미지에서 ReID 임베딩 추출
- **효율적인 배치 처리**: `@torch.inference_mode()` 사용으로 메모리 효율성 향상
- **유사도 계산**: 임베딩 간 유사도 측정 및 유사도 행렬 계산
- **다양한 출력 형식**: NumPy, JSON, 텍스트 형식 지원
- **GPU/CPU 지원**: CUDA 및 CPU 환경에서 실행 가능
- **최적화된 전처리**: 최신 torchvision InterpolationMode 사용

## 📦 설치

### 1. 의존성 설치

```bash
pip install -r requirements.txt
```

### 2. 패키지 설치 (선택사항)

```bash
pip install -e .
```

## 🎯 사용법

### 1. 명령행 인터페이스

#### 기본 사용법
```bash
python inference.py \
    --model_path path/to/trained_model.pth \
    --input path/to/image.jpg \
    --output embeddings.npy
```

#### 여러 이미지 처리
```bash
python inference.py \
    --model_path path/to/trained_model.pth \
    --input path/to/image_folder/ \
    --output embeddings.npy \
    --output_format json \
    --batch_size 32
```

#### 고급 옵션
```bash
python inference.py \
    --model_path path/to/trained_model.pth \
    --input path/to/images/ \
    --config_path path/to/config.yml \
    --output embeddings.npy \
    --device cuda \
    --semantic_weight 0.2 \
    --image_size 384 128 \
    --batch_size 16 \
    --normalize
```

### 2. Python API 사용

```python
from embedding_extractor import ReIDEmbeddingExtractor
import numpy as np

# 임베딩 추출기 초기화
extractor = ReIDEmbeddingExtractor(
    model_path="path/to/trained_model.pth",
    config_path="path/to/config.yml",
    device="cuda",  # 또는 None (자동 선택)
    semantic_weight=0.2,
    image_size=(384, 128),
    normalize_features=True
)

# 단일 이미지 임베딩 추출
embedding = extractor.extract_embedding("path/to/image.jpg")
print(f"임베딩 차원: {len(embedding)}")

# 여러 이미지 배치 처리
image_paths = ["img1.jpg", "img2.jpg", "img3.jpg"]
embeddings = extractor.extract_embeddings_batch(image_paths, batch_size=32)

# 유사도 계산
similarity = extractor.compute_similarity(embeddings[0], embeddings[1])
print(f"유사도: {similarity:.4f}")

# 유사도 행렬 계산
similarity_matrix = extractor.compute_similarity_matrix(embeddings[:2], embeddings[2:])
print(f"유사도 행렬 크기: {similarity_matrix.shape}")

# 가장 유사한 이미지 찾기
query_embedding = embeddings[0]
gallery_embeddings = embeddings[1:]
best_idx, best_sim = extractor.find_most_similar(query_embedding, gallery_embeddings)
print(f"가장 유사한 이미지: {best_idx}, 유사도: {best_sim:.4f}")
```

## ⚙️ 설정

### 모델 설정 (YAML 파일)

```yaml
MODEL:
  NAME: 'transformer'
  TRANSFORMER_TYPE: 'swin_base_patch4_window7_224'
  SEMANTIC_WEIGHT: 0.2

INPUT:
  SIZE_TRAIN: [384, 128]
  SIZE_TEST: [384, 128]
  PIXEL_MEAN: [0.5, 0.5, 0.5]
  PIXEL_STD: [0.5, 0.5, 0.5]

TEST:
  NECK_FEAT: 'before'
  FEAT_NORM: 'yes'
```

### 명령행 인자

| 인자 | 설명 | 기본값 |
|------|------|--------|
| `--model_path` | 훈련된 모델 경로 | 필수 |
| `--input` | 입력 이미지/폴더 경로 | 필수 |
| `--output` | 출력 파일 경로 | `embeddings.npy` |
| `--config_path` | 설정 파일 경로 | `default_config.yml` |
| `--device` | 사용 디바이스 | 자동 선택 |
| `--semantic_weight` | 시맨틱 가중치 | `0.2` |
| `--image_size` | 이미지 크기 | `384 128` |
| `--batch_size` | 배치 크기 | `32` |
| `--output_format` | 출력 형식 | `npy` |
| `--normalize` | L2 정규화 | `True` |

## 🔧 PyTorch 최신 버전 개선사항

### 1. 메모리 효율성
```python
@torch.inference_mode()  # PyTorch 1.9+에서 권장
def extract_embedding(self, image):
    # 추론 모드로 메모리 사용량 최적화
    pass
```

### 2. 최신 torchvision 호환
```python
# 최신 InterpolationMode 사용
transforms.Resize(
    image_size, 
    interpolation=transforms.InterpolationMode.BICUBIC
)
```

### 3. 타입 힌트 개선
```python
def extract_embedding(
    self, 
    image: Union[str, Image.Image, np.ndarray]
) -> np.ndarray:
    # 명확한 타입 힌트로 코드 가독성 향상
    pass
```

### 4. 오류 처리 강화
```python
try:
    from model import make_model
    from config import cfg
except ImportError as e:
    print(f"SOLIDER_REID 모듈을 찾을 수 없습니다: {e}")
    sys.exit(1)
```

## 📁 출력 형식

### 1. NumPy 형식 (.npy)
```python
import numpy as np
embeddings = np.load("embeddings.npy")
print(embeddings.shape)  # (num_images, embedding_dim)
```

### 2. JSON 형식 (.json)
```json
{
  "embeddings": [[0.1, 0.2, ...], [0.3, 0.4, ...]],
  "image_paths": ["img1.jpg", "img2.jpg"],
  "embedding_dim": 2048
}
```

### 3. 텍스트 형식 (.txt)
```
# Image 0: img1.jpg
0.1 0.2 0.3 ...

# Image 1: img2.jpg
0.4 0.5 0.6 ...
```

## 🔧 고급 사용법

### 1. 커스텀 전처리

```python
from PIL import Image
import torchvision.transforms as T

# 커스텀 전처리 파이프라인
custom_transform = T.Compose([
    T.Resize((384, 128), interpolation=T.InterpolationMode.BICUBIC),
    T.ColorJitter(brightness=0.2, contrast=0.2),
    T.ToTensor(),
    T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# 이미지 전처리
image = Image.open("image.jpg")
processed = custom_transform(image)
```

### 2. 배치 처리 최적화

```python
# 큰 데이터셋의 경우 배치 크기 조정
embeddings = extractor.extract_embeddings_batch(
    image_paths, 
    batch_size=64  # GPU 메모리에 따라 조정
)
```

### 3. 메모리 효율적인 처리

```python
# 대용량 데이터셋 처리
def process_large_dataset(image_paths, batch_size=32):
    all_embeddings = []
    
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i + batch_size]
        batch_embeddings = extractor.extract_embeddings_batch(batch_paths)
        all_embeddings.extend(batch_embeddings)
        
        # 메모리 정리
        torch.cuda.empty_cache()
    
    return all_embeddings
```

## 📊 성능 최적화

### 1. GPU 사용
```python
# CUDA 사용 시
extractor = ReIDEmbeddingExtractor(
    model_path="model.pth",
    config_path="config.yml",
    device="cuda"
)
```

### 2. 배치 크기 조정
- GPU 메모리에 따라 배치 크기 조정
- 일반적으로 16-64 사이에서 최적 성능

### 3. 추론 모드 사용
```python
# @torch.inference_mode() 자동 적용
# 메모리 사용량 최적화 및 성능 향상
```

## 🐛 문제 해결

### 1. CUDA 메모리 부족
```bash
# 배치 크기 줄이기
python inference.py --batch_size 8

# CPU 사용
python inference.py --device cpu
```

### 2. 모델 로드 실패
```python
# 모델 파일 경로 확인
import os
print(os.path.exists("path/to/model.pth"))

# 설정 파일 확인
print(os.path.exists("path/to/config.yml"))
```

### 3. 이미지 로드 실패
```python
# 지원되는 이미지 형식 확인
from PIL import Image
try:
    img = Image.open("image.jpg")
    print("이미지 로드 성공")
except Exception as e:
    print(f"이미지 로드 실패: {e}")
```

### 4. PyTorch 버전 호환성
```python
import torch
print(f"PyTorch 버전: {torch.__version__}")

# 최소 요구사항: PyTorch 2.0+
if torch.__version__ < "2.0.0":
    print("PyTorch 2.0 이상 버전을 사용하세요")
```

## 📝 예제

자세한 사용 예제는 `example_usage.py`를 참조하세요:

```bash
python example_usage.py
```

## 🔄 버전 업데이트

### v1.0.0 (PyTorch 최신 버전 호환)
- PyTorch 2.0+ 완전 지원
- `@torch.inference_mode()` 사용으로 메모리 효율성 향상
- 최신 torchvision InterpolationMode 사용
- 타입 힌트 개선
- 오류 처리 강화
- 배치 처리 최적화

## 🤝 기여

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## 📄 라이선스

MIT License

## 📞 지원

문제가 있거나 질문이 있으시면 이슈를 생성해주세요.