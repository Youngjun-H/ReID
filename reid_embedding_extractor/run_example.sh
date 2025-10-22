#!/bin/bash
# ReID Embedding Extractor 예제 실행 스크립트 - PyTorch 최신 버전 호환

echo "ReID Embedding Extractor 예제 실행 - PyTorch 최신 버전 호환"
echo "=========================================================="

# PyTorch 버전 확인
echo "1. PyTorch 버전 확인 중..."
python -c "
import torch
print(f'PyTorch 버전: {torch.__version__}')
print(f'CUDA 사용 가능: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA 버전: {torch.version.cuda}')
    print(f'GPU 개수: {torch.cuda.device_count()}')
"

# 기본 설정 파일 생성
echo ""
echo "2. 기본 설정 파일 생성 중..."
python -c "
from embedding_extractor import create_default_config
with open('default_config.yml', 'w') as f:
    f.write(create_default_config())
print('기본 설정 파일 생성 완료: default_config.yml')
"

# 예제 이미지 디렉토리 생성
echo ""
echo "3. 예제 이미지 디렉토리 생성 중..."
mkdir -p sample_images

# 더미 이미지 생성 (PIL 사용)
echo "4. 더미 이미지 생성 중..."
python -c "
from PIL import Image
import os

# 샘플 이미지들 생성
colors = ['red', 'green', 'blue', 'yellow', 'purple']
for i, color in enumerate(colors):
    img = Image.new('RGB', (128, 384), color=color)
    img.save(f'sample_images/person_{i+1}.jpg')
    print(f'생성됨: sample_images/person_{i+1}.jpg')
"

# 예제 실행
echo ""
echo "5. 사용 예제 실행 중..."
python example_usage.py

echo ""
echo "=========================================================="
echo "예제 실행 완료!"
echo ""
echo "PyTorch 최신 버전의 장점:"
echo "- @torch.inference_mode() 사용으로 메모리 효율성 향상"
echo "- 최신 torchvision InterpolationMode 사용"
echo "- 향상된 성능과 안정성"
echo ""
echo "실제 사용을 위해서는:"
echo "1. 훈련된 SOLIDER_REID 모델을 준비하세요"
echo "2. inference.py 스크립트를 사용하세요"
echo ""
echo "예제 명령어:"
echo "python inference.py --model_path path/to/model.pth --input sample_images/ --output embeddings.npy"