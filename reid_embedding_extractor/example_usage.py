#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ReID 임베딩 추출기 사용 예제 - PyTorch 최신 버전 호환
다양한 시나리오에서 ReIDEmbeddingExtractor를 사용하는 방법을 보여줍니다.
"""

import os
import sys
import numpy as np
from PIL import Image
import logging

# 현재 디렉토리를 Python 경로에 추가
sys.path.append(os.path.dirname(__file__))

from embedding_extractor import ReIDEmbeddingExtractor


def setup_logging():
    """로깅 설정"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def example_single_image():
    """단일 이미지에서 임베딩 추출 예제"""
    print("=" * 60)
    print("예제 1: 단일 이미지에서 임베딩 추출")
    print("=" * 60)
    
    # 모델 경로 설정 (실제 경로로 변경 필요)
    model_path = "../models/SOLIDER_REID/log/msmt17/swin_base/transformer_120.pth"
    config_path = "../models/SOLIDER_REID/configs/msmt17/swin_base.yml"
    
    # 더미 이미지 생성 (실제 사용 시에는 실제 이미지 경로 사용)
    dummy_image = Image.new('RGB', (128, 384), color='red')
    
    try:
        # 임베딩 추출기 초기화
        extractor = ReIDEmbeddingExtractor(
            model_path=model_path,
            config_path=config_path,
            device='cpu',  # GPU가 없는 경우
            semantic_weight=0.2,
            image_size=(384, 128),
            normalize_features=True
        )
        
        # 임베딩 추출
        embedding = extractor.extract_embedding(dummy_image)
        
        print(f"임베딩 차원: {len(embedding)}")
        print(f"임베딩 샘플 (처음 10개): {embedding[:10]}")
        print(f"임베딩 L2 노름: {np.linalg.norm(embedding):.4f}")
        
    except FileNotFoundError as e:
        print(f"모델 파일을 찾을 수 없습니다: {e}")
        print("실제 모델 경로로 변경해주세요.")
    except Exception as e:
        print(f"오류 발생: {e}")


def example_batch_processing():
    """배치 처리 예제"""
    print("\n" + "=" * 60)
    print("예제 2: 배치 처리로 여러 이미지 임베딩 추출")
    print("=" * 60)
    
    # 더미 이미지들 생성
    images = []
    colors = ['red', 'green', 'blue', 'yellow', 'purple']
    
    for i, color in enumerate(colors):
        img = Image.new('RGB', (128, 384), color=color)
        images.append(img)
    
    print(f"처리할 이미지 수: {len(images)}")
    
    # 실제 사용 시에는 다음과 같이 사용:
    # model_path = "path/to/your/model.pth"
    # config_path = "path/to/your/config.yml"
    # extractor = ReIDEmbeddingExtractor(model_path, config_path)
    # embeddings = extractor.extract_embeddings_batch(images, batch_size=2)
    
    # 더미 임베딩 생성 (실제 사용 시에는 위의 코드 사용)
    embeddings = [np.random.randn(2048) for _ in images]
    
    print(f"추출된 임베딩 수: {len(embeddings)}")
    print(f"각 임베딩 차원: {len(embeddings[0])}")


def example_similarity_computation():
    """유사도 계산 예제"""
    print("\n" + "=" * 60)
    print("예제 3: 임베딩 간 유사도 계산")
    print("=" * 60)
    
    # 더미 임베딩 생성
    embedding1 = np.random.randn(2048)
    embedding2 = np.random.randn(2048)
    embedding3 = embedding1 + np.random.randn(2048) * 0.1  # 유사한 임베딩
    
    # L2 정규화
    embedding1 = embedding1 / np.linalg.norm(embedding1)
    embedding2 = embedding2 / np.linalg.norm(embedding2)
    embedding3 = embedding3 / np.linalg.norm(embedding3)
    
    # 유사도 계산 (실제 사용 시에는 ReIDEmbeddingExtractor 인스턴스 사용)
    sim_1_2 = np.dot(embedding1, embedding2)
    sim_1_3 = np.dot(embedding1, embedding3)
    sim_2_3 = np.dot(embedding2, embedding3)
    
    print(f"임베딩 1 vs 임베딩 2 유사도: {sim_1_2:.4f}")
    print(f"임베딩 1 vs 임베딩 3 유사도: {sim_1_3:.4f}")
    print(f"임베딩 2 vs 임베딩 3 유사도: {sim_2_3:.4f}")
    
    # 가장 유사한 임베딩 찾기
    gallery_embeddings = [embedding2, embedding3]
    similarities = [sim_1_2, sim_1_3]
    best_idx = np.argmax(similarities)
    best_similarity = similarities[best_idx]
    
    print(f"임베딩 1과 가장 유사한 것은 갤러리 {best_idx + 1} (유사도: {best_similarity:.4f})")


def example_similarity_matrix():
    """유사도 행렬 계산 예제"""
    print("\n" + "=" * 60)
    print("예제 4: 유사도 행렬 계산")
    print("=" * 60)
    
    # 더미 임베딩 세트 생성
    query_embeddings = [np.random.randn(2048) for _ in range(3)]
    gallery_embeddings = [np.random.randn(2048) for _ in range(5)]
    
    # 정규화
    query_embeddings = [emb / np.linalg.norm(emb) for emb in query_embeddings]
    gallery_embeddings = [emb / np.linalg.norm(emb) for emb in gallery_embeddings]
    
    # 유사도 행렬 계산 (실제 사용 시에는 ReIDEmbeddingExtractor 인스턴스 사용)
    similarity_matrix = np.dot(np.array(query_embeddings), np.array(gallery_embeddings).T)
    
    print(f"쿼리 임베딩 수: {len(query_embeddings)}")
    print(f"갤러리 임베딩 수: {len(gallery_embeddings)}")
    print(f"유사도 행렬 크기: {similarity_matrix.shape}")
    print(f"유사도 행렬 샘플:")
    print(similarity_matrix[:2, :3])  # 처음 2x3 부분만 출력


def example_file_processing():
    """파일 처리 예제"""
    print("\n" + "=" * 60)
    print("예제 5: 파일에서 이미지 로드 및 처리")
    print("=" * 60)
    
    # 이미지 파일 경로 (실제 경로로 변경 필요)
    image_paths = [
        "sample_images/person1.jpg",
        "sample_images/person2.jpg",
        "sample_images/person3.jpg"
    ]
    
    print("처리할 이미지 파일들:")
    for i, path in enumerate(image_paths):
        exists = "✓" if os.path.exists(path) else "✗"
        print(f"  {i+1}. {path} {exists}")
    
    # 실제 사용 시에는 다음과 같이 사용:
    # extractor = ReIDEmbeddingExtractor(model_path, config_path)
    # embeddings = []
    # for path in image_paths:
    #     if os.path.exists(path):
    #         emb = extractor.extract_embedding(path)
    #         embeddings.append(emb)
    #         print(f"처리 완료: {path}")
    
    print("\n실제 사용 시에는 위의 주석 처리된 코드를 사용하세요.")


def example_configuration():
    """설정 예제"""
    print("\n" + "=" * 60)
    print("예제 6: 다양한 설정 옵션")
    print("=" * 60)
    
    print("ReIDEmbeddingExtractor 설정 옵션:")
    print("  - model_path: 훈련된 모델 파일 경로 (필수)")
    print("  - config_path: 설정 파일 경로 (선택사항)")
    print("  - device: 'cuda' 또는 'cpu' (기본값: 자동 선택)")
    print("  - semantic_weight: 시맨틱 가중치 (기본값: 0.2)")
    print("  - image_size: 입력 이미지 크기 (height, width)")
    print("  - normalize_features: L2 정규화 여부 (기본값: True)")
    
    print("\n추론 스크립트 사용법:")
    print("  python inference.py \\")
    print("    --model_path path/to/model.pth \\")
    print("    --input path/to/image.jpg \\")
    print("    --output embeddings.npy \\")
    print("    --device cuda \\")
    print("    --semantic_weight 0.2 \\")
    print("    --normalize")


def example_pytorch_compatibility():
    """PyTorch 호환성 예제"""
    print("\n" + "=" * 60)
    print("예제 7: PyTorch 최신 버전 호환성")
    print("=" * 60)
    
    import torch
    print(f"PyTorch 버전: {torch.__version__}")
    print(f"CUDA 사용 가능: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA 버전: {torch.version.cuda}")
        print(f"GPU 개수: {torch.cuda.device_count()}")
    
    print("\n주요 개선사항:")
    print("  - @torch.inference_mode() 사용으로 메모리 효율성 향상")
    print("  - InterpolationMode.BICUBIC 사용으로 최신 torchvision 호환")
    print("  - 타입 힌트 개선으로 코드 가독성 향상")
    print("  - 오류 처리 강화")
    print("  - 배치 처리 최적화")


def create_sample_config():
    """샘플 설정 파일 생성"""
    print("\n" + "=" * 60)
    print("예제 8: 샘플 설정 파일 생성")
    print("=" * 60)
    
    from embedding_extractor import create_default_config
    
    config_content = create_default_config()
    
    # 설정 파일 저장
    config_path = "sample_config.yml"
    with open(config_path, 'w') as f:
        f.write(config_content)
    
    print(f"샘플 설정 파일이 생성되었습니다: {config_path}")
    print("이 파일을 수정하여 사용하세요.")


def main():
    """모든 예제 실행"""
    setup_logging()
    
    print("ReID 임베딩 추출기 사용 예제 - PyTorch 최신 버전 호환")
    print("=" * 60)
    
    # 예제 실행
    example_single_image()
    example_batch_processing()
    example_similarity_computation()
    example_similarity_matrix()
    example_file_processing()
    example_configuration()
    example_pytorch_compatibility()
    create_sample_config()
    
    print("\n" + "=" * 60)
    print("모든 예제가 완료되었습니다!")
    print("=" * 60)
    print("\n실제 사용을 위해서는:")
    print("1. 훈련된 SOLIDER_REID 모델을 준비하세요")
    print("2. 적절한 설정 파일을 준비하세요")
    print("3. 이미지 파일들을 준비하세요")
    print("4. inference.py 스크립트를 사용하거나")
    print("5. ReIDEmbeddingExtractor 클래스를 직접 사용하세요")
    print("\nPyTorch 최신 버전의 장점:")
    print("- 더 나은 메모리 효율성")
    print("- 향상된 성능")
    print("- 더 안정적인 추론")


if __name__ == "__main__":
    main()