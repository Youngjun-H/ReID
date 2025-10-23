#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
독립적인 ReID 임베딩 추출기 사용 예제
SOLIDER_REID 의존성 없이 완전히 독립적으로 작동
"""

import os
import sys
import numpy as np
from PIL import Image
import torch

# 독립적인 임베딩 추출기 import
from embedding_extractor import ReIDEmbeddingExtractor


def create_sample_images():
    """샘플 이미지 생성"""
    images = []
    colors = ['red', 'green', 'blue', 'yellow', 'purple']
    
    for i, color in enumerate(colors):
        img = Image.new('RGB', (128, 384), color=color)
        images.append(img)
        print(f"샘플 이미지 {i+1} 생성: {color} 색상")
    
    return images


def main():
    """메인 함수"""
    print("=" * 60)
    print("독립적인 ReID 임베딩 추출기 사용 예제")
    print("=" * 60)
    
    # 1. 샘플 이미지 생성
    print("\n1. 샘플 이미지 생성...")
    sample_images = create_sample_images()
    
    # 2. 더미 모델 생성 (실제 사용 시에는 훈련된 모델 사용)
    print("\n2. 더미 모델 생성...")
    from model_factory import make_model
    from config import cfg
    
    model = make_model(cfg, num_class=1000, camera_num=6, view_num=1, semantic_weight=0.2)
    model_path = "dummy_model.pth"
    torch.save(model.state_dict(), model_path)
    print(f"더미 모델 저장: {model_path}")
    
    # 3. 임베딩 추출기 초기화
    print("\n3. 임베딩 추출기 초기화...")
    try:
        extractor = ReIDEmbeddingExtractor(
            model_path=model_path,
            device='cpu',  # GPU가 있는 경우 'cuda' 사용
            semantic_weight=0.2,
            image_size=(384, 128),
            normalize_features=True
        )
        print("✓ 임베딩 추출기 초기화 성공!")
    except Exception as e:
        print(f"✗ 임베딩 추출기 초기화 실패: {e}")
        return
    
    # 4. 단일 이미지 임베딩 추출
    print("\n4. 단일 이미지 임베딩 추출...")
    try:
        embedding = extractor.extract_embedding(sample_images[0])
        print(f"✓ 임베딩 추출 성공!")
        print(f"  - 임베딩 차원: {len(embedding)}")
        print(f"  - 임베딩 샘플 (처음 5개): {embedding[:5]}")
        print(f"  - L2 노름: {np.linalg.norm(embedding):.4f}")
    except Exception as e:
        print(f"✗ 임베딩 추출 실패: {e}")
    
    # 5. 배치 임베딩 추출
    print("\n5. 배치 임베딩 추출...")
    try:
        embeddings = extractor.extract_embeddings_batch(sample_images, batch_size=2)
        print(f"✓ 배치 임베딩 추출 성공!")
        print(f"  - 추출된 임베딩 수: {len(embeddings)}")
        print(f"  - 각 임베딩 차원: {len(embeddings[0])}")
    except Exception as e:
        print(f"✗ 배치 임베딩 추출 실패: {e}")
    
    # 6. 유사도 계산
    print("\n6. 유사도 계산...")
    try:
        if len(embeddings) >= 2:
            similarity = extractor.compute_similarity(embeddings[0], embeddings[1])
            print(f"✓ 유사도 계산 성공!")
            print(f"  - 이미지 0 vs 이미지 1 유사도: {similarity:.4f}")
            
            # 가장 유사한 이미지 찾기
            query_embedding = embeddings[0]
            gallery_embeddings = embeddings[1:]
            best_idx, best_sim = extractor.find_most_similar(query_embedding, gallery_embeddings)
            print(f"  - 가장 유사한 이미지: {best_idx + 1} (유사도: {best_sim:.4f})")
    except Exception as e:
        print(f"✗ 유사도 계산 실패: {e}")
    
    # 7. 유사도 행렬 계산
    print("\n7. 유사도 행렬 계산...")
    try:
        if len(embeddings) >= 3:
            query_embeddings = embeddings[:2]
            gallery_embeddings = embeddings[2:]
            similarity_matrix = extractor.compute_similarity_matrix(query_embeddings, gallery_embeddings)
            print(f"✓ 유사도 행렬 계산 성공!")
            print(f"  - 유사도 행렬 크기: {similarity_matrix.shape}")
            print(f"  - 유사도 행렬:")
            print(similarity_matrix)
    except Exception as e:
        print(f"✗ 유사도 행렬 계산 실패: {e}")
    
    # 8. 정리
    print("\n8. 정리...")
    if os.path.exists(model_path):
        os.remove(model_path)
        print(f"✓ 더미 모델 파일 삭제: {model_path}")
    
    print("\n" + "=" * 60)
    print("독립적인 ReID 임베딩 추출기 예제 완료!")
    print("=" * 60)
    print("\n주요 특징:")
    print("✓ SOLIDER_REID 의존성 완전 제거")
    print("✓ 독립적으로 작동")
    print("✓ PyTorch 최신 버전 호환")
    print("✓ 다양한 백본 모델 지원 (Swin Transformer, ViT)")
    print("✓ 효율적인 배치 처리")
    print("✓ 유사도 계산 및 검색 기능")


if __name__ == "__main__":
    main()
