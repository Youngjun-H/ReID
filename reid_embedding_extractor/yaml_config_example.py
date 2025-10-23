#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YAML 설정 파일을 사용한 ReID 임베딩 추출기 예제
다양한 모델별 설정 파일을 자동으로 로드
"""

import os
import sys
import numpy as np
from PIL import Image
import torch

# 독립적인 임베딩 추출기 import
sys.path.append(os.path.dirname(__file__))

try:
    # 패키지로 import될 때
    from .embedding_extractor import ReIDEmbeddingExtractor
    from .config import get_available_configs, find_config_file
except ImportError:
    # 직접 실행될 때
    from embedding_extractor import ReIDEmbeddingExtractor
    from config import get_available_configs, find_config_file


def create_sample_images():
    """샘플 이미지 생성"""
    images = []
    colors = ['red', 'green', 'blue', 'yellow', 'purple']
    
    for i, color in enumerate(colors):
        img = Image.new('RGB', (128, 384), color=color)
        images.append(img)
        print(f"샘플 이미지 {i+1} 생성: {color} 색상")
    
    return images


def test_model_with_config(model_name: str, model_path: str):
    """특정 모델과 설정으로 테스트"""
    print(f"\n{'='*60}")
    print(f"모델: {model_name}")
    print(f"{'='*60}")
    
    # 1. 설정 파일 확인
    config_path = find_config_file(model_name)
    if config_path:
        print(f"✓ 설정 파일 발견: {config_path}")
    else:
        print(f"⚠ 설정 파일을 찾을 수 없습니다. 기본 설정 사용")
    
    # 2. 더미 모델 생성 (실제 사용 시에는 훈련된 모델 사용)
    print(f"\n1. 더미 모델 생성...")
    from model_factory import make_model
    from config import cfg
    
    # 설정 파일이 있으면 로드
    if config_path:
        cfg.merge_from_file(config_path)
        print(f"   - 설정 파일 로드: {config_path}")
        print(f"   - 모델 타입: {cfg.MODEL.transformer_type}")
        print(f"   - 입력 크기: {cfg.INPUT.size_train}")
    
    model = make_model(cfg, num_class=1000, camera_num=6, view_num=1, semantic_weight=0.2)
    torch.save(model.state_dict(), model_path)
    print(f"   ✓ 더미 모델 저장: {model_path}")
    
    # 3. 임베딩 추출기 초기화
    print(f"\n2. 임베딩 추출기 초기화...")
    try:
        extractor = ReIDEmbeddingExtractor(
            model_path=model_path,
            device='cpu',
            semantic_weight=0.2,
            image_size=(384, 128),
            normalize_features=True
        )
        print("   ✓ 임베딩 추출기 초기화 성공!")
    except Exception as e:
        print(f"   ✗ 임베딩 추출기 초기화 실패: {e}")
        return False
    
    # 4. 이미지 처리 테스트
    print(f"\n3. 이미지 처리 테스트...")
    try:
        dummy_image = Image.new('RGB', (128, 384), color='blue')
        processed = extractor.preprocess_image(dummy_image)
        print(f"   ✓ 이미지 전처리 성공! 크기: {processed.shape}")
    except Exception as e:
        print(f"   ✗ 이미지 전처리 실패: {e}")
        return False
    
    # 5. 정리
    if os.path.exists(model_path):
        os.remove(model_path)
        print(f"   ✓ 테스트 파일 정리: {model_path}")
    
    return True


def main():
    """메인 함수"""
    print("=" * 80)
    print("YAML 설정 파일을 사용한 ReID 임베딩 추출기 예제")
    print("=" * 80)
    
    # 1. 사용 가능한 설정 파일 확인
    print("\n1. 사용 가능한 설정 파일 확인...")
    available_configs = get_available_configs()
    if available_configs:
        print(f"   ✓ 사용 가능한 설정: {', '.join(available_configs)}")
    else:
        print("   ⚠ 설정 파일을 찾을 수 없습니다.")
    
    # 2. 각 모델별로 테스트
    models_to_test = [
        ('swin_base', 'swin_base_model.pth'),
        ('swin_tiny', 'swin_tiny_model.pth'),
        ('swin_small', 'swin_small_model.pth'),
    ]
    
    print(f"\n2. 모델별 테스트 시작...")
    success_count = 0
    
    for model_name, model_path in models_to_test:
        try:
            if test_model_with_config(model_name, model_path):
                success_count += 1
        except Exception as e:
            print(f"   ✗ {model_name} 테스트 실패: {e}")
    
    # 3. 결과 요약
    print(f"\n{'='*80}")
    print(f"테스트 결과 요약")
    print(f"{'='*80}")
    print(f"✓ 성공한 모델: {success_count}/{len(models_to_test)}")
    print(f"✓ 사용 가능한 설정: {len(available_configs)}개")
    
    if success_count == len(models_to_test):
        print(f"\n🎉 모든 모델이 성공적으로 작동합니다!")
    else:
        print(f"\n⚠ 일부 모델에서 문제가 발생했습니다.")
    
    print(f"\n주요 특징:")
    print(f"✓ YAML 설정 파일 자동 로드")
    print(f"✓ 모델별 맞춤 설정")
    print(f"✓ 확장 가능한 구조")
    print(f"✓ SOLIDER_REID 독립성 유지")


if __name__ == "__main__":
    main()
