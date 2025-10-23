#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
다중 데이터셋 설정 파일을 사용한 ReID 임베딩 추출기 예제
MSMT17과 Market1501 데이터셋용 설정 파일 지원
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
    from .config import get_available_configs, get_available_datasets, find_config_file
except ImportError:
    # 직접 실행될 때
    from embedding_extractor import ReIDEmbeddingExtractor
    from config import get_available_configs, get_available_datasets, find_config_file


def test_dataset_configs():
    """다양한 데이터셋 설정 테스트"""
    print("=" * 80)
    print("다중 데이터셋 설정 파일 지원 테스트")
    print("=" * 80)
    
    # 1. 사용 가능한 데이터셋 확인
    print("\n1. 사용 가능한 데이터셋 확인...")
    datasets = get_available_datasets()
    print(f"   사용 가능한 데이터셋: {datasets}")
    
    # 2. 전체 설정 파일 확인
    print("\n2. 전체 설정 파일 확인...")
    all_configs = get_available_configs('all')
    print(f"   전체 설정 파일: {len(all_configs)}개")
    for config in all_configs:
        print(f"     - {config}")
    
    # 3. 각 데이터셋별 설정 파일 확인
    print("\n3. 각 데이터셋별 설정 파일 확인...")
    for dataset in datasets:
        configs = get_available_configs(dataset)
        print(f"   {dataset}: {configs}")
    
    return datasets


def test_model_with_dataset(model_name: str, dataset: str):
    """특정 모델과 데이터셋으로 테스트"""
    print(f"\n{'='*60}")
    print(f"모델: {model_name} | 데이터셋: {dataset}")
    print(f"{'='*60}")
    
    # 1. 설정 파일 확인
    config_path = find_config_file(model_name, dataset)
    if config_path:
        print(f"✓ 설정 파일 발견: {config_path}")
        
        # 설정 파일 내용 확인
        with open(config_path, 'r', encoding='utf-8') as f:
            content = f.read()
            if 'market1501' in content:
                print(f"  - Market1501 데이터셋 설정 확인")
            elif 'msmt17' in content:
                print(f"  - MSMT17 데이터셋 설정 확인")
    else:
        print(f"⚠ 설정 파일을 찾을 수 없습니다.")
        return False
    
    # 2. 더미 모델 생성
    print(f"\n1. 더미 모델 생성...")
    from model_factory import make_model
    from config import cfg
    
    # 설정 파일 로드
    cfg.merge_from_file(config_path)
    print(f"   - 설정 파일 로드: {config_path}")
    print(f"   - 모델 타입: {cfg.MODEL.transformer_type}")
    print(f"   - 입력 크기: {cfg.INPUT.size_train}")
    
    model = make_model(cfg, num_class=1000, camera_num=6, view_num=1, semantic_weight=0.2)
    model_path = f"{model_name}_{dataset}_model.pth"
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
    print("다중 데이터셋 설정 파일을 사용한 ReID 임베딩 추출기 예제")
    print("=" * 80)
    
    # 1. 데이터셋 설정 확인
    datasets = test_dataset_configs()
    
    # 2. 각 데이터셋별로 모델 테스트
    print(f"\n4. 각 데이터셋별 모델 테스트...")
    models = ['swin_base', 'swin_tiny', 'swin_small']
    
    success_count = 0
    total_tests = 0
    
    for dataset in datasets:
        for model in models:
            total_tests += 1
            try:
                if test_model_with_dataset(model, dataset):
                    success_count += 1
            except Exception as e:
                print(f"   ✗ {model} ({dataset}) 테스트 실패: {e}")
    
    # 3. 결과 요약
    print(f"\n{'='*80}")
    print(f"테스트 결과 요약")
    print(f"{'='*80}")
    print(f"✓ 성공한 테스트: {success_count}/{total_tests}")
    print(f"✓ 지원하는 데이터셋: {len(datasets)}개")
    print(f"✓ 지원하는 모델: {len(models)}개")
    
    if success_count == total_tests:
        print(f"\n🎉 모든 테스트가 성공적으로 완료되었습니다!")
    else:
        print(f"\n⚠ 일부 테스트에서 문제가 발생했습니다.")
    
    print(f"\n주요 특징:")
    print(f"✓ 다중 데이터셋 지원 (MSMT17, Market1501)")
    print(f"✓ 모델별 맞춤 설정")
    print(f"✓ 자동 설정 파일 감지")
    print(f"✓ 확장 가능한 구조")
    print(f"✓ SOLIDER_REID 독립성 유지")


if __name__ == "__main__":
    main()
