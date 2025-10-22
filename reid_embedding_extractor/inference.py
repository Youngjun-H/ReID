#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ReID 임베딩 추출 추론 스크립트 - PyTorch 최신 버전 호환
단일 이미지 또는 이미지 폴더에서 ReID 임베딩을 추출합니다.
"""

import os
import sys
import argparse
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any
import logging

from embedding_extractor import ReIDEmbeddingExtractor


def setup_logging(log_level: str = "INFO"):
    """로깅 설정"""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def find_image_files(directory: str, extensions: List[str] = None) -> List[str]:
    """
    디렉토리에서 이미지 파일 찾기
    
    Args:
        directory: 검색할 디렉토리
        extensions: 지원할 확장자 리스트
        
    Returns:
        List[str]: 이미지 파일 경로 리스트
    """
    if extensions is None:
        extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
    
    image_files = []
    directory = Path(directory)
    
    if directory.is_file():
        # 단일 파일인 경우
        if directory.suffix.lower() in extensions:
            image_files.append(str(directory))
    else:
        # 디렉토리인 경우
        for ext in extensions:
            image_files.extend(directory.glob(f"**/*{ext}"))
            image_files.extend(directory.glob(f"**/*{ext.upper()}"))
    
    return [str(f) for f in image_files]


def save_embeddings(embeddings: List[np.ndarray], 
                   image_paths: List[str], 
                   output_path: str,
                   save_format: str = "npy"):
    """
    임베딩을 파일로 저장
    
    Args:
        embeddings: 임베딩 리스트
        image_paths: 이미지 경로 리스트
        output_path: 출력 파일 경로
        save_format: 저장 형식 ('npy', 'json', 'txt')
    """
    if save_format == "npy":
        # NumPy 형식으로 저장
        np.save(output_path, np.array(embeddings))
        print(f"임베딩이 NumPy 형식으로 저장되었습니다: {output_path}")
        
    elif save_format == "json":
        # JSON 형식으로 저장
        data = {
            "embeddings": [emb.tolist() for emb in embeddings],
            "image_paths": image_paths,
            "embedding_dim": len(embeddings[0]) if embeddings else 0
        }
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"임베딩이 JSON 형식으로 저장되었습니다: {output_path}")
        
    elif save_format == "txt":
        # 텍스트 형식으로 저장
        with open(output_path, 'w') as f:
            for i, (path, emb) in enumerate(zip(image_paths, embeddings)):
                f.write(f"# Image {i}: {path}\n")
                f.write(" ".join(map(str, emb)) + "\n\n")
        print(f"임베딩이 텍스트 형식으로 저장되었습니다: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="ReID 임베딩 추출 - PyTorch 최신 버전 호환")
    
    # 필수 인자
    parser.add_argument("--model_path", type=str, required=True,
                       help="훈련된 SOLIDER_REID 모델 경로")
    parser.add_argument("--input", type=str, required=True,
                       help="입력 이미지 파일 또는 디렉토리 경로")
    
    # 선택 인자
    parser.add_argument("--config_path", type=str, default=None,
                       help="설정 파일 경로 (기본값: default_config.yml)")
    parser.add_argument("--output", type=str, default="embeddings.npy",
                       help="출력 파일 경로 (기본값: embeddings.npy)")
    parser.add_argument("--output_format", type=str, default="npy",
                       choices=["npy", "json", "txt"],
                       help="출력 형식 (기본값: npy)")
    parser.add_argument("--device", type=str, default=None,
                       choices=["cuda", "cpu"],
                       help="사용할 디바이스 (기본값: 자동 선택)")
    parser.add_argument("--semantic_weight", type=float, default=0.2,
                       help="시맨틱 가중치 (기본값: 0.2)")
    parser.add_argument("--image_size", type=int, nargs=2, default=[384, 128],
                       help="이미지 크기 [height width] (기본값: 384 128)")
    parser.add_argument("--batch_size", type=int, default=32,
                       help="배치 크기 (기본값: 32)")
    parser.add_argument("--normalize", action="store_true", default=True,
                       help="특징 벡터 L2 정규화 (기본값: True)")
    parser.add_argument("--log_level", type=str, default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="로깅 레벨 (기본값: INFO)")
    
    args = parser.parse_args()
    
    # 로깅 설정
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    # 설정 파일 경로 설정
    if args.config_path is None:
        config_path = os.path.join(os.path.dirname(__file__), "default_config.yml")
        if not os.path.exists(config_path):
            logger.warning("기본 설정 파일이 없습니다. 기본 설정을 사용합니다.")
            config_path = None
    else:
        config_path = args.config_path
    
    try:
        # ReID 임베딩 추출기 초기화
        logger.info("ReID 임베딩 추출기 초기화 중...")
        extractor = ReIDEmbeddingExtractor(
            model_path=args.model_path,
            config_path=config_path,
            device=args.device,
            semantic_weight=args.semantic_weight,
            image_size=tuple(args.image_size),
            normalize_features=args.normalize
        )
        
        # 이미지 파일 찾기
        logger.info(f"이미지 파일 검색 중: {args.input}")
        image_files = find_image_files(args.input)
        
        if not image_files:
            logger.error("이미지 파일을 찾을 수 없습니다.")
            return
        
        logger.info(f"총 {len(image_files)}개의 이미지 파일을 찾았습니다.")
        
        # 임베딩 추출
        logger.info("임베딩 추출 시작...")
        if len(image_files) == 1:
            # 단일 이미지
            embedding = extractor.extract_embedding(image_files[0])
            embeddings = [embedding]
        else:
            # 여러 이미지 (배치 처리)
            embeddings = extractor.extract_embeddings_batch(
                image_files, 
                batch_size=args.batch_size
            )
        
        logger.info(f"임베딩 추출 완료: {len(embeddings)}개")
        logger.info(f"임베딩 차원: {len(embeddings[0]) if embeddings else 0}")
        
        # 결과 저장
        save_embeddings(embeddings, image_files, args.output, args.output_format)
        
        # 유사도 분석 (여러 이미지인 경우)
        if len(embeddings) > 1:
            logger.info("이미지 간 유사도 분석...")
            similarities = []
            for i in range(len(embeddings)):
                for j in range(i + 1, len(embeddings)):
                    sim = extractor.compute_similarity(embeddings[i], embeddings[j])
                    similarities.append((i, j, sim))
                    logger.info(f"이미지 {i} vs {j}: 유사도 = {sim:.4f}")
            
            if similarities:
                avg_similarity = np.mean([s[2] for s in similarities])
                logger.info(f"평균 유사도: {avg_similarity:.4f}")
        
        logger.info("임베딩 추출 완료!")
        
    except Exception as e:
        logger.error(f"오류 발생: {e}")
        raise


if __name__ == "__main__":
    main()