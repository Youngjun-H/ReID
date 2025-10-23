#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ReID 결과 시각화 도구
Query와 Gallery 디렉토리를 입력으로 받아 ReID 결과를 시각화합니다.
"""

import os
import sys
import argparse
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import logging

# ReID 임베딩 추출기 임포트
from models import SOLIDEREmbeddingExtractor


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
    
    return sorted([str(f) for f in image_files])


def extract_embeddings_for_directory(extractor: SOLIDEREmbeddingExtractor, 
                                   directory: str) -> Tuple[List[str], np.ndarray]:
    """
    디렉토리 내 모든 이미지에서 임베딩 추출
    
    Args:
        extractor: ReID 임베딩 추출기
        directory: 이미지 디렉토리
        
    Returns:
        Tuple[List[str], np.ndarray]: (이미지 경로 리스트, 임베딩 배열)
    """
    image_files = find_image_files(directory)
    
    if not image_files:
        raise ValueError(f"디렉토리에서 이미지를 찾을 수 없습니다: {directory}")
    
    print(f"임베딩 추출 중: {len(image_files)}개 이미지")
    embeddings = extractor.extract_embeddings_batch(image_files)
    
    return image_files, np.array(embeddings)


def compute_similarity_matrix(query_embeddings: np.ndarray, 
                            gallery_embeddings: np.ndarray) -> np.ndarray:
    """
    Query와 Gallery 임베딩 간의 유사도 행렬 계산
    
    Args:
        query_embeddings: Query 임베딩 배열 (N, D)
        gallery_embeddings: Gallery 임베딩 배열 (M, D)
        
    Returns:
        np.ndarray: 유사도 행렬 (N, M)
    """
    # L2 정규화
    query_norm = query_embeddings / np.linalg.norm(query_embeddings, axis=1, keepdims=True)
    gallery_norm = gallery_embeddings / np.linalg.norm(gallery_embeddings, axis=1, keepdims=True)
    
    # 코사인 유사도 계산
    similarity_matrix = np.dot(query_norm, gallery_norm.T)
    
    return similarity_matrix


def get_top_k_matches(similarity_matrix: np.ndarray, 
                     query_paths: List[str], 
                     gallery_paths: List[str], 
                     k: int = 5) -> List[List[Tuple[str, float]]]:
    """
    각 Query에 대해 상위 K개 Gallery 매치 반환
    
    Args:
        similarity_matrix: 유사도 행렬 (N, M)
        query_paths: Query 이미지 경로 리스트
        gallery_paths: Gallery 이미지 경로 리스트
        k: 상위 K개
        
    Returns:
        List[List[Tuple[str, float]]]: 각 Query의 상위 K개 매치 리스트
    """
    top_k_matches = []
    
    for i, query_path in enumerate(query_paths):
        # 상위 k개 인덱스와 유사도 점수
        top_k_indices = np.argsort(similarity_matrix[i])[-k:][::-1]
        top_k_scores = similarity_matrix[i][top_k_indices]
        
        matches = []
        for idx, score in zip(top_k_indices, top_k_scores):
            matches.append((gallery_paths[idx], float(score)))
        
        top_k_matches.append(matches)
    
    return top_k_matches


def create_visualization(query_paths: List[str], 
                        gallery_paths: List[str], 
                        top_k_matches: List[List[Tuple[str, float]]], 
                        k: int = 5,
                        output_path: str = "reid_results.png") -> None:
    """
    ReID 결과 시각화 생성
    
    Args:
        query_paths: Query 이미지 경로 리스트
        gallery_paths: Gallery 이미지 경로 리스트
        top_k_matches: 각 Query의 상위 K개 매치
        k: 상위 K개
        output_path: 출력 이미지 경로
    """
    num_queries = len(query_paths)
    
    # 이미지 크기 설정
    img_width = 160  # 이미지 크기 약간 증가
    img_height = 320  # 이미지 크기 약간 증가
    margin = 20  # 여백 더 증가
    title_height = 50  # 제목 공간 증가
    filename_height = 40  # 파일명 공간 증가
    
    # 전체 캔버스 크기 계산 (제목과 파일명 공간 포함)
    canvas_width = (k + 1) * (img_width + margin) + margin
    canvas_height = num_queries * (img_height + title_height + filename_height + margin) + margin + 80  # 전체 제목 공간 더 추가
    
    # Figure 생성 (더 큰 크기로)
    fig, axes = plt.subplots(num_queries, k + 1, 
                            figsize=(canvas_width / 70, canvas_height / 70))
    
    if num_queries == 1:
        axes = axes.reshape(1, -1)
    
    for i, query_path in enumerate(query_paths):
        # Query 이미지 로드 및 표시
        try:
            query_img = Image.open(query_path).convert('RGB')
            query_img = query_img.resize((img_width, img_height), Image.Resampling.LANCZOS)
            axes[i, 0].imshow(query_img)
            
            # Query 제목 (위쪽에)
            axes[i, 0].set_title(f"Query {i+1}", fontsize=14, fontweight='bold', pad=25)
            axes[i, 0].axis('off')
            
            # Query 이미지 경로 표시 (아래쪽에)
            query_name = Path(query_path).name
            # 파일명이 너무 길면 줄임
            if len(query_name) > 25:
                query_name = query_name[:22] + "..."
            axes[i, 0].text(0.5, -0.12, query_name, fontsize=10, ha='center', va='top', 
                          transform=axes[i, 0].transAxes, wrap=True)
            
        except Exception as e:
            print(f"Query 이미지 로드 실패: {query_path}, 오류: {e}")
            axes[i, 0].text(0.5, 0.5, f"Error\n{Path(query_path).name}", 
                          ha='center', va='center', transform=axes[i, 0].transAxes)
            axes[i, 0].axis('off')
        
        # Gallery 이미지들 표시
        for j, (gallery_path, score) in enumerate(top_k_matches[i]):
            try:
                gallery_img = Image.open(gallery_path).convert('RGB')
                gallery_img = gallery_img.resize((img_width, img_height), Image.Resampling.LANCZOS)
                axes[i, j + 1].imshow(gallery_img)
                
                # 유사도 점수 표시 (위쪽에)
                axes[i, j + 1].set_title(f"Rank {j+1}\nScore: {score:.3f}", 
                                       fontsize=11, color='blue', pad=25)
                axes[i, j + 1].axis('off')
                
                # Gallery 이미지 경로 표시 (아래쪽에)
                gallery_name = Path(gallery_path).name
                # 파일명이 너무 길면 줄임
                if len(gallery_name) > 25:
                    gallery_name = gallery_name[:22] + "..."
                axes[i, j + 1].text(0.5, -0.12, gallery_name, fontsize=10, ha='center', va='top',
                                  transform=axes[i, j + 1].transAxes, wrap=True)
                
                # 상위 매치에 테두리 추가
                if j == 0:  # 가장 유사한 이미지
                    rect = patches.Rectangle((0, 0), img_width-1, img_height-1, 
                                           linewidth=3, edgecolor='red', facecolor='none')
                    axes[i, j + 1].add_patch(rect)
                
            except Exception as e:
                print(f"Gallery 이미지 로드 실패: {gallery_path}, 오류: {e}")
                axes[i, j + 1].text(0.5, 0.5, f"Error\n{Path(gallery_path).name}", 
                                  ha='center', va='center', 
                                  transform=axes[i, j + 1].transAxes)
                axes[i, j + 1].axis('off')
    
    # 레이아웃 조정 (더 많은 여백 제공)
    plt.tight_layout()
    plt.subplots_adjust(
        top=0.92,      # 상단 여백 증가
        bottom=0.08,   # 하단 여백 증가
        left=0.05, 
        right=0.95,
        hspace=0.4,    # 행 간격 증가
        wspace=0.1     # 열 간격 증가
    )
    
    # 제목 추가 (더 큰 여백으로)
    fig.suptitle(f'ReID Results - Top {k} Matches', fontsize=18, fontweight='bold', y=0.98)
    
    # 저장 (더 높은 해상도와 여백)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none', pad_inches=0.5)
    print(f"결과 이미지 저장: {output_path}")
    
    # 표시
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="ReID 결과 시각화 도구")
    
    # 필수 인자
    parser.add_argument("--query_dir", type=str, required=True,
                       help="Query 이미지 디렉토리")
    parser.add_argument("--gallery_dir", type=str, required=True,
                       help="Gallery 이미지 디렉토리")
    parser.add_argument("--model_path", type=str, required=True,
                       help="훈련된 SOLIDER_REID 모델 경로")
    
    # 선택 인자
    parser.add_argument("--config_path", type=str, default=None,
                       help="설정 파일 경로")
    parser.add_argument("--output", type=str, default="reid_results.png",
                       help="출력 이미지 경로 (기본값: reid_results.png)")
    parser.add_argument("--rank", type=int, default=5,
                       help="상위 K개 매치 표시 (기본값: 5)")
    parser.add_argument("--device", type=str, default=None,
                       help="사용할 디바이스 (기본값: 자동 선택)")
    parser.add_argument("--semantic_weight", type=float, default=0.2,
                       help="시맨틱 가중치 (기본값: 0.2)")
    parser.add_argument("--log_level", type=str, default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="로깅 레벨 (기본값: INFO)")
    
    args = parser.parse_args()
    
    # 로깅 설정
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    try:
        # ReID 임베딩 추출기 초기화
        logger.info("ReID 임베딩 추출기 초기화 중...")
        extractor = SOLIDEREmbeddingExtractor(
            model_path=args.model_path,
            config_path=args.config_path,
            device=args.device,
            semantic_weight=args.semantic_weight,
            image_size=(384, 128),
            normalize_features=True
        )
        
        # Query 이미지 임베딩 추출
        logger.info(f"Query 이미지 임베딩 추출 중: {args.query_dir}")
        query_paths, query_embeddings = extract_embeddings_for_directory(
            extractor, args.query_dir
        )
        
        # Gallery 이미지 임베딩 추출
        logger.info(f"Gallery 이미지 임베딩 추출 중: {args.gallery_dir}")
        gallery_paths, gallery_embeddings = extract_embeddings_for_directory(
            extractor, args.gallery_dir
        )
        
        # 유사도 행렬 계산
        logger.info("유사도 행렬 계산 중...")
        similarity_matrix = compute_similarity_matrix(query_embeddings, gallery_embeddings)
        
        # 상위 K개 매치 찾기
        logger.info(f"상위 {args.rank}개 매치 찾는 중...")
        top_k_matches = get_top_k_matches(
            similarity_matrix, query_paths, gallery_paths, args.rank
        )
        
        # 결과 시각화
        logger.info("결과 시각화 생성 중...")
        create_visualization(
            query_paths, gallery_paths, top_k_matches, 
            k=args.rank, output_path=args.output
        )
        
        # 결과 요약 출력
        logger.info("=" * 60)
        logger.info("ReID 결과 요약")
        logger.info("=" * 60)
        logger.info(f"Query 이미지 수: {len(query_paths)}")
        logger.info(f"Gallery 이미지 수: {len(gallery_paths)}")
        logger.info(f"상위 {args.rank}개 매치 표시")
        logger.info(f"결과 이미지: {args.output}")
        
        # 각 Query의 최고 매치 점수 출력
        for i, query_path in enumerate(query_paths):
            best_score = top_k_matches[i][0][1] if top_k_matches[i] else 0.0
            best_match = Path(top_k_matches[i][0][0]).name if top_k_matches[i] else "None"
            logger.info(f"Query {i+1}: {Path(query_path).name} -> {best_match} (Score: {best_score:.3f})")
        
        logger.info("ReID 시각화 완료!")
        
    except Exception as e:
        logger.error(f"오류 발생: {e}")
        raise


if __name__ == "__main__":
    main()
