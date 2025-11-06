"""
디렉토리 처리 모듈
디렉토리 내 이미지를 처리하고 label.txt를 생성하는 기능
"""
import os
from pathlib import Path
from typing import List, Optional
try:
    from .ocr_client import VLLMOCRClient
except ImportError:
    from ocr_client import VLLMOCRClient


# 지원하는 이미지 확장자
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'}


def get_image_files(directory: str) -> List[str]:
    """
    디렉토리 내의 모든 이미지 파일 경로 반환
    
    Args:
        directory: 디렉토리 경로
        
    Returns:
        이미지 파일 경로 리스트
    """
    directory_path = Path(directory)
    if not directory_path.exists():
        raise ValueError(f"디렉토리가 존재하지 않습니다: {directory}")
    
    image_files = []
    for ext in IMAGE_EXTENSIONS:
        image_files.extend(directory_path.glob(f"*{ext}"))
        image_files.extend(directory_path.glob(f"*{ext.upper()}"))
    
    # 문자열로 변환하고 정렬
    return sorted([str(f) for f in image_files])


def process_directory(
    directory: str,
    ocr_client: VLLMOCRClient,
    label_filename: str = "label.txt"
) -> Optional[str]:
    """
    디렉토리 내의 모든 이미지에 대해 OCR을 수행하고 label.txt 생성
    
    Args:
        directory: 처리할 디렉토리 경로
        ocr_client: OCR 클라이언트 인스턴스
        label_filename: 생성할 라벨 파일 이름
        
    Returns:
        생성된 라벨 텍스트, 실패 시 None
    """
    print(f"\n{'='*80}")
    print(f"디렉토리 처리 시작: {directory}")
    print(f"{'='*80}")
    
    # 이미지 파일 찾기
    image_files = get_image_files(directory)
    
    if not image_files:
        print(f"이미지 파일을 찾을 수 없습니다: {directory}")
        return None
    
    print(f"발견된 이미지 파일 수: {len(image_files)}")
    
    # 모든 이미지에 대해 OCR 수행
    texts = ocr_client.process_images(image_files)
    
    # 가장 많이 나온 결과를 라벨로 사용
    label = ocr_client.get_most_common_label(texts)
    
    if not label:
        print(f"유효한 라벨을 찾을 수 없습니다: {directory}")
        return None
    
    # label.txt 저장
    label_path = Path(directory) / label_filename
    with open(label_path, 'w', encoding='utf-8') as f:
        f.write(label)
    
    print(f"라벨 저장 완료: {label_path}")
    print(f"라벨 내용: {label}")
    print(f"통계: 총 {len(texts)}개 결과 중 '{label}'가 {texts.count(label)}회 나타남")
    
    return label


def process_recursive(
    root_directory: str,
    ocr_client: VLLMOCRClient,
    label_filename: str = "label.txt",
    max_depth: Optional[int] = None
) -> dict:
    """
    루트 디렉토리부터 재귀적으로 하위 디렉토리를 처리
    
    Args:
        root_directory: 루트 디렉토리 경로
        ocr_client: OCR 클라이언트 인스턴스
        label_filename: 생성할 라벨 파일 이름
        max_depth: 최대 탐색 깊이 (None이면 제한 없음)
        
    Returns:
        처리 결과 딕셔너리 {디렉토리: 라벨}
    """
    root_path = Path(root_directory)
    if not root_path.exists():
        raise ValueError(f"디렉토리가 존재하지 않습니다: {root_directory}")
    
    results = {}
    
    # 현재 디렉토리에 이미지가 있는지 확인
    image_files = get_image_files(str(root_path))
    
    if image_files:
        # 이미지가 있으면 현재 디렉토리 처리
        label = process_directory(str(root_path), ocr_client, label_filename)
        if label:
            results[str(root_path)] = label
    else:
        # 이미지가 없으면 하위 디렉토리 탐색
        print(f"이미지가 없는 디렉토리, 하위 디렉토리 탐색: {root_path}")
        
        def process_dir_recursive(current_dir: Path, depth: int = 0):
            if max_depth is not None and depth > max_depth:
                return
            
            # 현재 디렉토리의 직접 하위 디렉토리만 확인
            for item in current_dir.iterdir():
                if item.is_dir():
                    # 하위 디렉토리에 이미지가 있는지 확인
                    sub_image_files = get_image_files(str(item))
                    if sub_image_files:
                        # 이미지가 있으면 처리
                        label = process_directory(str(item), ocr_client, label_filename)
                        if label:
                            results[str(item)] = label
                    else:
                        # 이미지가 없으면 더 깊이 탐색
                        process_dir_recursive(item, depth + 1)
        
        process_dir_recursive(root_path)
    
    return results

