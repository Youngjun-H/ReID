import os
import re
from pathlib import Path
import argparse


def remove_korean(text: str) -> str:
    """한글 문자를 제거"""
    # 한글 유니코드 범위: \uAC00-\uD7A3
    korean_pattern = re.compile(r'[\uAC00-\uD7A3]+')
    return korean_pattern.sub('', text)


def normalize_filename(filename: str) -> str:
    """
    파일명을 정규화
    - '외부간판' → 'cctv0'로 변경
    - ' (숫자)' → '-숫자' 형태로 변경
    - 나머지 한글은 모두 제거
    - 연속된 공백과 하이픈 정리
    """
    # 먼저 '외부간판'을 'cctv0'로 변경
    filename = filename.replace('외부간판', 'cctv0')
    
    # ' (숫자)' 패턴을 '-숫자'로 변경
    filename = re.sub(r'\s*\(\s*(\d+)\s*\)', r'-\1', filename)
    
    # 나머지 한글 제거
    filename = remove_korean(filename)
    
    # 연속된 공백과 하이픈을 하나로 정리
    filename = re.sub(r'[\s-]+', '-', filename)
    
    # 앞뒤 하이픈 제거
    filename = filename.strip('-')
    
    return filename


def rename_files_in_directory(directory: str, dry_run: bool = True) -> None:
    """
    디렉토리 내의 모든 파일명을 변경
    
    Args:
        directory: 대상 디렉토리 경로
        dry_run: True이면 실제로 변경하지 않고 미리보기만 출력
    """
    dir_path = Path(directory)
    
    if not dir_path.exists():
        print(f"Error: {directory} 디렉토리가 존재하지 않습니다.")
        return
    
    if not dir_path.is_dir():
        print(f"Error: {directory}는 디렉토리가 아닙니다.")
        return
    
    # 디렉토리 내의 모든 파일 찾기
    files = [f for f in dir_path.iterdir() if f.is_file()]
    
    if not files:
        print(f"Warning: {directory} 디렉토리에 파일이 없습니다.")
        return
    
    print(f"{'[DRY RUN]' if dry_run else '[실제 변경]'} {directory} 디렉토리의 파일명 변경:")
    print("-" * 80)
    
    rename_count = 0
    for file_path in sorted(files):
        old_name = file_path.name
        stem = file_path.stem  # 확장자 제외한 파일명
        ext = file_path.suffix  # 확장자
        
        # 파일명 정규화
        new_stem = normalize_filename(stem)
        new_name = new_stem + ext
        
        if old_name != new_name:
            print(f"  {old_name}")
            print(f"  → {new_name}")
            print()
            
            if not dry_run:
                new_path = file_path.parent / new_name
                # 동일한 이름의 파일이 이미 존재하는지 확인
                if new_path.exists():
                    print(f"  Warning: {new_name} 파일이 이미 존재합니다. 건너뜁니다.")
                    print()
                    continue
                
                try:
                    file_path.rename(new_path)
                    rename_count += 1
                except Exception as e:
                    print(f"  Error: 파일명 변경 실패 - {e}")
                    print()
            else:
                rename_count += 1
    
    print("-" * 80)
    if dry_run:
        print(f"총 {rename_count}개의 파일이 변경될 예정입니다. (실제 변경을 원하면 --execute 옵션을 사용하세요)")
    else:
        print(f"총 {rename_count}개의 파일명이 변경되었습니다.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="비디오 파일명에서 한글 제거 및 정규화")
    parser.add_argument("--directory", type=str, required=True, help="변경할 파일이 있는 디렉토리 경로")
    parser.add_argument("--execute", action="store_true", help="실제로 파일명을 변경 (기본값: False, 미리보기만)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    rename_files_in_directory(
        directory=args.directory,
        dry_run=not args.execute
    )



