from ultralytics import YOLO
import os
import argparse
import cv2
from pathlib import Path
from tqdm import tqdm


def ensure_dir(path: str) -> None:
    """디렉토리가 없으면 생성"""
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def has_license_plate(image_path: str, lp_detection_model: YOLO) -> bool:
    """이미지에 번호판이 있는지 확인"""
    try:
        image_crop = cv2.imread(image_path)
        if image_crop is None:
            return False
        
        lp_detection_results = lp_detection_model(image_crop, conf=0.6, verbose=False)[0]
        
        # detection 결과가 있고 boxes가 있으면 번호판이 감지된 것
        if lp_detection_results.boxes is not None and len(lp_detection_results.boxes) > 0:
            return True
        return False
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return False


def process_runs_directory(runs_dir: str, output_dir: str, lp_detection_model_path: str):
    """
    runs 디렉토리 내의 모든 track_id 디렉토리를 순회하며 번호판이 감지된 이미지만 필터링
    
    Args:
        runs_dir: 원본 runs 디렉토리 경로
        output_dir: 필터링된 결과를 저장할 디렉토리 경로
        lp_detection_model_path: 번호판 detection 모델 경로
    """
    runs_path = Path(runs_dir)
    output_path = Path(output_dir)
    
    if not runs_path.exists():
        print(f"Error: {runs_dir} 디렉토리가 존재하지 않습니다.")
        return
    
    ensure_dir(output_dir)
    
    # 번호판 detection 모델 로드
    print(f"번호판 detection 모델 로드 중: {lp_detection_model_path}")
    lp_detection_model = YOLO(lp_detection_model_path)
    
    # runs 디렉토리 내의 모든 서브디렉토리 순회
    subdirs = [d for d in runs_path.iterdir() if d.is_dir()]
    
    for subdir in tqdm(subdirs, desc="서브디렉토리 처리"):
        subdir_name = subdir.name
        output_subdir = output_path / subdir_name
        ensure_dir(str(output_subdir))
        
        # track_* 디렉토리 찾기
        track_dirs = sorted([d for d in subdir.iterdir() if d.is_dir() and d.name.startswith("track_")])
        
        for track_dir in tqdm(track_dirs, desc=f"  {subdir_name} 처리", leave=False):
            track_name = track_dir.name
            
            # 이미지 파일 찾기
            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
            image_files = []
            for ext in image_extensions:
                image_files.extend(list(track_dir.glob(f"*{ext}")))
                image_files.extend(list(track_dir.glob(f"*{ext.upper()}")))
            
            if not image_files:
                continue
            
            # 번호판이 감지된 이미지만 필터링
            filtered_images = []
            for image_path in tqdm(image_files, desc=f"    {track_name} 이미지 검사", leave=False):
                if has_license_plate(str(image_path), lp_detection_model):
                    filtered_images.append(image_path)
            
            # 번호판이 감지된 이미지가 하나라도 있으면 해당 track_id 디렉토리 생성 및 저장
            if filtered_images:
                output_track_dir = output_subdir / track_name
                ensure_dir(str(output_track_dir))
                
                for image_path in filtered_images:
                    output_image_path = output_track_dir / image_path.name
                    # 이미지 복사
                    image = cv2.imread(str(image_path))
                    if image is not None:
                        cv2.imwrite(str(output_image_path), image)
                
                print(f"  {subdir_name}/{track_name}: {len(filtered_images)}/{len(image_files)} 이미지 저장")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="번호판 detection으로 이미지 필터링")
    parser.add_argument("--runs_dir", type=str, default="runs", help="원본 runs 디렉토리 경로")
    parser.add_argument("--output_dir", type=str, default="runs_filtered_by_lp", help="필터링된 결과를 저장할 디렉토리 경로")
    parser.add_argument("--lp_detection_model", type=str, required=True, help="번호판 detection 모델 경로")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    process_runs_directory(
        runs_dir=args.runs_dir,
        output_dir=args.output_dir,
        lp_detection_model_path=args.lp_detection_model
    )
    print("처리 완료!")

