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


def has_license_plate(image_path: str, lp_detection_model: YOLO) -> tuple[bool, float]:
    """
    이미지에 번호판이 있는지 확인하고 최고 confidence 값을 반환
    
    Returns:
        (bool, float): (번호판 감지 여부, 최고 confidence 값)
    """
    try:
        image_crop = cv2.imread(image_path)
        if image_crop is None:
            return False, 0.0
        
        lp_detection_results = lp_detection_model(image_crop, conf=0.6, verbose=False)[0]
        
        # detection 결과가 있고 boxes가 있으면 번호판이 감지된 것
        if lp_detection_results.boxes is not None and len(lp_detection_results.boxes) > 0:
            # 최고 confidence 값 반환
            confidences = lp_detection_results.boxes.conf.cpu().numpy()
            max_conf = float(confidences.max())
            return True, max_conf
        return False, 0.0
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return False, 0.0


def process_runs_directory(runs_dir: str, output_dir: str, lp_detection_model_path: str):
    """
    runs 디렉토리 내의 모든 하위 디렉토리를 재귀적으로 순회하며 track_* 디렉토리의 번호판이 감지된 이미지만 필터링
    
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
    
    # runs 디렉토리 내의 모든 track_* 디렉토리를 재귀적으로 찾기
    track_dirs = sorted([d for d in runs_path.rglob("track_*") if d.is_dir()])
    
    if not track_dirs:
        print(f"Warning: {runs_dir} 디렉토리 내에 track_* 패턴의 디렉토리를 찾을 수 없습니다.")
        return
    
    for track_dir in tqdm(track_dirs, desc="track 디렉토리 처리"):
        # 원본 디렉토리 기준 상대 경로 계산
        relative_path = track_dir.relative_to(runs_path)
        
        # 출력 디렉토리 경로 생성 (상대 경로 구조 유지)
        output_track_dir = output_path / relative_path
        ensure_dir(str(output_track_dir.parent))
        
        track_name = track_dir.name
        
        # 이미지 파일 찾기
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        image_files = []
        for ext in image_extensions:
            image_files.extend(list(track_dir.glob(f"*{ext}")))
            image_files.extend(list(track_dir.glob(f"*{ext.upper()}")))
        
        if not image_files:
            continue
        
        # 번호판이 감지된 이미지만 필터링 (confidence 값 포함)
        filtered_images = []  # (image_path, confidence) 튜플 리스트
        for image_path in tqdm(image_files, desc=f"    {relative_path} 이미지 검사", leave=False):
            has_lp, conf_value = has_license_plate(str(image_path), lp_detection_model)
            if has_lp:
                filtered_images.append((image_path, conf_value))
        
        # 번호판이 감지된 이미지가 하나라도 있으면 해당 track_id 디렉토리 생성 및 저장
        if filtered_images:
            ensure_dir(str(output_track_dir))
            
            for image_path, conf_value in filtered_images:
                # 파일명에 confidence 값 추가
                # 예: original_name.jpg -> original_name_conf0.85.jpg
                original_name = image_path.stem
                original_ext = image_path.suffix
                new_filename = f"{original_name}_conf{conf_value:.2f}{original_ext}"
                output_image_path = output_track_dir / new_filename
                
                # 이미지 복사
                image = cv2.imread(str(image_path))
                if image is not None:
                    cv2.imwrite(str(output_image_path), image)
            
            print(f"  {relative_path}: {len(filtered_images)}/{len(image_files)} 이미지 저장")


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

