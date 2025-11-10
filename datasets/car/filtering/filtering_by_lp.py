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


def detect_license_plate(image_path: str, lp_detection_model: YOLO) -> tuple[bool, float, list]:
    """
    이미지에 번호판이 있는지 확인하고 최고 confidence 값과 bounding box 정보를 반환
    
    Returns:
        (bool, float, list): (번호판 감지 여부, 최고 confidence 값, bounding box 리스트)
                            bounding box는 [x1, y1, x2, y2] 형식의 리스트
    """
    try:
        image_crop = cv2.imread(image_path)
        if image_crop is None:
            return False, 0.0, []
        
        lp_detection_results = lp_detection_model(image_crop, conf=0.6, verbose=False)[0]
        
        # detection 결과가 있고 boxes가 있으면 번호판이 감지된 것
        if lp_detection_results.boxes is not None and len(lp_detection_results.boxes) > 0:
            # 최고 confidence 값을 가진 box 찾기
            confidences = lp_detection_results.boxes.conf.cpu().numpy()
            max_idx = confidences.argmax()
            max_conf = float(confidences[max_idx])
            
            # 최고 confidence를 가진 box의 좌표 추출
            box = lp_detection_results.boxes.xyxy[max_idx].cpu().numpy()
            bbox = [int(box[0]), int(box[1]), int(box[2]), int(box[3])]  # [x1, y1, x2, y2]
            
            return True, max_conf, bbox
        return False, 0.0, []
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return False, 0.0, []


def crop_license_plate(image, bbox: list):
    """
    이미지에서 bounding box 영역을 crop
    
    Args:
        image: 원본 이미지
        bbox: [x1, y1, x2, y2] 형식의 bounding box 좌표
    
    Returns:
        crop된 이미지
    """
    x1, y1, x2, y2 = bbox
    # 이미지 범위를 벗어나지 않도록 처리
    h, w = image.shape[:2]
    x1 = max(0, min(x1, w))
    y1 = max(0, min(y1, h))
    x2 = max(0, min(x2, w))
    y2 = max(0, min(y2, h))
    
    if x2 > x1 and y2 > y1:
        return image[y1:y2, x1:x2]
    return None


def process_runs_directory(runs_dir: str, output_dir: str, lp_detection_model_path: str, output_lp_dir: str = None):
    """
    runs 디렉토리 내의 모든 하위 디렉토리를 재귀적으로 순회하며 track_* 디렉토리의 번호판이 감지된 이미지만 필터링
    
    Args:
        runs_dir: 원본 runs 디렉토리 경로
        output_dir: 필터링된 결과를 저장할 디렉토리 경로
        lp_detection_model_path: 번호판 detection 모델 경로
        output_lp_dir: crop된 번호판 이미지를 저장할 디렉토리 경로 (None이면 저장하지 않음)
    """
    runs_path = Path(runs_dir)
    output_path = Path(output_dir)
    output_lp_path = Path(output_lp_dir) if output_lp_dir else None
    
    if not runs_path.exists():
        print(f"Error: {runs_dir} 디렉토리가 존재하지 않습니다.")
        return
    
    ensure_dir(output_dir)
    if output_lp_dir:
        ensure_dir(output_lp_dir)
    
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
        
        # 이미 저장된 파일 목록 확인 (중복 방지)
        # 저장된 파일명 형식: {original_name}_conf{conf_value:.2f}
        # 원본 파일명으로 매칭하기 위해 저장된 파일명에서 _conf 부분을 제거한 원본 이름 추출
        existing_original_names = set()
        if output_track_dir.exists():
            for ext in image_extensions:
                for f in output_track_dir.glob(f"*{ext}"):
                    # 파일명에서 _conf{숫자} 부분을 제거하여 원본 이름 추출
                    stem = f.stem
                    if "_conf" in stem:
                        original_name = stem.rsplit("_conf", 1)[0]
                        existing_original_names.add(original_name)
                    else:
                        existing_original_names.add(stem)
                for f in output_track_dir.glob(f"*{ext.upper()}"):
                    stem = f.stem
                    if "_conf" in stem:
                        original_name = stem.rsplit("_conf", 1)[0]
                        existing_original_names.add(original_name)
                    else:
                        existing_original_names.add(stem)
        
        # 번호판이 감지된 이미지만 필터링 (confidence 값 및 bounding box 포함)
        filtered_images = []  # (image_path, confidence, bbox) 튜플 리스트
        new_images_count = 0
        skipped_count = 0
        
        for image_path in tqdm(image_files, desc=f"    {relative_path} 이미지 검사", leave=False):
            # 이미 저장된 파일인지 확인 (원본 이름으로 비교)
            original_name = image_path.stem
            if original_name in existing_original_names:
                skipped_count += 1
                continue
            
            has_lp, conf_value, bbox = detect_license_plate(str(image_path), lp_detection_model)
            if has_lp:
                filtered_images.append((image_path, conf_value, bbox))
        
        # 번호판이 감지된 이미지만 저장 (번호판이 감지된 이미지가 하나도 없으면 해당 track 디렉토리는 생성하지 않음)
        if filtered_images:
            ensure_dir(str(output_track_dir))
            
            # 번호판 crop 저장용 디렉토리 설정
            output_lp_track_dir = None
            if output_lp_path:
                output_lp_track_dir = output_lp_path / relative_path
                ensure_dir(str(output_lp_track_dir))
            
            # 번호판이 감지된 이미지만 저장
            for image_path, conf_value, bbox in filtered_images:
                # 원본 이미지 읽기
                image = cv2.imread(str(image_path))
                if image is None:
                    continue
                
                # 필터링된 전체 이미지 저장
                original_name = image_path.stem
                original_ext = image_path.suffix
                new_filename = f"{original_name}_conf{conf_value:.2f}{original_ext}"
                output_image_path = output_track_dir / new_filename
                
                # 파일이 이미 존재하는지 확인 (중복 방지)
                if not output_image_path.exists():
                    cv2.imwrite(str(output_image_path), image)
                    new_images_count += 1
                
                # 번호판 crop 이미지 저장
                if output_lp_path and bbox:
                    lp_cropped = crop_license_plate(image, bbox)
                    if lp_cropped is not None:
                        lp_filename = f"{original_name}_lp_conf{conf_value:.2f}{original_ext}"
                        output_lp_path_full = output_lp_track_dir / lp_filename
                        # 파일이 이미 존재하는지 확인 (중복 방지)
                        if not output_lp_path_full.exists():
                            cv2.imwrite(str(output_lp_path_full), lp_cropped)
            
            lp_info = f", 번호판 crop: {len(filtered_images)}개" if output_lp_path else ""
            skip_info = f", 건너뛴 파일: {skipped_count}개" if skipped_count > 0 else ""
            new_info = f", 새로 저장: {new_images_count}개" if new_images_count > 0 else ""
            print(f"  {relative_path}: {len(filtered_images)}/{len(image_files)} 번호판 감지{skip_info}{new_info}{lp_info}")
        elif skipped_count > 0:
            print(f"  {relative_path}: 모든 파일 이미 처리됨 (건너뛴 파일: {skipped_count}개)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="번호판 detection으로 이미지 필터링")
    parser.add_argument("--runs_dir", type=str, default="runs_1031", help="원본 runs 디렉토리 경로")
    parser.add_argument("--output_dir", type=str, default="runs_1031_filtered_by_lp", help="필터링된 결과를 저장할 디렉토리 경로")
    parser.add_argument("--output_lp_dir", type=str, default=None, help="crop된 번호판 이미지를 저장할 디렉토리 경로 (선택사항)")
    parser.add_argument("--lp_detection_model", type=str, required=True, help="번호판 detection 모델 경로")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    process_runs_directory(
        runs_dir=args.runs_dir,
        output_dir=args.output_dir,
        lp_detection_model_path=args.lp_detection_model,
        output_lp_dir=args.output_lp_dir
    )
    print("처리 완료!")

