from ultralytics import YOLO
import os
import argparse
from collections import defaultdict, deque
import cv2
import numpy as np
from pathlib import Path
import re


def ensure_dir(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def clamp(val: int, low: int, high: int) -> int:
    return max(low, min(high, val))


def iou_xyxy(a, b):
    """두 바운딩 박스의 IoU를 계산"""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    a_area = max(0, (ax2 - ax1)) * max(0, (ay2 - ay1))
    b_area = max(0, (bx2 - bx1)) * max(0, (by2 - by1))
    denom = a_area + b_area - inter
    return inter / (denom + 1e-6) if denom > 0 else 0.0


# 박스 유사도 판단 함수 (크기/거리 기반)
def is_same_object(b1, b2, diag, size_tol=0.25, dist_tol=0.03):
    x1, y1, x2, y2 = b1
    u1, v1, u2, v2 = b2
    cx1, cy1 = (x1 + x2) / 2, (y1 + y2) / 2
    cx2, cy2 = (u1 + u2) / 2, (v1 + v2) / 2
    w1, h1 = x2 - x1, y2 - y1
    w2, h2 = u2 - u1, v2 - v1

    size_ratio = abs((w1 * h1) - (w2 * h2)) / (w1 * h1 + 1e-6)
    dist = np.sqrt((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2) / diag
    return size_ratio < size_tol and dist < dist_tol


def parse_roi_file(roi_file_path: str) -> list:
    """ROI 파일에서 ROI 좌표를 파싱하여 반환
    
    ROI 파일 형식:
    roi: (x1, y1, x2, y2)
    main roi: (x1, y1, x2, y2)
    sub roi: (x1, y1, x2, y2)
    
    Returns:
        list: [(x1, y1, x2, y2), ...] 형식의 ROI 리스트
    """
    rois = []
    if not os.path.exists(roi_file_path):
        print(f"경고: ROI 파일을 찾을 수 없습니다: {roi_file_path}")
        return rois
    
    with open(roi_file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            # "roi: (x1, y1, x2, y2)" 또는 "main roi: (x1, y1, x2, y2)" 형식 파싱
            match = re.search(r'\((\d+),\s*(\d+),\s*(\d+),\s*(\d+)\)', line)
            if match:
                x1, y1, x2, y2 = map(int, match.groups())
                rois.append((x1, y1, x2, y2))
                print(f"ROI 로드: ({x1}, {y1}, {x2}, {y2})")
    
    return rois


def is_bbox_in_roi(bbox, rois, min_iou=0.7):
    """바운딩 박스가 ROI 안에 있는지 IoU 기준으로 확인
    
    Args:
        bbox: (x1, y1, x2, y2) 형식의 바운딩 박스
        rois: [(x1, y1, x2, y2), ...] 형식의 ROI 리스트
        min_iou: 최소 IoU 임계값 (기본값 0.7 = 70% 이상 겹쳐야 함)
    
    Returns:
        bool: 바운딩 박스가 하나 이상의 ROI와 min_iou 이상 겹치면 True
    """
    if not rois:
        return True  # ROI가 없으면 모든 박스 허용
    
    for roi_x1, roi_y1, roi_x2, roi_y2 in rois:
        # ROI 좌표 정규화 (x1 < x2, y1 < y2 보장)
        roi_x_min, roi_x_max = min(roi_x1, roi_x2), max(roi_x1, roi_x2)
        roi_y_min, roi_y_max = min(roi_y1, roi_y2), max(roi_y1, roi_y2)
        roi_bbox = [roi_x_min, roi_y_min, roi_x_max, roi_y_max]
        
        # IoU 계산
        iou = iou_xyxy(bbox, roi_bbox)
        
        if iou >= min_iou:
            return True
    
    return False


def get_video_files(source_path: str) -> list:
    """디렉토리 또는 파일 경로에서 비디오 파일 목록을 반환"""
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.m4v', '.webm'}
    source = Path(source_path)
    
    if source.is_file():
        if source.suffix.lower() in video_extensions:
            return [str(source)]
        else:
            return []
    elif source.is_dir():
        video_files = []
        for ext in video_extensions:
            video_files.extend(source.glob(f'*{ext}'))
            video_files.extend(source.glob(f'*{ext.upper()}'))
        return sorted([str(f) for f in video_files])
    else:
        return []


def run_single_video(
    video_path: str,
    output_dir: str,
    model_path: str = "yolo11x.pt",
    conf: float = 0.3,
    iou: float = 0.5,
    show: bool = False,
    max_frames_per_track: int = 200,
    stationary_rel_thresh: float = 0.002,
    stationary_patience: int = 30,
    rois: list = None,
    roi_min_iou: float = 0.7,
) -> None:
    """단일 비디오 파일에 대해 추적 및 크롭 저장 수행
    
    Args:
        rois: [(x1, y1, x2, y2), ...] 형식의 ROI 리스트. None이면 ROI 필터링 없음
        roi_min_iou: ROI 필터링에 사용할 최소 IoU 임계값 (기본값 0.7 = 70%)
    """
    
    # 영상 이름으로 출력 디렉토리 생성
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    video_output_dir = os.path.join(output_dir, video_name)
    ensure_dir(video_output_dir)
    
    vehicle_classes = [2, 3, 5, 7]
    model = YOLO(model_path)

    saved_counts = defaultdict(int)
    first_bbox = {}  # track_id별 첫 프레임의 bbox 저장
    first_frame = {}  # track_id별 첫 프레임 번호 저장
    frozen_ids = set()
    frozen_db = []  # [(bbox, last_seen_frame, diag)]
    last_boxes = {}
    recent_lost = deque(maxlen=200)
    id_remap = {}
    last_seen = {}
    check_interval = 30  # 주기적으로 체크할 프레임 간격

    frame_index = 0
    roi_filtered_count = 0  # ROI 필터링으로 제외된 객체 수

    for result in model.track(
        source=video_path,
        conf=conf,
        iou=iou,
        show=show,
        classes=vehicle_classes,
        tracker="bytetrack.yaml",
        persist=True,
        stream=True,
        device=0,  # GPU 사용 (CUDA device 0)
    ):
        frame = result.orig_img
        h, w = frame.shape[:2]
        diag = (h * h + w * w) ** 0.5

        if result.boxes is None or result.boxes.id is None:
            frame_index += 1
            continue

        ids = result.boxes.id.int().tolist()
        xyxys = result.boxes.xyxy.int().tolist()
        clss = result.boxes.cls.int().tolist()

        src_path = getattr(result, "path", None)
        src_name = os.path.splitext(os.path.basename(src_path if src_path else "unknown"))[0]

        # ------------------ ID 병합 ------------------
        canonical_ids = []
        for raw_id, bbox in zip(ids, xyxys):
            cand_id = id_remap.get(raw_id, raw_id)
            best_id = cand_id
            best_iou = 0.0

            # 기존 트랙과 비교
            for prev_id, prev_box in last_boxes.items():
                iou_val = iou_xyxy(bbox, prev_box)
                if iou_val > best_iou:
                    best_iou = iou_val
                    best_id = prev_id

            # 최근 잃은 트랙도 비교
            if best_iou < 0.6:
                for prev_id, prev_box, seen in reversed(recent_lost):
                    if frame_index - seen > 60:
                        break
                    if is_same_object(prev_box, bbox, diag):
                        best_iou = 1.0
                        best_id = prev_id
                        break

            if best_iou >= 0.6:
                id_remap[raw_id] = best_id
                canonical_ids.append(best_id)
            else:
                canonical_ids.append(cand_id)

        # ------------------ 트랙 관리 ------------------
        alive = set(canonical_ids)
        for tid in list(last_boxes.keys()):
            if tid not in alive:
                # 트랙이 사라지기 전에 마지막 체크: 첫 프레임과 마지막 프레임 비교
                if tid in first_bbox and tid in last_boxes and tid not in frozen_ids:
                    frames_since_first = last_seen.get(tid, frame_index - 1) - first_frame.get(tid, frame_index - 1)
                    if frames_since_first >= 30:  # 최소 30프레임 이상 관찰된 경우만 체크
                        if is_same_object(first_bbox[tid], last_boxes[tid], diag, size_tol=0.3, dist_tol=0.05):
                            # 거의 움직이지 않았으면 정지 차량으로 판단
                            frozen_ids.add(tid)
                            frozen_db.append((last_boxes[tid], last_seen.get(tid, frame_index - 1), diag))
                
                recent_lost.append((tid, last_boxes[tid], last_seen.get(tid, frame_index - 1)))
                # 정리: 사라진 트랙의 정보 제거
                first_bbox.pop(tid, None)
                first_frame.pop(tid, None)
                last_boxes.pop(tid, None)

        for raw_id, canon_id, bbox, cls_idx in zip(ids, canonical_ids, xyxys, clss):
            track_id = canon_id
            x1, y1, x2, y2 = bbox
            x1, y1, x2, y2 = clamp(x1, 0, w - 1), clamp(y1, 0, h - 1), clamp(x2, 0, w - 1), clamp(y2, 0, h - 1)
            if x2 <= x1 or y2 <= y1:
                continue

            # ----------- ROI 필터링 -----------
            if rois and not is_bbox_in_roi([x1, y1, x2, y2], rois, min_iou=roi_min_iou):
                roi_filtered_count += 1
                continue

            # ----------- 정지 차량 DB와 비교 -----------
            skip_save = False
            for fb, f_seen, f_diag in frozen_db:
                if frame_index - f_seen < 2000:  # 약 1분 내 재등장
                    if is_same_object(fb, bbox, diag=f_diag):
                        skip_save = True
                        break
            if skip_save:
                continue

            # ----------- 첫 프레임 bbox 초기화 -----------
            if track_id not in first_bbox:
                first_bbox[track_id] = [x1, y1, x2, y2]
                first_frame[track_id] = frame_index

            # ----------- 주기적으로 첫 프레임과 현재 프레임 비교 -----------
            # 체크 조건: 일정 간격마다 또는 일정 프레임 수 이상 관찰되었을 때
            should_check = False
            if track_id in first_frame:
                frames_since_first = frame_index - first_frame[track_id]
                # check_interval 간격마다 체크하거나, 60프레임 이상 관찰되었을 때 체크
                if frames_since_first >= check_interval and frames_since_first % check_interval == 0:
                    should_check = True
                elif frames_since_first >= 60 and saved_counts[track_id] >= 10:
                    should_check = True

            if should_check and track_id not in frozen_ids:
                # 첫 프레임과 현재 프레임의 bbox 비교
                fb = first_bbox[track_id]
                if is_same_object(fb, [x1, y1, x2, y2], diag, size_tol=0.3, dist_tol=0.05):
                    # 거의 움직이지 않았으면 정지 차량으로 판단
                    frozen_ids.add(track_id)
                    frozen_db.append(([x1, y1, x2, y2], frame_index, diag))
                    continue

            # 프레임당 저장 제한
            if saved_counts[track_id] >= max_frames_per_track:
                frozen_ids.add(track_id)
                frozen_db.append(([x1, y1, x2, y2], frame_index, diag))
                continue

            # 동결된 트랙은 저장하지 않음
            if track_id in frozen_ids:
                continue

            # ----------- 저장 -----------
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            track_dir = os.path.join(video_output_dir, f"track_{int(track_id)}")
            ensure_dir(track_dir)
            idx = saved_counts[track_id]
            out_name = f"{src_name}_f{frame_index:06d}_{idx:03d}.jpg"
            out_path = os.path.join(track_dir, out_name)
            cv2.imwrite(out_path, crop)

            saved_counts[track_id] += 1
            last_boxes[track_id] = [x1, y1, x2, y2]
            last_seen[track_id] = frame_index

        frame_index += 1
    
    if rois:
        print(f"  ROI 필터링으로 제외된 객체 수: {roi_filtered_count}")


def run(
    source_path: str,
    output_dir: str,
    model_path: str = "yolo11x.pt",
    conf: float = 0.3,
    iou: float = 0.5,
    show: bool = False,
    max_frames_per_track: int = 200,
    stationary_rel_thresh: float = 0.002,
    stationary_patience: int = 30,
    roi_file: str = None,
    roi_min_iou: float = 0.7,
) -> None:
    """source_path가 디렉토리인 경우 모든 비디오 파일을 처리, 파일인 경우 단일 파일 처리"""
    
    ensure_dir(output_dir)
    video_files = get_video_files(source_path)
    
    if not video_files:
        print(f"경고: {source_path}에서 비디오 파일을 찾을 수 없습니다.")
        return
    
    # ROI 파일 로드
    rois = []
    if roi_file:
        rois = parse_roi_file(roi_file)
        if rois:
            print(f"총 {len(rois)}개의 ROI가 로드되었습니다.")
        else:
            print("경고: ROI 파일에서 유효한 ROI를 찾을 수 없습니다. ROI 필터링 없이 진행합니다.")
    
    print(f"총 {len(video_files)}개의 비디오 파일을 처리합니다.")
    
    for idx, video_path in enumerate(video_files, 1):
        print(f"\n[{idx}/{len(video_files)}] 처리 중: {video_path}")
        try:
            run_single_video(
                video_path=video_path,
                output_dir=output_dir,
                model_path=model_path,
                conf=conf,
                iou=iou,
                show=show,
                max_frames_per_track=max_frames_per_track,
                stationary_rel_thresh=stationary_rel_thresh,
                stationary_patience=stationary_patience,
                rois=rois,
                roi_min_iou=roi_min_iou,
            )
            print(f"완료: {video_path}")
        except Exception as e:
            print(f"오류 발생 ({video_path}): {e}")
            continue
    
    print(f"\n모든 처리 완료! 결과는 {output_dir}에 저장되었습니다.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Vehicle-only tracking and crop saver with ROI filtering")
    parser.add_argument("--source", type=str, default="/data/reid/reid_master/dataset_builder_car/video/test", help="영상 경로 또는 디렉토리")
    parser.add_argument("--output", type=str, default="/data/reid/reid_master/dataset_builder_car/runs_test", help="크롭 저장 경로")
    parser.add_argument("--weights", type=str, default="yolo11n.pt", help="YOLO 가중치")
    parser.add_argument("--conf", type=float, default=0.7, help="confidence threshold")
    parser.add_argument("--iou", type=float, default=0.5, help="IoU threshold")
    parser.add_argument("--show", action="store_true", help="시각화 여부")
    parser.add_argument("--max-per-track", type=int, default=200, help="트랙별 저장 제한")
    parser.add_argument("--stationary-rel-thresh", type=float, default=0.002, help="정지 임계비 (사용 안함, 호환성 유지)")
    parser.add_argument("--stationary-patience", type=int, default=30, help="정지 프레임 수 (사용 안함, 호환성 유지)")
    parser.add_argument("--roi-file", type=str, default="/data/reid/reid_master/roi.txt", help="ROI 파일 경로 (None이면 ROI 필터링 없음)")
    parser.add_argument("--roi-min-iou", type=float, default=0.7, help="ROI 필터링에 사용할 최소 IoU 임계값 (기본값 0.7 = 70%%)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(
        source_path=args.source,
        output_dir=args.output,
        model_path=args.weights,
        conf=args.conf,
        iou=args.iou,
        show=args.show,
        max_frames_per_track=args.max_per_track,
        stationary_rel_thresh=args.stationary_rel_thresh,
        stationary_patience=args.stationary_patience,
        roi_file=args.roi_file if args.roi_file else None,
        roi_min_iou=args.roi_min_iou,
    )

