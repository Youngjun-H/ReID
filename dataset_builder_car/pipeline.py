from ultralytics import YOLO
import os
import argparse
from collections import defaultdict
from collections import deque
import cv2


def ensure_dir(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def clamp(val: int, low: int, high: int) -> int:
    return max(low, min(high, val))


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
) -> None:
    """
    - source_path: 로컬 영상 파일 또는 디렉토리 경로
    - output_dir: 트랙별 크롭 저장 경로
    - model_path: YOLO 가중치 경로
    - max_frames_per_track: 트랙별 저장 프레임 상한
    - stationary_rel_thresh: 프레임 대각선 대비 이동 임계비 (정지 판단)
    - stationary_patience: 임계 이하 이동이 연속으로 발생해야 하는 프레임 수
    """

    ensure_dir(output_dir)

    # COCO 차량 계열 클래스 인덱스: bicycle(1), car(2), motorcycle(3), bus(5), truck(7)
    vehicle_classes = [1, 2, 3, 5, 7]

    model = YOLO(model_path)

    saved_counts = defaultdict(int)  # track_id -> saved image count
    last_centers = {}  # track_id -> (cx, cy)
    stationary_streak = defaultdict(int)  # track_id -> consecutive frames below movement threshold
    frozen_ids = set()  # track_ids no longer saved (cap reached or stationary)
    last_boxes = {}  # canonical_track_id -> [x1, y1, x2, y2]
    recent_lost = deque(maxlen=200)  # deque of (track_id, bbox, last_seen_frame)
    id_remap = {}  # raw_id -> canonical_id
    last_seen = {}  # track_id -> frame_index

    frame_index = 0

    # stream=True 로 프레임 단위 결과를 순회하면서 사용자 정의 처리
    for result in model.track(
        source=source_path,
        conf=conf,
        iou=iou,
        show=show,
        classes=vehicle_classes,
        tracker="bytetrack.yaml",
        persist=True,
        stream=True,
    ):
        frame = result.orig_img  # numpy array (H, W, C)
        h, w = frame.shape[:2]
        diag = (h * h + w * w) ** 0.5
        move_thresh = stationary_rel_thresh * diag

        if result.boxes is None or result.boxes.id is None:
            frame_index += 1
            continue

        ids = result.boxes.id.int().tolist()
        xyxys = result.boxes.xyxy.int().tolist()
        clss = result.boxes.cls.int().tolist()

        # 비디오 파일명 힌트를 저장 경로에 반영 (소스 경로가 있다면)
        src_path = getattr(result, "path", None)
        src_name = os.path.splitext(os.path.basename(src_path if src_path else "unknown"))[0]

        # ---------- ID 스티칭: raw ID를 canonical ID로 매핑 ----------
        def iou_xyxy(a, b):
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

        canonical_ids = []
        for raw_id, bbox in zip(ids, xyxys):
            cand_id = id_remap.get(raw_id, raw_id)
            best_id = cand_id
            best_iou = 0.0

            # 현존 트랙과 비교
            for prev_id, prev_box in last_boxes.items():
                iou_val = iou_xyxy(bbox, prev_box)
                if iou_val > best_iou:
                    best_iou = iou_val
                    best_id = prev_id

            # 최근 잃은 트랙과도 비교 (최대 30프레임 갭)
            if best_iou < 0.6:
                for prev_id, prev_box, seen in reversed(recent_lost):
                    if frame_index - seen > 30:
                        break
                    iou_val = iou_xyxy(bbox, prev_box)
                    if iou_val > best_iou:
                        best_iou = iou_val
                        best_id = prev_id

            if best_iou >= 0.6:
                id_remap[raw_id] = best_id
                canonical_ids.append(best_id)
            else:
                canonical_ids.append(cand_id)

        # 이번 프레임에 보이지 않는 이전 트랙을 recent_lost에 기록
        alive = set(canonical_ids)
        for tid in list(last_boxes.keys()):
            if tid not in alive:
                recent_lost.append((tid, last_boxes[tid], last_seen.get(tid, frame_index - 1)))
                last_boxes.pop(tid, None)

        for raw_id, canon_id, bbox, cls_idx in zip(ids, canonical_ids, xyxys, clss):
            track_id = canon_id
            if track_id in frozen_ids:
                continue

            x1, y1, x2, y2 = bbox
            x1 = clamp(x1, 0, w - 1)
            y1 = clamp(y1, 0, h - 1)
            x2 = clamp(x2, 0, w - 1)
            y2 = clamp(y2, 0, h - 1)
            if x2 <= x1 or y2 <= y1:
                continue

            # 정지 차량 억제: 중심 이동량 기반
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            if track_id in last_centers:
                px, py = last_centers[track_id]
                dist = ((cx - px) ** 2 + (cy - py) ** 2) ** 0.5
                if dist < move_thresh:
                    stationary_streak[track_id] += 1
                else:
                    stationary_streak[track_id] = 0
            last_centers[track_id] = (cx, cy)

            if stationary_streak[track_id] >= stationary_patience:
                frozen_ids.add(track_id)
                continue

            # 트랙별 저장 상한
            if saved_counts[track_id] >= max_frames_per_track:
                frozen_ids.add(track_id)
                continue

            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            track_dir = os.path.join(output_dir, f"track_{int(track_id)}")
            ensure_dir(track_dir)
            idx = saved_counts[track_id]
            out_name = f"{src_name}_f{frame_index:06d}_{idx:03d}.jpg"
            out_path = os.path.join(track_dir, out_name)
            cv2.imwrite(out_path, crop)
            saved_counts[track_id] += 1
            last_boxes[track_id] = [x1, y1, x2, y2]
            last_seen[track_id] = frame_index

        frame_index += 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Vehicle-only tracking and crop saver")
    parser.add_argument(
        "--source",
        type=str,
        default="/data/reid/reid_master/dataset_builder_car/2025-10-29-11-30-00-외부간판.mp4",
        help="로컬 영상 파일 경로 또는 디렉터리 경로 (이미지/비디오 지원)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="runs/vehicle_crops",
        help="크롭 이미지 저장 경로",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="yolo11x.pt",
        help="YOLO 가중치 경로",
    )
    parser.add_argument("--conf", type=float, default=0.7, help="confidence threshold")
    parser.add_argument("--iou", type=float, default=0.5, help="NMS IoU threshold")
    parser.add_argument(
        "--show",
        action="store_true",
        help="시각화 윈도우 표시",
    )
    parser.add_argument(
        "--max-per-track",
        type=int,
        default=100,
        help="트랙별 저장 프레임 상한",
    )
    parser.add_argument(
        "--stationary-rel-thresh",
        type=float,
        default=0.002,
        help="정지 판단 임계비 (프레임 대각선 대비 이동량)",
    )
    parser.add_argument(
        "--stationary-patience",
        type=int,
        default=30,
        help="정지 판단 연속 프레임 수",
    )
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
    )