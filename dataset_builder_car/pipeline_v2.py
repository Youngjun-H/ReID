from ultralytics import YOLO
import os
import argparse
from collections import defaultdict, deque
import cv2
import numpy as np


def ensure_dir(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def clamp(val: int, low: int, high: int) -> int:
    return max(low, min(high, val))


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

    ensure_dir(output_dir)
    vehicle_classes = [2, 3, 5, 7]
    model = YOLO(model_path)

    saved_counts = defaultdict(int)
    last_centers = {}
    stationary_streak = defaultdict(int)
    frozen_ids = set()
    frozen_db = []  # [(bbox, last_seen_frame, diag)]
    last_boxes = {}
    recent_lost = deque(maxlen=200)
    id_remap = {}
    last_seen = {}

    frame_index = 0

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
        frame = result.orig_img
        h, w = frame.shape[:2]
        diag = (h * h + w * w) ** 0.5
        move_thresh = stationary_rel_thresh * diag

        if result.boxes is None or result.boxes.id is None:
            frame_index += 1
            continue

        ids = result.boxes.id.int().tolist()
        xyxys = result.boxes.xyxy.int().tolist()
        clss = result.boxes.cls.int().tolist()

        src_path = getattr(result, "path", None)
        src_name = os.path.splitext(os.path.basename(src_path if src_path else "unknown"))[0]

        # ------------------ ID 병합 ------------------
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
                recent_lost.append((tid, last_boxes[tid], last_seen.get(tid, frame_index - 1)))
                last_boxes.pop(tid, None)

        for raw_id, canon_id, bbox, cls_idx in zip(ids, canonical_ids, xyxys, clss):
            track_id = canon_id
            x1, y1, x2, y2 = bbox
            x1, y1, x2, y2 = clamp(x1, 0, w - 1), clamp(y1, 0, h - 1), clamp(x2, 0, w - 1), clamp(y2, 0, h - 1)
            if x2 <= x1 or y2 <= y1:
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

            # ----------- 중심 이동 기반 정지 판별 -----------
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            if track_id in last_centers:
                px, py = last_centers[track_id]
                dist = ((cx - px) ** 2 + (cy - py) ** 2) ** 0.5
                if dist < move_thresh:
                    stationary_streak[track_id] += 1
                else:
                    stationary_streak[track_id] = 0
            last_centers[track_id] = (cx, cy)

            # 정지 확정 → DB 저장 & 동결
            if stationary_streak[track_id] >= stationary_patience:
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
    parser = argparse.ArgumentParser(description="Vehicle-only tracking and crop saver (enhanced)")
    parser.add_argument("--source", type=str, required=True, help="영상 경로 또는 디렉토리")
    parser.add_argument("--output", type=str, default="runs/vehicle_crops_2025-10-21-09-00-00-외부간판-yolo11n", help="크롭 저장 경로")
    parser.add_argument("--weights", type=str, default="yolo11n.pt", help="YOLO 가중치")
    parser.add_argument("--conf", type=float, default=0.7, help="confidence threshold")
    parser.add_argument("--iou", type=float, default=0.5, help="IoU threshold")
    parser.add_argument("--show", action="store_true", help="시각화 여부")
    parser.add_argument("--max-per-track", type=int, default=200, help="트랙별 저장 제한")
    parser.add_argument("--stationary-rel-thresh", type=float, default=0.002, help="정지 임계비")
    parser.add_argument("--stationary-patience", type=int, default=30, help="정지 프레임 수")
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
