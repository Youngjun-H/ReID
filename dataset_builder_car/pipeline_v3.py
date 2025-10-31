from ultralytics import YOLO
import cv2
import os

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def run_with_roi(video_path, output_dir, model_path="yolo11x.pt", conf=0.5, iou=0.5, show=True):
    ensure_dir(output_dir)
    model = YOLO(model_path)
    vehicle_classes = [2, 3, 5, 7]

    # 1️⃣ 첫 프레임에서 ROI 지정
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    if not ret:
        print("❌ 영상 로드 실패")
        return
    roi = cv2.selectROI("Select ROI", frame, showCrosshair=True, fromCenter=False)
    cv2.destroyWindow("Select ROI")
    rx1, ry1, rw, rh = roi
    rx2, ry2 = rx1 + rw, ry1 + rh

    # ROI 시각 확인
    preview = frame.copy()
    cv2.rectangle(preview, (rx1, ry1), (rx2, ry2), (0, 255, 0), 2)
    cv2.imshow("Selected ROI", preview)
    cv2.waitKey(500)
    cv2.destroyAllWindows()

    # 2️⃣ YOLO 추론 + 트래킹
    results = model.track(
        source=video_path,
        conf=conf,
        iou=iou,
        classes=vehicle_classes,
        tracker="bytetrack.yaml",
        persist=True,
        stream=True
    )

    saved_counts = {}

    for i, r in enumerate(results):
        frame = r.orig_img
        if r.boxes is None or r.boxes.id is None:
            continue

        ids = r.boxes.id.int().tolist()
        xyxys = r.boxes.xyxy.int().tolist()
        clss = r.boxes.cls.int().tolist()

        for track_id, bbox, cls_idx in zip(ids, xyxys, clss):
            x1, y1, x2, y2 = bbox

            # 3️⃣ ROI 교차 여부 판단
            inter_x1 = max(rx1, x1)
            inter_y1 = max(ry1, y1)
            inter_x2 = min(rx2, x2)
            inter_y2 = min(ry2, y2)
            inter_w = max(0, inter_x2 - inter_x1)
            inter_h = max(0, inter_y2 - inter_y1)
            inter_area = inter_w * inter_h
            bbox_area = (x2 - x1) * (y2 - y1)
            iou_with_roi = inter_area / (bbox_area + 1e-6)

            # ROI와 30% 이상 겹치는 경우만 저장
            if iou_with_roi < 0.3:
                continue

            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            # 4️⃣ 저장
            track_dir = os.path.join(output_dir, f"track_{int(track_id)}")
            ensure_dir(track_dir)
            idx = saved_counts.get(track_id, 0)
            out_path = os.path.join(track_dir, f"f{i:06d}_{idx:03d}.jpg")
            cv2.imwrite(out_path, crop)
            saved_counts[track_id] = idx + 1

        if show:
            # ROI 시각화
            cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), (0, 255, 0), 2)
            cv2.imshow("Tracking (ROI)", frame)
            if cv2.waitKey(1) & 0xFF == 27:  # ESC
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_with_roi(
        video_path="2025-10-21-09-00-00-외부간판.mp4",
        output_dir="roi_tracks",
        model_path="yolo11x.pt",
        conf=0.5,
        iou=0.5,
        show=True
    )
