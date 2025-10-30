import cv2
from pathlib import Path


def save_tracklets(frame, tracks, cam_id, output_dir, frame_id=None, timestamp=None):
    for x1, y1, x2, y2, tid, conf in tracks:
        crop = frame[int(y1):int(y2), int(x1):int(x2)]
        track_dir = Path(output_dir) / cam_id / f"id_{int(tid):04d}"
        track_dir.mkdir(parents=True, exist_ok=True)
        fname = f"id_{int(tid):04d}_{frame_id or 0:06d}.jpg"
        cv2.imwrite(str(track_dir / fname), crop)