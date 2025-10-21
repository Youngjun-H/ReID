import csv
from pathlib import Path


def append_metadata(cam_id, track_id, frame_id, conf, timestamp, out_dir):
    csv_path = Path(out_dir) / cam_id / "metadata.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    header = ["cam_id", "track_id", "frame_id", "timestamp", "conf"]

    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(header)
        writer.writerow([cam_id, track_id, frame_id, timestamp, conf])