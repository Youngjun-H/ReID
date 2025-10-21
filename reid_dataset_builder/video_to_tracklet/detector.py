from ultralytics import YOLO
import torch


class PersonDetector:
    def __init__(self, model_path="yolo11x.pt", conf=0.85, device=None):
        self.model = YOLO(model_path)
        self.conf = conf
        print(f"conf: {self.conf}")
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    def detect(self, frame):
        """사람 검출만 수행 (tracking 없음)"""
        results = self.model.predict(frame, conf=self.conf, classes=[0], device=self.device, verbose=False)
        boxes = results[0].boxes.xyxy.cpu().numpy()
        confs = results[0].boxes.conf.cpu().numpy()
        return boxes, confs

    def track(self, frame):
        """YOLO 내장 tracking 기능 사용"""
        results = self.model.track(
            frame, 
            conf=self.conf, 
            classes=[0], 
            device=self.device, 
            verbose=False,
            persist=True,  # tracking 상태 유지
            tracker="bytetrack.yaml",  # ByteTrack 트래커 사용
        )
        
        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            confs = results[0].boxes.conf.cpu().numpy()
            track_ids = results[0].boxes.id.cpu().numpy()
            return boxes, confs, track_ids
        else:
            return [], [], []