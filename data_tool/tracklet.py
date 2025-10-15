from ultralytics import YOLO
import cv2
import os
from pathlib import Path
from datetime import datetime
import json
import numpy as np

class TrackletCollector:
    def __init__(self, 
                 camera_id,
                 model_path='yolov8n.pt',
                 tracker='bytetrack.yaml',
                 output_dir='./tracklets'):
        """
        YOLO tracking 기반 tracklet 수집기
        
        Args:
            camera_id: 카메라 식별자 (예: '1F', '2F')
            model_path: YOLO 모델 경로
            tracker: tracking 알고리즘 (bytetrack, botsort)
            output_dir: tracklet 저장 디렉토리
        """
        self.camera_id = camera_id
        self.model = YOLO(model_path)
        self.tracker = tracker
        self.output_dir = Path(output_dir) / camera_id
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Tracklet 저장소
        self.tracklets = {}  # {track_id: tracklet_data}
        
        # 설정
        self.min_frames = 10  # 최소 프레임 수
        self.save_interval = 5  # N프레임마다 저장
        self.person_class_id = 0  # COCO dataset person class
        
    def process_video(self, video_path, conf_threshold=0.8):
        """
        비디오에서 tracklet 추출
        """
        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_count = 0
        
        print(f"Processing {video_path}...")
        print(f"FPS: {fps}")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # YOLO tracking 실행
            results = self.model.track(
                frame,
                persist=True,  # track ID 유지
                tracker=self.tracker,
                classes=[self.person_class_id],  # person만
                conf=conf_threshold,
                verbose=False
            )
            
            # 프레임 처리
            self._process_frame(results[0], frame, frame_count, fps)
            
            frame_count += 1
            
            if frame_count % 100 == 0:
                print(f"Processed {frame_count} frames, "
                      f"Active tracks: {len(self.tracklets)}")
        
        cap.release()
        
        # 최종 저장
        self._finalize_tracklets()
        
        print(f"✓ Collected {len(self.tracklets)} tracklets")
        return self.tracklets
    
    def _process_frame(self, result, frame, frame_idx, fps):
        """
        단일 프레임에서 detection 처리
        """
        if result.boxes is None or result.boxes.id is None:
            return
        
        boxes = result.boxes.xyxy.cpu().numpy()
        track_ids = result.boxes.id.cpu().numpy().astype(int)
        confidences = result.boxes.conf.cpu().numpy()
        
        # NumPy 타입을 Python 기본 타입으로 변환
        track_ids = [int(tid) for tid in track_ids]
        confidences = [float(conf) for conf in confidences]
        
        for box, track_id, conf in zip(boxes, track_ids, confidences):
            x1, y1, x2, y2 = box
            
            # 품질 체크
            if not self._is_good_detection(box, frame.shape, conf):
                continue
            
            # Tracklet 초기화 or 업데이트
            if track_id not in self.tracklets:
                self._init_tracklet(track_id, frame_idx, fps)
            
            # N프레임마다 저장 (간헐적 샘플링)
            if frame_idx % self.save_interval == 0:
                self._add_detection(track_id, frame, box, frame_idx, conf)
    
    def _init_tracklet(self, track_id, frame_idx, fps):
        """
        새로운 tracklet 초기화
        """
        self.tracklets[track_id] = {
            'track_id': track_id,
            'camera_id': self.camera_id,
            'start_frame': frame_idx,
            'end_frame': frame_idx,
            'start_time': None,  # 실시간 처리시 datetime
            'end_time': None,
            'detections': [],
            'image_paths': [],
            'fps': fps
        }
    
    def _add_detection(self, track_id, frame, box, frame_idx, conf):
        """
        Detection을 tracklet에 추가하고 이미지 저장
        """
        x1, y1, x2, y2 = box
        
        # Person crop
        crop = frame[int(y1):int(y2), int(x1):int(x2)]
        
        # 이미지 저장
        img_filename = f"track{track_id:04d}_frame{frame_idx:06d}.jpg"
        img_path = self.output_dir / f"track_{track_id:04d}" / img_filename
        
        # 이미지 저장 시도
        success = cv2.imwrite(str(img_path), crop)
        
        # 이미지 저장이 성공한 경우에만 폴더 생성 및 tracklet 업데이트
        if success:
            # 폴더가 아직 생성되지 않았다면 생성
            if not img_path.parent.exists():
                img_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Tracklet 업데이트
            self.tracklets[track_id]['detections'].append({
                'frame_idx': frame_idx,
                'bbox': [float(x1), float(y1), float(x2), float(y2)],
                'confidence': float(conf),
                'timestamp': datetime.now().isoformat()
            })
            self.tracklets[track_id]['image_paths'].append(str(img_path))
            self.tracklets[track_id]['end_frame'] = frame_idx
    
    def _is_good_detection(self, box, frame_shape, conf):
        """
        Detection 품질 검사
        """
        x1, y1, x2, y2 = box
        h, w = frame_shape[:2]
        
        # Bbox 크기
        bbox_w = x2 - x1
        bbox_h = y2 - y1
        
        # 너무 작거나 큰 detection 제외
        if bbox_h < 50 or bbox_w < 30:  # 최소 크기
            return False
        if bbox_h > h * 0.9 or bbox_w > w * 0.9:  # 너무 큼
            return False
        
        # Aspect ratio (사람은 세로로 긴 형태)
        aspect_ratio = bbox_h / bbox_w
        if aspect_ratio < 1.2 or aspect_ratio > 5.0:
            return False
        
        # Confidence
        if conf < 0.5:
            return False
        
        # 화면 가장자리는 제외 (잘림 가능성)
        margin = 10
        if x1 < margin or y1 < margin or x2 > w - margin or y2 > h - margin:
            return False
        
        return True
    
    def _convert_numpy_types(self, obj):
        """
        NumPy 타입을 Python 기본 타입으로 변환
        """
        if isinstance(obj, dict):
            return {key: self._convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_types(item) for item in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj
    
    def _finalize_tracklets(self):
        """
        최종 처리: 짧은 tracklet 제거 및 메타데이터 저장
        """
        valid_tracklets = {}
        
        for track_id, tracklet in self.tracklets.items():
            # 최소 프레임 수 체크 또는 이미지가 없는 경우
            if len(tracklet['detections']) < self.min_frames or not tracklet['image_paths']:
                # 이미지 삭제
                for img_path in tracklet['image_paths']:
                    if os.path.exists(img_path):
                        os.remove(img_path)
                # 빈 폴더가 있다면 삭제
                track_folder = self.output_dir / f"track_{track_id:04d}"
                if track_folder.exists() and not any(track_folder.iterdir()):
                    track_folder.rmdir()
                continue
            
            # 시간 정보 계산
            tracklet['duration_frames'] = tracklet['end_frame'] - tracklet['start_frame']
            tracklet['duration_seconds'] = tracklet['duration_frames'] / tracklet['fps']
            
            # NumPy 타입을 Python 기본 타입으로 변환
            tracklet_serializable = self._convert_numpy_types(tracklet)
            
            # 메타데이터 저장 (이미지가 있는 경우에만)
            if tracklet['image_paths']:  # 이미지가 있는 경우에만 메타데이터 저장
                metadata_path = self.output_dir / f"track_{track_id:04d}" / "metadata.json"
                # 폴더가 이미 존재하는지 확인 (이미지 저장 시 생성됨)
                if metadata_path.parent.exists():
                    with open(metadata_path, 'w') as f:
                        json.dump(tracklet_serializable, f, indent=2)
            
            valid_tracklets[track_id] = tracklet
        
        self.tracklets = valid_tracklets

# 사용 예시
collector = TrackletCollector(
    camera_id='3F',
    model_path='yolo11x.pt',  # medium 모델 (정확도↑)
    tracker='botsort.yaml'
)

tracklets = collector.process_video('/data/reid/reid_master/cctv_dataset/250915/2025-09-15-08-50-00-사내-3층-외부.avi', conf_threshold=0.8)