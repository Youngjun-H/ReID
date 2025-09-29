from ultralytics import YOLO
import torch
import cv2
import os
import numpy as np
from collections import defaultdict
import json
import argparse
import asyncio
import aiofiles
from concurrent.futures import ThreadPoolExecutor
import time
from pathlib import Path
import threading
from queue import Queue
import gc

class OptimizedPersonTracker:
    def __init__(self, model_path="yolo11x.pt", max_workers=4):
        """
        최적화된 사람 detection 및 tracking을 위한 클래스
        
        Args:
            model_path (str): YOLO 모델 경로
            max_workers (int): I/O 작업을 위한 최대 워커 수
        """
        # GPU 사용 가능 여부 확인
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"사용 중인 디바이스: {self.device}")
        
        # YOLO 모델 로드 및 최적화
        self.model = YOLO(model_path)
        self.model.to(self.device)
        
        # 모델 최적화 설정
        if self.device == 'cuda':
            # GPU 메모리 최적화
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            
        # I/O 워커 풀
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        
        # 비동기 I/O를 위한 큐
        self.save_queue = Queue(maxsize=1000)
        self.save_thread = None
        
        # 성능 통계
        self.stats = {
            'total_frames': 0,
            'processed_frames': 0,
            'detections': 0,
            'saved_images': 0,
            'start_time': None,
            'inference_time': 0,
            'io_time': 0
        }
        
    def _preprocess_frame(self, frame, target_size=(640, 640)):
        """프레임 전처리 최적화"""
        # OpenCV는 BGR, YOLO는 RGB 사용
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 리사이징 최적화
        h, w = frame_rgb.shape[:2]
        if h != target_size[0] or w != target_size[1]:
            frame_rgb = cv2.resize(frame_rgb, target_size, interpolation=cv2.INTER_LINEAR)
            
        return frame_rgb
    
    def _batch_inference(self, frames, conf_threshold=0.5, classes=[0]):
        """배치 추론으로 성능 최적화"""
        if not frames:
            return []
            
        # 배치 크기 조정 (GPU 메모리에 따라)
        batch_size = min(8, len(frames))  # GPU 메모리에 따라 조정
        
        results = []
        for i in range(0, len(frames), batch_size):
            batch_frames = frames[i:i + batch_size]
            
            # 배치 추론 실행
            batch_results = self.model.track(
                batch_frames,
                conf=conf_threshold,
                classes=classes,
                persist=True,
                verbose=False,
                device=self.device
            )
            
            results.extend(batch_results)
            
        return results
    
    def _async_save_image(self, img_data, img_path, track_id, metadata):
        """비동기 이미지 저장"""
        def save_task():
            try:
                # 디렉토리 생성 (한 번만)
                os.makedirs(os.path.dirname(img_path), exist_ok=True)
                
                # 이미지 저장
                cv2.imwrite(img_path, img_data)
                
                # 메타데이터 저장
                metadata['image_path'] = img_path
                return metadata
                
            except Exception as e:
                print(f"이미지 저장 오류: {e}")
                return None
        
        # 비동기 실행
        future = self.executor.submit(save_task)
        return future
    
    def _process_detections_batch(self, results, frame_numbers, fps, save_dir, conf_threshold):
        """배치 단위로 detection 결과 처리"""
        futures = []
        metadata_batch = []
        
        for result, frame_num in zip(results, frame_numbers):
            if result.boxes is not None and len(result.boxes) > 0:
                boxes = result.boxes
                
                for box in boxes:
                    conf = float(box.conf[0])
                    if conf < conf_threshold:
                        continue
                    
                    # track_id 확인
                    if hasattr(box, 'id') and box.id is not None:
                        track_id = int(box.id[0])
                    else:
                        continue
                    
                    # bounding box 좌표 (GPU에서 직접 처리)
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    
                    # 프레임 이미지에서 해당 영역 추출
                    frame_img = result.orig_img
                    person_crop = frame_img[int(y1):int(y2), int(x1):int(x2)]
                    
                    if person_crop.size > 0:
                        # 이미지 저장 경로
                        track_dir = os.path.join(save_dir, f"person_{track_id:04d}")
                        img_filename = f"frame_{frame_num:06d}_conf_{conf:.3f}.jpg"
                        img_path = os.path.join(track_dir, img_filename)
                        
                        # 메타데이터 준비
                        metadata = {
                            'frame_number': frame_num,
                            'track_id': track_id,
                            'confidence': conf,
                            'bbox': [float(x1), float(y1), float(x2), float(y2)],
                            'timestamp': frame_num / fps
                        }
                        
                        # 비동기 저장 작업 추가
                        future = self._async_save_image(person_crop, img_path, track_id, metadata)
                        futures.append((future, track_id, metadata))
                        metadata_batch.append(metadata)
        
        return futures, metadata_batch
    
    def track_persons_optimized(self, video_path, 
                               conf_threshold=0.85, 
                               frame_interval=1, 
                               save_dir="./tracked_persons",
                               classes=[0],
                               batch_size=8,
                               enable_async=True):
        """
        최적화된 영상에서 사람 detection 및 tracking
        
        Args:
            video_path (str): 입력 영상 경로
            conf_threshold (float): confidence threshold
            frame_interval (int): 프레임 간격
            save_dir (str): 저장 디렉토리
            classes (list): detection할 클래스 ID
            batch_size (int): 배치 크기
            enable_async (bool): 비동기 I/O 사용 여부
        """
        
        # 성능 측정 시작
        self.stats['start_time'] = time.time()
        
        # 저장 디렉토리 생성
        os.makedirs(save_dir, exist_ok=True)
        
        # 영상 정보 가져오기
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        print(f"최적화된 처리 시작:")
        print(f"- 총 프레임 수: {total_frames}")
        print(f"- FPS: {fps}")
        print(f"- Confidence threshold: {conf_threshold}")
        print(f"- Frame interval: {frame_interval}")
        print(f"- Batch size: {batch_size}")
        print(f"- 비동기 I/O: {enable_async}")
        print(f"- 저장 디렉토리: {save_dir}")
        print("-" * 50)
        
        # 배치 처리를 위한 변수들
        frame_batch = []
        frame_numbers = []
        tracked_persons = defaultdict(list)
        all_futures = []
        
        frame_count = 0
        processed_frames = 0
        
        # 스트리밍 처리
        results = self.model.track(video_path, 
                                 stream=True, 
                                 conf=conf_threshold,
                                 classes=classes,
                                 persist=True,
                                 verbose=False)
        
        for result in results:
            frame_count += 1
            
            # frame_interval에 따라 처리할 프레임만 선택
            if frame_count % frame_interval != 0:
                continue
            
            # 배치에 프레임 추가
            frame_batch.append(result)
            frame_numbers.append(frame_count)
            processed_frames += 1
            
            # 배치 크기에 도달하면 처리
            if len(frame_batch) >= batch_size:
                # 배치 처리
                futures, metadata_batch = self._process_detections_batch(
                    frame_batch, frame_numbers, fps, save_dir, conf_threshold
                )
                
                all_futures.extend(futures)
                
                # 메타데이터 수집
                for metadata in metadata_batch:
                    tracked_persons[metadata['track_id']].append(metadata)
                
                # 배치 초기화
                frame_batch = []
                frame_numbers = []
                
                # 메모리 정리
                if frame_count % 100 == 0:
                    gc.collect()
                    if self.device == 'cuda':
                        torch.cuda.empty_cache()
            
            # 진행상황 출력
            if processed_frames % 50 == 0:
                elapsed = time.time() - self.stats['start_time']
                fps_current = processed_frames / elapsed if elapsed > 0 else 0
                print(f"처리된 프레임: {processed_frames}/{total_frames//frame_interval} "
                      f"({processed_frames/(total_frames//frame_interval)*100:.1f}%) "
                      f"현재 FPS: {fps_current:.1f}")
        
        # 남은 배치 처리
        if frame_batch:
            futures, metadata_batch = self._process_detections_batch(
                frame_batch, frame_numbers, fps, save_dir, conf_threshold
            )
            all_futures.extend(futures)
            
            for metadata in metadata_batch:
                tracked_persons[metadata['track_id']].append(metadata)
        
        # 비동기 저장 작업 완료 대기
        if enable_async and all_futures:
            print("비동기 저장 작업 완료 대기 중...")
            for future, track_id, metadata in all_futures:
                try:
                    result = future.result(timeout=30)  # 30초 타임아웃
                    if result:
                        tracked_persons[track_id].append(result)
                except Exception as e:
                    print(f"저장 작업 오류: {e}")
        
        cap.release()
        
        # 성능 통계 계산
        total_time = time.time() - self.stats['start_time']
        self.stats.update({
            'total_frames': total_frames,
            'processed_frames': processed_frames,
            'total_time': total_time,
            'fps_processed': processed_frames / total_time if total_time > 0 else 0
        })
        
        # 최종 결과 저장
        summary = {
            'video_path': video_path,
            'total_frames': total_frames,
            'processed_frames': processed_frames,
            'conf_threshold': conf_threshold,
            'frame_interval': frame_interval,
            'batch_size': batch_size,
            'tracked_persons_count': len(tracked_persons),
            'performance_stats': self.stats,
            'tracked_persons': dict(tracked_persons)
        }
        
        # JSON 파일로 메타데이터 저장
        json_path = os.path.join(save_dir, "tracking_summary.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print(f"\n최적화된 처리 완료!")
        print(f"- 총 처리된 프레임: {processed_frames}")
        print(f"- 감지된 사람 수: {len(tracked_persons)}")
        print(f"- 총 처리 시간: {total_time:.2f}초")
        print(f"- 평균 처리 FPS: {self.stats['fps_processed']:.2f}")
        print(f"- 메타데이터 저장: {json_path}")
        
        # 워커 풀 종료
        self.executor.shutdown(wait=True)
        
        return tracked_persons, summary

    def analyze_tracking_results(self, summary_path):
        """tracking 결과를 분석하는 함수"""
        with open(summary_path, 'r', encoding='utf-8') as f:
            summary = json.load(f)
        
        print("=== 최적화된 Tracking 결과 분석 ===")
        print(f"영상: {summary['video_path']}")
        print(f"총 프레임 수: {summary['total_frames']}")
        print(f"처리된 프레임 수: {summary['processed_frames']}")
        print(f"감지된 사람 수: {summary['tracked_persons_count']}")
        print(f"Confidence threshold: {summary['conf_threshold']}")
        print(f"Frame interval: {summary['frame_interval']}")
        print(f"Batch size: {summary['batch_size']}")
        
        # 성능 통계
        if 'performance_stats' in summary:
            stats = summary['performance_stats']
            print(f"\n=== 성능 통계 ===")
            print(f"총 처리 시간: {stats['total_time']:.2f}초")
            print(f"평균 처리 FPS: {stats['fps_processed']:.2f}")
            print(f"실제 영상 FPS: {summary['total_frames'] / stats['total_time']:.2f}")
        
        # 각 사람별 통계
        print("\n=== 각 사람별 통계 ===")
        for track_id, person_data in summary['tracked_persons'].items():
            print(f"Person {track_id}: {len(person_data)}개 프레임에서 감지")
            if person_data:
                confidences = [data['confidence'] for data in person_data]
                print(f"  - 평균 confidence: {np.mean(confidences):.3f}")
                print(f"  - 최대 confidence: {np.max(confidences):.3f}")
                print(f"  - 최소 confidence: {np.min(confidences):.3f}")

def main():
    parser = argparse.ArgumentParser(description='최적화된 사람 detection 및 tracking')
    parser.add_argument('--video', type=str, required=True, help='입력 영상 경로')
    parser.add_argument('--conf', type=float, default=0.85, help='Confidence threshold (0.0-1.0)')
    parser.add_argument('--interval', type=int, default=1, help='프레임 간격')
    parser.add_argument('--save_dir', type=str, default='./tracked_persons_optimized', help='저장 디렉토리')
    parser.add_argument('--model', type=str, default='yolo11x.pt', help='YOLO 모델 경로')
    parser.add_argument('--batch_size', type=int, default=8, help='배치 크기')
    parser.add_argument('--workers', type=int, default=4, help='I/O 워커 수')
    parser.add_argument('--no_async', action='store_true', help='비동기 I/O 비활성화')
    
    args = parser.parse_args()
    
    # 최적화된 PersonTracker 인스턴스 생성
    tracker = OptimizedPersonTracker(model_path=args.model, max_workers=args.workers)
    
    # 사람 tracking 실행
    tracked_persons, summary = tracker.track_persons_optimized(
        video_path=args.video,
        conf_threshold=args.conf,
        frame_interval=args.interval,
        save_dir=args.save_dir,
        batch_size=args.batch_size,
        enable_async=not args.no_async
    )
    
    # 결과 분석
    summary_path = os.path.join(args.save_dir, "tracking_summary.json")
    tracker.analyze_tracking_results(summary_path)

if __name__ == "__main__":
    main()
