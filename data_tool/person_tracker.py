from ultralytics import YOLO
import torch
import cv2
import os
import numpy as np
from collections import defaultdict
import json
import argparse

class PersonTracker:
    def __init__(self, model_path="yolo11x.pt"):
        """
        사람 detection 및 tracking을 위한 클래스
        
        Args:
            model_path (str): YOLO 모델 경로
        """
        # GPU 사용 가능 여부 확인
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"사용 중인 디바이스: {self.device}")
        
        # YOLO 모델 로드
        self.model = YOLO(model_path)
        self.model.to(self.device)
        
    def track_persons(self, video_path, 
                     conf_threshold=0.85, 
                     frame_interval=1, 
                     save_dir="./tracked_persons",
                     classes=[0]):  # 0은 COCO 데이터셋에서 person 클래스
        """
        영상에서 사람만 detection하고 track_id별로 저장하는 함수
        
        Args:
            video_path (str): 입력 영상 경로
            conf_threshold (float): confidence threshold (0.0 ~ 1.0)
            frame_interval (int): 몇 프레임당 한번 모델로 인식할지 설정
            save_dir (str): 저장할 디렉토리 경로
            classes (list): detection할 클래스 ID (0=person)
        
        Returns:
            tuple: (tracked_persons, summary)
        """
        
        # 저장 디렉토리 생성
        os.makedirs(save_dir, exist_ok=True)
        
        # track_id별로 데이터를 저장할 딕셔너리
        tracked_persons = defaultdict(list)
        
        # 영상 정보 가져오기
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        print(f"영상 정보:")
        print(f"- 총 프레임 수: {total_frames}")
        print(f"- FPS: {fps}")
        print(f"- Confidence threshold: {conf_threshold}")
        print(f"- Frame interval: {frame_interval}")
        print(f"- 저장 디렉토리: {save_dir}")
        print("-" * 50)
        
        frame_count = 0
        processed_frames = 0
        
        # stream=True로 설정하여 메모리 효율적으로 처리
        results = self.model.track(video_path, 
                                 stream=True, 
                                 conf=conf_threshold,
                                 classes=classes,
                                 persist=True)
        
        for result in results:
            frame_count += 1
            
            # frame_interval에 따라 처리할 프레임만 선택
            if frame_count % frame_interval != 0:
                continue
                
            processed_frames += 1
            
            # detection 결과가 있는 경우
            if result.boxes is not None and len(result.boxes) > 0:
                boxes = result.boxes
                
                # 각 detection에 대해 처리
                for i, box in enumerate(boxes):
                    # confidence 확인
                    conf = float(box.conf[0])
                    if conf < conf_threshold:
                        continue
                    
                    # track_id 확인
                    if hasattr(box, 'id') and box.id is not None:
                        track_id = int(box.id[0])
                    else:
                        # track_id가 없는 경우 건너뛰기
                        continue
                    
                    # bounding box 좌표
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    
                    # 프레임 이미지에서 해당 영역 추출
                    frame_img = result.orig_img
                    person_crop = frame_img[int(y1):int(y2), int(x1):int(x2)]
                    
                    # 빈 이미지가 아닌 경우에만 저장
                    if person_crop.size > 0:
                        # track_id별 디렉토리 생성
                        track_dir = os.path.join(save_dir, f"person_{track_id:04d}")
                        os.makedirs(track_dir, exist_ok=True)
                        
                        # 이미지 저장
                        img_filename = f"frame_{frame_count:06d}_conf_{conf:.3f}.jpg"
                        img_path = os.path.join(track_dir, img_filename)
                        cv2.imwrite(img_path, person_crop)
                        
                        # 메타데이터 저장
                        metadata = {
                            'frame_number': frame_count,
                            'track_id': track_id,
                            'confidence': conf,
                            'bbox': [float(x1), float(y1), float(x2), float(y2)],
                            'timestamp': frame_count / fps,
                            'image_path': img_path
                        }
                        
                        tracked_persons[track_id].append(metadata)
            
            # 진행상황 출력 (100프레임마다)
            if processed_frames % 100 == 0:
                print(f"처리된 프레임: {processed_frames}/{total_frames//frame_interval} "
                      f"({processed_frames/(total_frames//frame_interval)*100:.1f}%)")
        
        cap.release()
        
        # 최종 결과 저장
        summary = {
            'video_path': video_path,
            'total_frames': total_frames,
            'processed_frames': processed_frames,
            'conf_threshold': conf_threshold,
            'frame_interval': frame_interval,
            'tracked_persons_count': len(tracked_persons),
            'tracked_persons': dict(tracked_persons)
        }
        
        # JSON 파일로 메타데이터 저장
        json_path = os.path.join(save_dir, "tracking_summary.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print(f"\n처리 완료!")
        print(f"- 총 처리된 프레임: {processed_frames}")
        print(f"- 감지된 사람 수: {len(tracked_persons)}")
        print(f"- 메타데이터 저장: {json_path}")
        
        return tracked_persons, summary

    def analyze_tracking_results(self, summary_path):
        """
        tracking 결과를 분석하는 함수
        
        Args:
            summary_path (str): tracking_summary.json 파일 경로
        """
        with open(summary_path, 'r', encoding='utf-8') as f:
            summary = json.load(f)
        
        print("=== Tracking 결과 분석 ===")
        print(f"영상: {summary['video_path']}")
        print(f"총 프레임 수: {summary['total_frames']}")
        print(f"처리된 프레임 수: {summary['processed_frames']}")
        print(f"감지된 사람 수: {summary['tracked_persons_count']}")
        print(f"Confidence threshold: {summary['conf_threshold']}")
        print(f"Frame interval: {summary['frame_interval']}")
        
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
    parser = argparse.ArgumentParser(description='사람 detection 및 tracking')
    parser.add_argument('--video', type=str, required=True, help='입력 영상 경로')
    parser.add_argument('--conf', type=float, default=0.5, help='Confidence threshold (0.0-1.0)')
    parser.add_argument('--interval', type=int, default=1, help='프레임 간격 (몇 프레임마다 처리할지)')
    parser.add_argument('--save_dir', type=str, default='./tracked_persons', help='저장 디렉토리')
    parser.add_argument('--model', type=str, default='yolo11x.pt', help='YOLO 모델 경로')
    
    args = parser.parse_args()
    
    # PersonTracker 인스턴스 생성
    tracker = PersonTracker(model_path=args.model)
    
    # 사람 tracking 실행
    tracked_persons, summary = tracker.track_persons(
        video_path=args.video,
        conf_threshold=args.conf,
        frame_interval=args.interval,
        save_dir=args.save_dir
    )
    
    # 결과 분석
    summary_path = os.path.join(args.save_dir, "tracking_summary.json")
    tracker.analyze_tracking_results(summary_path)

if __name__ == "__main__":
    main()
