import cv2
import time
from pathlib import Path
from detector import PersonDetector
from cropper import save_tracklets
from metadata_writer import append_metadata
from visualizer import ProgressVisualizer

def process_video(video_path, cam_id="cam01", out_dir="dataset_raw", conf=0.85, frame_interval=1):
    detector = PersonDetector(conf=conf)
    cap = cv2.VideoCapture(video_path)
    
    # 비디오 정보 가져오기
    reported_total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # FPS 및 총 프레임 수 검증
    print(f"📊 OpenCV 보고 정보:")
    print(f"   📏 보고된 총 프레임: {reported_total_frames:,}")
    print(f"   ⏱️  보고된 FPS: {fps:.1f}")
    
    # 실제 FPS와 총 프레임 수를 계산
    print("🔍 실제 비디오 정보를 계산합니다...")
    
    # 처음 100프레임으로 FPS 계산
    test_frames = 100
    start_time = time.time()
    actual_frames_counted = 0
    
    for i in range(test_frames):
        ret, _ = cap.read()
        if not ret:
            break
        actual_frames_counted += 1
    
    end_time = time.time()
    
    if end_time > start_time and actual_frames_counted > 0:
        actual_fps = actual_frames_counted / (end_time - start_time)
        print(f"   ✅ 계산된 실제 FPS: {actual_fps:.1f}")
        fps = actual_fps
    else:
        fps = 30.0  # 기본값 사용
        print(f"   ⚠️  기본 FPS 사용: {fps}")
    
    # AVI 파일의 경우 더 엄격한 검증 필요
    video_extension = Path(video_path).suffix.lower()
    is_avi = video_extension == '.avi'
    
    if is_avi:
        print(f"   🎬 AVI 파일 감지: 동적 계산 모드로 강제 전환")
        print(f"   📊 보고된 프레임 수: {reported_total_frames:,} (신뢰하지 않음)")
        print(f"   💡 AVI 파일은 OpenCV에서 부정확한 프레임 수를 보고하는 경우가 많습니다")
        total_frames = -1  # AVI는 항상 동적 계산 모드
    else:
        # MP4 등 다른 포맷
        if reported_total_frames > 0 and reported_total_frames < 1000000:  # 합리적인 범위
            total_frames = reported_total_frames
            print(f"   ✅ OpenCV 보고 프레임 수 사용: {total_frames:,}")
        else:
            print(f"   ⚠️  OpenCV 보고 프레임 수가 비정상적입니다 ({reported_total_frames:,})")
            print("   📏 동적 계산 모드로 전환합니다...")
            total_frames = -1  # 동적 계산 모드
    
    # 비디오 캡처 재시작
    cap.release()
    cap = cv2.VideoCapture(video_path)
    
    # 시각화 도구 초기화
    visualizer = ProgressVisualizer(video_path, total_frames, fps, conf, out_dir, cam_id, frame_interval)
    visualizer.print_video_info()

    frame_count = 0
    processed_frames = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        
        # frame_interval에 따라 처리 여부 결정
        if frame_count % frame_interval == 0:
            processed_frames += 1
            
            # YOLO 내장 tracking 사용
            boxes, confs, track_ids = detector.track(frame)
            
            # tracking 결과를 기존 형식으로 변환
            tracks = []
            for i, (box, conf, tid) in enumerate(zip(boxes, confs, track_ids)):
                x1, y1, x2, y2 = box
                tracks.append((x1, y1, x2, y2, tid, conf))

            # 진행상황 업데이트 (처리된 프레임 수 증가)
            visualizer.update_progress(tracks, frame_count, processed_frames)
            
            # 이미지 저장 및 메타데이터 기록
            timestamp = time.strftime("%Y-%m-%dT%H:%M:%S")
            frame_id = visualizer.get_frame_id()
            save_tracklets(frame, tracks, cam_id, out_dir, frame_id, timestamp)

            for x1, y1, x2, y2, tid, conf in tracks:
                append_metadata(cam_id, tid, frame_id, conf, timestamp, out_dir)
        else:
            # 처리하지 않는 프레임은 빈 tracks로 진행상황만 업데이트
            visualizer.update_progress([], frame_count, processed_frames)

    cap.release()
    visualizer.close()

if __name__ == "__main__":
    import argparse


    parser = argparse.ArgumentParser(description="Extract tracklets from CCTV video")
    parser.add_argument("--video", type=str, required=True, help="Path to video file or RTSP stream")
    parser.add_argument("--cam-id", type=str, default="cam01")
    parser.add_argument("--out", type=str, default="dataset_raw")
    parser.add_argument("--conf", type=float, default=0.85)
    parser.add_argument("--frame-interval", type=int, default=1, 
                       help="Process every N frames (default: 1, process every frame)")
    args = parser.parse_args()

    process_video(args.video, args.cam_id, args.out, args.conf, args.frame_interval)