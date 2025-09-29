from person_tracker import PersonTracker
from person_tracker_optimized import OptimizedPersonTracker

def basic_usage():
    """기본 사용 예제"""
    print("=== 기본 사용 예제 ===")
    
    # PersonTracker 인스턴스 생성
    tracker = PersonTracker(model_path="yolo11x.pt")
    
    # 영상 경로 설정
    video_path = "../cctv_dataset/0926_cctv1.avi"
    
    # 기본 설정으로 tracking 실행
    tracked_persons, summary = tracker.track_persons(
        video_path=video_path,
        conf_threshold=0.85,  # confidence threshold
        frame_interval=5,    # 5프레임마다 한번씩 처리
        save_dir="./tracked_cctv0"
    )
    
    return tracked_persons, summary

def optimized_usage():
    """기본 사용 예제"""
    print("=== 기본 사용 예제 ===")
    
    # PersonTracker 인스턴스 생성
    tracker = OptimizedPersonTracker(model_path="yolo11x.pt", max_workers=8)
    
    # 영상 경로 설정
    video_path = "../cctv_dataset/0926_cctv0.avi"

    tracked_persons, summary = tracker.track_persons_optimized(
        video_path=video_path,
        conf_threshold=0.8,  # confidence threshold
        frame_interval=5,    # 5프레임마다 한번씩 처리
        save_dir="./tracked_cctv0",
        batch_size=8,
        enable_async=True
    )
    
    return tracked_persons, summary

if __name__=="__main__":
    optimized_usage()