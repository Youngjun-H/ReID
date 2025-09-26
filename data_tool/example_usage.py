#!/usr/bin/env python3
"""
PersonTracker 사용 예제 스크립트
"""

from person_tracker import PersonTracker

def example_basic_usage():
    """기본 사용 예제"""
    print("=== 기본 사용 예제 ===")
    
    # PersonTracker 인스턴스 생성
    tracker = PersonTracker(model_path="yolo11x.pt")
    
    # 영상 경로 설정
    video_path = "../cctv_dataset/0926_cctv0.avi"
    
    # 기본 설정으로 tracking 실행
    tracked_persons, summary = tracker.track_persons(
        video_path=video_path,
        conf_threshold=0.6,  # confidence threshold
        frame_interval=5,    # 5프레임마다 한번씩 처리
        save_dir="./tracked_persons"
    )
    
    return tracked_persons, summary

def example_custom_settings():
    """커스텀 설정 사용 예제"""
    print("=== 커스텀 설정 사용 예제 ===")
    
    # PersonTracker 인스턴스 생성
    tracker = PersonTracker(model_path="yolo11x.pt")
    
    # 영상 경로 설정
    video_path = "../cctv_dataset/0926_cctv0.avi"
    
    # 높은 정확도를 위한 설정
    tracked_persons, summary = tracker.track_persons(
        video_path=video_path,
        conf_threshold=0.8,  # 높은 confidence threshold
        frame_interval=1,    # 모든 프레임 처리
        save_dir="./tracked_persons_high_conf"
    )
    
    return tracked_persons, summary

def example_fast_processing():
    """빠른 처리 예제"""
    print("=== 빠른 처리 예제 ===")
    
    # PersonTracker 인스턴스 생성
    tracker = PersonTracker(model_path="yolo11x.pt")
    
    # 영상 경로 설정
    video_path = "../cctv_dataset/0926_cctv0.avi"
    
    # 빠른 처리를 위한 설정
    tracked_persons, summary = tracker.track_persons(
        video_path=video_path,
        conf_threshold=0.4,  # 낮은 confidence threshold
        frame_interval=10,   # 10프레임마다 한번씩 처리
        save_dir="./tracked_persons_fast"
    )
    
    return tracked_persons, summary

def example_analyze_results():
    """결과 분석 예제"""
    print("=== 결과 분석 예제 ===")
    
    # PersonTracker 인스턴스 생성
    tracker = PersonTracker()
    
    # 저장된 결과 분석
    summary_path = "./tracked_persons/tracking_summary.json"
    if os.path.exists(summary_path):
        tracker.analyze_tracking_results(summary_path)
    else:
        print(f"결과 파일을 찾을 수 없습니다: {summary_path}")

if __name__ == "__main__":
    import os
    
    print("PersonTracker 사용 예제")
    print("=" * 50)
    
    # 사용할 예제 선택
    choice = input("실행할 예제를 선택하세요 (1: 기본, 2: 커스텀, 3: 빠른처리, 4: 결과분석): ")
    
    if choice == "1":
        example_basic_usage()
    elif choice == "2":
        example_custom_settings()
    elif choice == "3":
        example_fast_processing()
    elif choice == "4":
        example_analyze_results()
    else:
        print("잘못된 선택입니다. 기본 예제를 실행합니다.")
        example_basic_usage()
