from person_tracker import PersonTracker
from person_tracker_optimized import OptimizedPersonTracker
import os
import glob
from pathlib import Path

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

def optimized_usage(video_path, save_dir):
    """단일 파일에 대한 최적화된 사용 예제"""
    print(f"=== 처리 중: {os.path.basename(video_path)} ===")
    
    # PersonTracker 인스턴스 생성
    tracker = OptimizedPersonTracker(model_path="yolo11x.pt", max_workers=8)

    tracked_persons, summary = tracker.track_persons_optimized(
        video_path=video_path,
        conf_threshold=0.8,  # confidence threshold
        frame_interval=5,    # 5프레임마다 한번씩 처리
        save_dir=save_dir,
        batch_size=8,
        enable_async=True
    )
    
    return tracked_persons, summary

def process_all_cctv_files():
    """모든 CCTV 파일을 처리하는 메인 함수"""
    print("🎬 모든 CCTV 파일 처리 시작")
    print("=" * 60)
    
    # cctv_dataset 폴더의 모든 AVI 파일 찾기
    cctv_dataset_path = "../cctv_dataset"
    video_files = []
    
    # os.walk를 사용하여 모든 하위 디렉토리의 AVI 파일 찾기
    for root, dirs, files in os.walk(cctv_dataset_path):
        for file in files:
            if file.lower().endswith('.avi'):
                video_files.append(os.path.join(root, file))
    
    # 파일 경로 정렬
    video_files.sort()
    
    print(f"📁 발견된 비디오 파일: {len(video_files)}개")
    
    # 결과 저장을 위한 메인 폴더 생성
    main_output_dir = "./tracked_results"
    os.makedirs(main_output_dir, exist_ok=True)
    
    success_count = 0
    error_count = 0
    results = []
    
    for i, video_path in enumerate(video_files, 1):
        try:
            # 파일명에서 확장자 제거하고 안전한 폴더명 생성
            video_name = Path(video_path).stem
            # 특수문자를 언더스코어로 변경하여 안전한 폴더명 생성
            safe_folder_name = "".join(c if c.isalnum() or c in '-_' else '_' for c in video_name)
            
            # 각 파일별 저장 폴더 생성
            file_save_dir = os.path.join(main_output_dir, safe_folder_name)
            os.makedirs(file_save_dir, exist_ok=True)
            
            print(f"\n[{i}/{len(video_files)}] 처리 중: {os.path.basename(video_path)}")
            print(f"📂 저장 폴더: {file_save_dir}")
            
            # optimized_usage 함수 실행
            tracked_persons, summary = optimized_usage(video_path, file_save_dir)
            
            # 결과 저장
            result_info = {
                'video_path': video_path,
                'save_dir': file_save_dir,
                'tracked_persons': len(tracked_persons) if tracked_persons else 0,
                'summary': summary,
                'status': 'success'
            }
            results.append(result_info)
            success_count += 1
            
            print(f"✅ 완료: {len(tracked_persons) if tracked_persons else 0}명 추적됨")
            
        except Exception as e:
            print(f"❌ 오류 발생: {os.path.basename(video_path)} - {str(e)}")
            error_count += 1
            
            error_info = {
                'video_path': video_path,
                'save_dir': None,
                'tracked_persons': 0,
                'summary': None,
                'status': 'error',
                'error': str(e)
            }
            results.append(error_info)
    
    # 최종 결과 출력
    print("\n" + "=" * 60)
    print("📊 처리 결과 요약")
    print("=" * 60)
    print(f"✅ 성공: {success_count}개")
    print(f"❌ 실패: {error_count}개")
    print(f"📁 결과 저장 위치: {os.path.abspath(main_output_dir)}")
    
    # 상세 결과 출력
    print(f"\n📋 상세 결과:")
    for i, result in enumerate(results, 1):
        status_icon = "✅" if result['status'] == 'success' else "❌"
        print(f"  {i:2d}. {status_icon} {os.path.basename(result['video_path'])}")
        if result['status'] == 'success':
            print(f"      📂 저장: {result['save_dir']}")
            print(f"      👥 추적된 사람: {result['tracked_persons']}명")
        else:
            print(f"      ❌ 오류: {result.get('error', 'Unknown error')}")
    
    return results

def test_with_small_sample():
    """소규모 테스트용 함수 (처음 3개 파일만 처리)"""
    print("🧪 소규모 테스트 시작 (처음 3개 파일만 처리)")
    print("=" * 60)
    
    # cctv_dataset 폴더의 모든 AVI 파일 찾기
    cctv_dataset_path = "../cctv_dataset"
    video_files = []
    
    # os.walk를 사용하여 모든 하위 디렉토리의 AVI 파일 찾기
    for root, dirs, files in os.walk(cctv_dataset_path):
        for file in files:
            if file.lower().endswith('.avi'):
                video_files.append(os.path.join(root, file))
    
    # 파일 경로 정렬
    video_files.sort()
    
    # 처음 3개 파일만 처리
    test_files = video_files[:3]
    print(f"📁 테스트할 비디오 파일: {len(test_files)}개")
    
    # 결과 저장을 위한 메인 폴더 생성
    main_output_dir = "./tracked_test_results"
    os.makedirs(main_output_dir, exist_ok=True)
    
    success_count = 0
    error_count = 0
    results = []
    
    for i, video_path in enumerate(test_files, 1):
        try:
            # 파일명에서 확장자 제거하고 안전한 폴더명 생성
            video_name = Path(video_path).stem
            # 특수문자를 언더스코어로 변경하여 안전한 폴더명 생성
            safe_folder_name = "".join(c if c.isalnum() or c in '-_' else '_' for c in video_name)
            
            # 각 파일별 저장 폴더 생성
            file_save_dir = os.path.join(main_output_dir, safe_folder_name)
            os.makedirs(file_save_dir, exist_ok=True)
            
            print(f"\n[{i}/{len(test_files)}] 처리 중: {os.path.basename(video_path)}")
            print(f"📂 저장 폴더: {file_save_dir}")
            
            # optimized_usage 함수 실행
            tracked_persons, summary = optimized_usage(video_path, file_save_dir)
            
            # 결과 저장
            result_info = {
                'video_path': video_path,
                'save_dir': file_save_dir,
                'tracked_persons': len(tracked_persons) if tracked_persons else 0,
                'summary': summary,
                'status': 'success'
            }
            results.append(result_info)
            success_count += 1
            
            print(f"✅ 완료: {len(tracked_persons) if tracked_persons else 0}명 추적됨")
            
        except Exception as e:
            print(f"❌ 오류 발생: {os.path.basename(video_path)} - {str(e)}")
            error_count += 1
            
            error_info = {
                'video_path': video_path,
                'save_dir': None,
                'tracked_persons': 0,
                'summary': None,
                'status': 'error',
                'error': str(e)
            }
            results.append(error_info)
    
    # 최종 결과 출력
    print("\n" + "=" * 60)
    print("📊 테스트 결과 요약")
    print("=" * 60)
    print(f"✅ 성공: {success_count}개")
    print(f"❌ 실패: {error_count}개")
    print(f"📁 결과 저장 위치: {os.path.abspath(main_output_dir)}")
    
    return results

if __name__=="__main__":
    # 전체 파일 처리 (주석 해제하여 사용)
    # process_all_cctv_files()
    
    # 소규모 테스트 (현재 활성화)
    test_with_small_sample()