"""
차량 데이터셋 구축 통합 파이프라인

단계:
1. Tracking: CCTV 영상에서 차량 추적 및 crop
2. Filtering: 번호판 detection으로 필터링 및 번호판 crop
3. Pseudo Labeling: 번호판 이미지에 OCR로 라벨링
"""
import os
import argparse
from pathlib import Path
from typing import Optional

# 각 단계 모듈 import
try:
    from .tracking.pipeline_roi import run as run_tracking
    from .filtering.filtering_by_lp import process_runs_directory as run_filtering
    from .pseudo_labeling.vllm.ocr_client import VLLMOCRClient
    from .pseudo_labeling.vllm.directory_processor import process_recursive
except ImportError:
    # 직접 실행 시 상대 경로로 import
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from datasets.car.tracking.pipeline_roi import run as run_tracking
    from datasets.car.filtering.filtering_by_lp import process_runs_directory as run_filtering
    from datasets.car.pseudo_labeling.vllm.ocr_client import VLLMOCRClient
    from datasets.car.pseudo_labeling.vllm.directory_processor import process_recursive


def ensure_dir(path: str) -> None:
    """디렉토리가 없으면 생성"""
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def step1_tracking(
    video_dir: str,
    output_dir: str,
    model_path: str = "checkpoints/detection/yolo11n.pt",
    conf: float = 0.7,
    iou: float = 0.5,
    max_frames_per_track: int = 200,
    roi_file: Optional[str] = None,
    roi_min_iou: float = 0.7,
    num_workers: Optional[int] = None,
) -> str:
    """
    Step 1: CCTV 영상에서 차량 추적 및 crop
    
    Args:
        video_dir: CCTV 영상이 들어있는 디렉토리 경로
        output_dir: 추적 결과를 저장할 디렉토리 경로
        model_path: YOLO 모델 경로
        conf: confidence threshold
        iou: IoU threshold
        max_frames_per_track: 트랙별 최대 저장 프레임 수
        roi_file: ROI 파일 경로
        roi_min_iou: ROI 필터링 최소 IoU 임계값
        num_workers: 병렬 처리 프로세스 수 (None이면 자동)
    
    Returns:
        추적 결과 디렉토리 경로
    """
    print("\n" + "="*80)
    print("Step 1: 차량 추적 및 Crop")
    print("="*80)
    print(f"입력 비디오 디렉토리: {video_dir}")
    print(f"출력 디렉토리: {output_dir}")
    
    ensure_dir(output_dir)
    
    # Tracking 실행
    run_tracking(
        source_path=video_dir,
        output_dir=output_dir,
        model_path=model_path,
        conf=conf,
        iou=iou,
        show=False,
        max_frames_per_track=max_frames_per_track,
        stationary_rel_thresh=0.002,
        stationary_patience=30,
        roi_file=roi_file,
        roi_min_iou=roi_min_iou,
        num_workers=num_workers,
    )
    
    print(f"\nStep 1 완료! 결과: {output_dir}")
    return output_dir


def step2_filtering(
    tracking_dir: str,
    output_dir: str,
    lp_crop_dir: str,
    lp_detection_model_path: str = "checkpoints/detection/lp_detection.pt",
) -> tuple[str, str]:
    """
    Step 2: 번호판 detection으로 필터링 및 번호판 crop
    
    Args:
        tracking_dir: Step 1의 추적 결과 디렉토리
        output_dir: 필터링된 차량 이미지 저장 디렉토리
        lp_crop_dir: 번호판 crop 이미지 저장 디렉토리
        lp_detection_model_path: 번호판 detection 모델 경로
    
    Returns:
        (필터링된 차량 이미지 디렉토리, 번호판 crop 디렉토리) 튜플
    """
    print("\n" + "="*80)
    print("Step 2: 번호판 Detection 및 필터링")
    print("="*80)
    print(f"입력 디렉토리: {tracking_dir}")
    print(f"필터링된 차량 이미지 출력: {output_dir}")
    print(f"번호판 crop 이미지 출력: {lp_crop_dir}")
    
    ensure_dir(output_dir)
    ensure_dir(lp_crop_dir)
    
    # Filtering 실행
    run_filtering(
        runs_dir=tracking_dir,
        output_dir=output_dir,
        lp_detection_model_path=lp_detection_model_path,
        output_lp_dir=lp_crop_dir,
    )
    
    print(f"\nStep 2 완료!")
    print(f"  필터링된 차량 이미지: {output_dir}")
    print(f"  번호판 crop 이미지: {lp_crop_dir}")
    
    return output_dir, lp_crop_dir


def step3_pseudo_labeling(
    lp_crop_dir: str,
    host: str = "0.0.0.0",
    port: int = 8000,
    model: str = "Qwen/Qwen3-VL-4B-Instruct",
    gpu_util: float = 0.8,
    prompt: str = "차량 번호판의 문자를 추출해주세요.",
    max_tokens: int = 300,
    label_filename: str = "label.txt",
    auto_start: bool = True,
    show_log: bool = True,
) -> dict:
    """
    Step 3: 번호판 이미지에 OCR로 라벨링
    
    Args:
        lp_crop_dir: 번호판 crop 이미지 디렉토리
        host: VLLM 서버 호스트
        port: VLLM 서버 포트
        model: VLLM 모델 이름
        gpu_util: GPU 사용률
        prompt: OCR 프롬프트
        max_tokens: 최대 토큰 수
        label_filename: 생성할 라벨 파일 이름
        auto_start: 서버 자동 시작 여부
        show_log: 서버 로그 표시 여부
    
    Returns:
        처리 결과 딕셔너리 {디렉토리: 라벨}
    """
    print("\n" + "="*80)
    print("Step 3: 번호판 OCR 라벨링")
    print("="*80)
    print(f"입력 디렉토리: {lp_crop_dir}")
    print(f"VLLM 모델: {model}")
    
    # OCR 클라이언트 초기화
    print("\nOCR 클라이언트 초기화 중...")
    ocr_client = VLLMOCRClient(
        host=host,
        port=port,
        model=model,
        served_name=model,
        gpu_util=gpu_util,
        tp=1,
        log="vllm_server.log",
        max_model_len=40000,
        allowed_local_media_path=str(Path(lp_crop_dir).resolve().parent.parent.parent.parent),
        prompt=prompt,
        max_tokens=max_tokens,
        auto_start=auto_start,
        show_log=show_log
    )
    
    try:
        # 재귀적으로 디렉토리 처리
        print(f"\n재귀 디렉토리 처리 모드")
        results = process_recursive(
            root_directory=lp_crop_dir,
            ocr_client=ocr_client,
            label_filename=label_filename,
        )
        
        print(f"\n{'='*80}")
        print(f"Step 3 완료!")
        print(f"처리된 디렉토리 수: {len(results)}")
        print(f"{'='*80}")
        
        for dir_path, label in results.items():
            print(f"{dir_path}: {label}")
        
        return results
    
    except KeyboardInterrupt:
        print("\n\n사용자에 의해 중단되었습니다.")
        return {}
    except Exception as e:
        print(f"\n오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return {}
    finally:
        # 서버 종료
        print("\n서버 종료 중...")
        ocr_client.stop()


def run_full_pipeline(
    video_dir: str,
    output_base_dir: str,
    # Tracking 파라미터
    tracking_model_path: str = "checkpoints/detection/yolo11n.pt",
    tracking_conf: float = 0.7,
    tracking_iou: float = 0.5,
    max_frames_per_track: int = 200,
    roi_file: Optional[str] = None,
    roi_min_iou: float = 0.7,
    num_workers: Optional[int] = None,
    # Filtering 파라미터
    lp_detection_model_path: str = "checkpoints/detection/lp_detection.pt",
    # Pseudo Labeling 파라미터
    vllm_host: str = "0.0.0.0",
    vllm_port: int = 8000,
    vllm_model: str = "Qwen/Qwen3-VL-4B-Instruct",
    vllm_gpu_util: float = 0.8,
    vllm_prompt: str = "차량 번호판의 문자를 추출해주세요.",
    vllm_max_tokens: int = 300,
    label_filename: str = "label.txt",
    # 단계별 실행 제어
    skip_tracking: bool = False,
    skip_filtering: bool = False,
    skip_labeling: bool = False,
) -> dict:
    """
    전체 파이프라인 실행
    
    Args:
        video_dir: CCTV 영상이 들어있는 디렉토리 경로
        output_base_dir: 모든 출력 결과의 기본 디렉토리
        tracking_model_path: YOLO 모델 경로
        tracking_conf: confidence threshold
        tracking_iou: IoU threshold
        max_frames_per_track: 트랙별 최대 저장 프레임 수
        roi_file: ROI 파일 경로
        roi_min_iou: ROI 필터링 최소 IoU 임계값
        num_workers: 병렬 처리 프로세스 수
        lp_detection_model_path: 번호판 detection 모델 경로
        vllm_host: VLLM 서버 호스트
        vllm_port: VLLM 서버 포트
        vllm_model: VLLM 모델 이름
        vllm_gpu_util: GPU 사용률
        vllm_prompt: OCR 프롬프트
        vllm_max_tokens: 최대 토큰 수
        label_filename: 생성할 라벨 파일 이름
        skip_tracking: Step 1 건너뛰기
        skip_filtering: Step 2 건너뛰기
        skip_labeling: Step 3 건너뛰기
    
    Returns:
        각 단계의 출력 디렉토리 정보를 담은 딕셔너리
    """
    print("\n" + "="*80)
    print("차량 데이터셋 구축 통합 파이프라인 시작")
    print("="*80)
    print(f"입력 비디오 디렉토리: {video_dir}")
    print(f"출력 기본 디렉토리: {output_base_dir}")
    
    ensure_dir(output_base_dir)
    
    # 출력 디렉토리 구조
    tracking_dir = os.path.join(output_base_dir, "01_tracking")
    filtering_dir = os.path.join(output_base_dir, "02_filtered_vehicles")
    lp_crop_dir = os.path.join(output_base_dir, "03_license_plates")
    
    results = {
        "tracking_dir": tracking_dir,
        "filtering_dir": filtering_dir,
        "lp_crop_dir": lp_crop_dir,
    }
    
    # Step 1: Tracking
    if not skip_tracking:
        step1_tracking(
            video_dir=video_dir,
            output_dir=tracking_dir,
            model_path=tracking_model_path,
            conf=tracking_conf,
            iou=tracking_iou,
            max_frames_per_track=max_frames_per_track,
            roi_file=roi_file,
            roi_min_iou=roi_min_iou,
            num_workers=num_workers,
        )
    else:
        print("\nStep 1 (Tracking) 건너뛰기")
        if not os.path.exists(tracking_dir):
            print(f"경고: {tracking_dir} 디렉토리가 존재하지 않습니다.")
    
    # Step 2: Filtering
    if not skip_filtering:
        step2_filtering(
            tracking_dir=tracking_dir,
            output_dir=filtering_dir,
            lp_crop_dir=lp_crop_dir,
            lp_detection_model_path=lp_detection_model_path,
        )
    else:
        print("\nStep 2 (Filtering) 건너뛰기")
        if not os.path.exists(lp_crop_dir):
            print(f"경고: {lp_crop_dir} 디렉토리가 존재하지 않습니다.")
    
    # Step 3: Pseudo Labeling
    if not skip_labeling:
        labeling_results = step3_pseudo_labeling(
            lp_crop_dir=lp_crop_dir,
            host=vllm_host,
            port=vllm_port,
            model=vllm_model,
            gpu_util=vllm_gpu_util,
            prompt=vllm_prompt,
            max_tokens=vllm_max_tokens,
            label_filename=label_filename,
            auto_start=True,
            show_log=True,
        )
        results["labeling_results"] = labeling_results
    else:
        print("\nStep 3 (Pseudo Labeling) 건너뛰기")
    
    print("\n" + "="*80)
    print("전체 파이프라인 완료!")
    print("="*80)
    print(f"추적 결과: {tracking_dir}")
    print(f"필터링된 차량 이미지: {filtering_dir}")
    print(f"번호판 crop 이미지: {lp_crop_dir}")
    if "labeling_results" in results:
        print(f"라벨링 완료된 디렉토리 수: {len(results['labeling_results'])}")
    
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="차량 데이터셋 구축 통합 파이프라인",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예제:
  # 전체 파이프라인 실행
  python -m datasets.car.pipeline \\
      --video_dir /path/to/videos \\
      --output_dir /path/to/output

  # 특정 단계만 실행
  python -m datasets.car.pipeline \\
      --video_dir /path/to/videos \\
      --output_dir /path/to/output \\
      --skip_tracking \\
      --skip_filtering
        """
    )
    
    # 필수 인자
    parser.add_argument(
        "--video_dir",
        type=str,
        required=True,
        help="CCTV 영상이 들어있는 디렉토리 경로"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="모든 출력 결과의 기본 디렉토리"
    )
    
    # Tracking 파라미터
    parser.add_argument(
        "--tracking_model",
        type=str,
        default="checkpoints/detection/yolo11n.pt",
        help="YOLO 모델 경로 (기본값: checkpoints/detection/yolo11n.pt)"
    )
    parser.add_argument(
        "--tracking_conf",
        type=float,
        default=0.7,
        help="Tracking confidence threshold (기본값: 0.7)"
    )
    parser.add_argument(
        "--tracking_iou",
        type=float,
        default=0.5,
        help="Tracking IoU threshold (기본값: 0.5)"
    )
    parser.add_argument(
        "--max_frames_per_track",
        type=int,
        default=200,
        help="트랙별 최대 저장 프레임 수 (기본값: 200)"
    )
    parser.add_argument(
        "--roi_file",
        type=str,
        default="/data/reid/reid_master/roi.txt",
        help="ROI 파일 경로 (None이면 ROI 필터링 없음)"
    )
    parser.add_argument(
        "--roi_min_iou",
        type=float,
        default=0.3,
        help="ROI 필터링 최소 IoU 임계값 (기본값: 0.3)"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=None,
        help="병렬 처리 프로세스 수 (None이면 자동)"
    )
    
    # Filtering 파라미터
    parser.add_argument(
        "--lp_detection_model",
        type=str,
        default="checkpoints/detection/lp_detection.pt",
        help="번호판 detection 모델 경로 (기본값: checkpoints/detection/lp_detection.pt)"
    )
    
    # Pseudo Labeling 파라미터
    parser.add_argument(
        "--vllm_host",
        type=str,
        default="0.0.0.0",
        help="VLLM 서버 호스트 (기본값: 0.0.0.0)"
    )
    parser.add_argument(
        "--vllm_port",
        type=int,
        default=8000,
        help="VLLM 서버 포트 (기본값: 8000)"
    )
    parser.add_argument(
        "--vllm_model",
        type=str,
        default="Qwen/Qwen3-VL-4B-Instruct",
        help="VLLM 모델 이름 (기본값: Qwen/Qwen3-VL-4B-Instruct)"
    )
    parser.add_argument(
        "--vllm_gpu_util",
        type=float,
        default=0.8,
        help="GPU 사용률 (기본값: 0.8)"
    )
    parser.add_argument(
        "--vllm_prompt",
        type=str,
        default="차량 번호판의 문자를 추출해주세요.",
        help="OCR 프롬프트 (기본값: 차량 번호판의 문자를 추출해주세요.)"
    )
    parser.add_argument(
        "--vllm_max_tokens",
        type=int,
        default=300,
        help="최대 토큰 수 (기본값: 300)"
    )
    parser.add_argument(
        "--label_filename",
        type=str,
        default="label.txt",
        help="생성할 라벨 파일 이름 (기본값: label.txt)"
    )
    
    # 단계별 실행 제어
    parser.add_argument(
        "--skip_tracking",
        action="store_true",
        help="Step 1 (Tracking) 건너뛰기"
    )
    parser.add_argument(
        "--skip_filtering",
        action="store_true",
        help="Step 2 (Filtering) 건너뛰기"
    )
    parser.add_argument(
        "--skip_labeling",
        action="store_true",
        help="Step 3 (Pseudo Labeling) 건너뛰기"
    )
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    run_full_pipeline(
        video_dir=args.video_dir,
        output_base_dir=args.output_dir,
        # Tracking 파라미터
        tracking_model_path=args.tracking_model,
        tracking_conf=args.tracking_conf,
        tracking_iou=args.tracking_iou,
        max_frames_per_track=args.max_frames_per_track,
        roi_file=args.roi_file,
        roi_min_iou=args.roi_min_iou,
        num_workers=args.num_workers,
        # Filtering 파라미터
        lp_detection_model_path=args.lp_detection_model,
        # Pseudo Labeling 파라미터
        vllm_host=args.vllm_host,
        vllm_port=args.vllm_port,
        vllm_model=args.vllm_model,
        vllm_gpu_util=args.vllm_gpu_util,
        vllm_prompt=args.vllm_prompt,
        vllm_max_tokens=args.vllm_max_tokens,
        label_filename=args.label_filename,
        # 단계별 실행 제어
        skip_tracking=args.skip_tracking,
        skip_filtering=args.skip_filtering,
        skip_labeling=args.skip_labeling,
    )

