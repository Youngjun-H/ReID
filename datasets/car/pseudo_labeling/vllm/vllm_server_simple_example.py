"""
VLLM OCR 서버 예제 및 디렉토리 처리 스크립트
"""
import argparse
import sys
import os
from pathlib import Path
try:
    from .ocr_client import VLLMOCRClient
    from .directory_processor import process_directory, process_recursive, get_image_files
except ImportError:
    from ocr_client import VLLMOCRClient
    from directory_processor import process_directory, process_recursive, get_image_files


def main():
    # 프로젝트 루트 찾기 및 임시 파일 디렉토리 설정
    project_root = Path(__file__).parent.parent.parent.parent.parent
    cache_dir = project_root / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # 임시 파일 디렉토리 설정 (환경 변수가 없으면 설정)
    if "TMPDIR" not in os.environ:
        tmp_dir = cache_dir / "tmp"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        os.environ["TMPDIR"] = str(tmp_dir)
        os.environ["TEMP"] = str(tmp_dir)
        os.environ["TMP"] = str(tmp_dir)
    
    # HuggingFace 임시 파일 디렉토리 설정
    if "HF_HUB_TEMP" not in os.environ:
        os.environ["HF_HUB_TEMP"] = str(cache_dir / "tmp")
    
    # VLLM 캐시 디렉토리 설정 (중요!)
    if "VLLM_CACHE_ROOT" not in os.environ:
        vllm_cache_dir = cache_dir / "vllm"
        vllm_cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ["VLLM_CACHE_ROOT"] = str(vllm_cache_dir)
        print(f"VLLM 캐시 디렉토리 자동 설정: {vllm_cache_dir}")
    
    # VLLM Usage Stats 파일 경로 설정
    if "VLLM_USAGE_STATS_PATH" not in os.environ:
        os.environ["VLLM_USAGE_STATS_PATH"] = str(cache_dir / "vllm" / "usage_stats.json")
        print(f"VLLM Usage Stats 경로 자동 설정: {os.environ['VLLM_USAGE_STATS_PATH']}")
    parser = argparse.ArgumentParser(description="디렉토리 내 이미지에 대해 OCR을 수행하고 label.txt를 생성합니다.")
    parser.add_argument(
        "directory",
        type=str,
        help="처리할 디렉토리 경로 (하위 디렉토리까지 재귀적으로 처리)"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="서버 호스트 (기본값: 0.0.0.0)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="서버 포트 (기본값: 8000)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-VL-4B-Instruct",
        help="모델 이름 (기본값: Qwen/Qwen3-VL-4B-Instruct)"
    )
    parser.add_argument(
        "--gpu-util",
        type=float,
        default=0.8,
        help="GPU 사용률 (기본값: 0.8)"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="차량 번호판의 문자를 추출해주세요.",
        help="OCR 프롬프트 (기본값: 차량 번호판의 문자를 추출해주세요.)"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=300,
        help="최대 토큰 수 (기본값: 300)"
    )
    parser.add_argument(
        "--label-filename",
        type=str,
        default="label_qwen3.txt",
        help="생성할 라벨 파일 이름 (기본값: label.txt)"
    )
    parser.add_argument(
        "--no-auto-start",
        action="store_true",
        help="서버를 자동으로 시작하지 않음"
    )
    parser.add_argument(
        "--hide-log",
        action="store_true",
        help="서버 로그를 표시하지 않음"
    )
    
    args = parser.parse_args()
    
    # OCR 클라이언트 초기화
    print("OCR 클라이언트 초기화 중...")
    ocr_client = VLLMOCRClient(
        host=args.host,
        port=args.port,
        model=args.model,
        served_name=args.model,
        gpu_util=args.gpu_util,
        tp=1,
        log="vllm_server.log",
        max_model_len=40000,
        allowed_local_media_path="/data/reid/reid_master",
        prompt=args.prompt,
        max_tokens=args.max_tokens,
        auto_start=not args.no_auto_start,
        show_log=not args.hide_log
    )
    
    try:
        # 디렉토리 처리
        directory_path = Path(args.directory)
        
        if not directory_path.exists():
            print(f"오류: 디렉토리가 존재하지 않습니다: {args.directory}")
            sys.exit(1)
        
        # 디렉토리 내에 직접 이미지가 있는지 확인
        image_files = get_image_files(str(directory_path))
        
        if image_files:
            # 현재 디렉토리에 이미지가 있으면 단일 디렉토리 처리
            print(f"\n단일 디렉토리 처리 모드")
            label = process_directory(
                str(directory_path),
                ocr_client,
                args.label_filename
            )
            if label:
                print(f"\n처리 완료! 라벨: {label}")
            else:
                print("\n처리 실패")
                sys.exit(1)
        else:
            # 하위 디렉토리 재귀 처리
            print(f"\n재귀 디렉토리 처리 모드")
            results = process_recursive(
                str(directory_path),
                ocr_client,
                args.label_filename
            )
            
            print(f"\n{'='*80}")
            print(f"전체 처리 완료!")
            print(f"처리된 디렉토리 수: {len(results)}")
            print(f"{'='*80}")
            
            for dir_path, label in results.items():
                print(f"{dir_path}: {label}")
    
    except KeyboardInterrupt:
        print("\n\n사용자에 의해 중단되었습니다.")
    except Exception as e:
        print(f"\n오류 발생: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        # 서버 종료
        print("\n서버 종료 중...")
        ocr_client.stop()


if __name__ == "__main__":
    main()
