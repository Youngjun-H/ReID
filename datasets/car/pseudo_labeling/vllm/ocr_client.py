"""
OCR 클라이언트 모듈
VLLM 서버를 사용하여 이미지에서 번호판 문자를 추출하는 클라이언트
"""
import importlib.util
import sys
import os
import time
from pathlib import Path
from openai import OpenAI
from collections import Counter
from typing import List, Optional


class VLLMOCRClient:
    """VLLM 서버를 사용한 OCR 클라이언트"""
    
    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 8000,
        model: str = "Qwen/Qwen3-VL-4B-Instruct",
        served_name: str = "Qwen/Qwen3-VL-4B-Instruct",
        gpu_util: float = 0.8,
        tp: int = 1,
        log: str = "vllm_server.log",
        max_model_len: int = 40000,
        allowed_local_media_path: str = "/data/reid/reid_master",
        prompt: str = "차량 번호판의 문자를 추출해주세요.",
        max_tokens: int = 300,
        auto_start: bool = True,
        show_log: bool = True
    ):
        """
        Args:
            host: 서버 호스트
            port: 서버 포트
            model: 모델 이름
            served_name: 서빙 모델 이름
            gpu_util: GPU 사용률
            tp: 텐서 병렬 크기
            log: 로그 파일 경로
            max_model_len: 최대 모델 길이
            allowed_local_media_path: 허용된 로컬 미디어 경로
            prompt: OCR 프롬프트
            max_tokens: 최대 토큰 수
            auto_start: 자동으로 서버 시작 여부
            show_log: 로그 표시 여부
        """
        # 프로젝트 루트 찾기
        project_root = Path(__file__).parent.parent.parent.parent.parent
        cache_dir = project_root / "cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Triton 캐시 디렉토리 설정 (환경 변수가 없으면 프로젝트 루트의 cache/triton 사용)
        if "TRITON_CACHE_DIR" not in os.environ:
            triton_cache_dir = cache_dir / "triton"
            triton_cache_dir.mkdir(parents=True, exist_ok=True)
            os.environ["TRITON_CACHE_DIR"] = str(triton_cache_dir)
            print(f"Triton 캐시 디렉토리 자동 설정: {triton_cache_dir}")
        
        # 임시 파일 디렉토리 설정 (HuggingFace 다운로드 임시 파일용)
        if "TMPDIR" not in os.environ:
            tmp_dir = cache_dir / "tmp"
            tmp_dir.mkdir(parents=True, exist_ok=True)
            os.environ["TMPDIR"] = str(tmp_dir)
            os.environ["TEMP"] = str(tmp_dir)
            os.environ["TMP"] = str(tmp_dir)
            print(f"임시 파일 디렉토리 자동 설정: {tmp_dir}")
        
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
        # VLLMServer 모듈 로드
        server_path = Path(__file__).parent / "server.py"
        spec = importlib.util.spec_from_file_location("server", str(server_path))
        server = importlib.util.module_from_spec(spec)
        sys.modules["server"] = server
        spec.loader.exec_module(server)
        VLLMServer = server.VLLMServer
        
        # 서버 초기화
        self.server = VLLMServer(
            host=host,
            port=port,
            model=model,
            served_name=served_name,
            gpu_util=gpu_util,
            tp=tp,
            log=log,
            max_model_len=max_model_len,
            allowed_local_media_path=allowed_local_media_path,
        )
        
        self.host = host
        self.port = port
        self.served_name = served_name
        self.prompt = prompt
        self.max_tokens = max_tokens
        self._client = None
        
        if auto_start:
            self.start(show_log=show_log)
            self.wait_for_ready()
    
    def start(self, show_log: bool = True):
        """서버 시작"""
        self.server.start(show_log=show_log)
    
    def wait_for_ready(self, max_retries: int = 120, check_interval: float = 2.0):
        """
        서버가 준비될 때까지 대기
        
        Args:
            max_retries: 최대 재시도 횟수 (기본값: 120, 약 4분)
            check_interval: 각 체크 간격 (초, 기본값: 2.0)
        """
        print(f"서버 준비 대기 중... (최대 {max_retries * check_interval:.0f}초)")
        
        for attempt in range(max_retries):
            try:
                result = self.server.health_check()
                if "READY" in result:
                    print(f"서버 준비 완료! (시도 {attempt + 1}/{max_retries})")
                    return
                else:
                    if attempt % 10 == 0:  # 10번마다 진행 상황 출력
                        print(f"서버 준비 대기 중... (시도 {attempt + 1}/{max_retries})")
                    time.sleep(check_interval)
            except Exception as e:
                if attempt % 10 == 0:
                    print(f"헬스 체크 오류 (시도 {attempt + 1}/{max_retries}): {e}")
                time.sleep(check_interval)
        
        # 최종 체크
        result = self.server.health_check()
        if "READY" not in result:
            raise RuntimeError(f"서버 준비 실패 (최대 재시도 횟수 초과): {result}")
        print("서버 준비 완료")
    
    def stop(self):
        """서버 종료"""
        self.server.stop()
    
    @property
    def client(self):
        """OpenAI 클라이언트 인스턴스 반환 (lazy initialization)"""
        if self._client is None:
            self._client = OpenAI(
                base_url=f"http://{self.host}:{self.port}/v1",
                api_key="not-needed"
            )
        return self._client
    
    def extract_text(self, image_path: str) -> str:
        """
        이미지에서 텍스트 추출
        
        Args:
            image_path: 이미지 파일 경로
            
        Returns:
            추출된 텍스트
        """
        file_url = f"file://{os.path.abspath(image_path)}"
        
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": file_url
                        }
                    },
                    {
                        "type": "text",
                        "text": self.prompt
                    }
                ]
            }
        ]
        
        start = time.time()
        response = self.client.chat.completions.create(
            model=self.served_name,
            messages=messages,
            max_tokens=self.max_tokens
        )
        elapsed = time.time() - start
        
        text = response.choices[0].message.content.strip()
        print(f"[{os.path.basename(image_path)}] 추론 시간: {elapsed:.2f}s, 결과: {text}")
        
        return text
    
    def process_images(self, image_paths: List[str]) -> List[str]:
        """
        여러 이미지에 대해 텍스트 추출
        
        Args:
            image_paths: 이미지 파일 경로 리스트
            
        Returns:
            추출된 텍스트 리스트
        """
        results = []
        for image_path in image_paths:
            try:
                text = self.extract_text(image_path)
                results.append(text)
            except Exception as e:
                print(f"이미지 처리 실패 [{image_path}]: {e}")
                results.append("")
        return results
    
    def get_most_common_label(self, texts: List[str]) -> str:
        """
        텍스트 리스트에서 가장 많이 나온 텍스트 반환
        
        Args:
            texts: 텍스트 리스트
            
        Returns:
            가장 많이 나온 텍스트
        """
        # 빈 문자열 제거
        non_empty_texts = [t for t in texts if t.strip()]
        if not non_empty_texts:
            return ""
        
        counter = Counter(non_empty_texts)
        most_common = counter.most_common(1)[0]
        return most_common[0]
