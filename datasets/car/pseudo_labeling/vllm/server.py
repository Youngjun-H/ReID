import os, sys, time, subprocess, requests
import signal
class VLLMServer:
    def __init__(self, host: str, port: int, model: str, served_name: str, gpu_util: float, tp: int, log: str, **kwargs):
        self.host = host
        self.port = port
        self.model = model
        self.served_name = served_name
        self.gpu_util = gpu_util
        self.tp = tp
        self.log = log
        self.extra_args = kwargs
    def start(self, show_log: bool = False):
        # 환경 변수에서 캐시 디렉토리 가져오기
        triton_cache = os.environ.get("TRITON_CACHE_DIR", None)
        vllm_cache = os.environ.get("VLLM_CACHE_ROOT", None)
        vllm_usage_stats = os.environ.get("VLLM_USAGE_STATS_PATH", None)
        
        # 서브프로세스에 전달할 환경 변수 준비 (현재 환경 변수 복사)
        env = os.environ.copy()
        
        # VLLM 관련 환경 변수가 설정되어 있으면 확인
        if vllm_cache:
            print(f"VLLM 캐시 디렉토리: {vllm_cache}")
        if vllm_usage_stats:
            print(f"VLLM Usage Stats 경로: {vllm_usage_stats}")
        if triton_cache:
            print(f"Triton 캐시 디렉토리: {triton_cache}")
        
        cmd = [
            sys.executable, "-m", "vllm.entrypoints.openai.api_server",
            "--host", self.host, "--port", str(self.port),
            "--model", self.model,
            "--served-model-name", self.served_name,
            "--gpu-memory-utilization", str(self.gpu_util),
            "--tensor-parallel-size", str(self.tp),
        ]
        for key, value in self.extra_args.items():
            flag = f"--{key.replace('_', '-')}" # example: enable_auto_tool_choice -> --enable-auto-tool-choice
            if value is True:
                cmd.append(flag)
            elif value is not False and value is not None:
                cmd.append(flag)
                cmd.append(str(value))
        print(f"서버 시작 명령: {' '.join(cmd)}")
        if show_log:
            # 로그를 화면에도 출력하고 파일에도 저장 (tee 방식)
            self.proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
            self.pid = self.proc.pid
            print(f"서버 시작 완료: {self.pid}")
            print("=" * 80)
            # 백그라운드 스레드로 로그 출력 및 파일 저장
            import threading
            def log_output():
                with open(self.log, "w") as f:
                    for line in self.proc.stdout:
                        print(line, end='')
                        f.write(line)
                        f.flush()
            self.log_thread = threading.Thread(target=log_output, daemon=True)
            self.log_thread.start()
        else:
            # 기존 방식: 로그를 파일로만 저장
            # 로그 파일 디렉토리 생성
            log_dir = os.path.dirname(self.log)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir, exist_ok=True)
            
            try:
                self.proc = subprocess.Popen(
                    cmd, 
                    env=env,
                    stdout=open(self.log, "w", encoding="utf-8"), 
                    stderr=subprocess.STDOUT, 
                    text=True
                )
                self.pid = self.proc.pid
                print(f"서버 시작 완료 (PID: {self.pid})")
                print(f"로그 확인: tail -f {self.log}")
                print(f"서버가 준비될 때까지 대기 중...")
            except Exception as e:
                print(f"서버 시작 실패: {e}")
                raise
    def stop(self):
        try:
            os.kill(self.pid, signal.SIGTERM)
            print(f"서버 종료 완료: {self.pid}")
            return True
        except Exception:
            print(f"서버 종료 실패: {self.pid}")
            print(f"최근 로그 50줄: {open(self.log).readlines()[-50:]}")
            return False
    def health_check(self):
        """
        서버 헬스 체크
        
        Returns:
            "READY: {response_json}" 형식의 문자열 또는 실패 메시지
        """
        url = f"http://127.0.0.1:{self.port}/v1/models"
        try:
            r = requests.get(url, timeout=5)
            if r.status_code == 200:
                resp = f"READY: {r.json()}"
                return resp
            else:
                return f"서버 응답 오류: HTTP {r.status_code}"
        except requests.exceptions.ConnectionError:
            return "서버 연결 실패: 서버가 아직 시작 중이거나 포트가 열리지 않았습니다."
        except requests.exceptions.Timeout:
            return "서버 응답 시간 초과: 서버가 과부하 상태일 수 있습니다."
        except Exception as e:
            return f"헬스 체크 오류: {str(e)}"
    # def restart(self):
    #     self.stop()
    #     self.start()
    #     return self.health_check()