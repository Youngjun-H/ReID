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
            self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
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
            self.proc = subprocess.Popen(cmd, stdout=open(self.log, "w"), stderr=subprocess.STDOUT, text=True)
            self.pid = self.proc.pid
            print(f"서버 시작 완료: {self.pid}")
            print(f"로그 확인: tail -f {self.log}")
            print(f"준비되면 server.health_check()를 호출하세요.")
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
        url = f"http://127.0.0.1:{self.port}/v1/models"
        for _ in range(50):
            try:
                r = requests.get(url, timeout=2)
                if r.status_code == 200:
                    resp = f"READY: {r.json()}"
                    return resp
            except Exception as e:
                print(f"헬스 체크 실패: {e}")
                time.sleep(1)
        return f"서버 준비 실패. 최근 로그 50줄: {open(self.log).readlines()[-50:]}"
    # def restart(self):
    #     self.stop()
    #     self.start()
    #     return self.health_check()