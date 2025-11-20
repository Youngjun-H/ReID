"""
차량 데이터셋 구축 RTSP 실시간 파이프라인

기존 pipeline.py의 로직을 RTSP 실시간 스트림에 적용

단계:
1. Tracking: RTSP 스트림에서 차량 추적 및 crop (실시간)
2. Filtering: 번호판 detection으로 필터링 및 번호판 crop (주기적)
3. Pseudo Labeling: 번호판 이미지에 OCR로 라벨링 (주기적)
"""
import argparse
import logging
import os
import threading
import time
from collections import defaultdict, deque
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from ultralytics import YOLO

# 각 단계 모듈 import
try:
    from .filtering.filtering_by_lp import process_runs_directory as run_filtering
    from .pseudo_labeling.vllm.directory_processor import process_recursive
    from .pseudo_labeling.vllm.ocr_client import VLLMOCRClient
    from .tracking.pipeline_roi import (
        clamp,
        iou_xyxy,
        is_bbox_in_roi,
        is_same_object,
        parse_roi_file,
    )
except ImportError:
    # 직접 실행 시 상대 경로로 import
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from datasets.car.filtering.filtering_by_lp import process_runs_directory as run_filtering
    from datasets.car.pseudo_labeling.vllm.directory_processor import process_recursive
    from datasets.car.pseudo_labeling.vllm.ocr_client import VLLMOCRClient
    from datasets.car.tracking.pipeline_roi import (
        clamp,
        iou_xyxy,
        is_bbox_in_roi,
        is_same_object,
        parse_roi_file,
    )


def ensure_dir(path: str) -> None:
    """디렉토리가 없으면 생성"""
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def open_capture(url: str):
    """RTSP 스트림 연결"""
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # 버퍼 지연 최소화
    return cap


# 참조 코드 방식: 각 스레드마다 별도의 모델 인스턴스를 생성
# 락 없이 각 스레드가 독립적으로 모델을 로드하여 스레드 안전성 보장
# PyTorch는 멀티스레드 추론을 지원하므로 동시 실행 가능


class RTSPPipeline:
    """RTSP 실시간 파이프라인 클래스"""

    def __init__(
        self,
        rtsp_url: str,
        output_base_dir: str,
        # Tracking 파라미터
        tracking_model_path: str = "checkpoints/detection/yolo11n.pt",
        tracking_conf: float = 0.7,
        tracking_iou: float = 0.5,
        max_frames_per_track: int = 200,
        roi_file: Optional[str] = None,
        roi_min_iou: float = 0.7,
        # Filtering 파라미터
        lp_detection_model_path: str = "checkpoints/detection/lp_detection.pt",
        # Pseudo Labeling 파라미터
        ocr_client: Optional[VLLMOCRClient] = None,  # 외부에서 공유 OCR 클라이언트 전달
        vllm_host: str = "0.0.0.0",
        vllm_port: int = 8000,
        vllm_model: str = "Qwen/Qwen3-VL-4B-Instruct",
        vllm_gpu_util: float = 0.8,  # GPU 메모리 일부를 다른 모델을 위해 남김
        vllm_prompt: str = "차량 번호판의 문자를 추출해주세요.",
        vllm_max_tokens: int = 300,
        label_filename: str = "label.txt",
        # 실시간 처리 파라미터
        filtering_interval: int = 60,  # Filtering 실행 간격 (초)
        labeling_interval: int = 120,  # Labeling 실행 간격 (초)
        reconnect_delay: float = 2.0,  # 재연결 지연 (초)
        show_window: bool = False,
        save_clips: bool = False,
        output_video: str = "rtsp_output.mp4",
        manage_vllm_server: bool = True,  # VLLM 서버 종료를 이 인스턴스에서 관리할지 여부
    ):
        """
        Args:
            rtsp_url: RTSP 스트림 URL
            output_base_dir: 모든 출력 결과의 기본 디렉토리
            filtering_interval: Filtering 실행 간격 (초)
            labeling_interval: Labeling 실행 간격 (초)
            reconnect_delay: 재연결 지연 시간 (초)
            show_window: 윈도우 표시 여부
            save_clips: 동영상 저장 여부
            output_video: 저장할 동영상 파일 경로
        """
        self.rtsp_url = rtsp_url
        self.output_base_dir = output_base_dir
        self.tracking_model_path = tracking_model_path
        self.tracking_conf = tracking_conf
        self.tracking_iou = tracking_iou
        self.max_frames_per_track = max_frames_per_track
        self.roi_file = roi_file
        self.roi_min_iou = roi_min_iou
        self.lp_detection_model_path = lp_detection_model_path
        self.vllm_host = vllm_host
        self.vllm_port = vllm_port
        self.vllm_model = vllm_model
        self.vllm_gpu_util = vllm_gpu_util
        self.vllm_prompt = vllm_prompt
        self.vllm_max_tokens = vllm_max_tokens
        self.label_filename = label_filename
        self.filtering_interval = filtering_interval
        self.labeling_interval = labeling_interval
        self.reconnect_delay = reconnect_delay
        self.show_window = show_window
        self.save_clips = save_clips
        self.output_video = output_video

        # 출력 디렉토리 구조
        ensure_dir(output_base_dir)
        self.tracking_dir = os.path.join(output_base_dir, "01_tracking")
        self.filtering_dir = os.path.join(output_base_dir, "02_filtered_vehicles")
        self.lp_crop_dir = os.path.join(output_base_dir, "03_license_plates")

        ensure_dir(self.tracking_dir)
        ensure_dir(self.filtering_dir)
        ensure_dir(self.lp_crop_dir)

        # RTSP 스트림 이름으로 서브 디렉토리 생성
        stream_name = self._get_stream_name(rtsp_url)
        self.stream_output_dir = os.path.join(self.tracking_dir, stream_name)
        ensure_dir(self.stream_output_dir)

        # Tracking 상태
        self.vehicle_classes = [2, 3, 5, 7]  # car, motorcycle, bus, truck
        self.model = None
        self.saved_counts = defaultdict(int)
        self.last_boxes = {}
        self.recent_lost = deque(maxlen=200)
        self.id_remap = {}
        self.last_seen = {}
        self.frame_index = 0
        self.roi_filtered_count = 0
        self.rois = []

        # 실행 제어
        self.running = False
        self.stop_event = threading.Event()
        self.last_filtering_time = 0
        self.last_labeling_time = 0

        # 동영상 저장
        self.writer = None

        # 로깅 설정 (먼저 설정해야 이후 로깅 가능)
        self._setup_logging()

        # ROI 로드 (로깅 설정 후)
        if roi_file:
            self.rois = parse_roi_file(roi_file)
            if self.rois:
                self.logger.info(f"ROI 로드 완료: 총 {len(self.rois)}개의 ROI")
                self.logger.info(f"ROI 필터링 최소 IoU 임계값: {roi_min_iou}")

        # VLLM OCR 클라이언트 초기화 (공유 클라이언트가 있으면 사용, 없으면 새로 생성)
        self.manage_vllm_server = manage_vllm_server
        if ocr_client is not None:
            self.ocr_client = ocr_client
            self.logger.info("공유 VLLM OCR 클라이언트 사용")
        else:
            self.logger.info("VLLM OCR 클라이언트 초기화 중...")
            self.logger.info(f"VLLM GPU 사용률: {self.vllm_gpu_util} (다른 모델을 위해 메모리 일부 예약)")
            self.ocr_client = VLLMOCRClient(
                host=self.vllm_host,
                port=self.vllm_port,
                model=self.vllm_model,
                served_name=self.vllm_model,
                gpu_util=self.vllm_gpu_util,
                tp=1,
                log=os.path.join(output_base_dir, "vllm_server.log"),
                max_model_len=40000,
                allowed_local_media_path=str(Path(self.lp_crop_dir).resolve().parent.parent.parent.parent),
                prompt=self.vllm_prompt,
                max_tokens=self.vllm_max_tokens,
                auto_start=True,
                show_log=True,
            )
            self.logger.info("VLLM OCR 클라이언트 초기화 완료 (서버 지속 실행)")

    def _setup_logging(self):
        """로깅 설정"""
        log_dir = os.path.join(self.output_base_dir, "logs")
        ensure_dir(log_dir)

        # 로그 파일명: 날짜_시간_pipeline_rtsp.log
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"{timestamp}_pipeline_rtsp.log")

        # 로거 설정
        self.logger = logging.getLogger("RTSPPipeline")
        self.logger.setLevel(logging.INFO)

        # 기존 핸들러 제거 (중복 방지)
        if self.logger.handlers:
            self.logger.handlers.clear()

        # 파일 핸들러
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)

        # 콘솔 핸들러
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)

        self.logger.info(f"로깅 초기화 완료: {log_file}")

    def _get_stream_name(self, url: str) -> str:
        """RTSP URL에서 스트림 이름 추출"""
        # URL에서 IP나 호스트명 추출
        import re
        match = re.search(r'@([^:/]+)', url)
        if match:
            return f"rtsp_{match.group(1)}"
        return "rtsp_stream"

    def _initialize_model(self):
        """YOLO 모델 초기화 (각 스레드에서 독립적으로 모델 인스턴스 생성)"""
        if self.model is None:
            # 각 스레드마다 별도의 모델 인스턴스를 생성하여 스레드 안전성 보장
            # 참조 코드와 동일한 방식: 각 스레드가 자신만의 모델 인스턴스를 가짐
            self.logger.info(f"YOLO 모델 로드 중: {self.tracking_model_path} (스트림: {self.rtsp_url})")
            self.model = YOLO(self.tracking_model_path)
            self.logger.info(f"YOLO 모델 로드 완료 (스트림: {self.rtsp_url})")

    def _process_frame(self, frame):
        """단일 프레임 처리 (Tracking)"""
        if frame is None:
            return

        h, w = frame.shape[:2]
        diag = (h * h + w * w) ** 0.5

        # YOLO tracking (락 제거: PyTorch는 멀티스레드 추론을 지원하므로 동시 실행 가능)
        # GPU 한 개로도 여러 스트림을 동시에 처리할 수 있음
        results = self.model.track(
            frame,
            conf=self.tracking_conf,
            iou=self.tracking_iou,
            classes=self.vehicle_classes,
            tracker="bytetrack.yaml",
            persist=True,
            verbose=False,
            device=0,
        )

        if not results or len(results) == 0:
            self.frame_index += 1
            return

        result = results[0]

        if result.boxes is None or result.boxes.id is None:
            self.frame_index += 1
            return

        ids = result.boxes.id.int().tolist()
        xyxys = result.boxes.xyxy.int().tolist()
        clss = result.boxes.cls.int().tolist()

        src_name = self._get_stream_name(self.rtsp_url)

        # ID 병합
        canonical_ids = []
        for raw_id, bbox in zip(ids, xyxys):
            cand_id = self.id_remap.get(raw_id, raw_id)
            best_id = cand_id
            best_iou = 0.0

            # 기존 트랙과 비교
            for prev_id, prev_box in self.last_boxes.items():
                iou_val = iou_xyxy(bbox, prev_box)
                if iou_val > best_iou:
                    best_iou = iou_val
                    best_id = prev_id

            # 최근 잃은 트랙도 비교
            if best_iou < 0.6:
                for prev_id, prev_box, seen in reversed(self.recent_lost):
                    if self.frame_index - seen > 60:
                        break
                    if is_same_object(prev_box, bbox, diag):
                        best_iou = 1.0
                        best_id = prev_id
                        break

            if best_iou >= 0.6:
                self.id_remap[raw_id] = best_id
                canonical_ids.append(best_id)
            else:
                canonical_ids.append(cand_id)

        # 트랙 관리
        alive = set(canonical_ids)
        for tid in list(self.last_boxes.keys()):
            if tid not in alive:
                self.recent_lost.append((
                    tid,
                    self.last_boxes[tid],
                    self.last_seen.get(tid, self.frame_index - 1)
                ))
                self.last_boxes.pop(tid, None)

        # 이미지 저장
        for raw_id, canon_id, bbox, cls_idx in zip(ids, canonical_ids, xyxys, clss):
            track_id = canon_id
            x1, y1, x2, y2 = bbox
            x1, y1, x2, y2 = (
                clamp(x1, 0, w - 1),
                clamp(y1, 0, h - 1),
                clamp(x2, 0, w - 1),
                clamp(y2, 0, h - 1)
            )

            if x2 <= x1 or y2 <= y1:
                continue

            # ROI 필터링
            if self.rois and not is_bbox_in_roi([x1, y1, x2, y2], self.rois, min_iou=self.roi_min_iou):
                self.roi_filtered_count += 1
                continue

            # 프레임당 저장 제한
            if self.saved_counts[track_id] >= self.max_frames_per_track:
                continue

            # 저장
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            track_dir = os.path.join(self.stream_output_dir, f"track_{int(track_id)}")
            ensure_dir(track_dir)
            idx = self.saved_counts[track_id]
            out_name = f"{src_name}_f{self.frame_index:06d}_{idx:03d}.jpg"
            out_path = os.path.join(track_dir, out_name)
            cv2.imwrite(out_path, crop)

            # 로깅 (주기적으로만, 너무 많이 로그가 찍히지 않도록)
            if idx == 0 or idx % 50 == 0:
                self.logger.debug(f"트랙 {track_id} 이미지 저장: {out_name} (총 {idx+1}개)")

            self.saved_counts[track_id] += 1
            self.last_boxes[track_id] = [x1, y1, x2, y2]
            self.last_seen[track_id] = self.frame_index

        self.frame_index += 1

    def _run_filtering(self):
        """Step 2: Filtering 실행"""
        try:
            self.logger.info("="*80)
            self.logger.info("Step 2: 번호판 Detection 및 필터링 (주기적 실행)")
            self.logger.info("="*80)

            run_filtering(
                runs_dir=self.tracking_dir,
                output_dir=self.filtering_dir,
                lp_detection_model_path=self.lp_detection_model_path,
                output_lp_dir=self.lp_crop_dir,
            )

            self.logger.info("Step 2 완료!")
        except Exception as e:
            self.logger.error(f"Filtering 오류: {e}", exc_info=True)

    def _run_labeling(self):
        """Step 3: Pseudo Labeling 실행 (VLLM 서버는 이미 실행 중)"""
        try:
            self.logger.info("="*80)
            self.logger.info("Step 3: 번호판 OCR 라벨링 (주기적 실행)")
            self.logger.info("="*80)

            # OCR 클라이언트는 이미 초기화되어 있음 (서버 재시작하지 않음)
            results = process_recursive(
                root_directory=self.lp_crop_dir,
                ocr_client=self.ocr_client,
                label_filename=self.label_filename,
            )

            self.logger.info(f"Step 3 완료! 처리된 디렉토리 수: {len(results)}")

        except Exception as e:
            self.logger.error(f"Labeling 오류: {e}", exc_info=True)

    def _background_processor(self):
        """백그라운드에서 주기적으로 Filtering과 Labeling 실행"""
        while not self.stop_event.is_set():
            current_time = time.time()

            # Filtering 주기적 실행
            if current_time - self.last_filtering_time >= self.filtering_interval:
                self._run_filtering()
                self.last_filtering_time = current_time

            # Labeling 주기적 실행
            if current_time - self.last_labeling_time >= self.labeling_interval:
                self._run_labeling()
                self.last_labeling_time = current_time

            # 1초마다 체크
            self.stop_event.wait(1.0)

    def run(self):
        """RTSP 실시간 파이프라인 실행"""
        self.logger.info("="*80)
        self.logger.info("RTSP 실시간 차량 데이터셋 구축 파이프라인 시작")
        self.logger.info("="*80)
        self.logger.info(f"RTSP URL: {self.rtsp_url}")
        self.logger.info(f"출력 기본 디렉토리: {self.output_base_dir}")
        self.logger.info(f"Filtering 간격: {self.filtering_interval}초")
        self.logger.info(f"Labeling 간격: {self.labeling_interval}초")
        if self.rois:
            self.logger.info(f"ROI 필터링: {len(self.rois)}개 ROI, 최소 IoU: {self.roi_min_iou}")

        # 모델 초기화
        self._initialize_model()

        # 백그라운드 프로세서 시작
        bg_thread = threading.Thread(target=self._background_processor, daemon=True)
        bg_thread.start()

        self.running = True
        cap = None
        fps_t = time.time()
        fps = 0.0

        try:
            while self.running:
                # RTSP 연결
                if cap is None or not cap.isOpened():
                    self.logger.info(f"RTSP 연결 시도: {self.rtsp_url}")
                    cap = open_capture(self.rtsp_url)
                    if not cap.isOpened():
                        self.logger.warning(f"연결 실패. {self.reconnect_delay}초 후 재시도...")
                        time.sleep(self.reconnect_delay)
                        continue
                    self.logger.info("RTSP 연결 성공")

                    # 동영상 저장 초기화
                    if self.save_clips:
                        ret, frame = cap.read()
                        if ret and frame is not None:
                            h, w = frame.shape[:2]
                            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                            self.writer = cv2.VideoWriter(
                                self.output_video,
                                fourcc,
                                20.0,
                                (w, h)
                            )

                # 프레임 읽기
                ret, frame = cap.read()
                if not ret or frame is None:
                    time.sleep(0.05)
                    continue

                # 프레임 처리
                self._process_frame(frame)

                # FPS 계산
                now = time.time()
                dt = now - fps_t
                if dt >= 0.5:
                    fps = 1.0 / max(1e-6, dt)
                    fps_t = now

                # 동영상 저장
                if self.save_clips and self.writer is not None:
                    self.writer.write(frame)

                # 윈도우 표시
                if self.show_window:
                    display_frame = frame.copy()
                    cv2.putText(
                        display_frame,
                        f"FPS ~ {fps:.1f}",
                        (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (50, 200, 255),
                        2,
                        cv2.LINE_AA
                    )
                    cv2.imshow("RTSP Pipeline", display_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        self.logger.info("사용자에 의해 중단되었습니다.")
                        break

        except KeyboardInterrupt:
            self.logger.info("사용자에 의해 중단되었습니다.")
        except Exception as e:
            self.logger.error(f"오류 발생: {e}", exc_info=True)
        finally:
            self.running = False
            self.stop_event.set()

            # VLLM 서버 종료 (이 인스턴스가 관리하는 경우에만)
            if self.manage_vllm_server:
                self.logger.info("VLLM 서버 종료 중...")
                try:
                    self.ocr_client.stop()
                    self.logger.info("VLLM 서버 종료 완료")
                except Exception as e:
                    self.logger.error(f"VLLM 서버 종료 오류: {e}", exc_info=True)

            if cap is not None:
                cap.release()
            if self.writer is not None:
                self.writer.release()
            if self.show_window:
                cv2.destroyAllWindows()

            self.logger.info("="*80)
            self.logger.info("파이프라인 종료")
            self.logger.info("="*80)
            if self.rois:
                self.logger.info(f"ROI 필터링으로 제외된 객체 수: {self.roi_filtered_count}")
            self.logger.info(f"처리된 총 프레임 수: {self.frame_index}")
            self.logger.info(f"추적 결과: {self.tracking_dir}")
            self.logger.info(f"필터링된 차량 이미지: {self.filtering_dir}")
            self.logger.info(f"번호판 crop 이미지: {self.lp_crop_dir}")


class MultiRTSPPipeline:
    """여러 RTSP 스트림을 동시에 처리하는 파이프라인 클래스"""

    def __init__(
        self,
        rtsp_urls: list,
        output_base_dir: str,
        # Tracking 파라미터
        tracking_model_path: str = "checkpoints/detection/yolo11n.pt",
        tracking_conf: float = 0.7,
        tracking_iou: float = 0.5,
        max_frames_per_track: int = 200,
        roi_file: Optional[str] = None,
        roi_min_iou: float = 0.7,
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
        # 실시간 처리 파라미터
        filtering_interval: int = 60,
        labeling_interval: int = 120,
        reconnect_delay: float = 2.0,
        show_window: bool = False,
        save_clips: bool = False,
        output_video_prefix: str = "rtsp_output",
    ):
        """
        Args:
            rtsp_urls: RTSP 스트림 URL 리스트 (최소 2개)
            output_base_dir: 모든 출력 결과의 기본 디렉토리
            output_video_prefix: 동영상 파일명 접두사 (각 스트림별로 번호가 붙음)
        """
        if len(rtsp_urls) < 2:
            raise ValueError("최소 2개의 RTSP URL이 필요합니다.")

        self.rtsp_urls = rtsp_urls
        self.output_base_dir = output_base_dir
        self.pipelines = []
        self.threads = []
        self.stop_event = threading.Event()

        # 출력 디렉토리 생성
        ensure_dir(output_base_dir)

        # 로깅 설정
        self._setup_logging()

        # VLLM 서버 한 번만 시작 (공유)
        self.logger.info("="*80)
        self.logger.info("VLLM OCR 서버 초기화 (모든 스트림에서 공유)")
        self.logger.info("="*80)
        self.logger.info(f"VLLM GPU 사용률: {vllm_gpu_util}")

        # 공유 OCR 클라이언트 생성
        lp_crop_dir = os.path.join(output_base_dir, "03_license_plates")
        self.shared_ocr_client = VLLMOCRClient(
            host=vllm_host,
            port=vllm_port,
            model=vllm_model,
            served_name=vllm_model,
            gpu_util=vllm_gpu_util,
            tp=1,
            log=os.path.join(output_base_dir, "vllm_server.log"),
            max_model_len=40000,
            allowed_local_media_path=str(Path(lp_crop_dir).resolve().parent.parent.parent.parent),
            prompt=vllm_prompt,
            max_tokens=vllm_max_tokens,
            auto_start=True,
            show_log=True,
        )
        self.logger.info("VLLM OCR 서버 초기화 완료")

        # 각 RTSP 스트림별로 파이프라인 생성
        # 모델 초기화를 순차적으로 수행하여 GPU 리소스 경합 방지
        self.logger.info("="*80)
        self.logger.info("각 스트림별 파이프라인 초기화 시작")
        self.logger.info("="*80)
        for idx, rtsp_url in enumerate(rtsp_urls):
            self.logger.info(f"스트림 {idx+1}/{len(rtsp_urls)} 초기화 중: {rtsp_url}")
            # 각 스트림별 출력 디렉토리
            stream_output_dir = os.path.join(output_base_dir, f"stream_{idx+1}")

            # 동영상 파일명
            output_video = f"{output_video_prefix}_{idx+1}.mp4" if save_clips else "rtsp_output.mp4"

            pipeline = RTSPPipeline(
                rtsp_url=rtsp_url,
                output_base_dir=stream_output_dir,
                tracking_model_path=tracking_model_path,
                tracking_conf=tracking_conf,
                tracking_iou=tracking_iou,
                max_frames_per_track=max_frames_per_track,
                roi_file=roi_file,
                roi_min_iou=roi_min_iou,
                lp_detection_model_path=lp_detection_model_path,
                ocr_client=self.shared_ocr_client,  # 공유 OCR 클라이언트 전달
                vllm_host=vllm_host,
                vllm_port=vllm_port,
                vllm_model=vllm_model,
                vllm_gpu_util=vllm_gpu_util,
                vllm_prompt=vllm_prompt,
                vllm_max_tokens=vllm_max_tokens,
                label_filename=label_filename,
                filtering_interval=filtering_interval,
                labeling_interval=labeling_interval,
                reconnect_delay=reconnect_delay,
                show_window=show_window,
                save_clips=save_clips,
                output_video=output_video,
                manage_vllm_server=False,  # VLLM 서버는 이 클래스에서 관리
            )
            self.pipelines.append(pipeline)
            self.logger.info(f"스트림 {idx+1} 초기화 완료")
        self.logger.info("="*80)
        self.logger.info("모든 스트림 파이프라인 초기화 완료")
        self.logger.info("="*80)

    def _setup_logging(self):
        """로깅 설정"""
        log_dir = os.path.join(self.output_base_dir, "logs")
        ensure_dir(log_dir)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"{timestamp}_multi_rtsp_pipeline.log")

        self.logger = logging.getLogger("MultiRTSPPipeline")
        self.logger.setLevel(logging.INFO)

        if self.logger.handlers:
            self.logger.handlers.clear()

        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)

        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)

        self.logger.info(f"로깅 초기화 완료: {log_file}")

    def _run_pipeline(self, pipeline: RTSPPipeline, stream_idx: int):
        """개별 파이프라인을 스레드에서 실행 (참조 코드 방식)"""
        try:
            # 각 스레드에서 독립적으로 실행
            # 모델 초기화는 pipeline.run() 내부에서 수행됨
            self.logger.info(f"스트림 {stream_idx+1} 시작: {pipeline.rtsp_url}")
            pipeline.run()
        except Exception as e:
            self.logger.error(f"스트림 {stream_idx+1} 오류: {e}", exc_info=True)

    def run(self):
        """모든 RTSP 스트림을 동시에 실행"""
        self.logger.info("="*80)
        self.logger.info("다중 RTSP 실시간 차량 데이터셋 구축 파이프라인 시작")
        self.logger.info("="*80)
        self.logger.info(f"총 {len(self.rtsp_urls)}개의 RTSP 스트림 처리")
        for idx, url in enumerate(self.rtsp_urls):
            self.logger.info(f"  스트림 {idx+1}: {url}")
        self.logger.info(f"출력 기본 디렉토리: {self.output_base_dir}")
        self.logger.info(f"Filtering 간격: {self.pipelines[0].filtering_interval}초")
        self.logger.info(f"Labeling 간격: {self.pipelines[0].labeling_interval}초")

        # 각 파이프라인을 별도 스레드에서 실행
        for idx, pipeline in enumerate(self.pipelines):
            thread = threading.Thread(
                target=self._run_pipeline,
                args=(pipeline, idx),
                daemon=False,
                name=f"RTSPStream-{idx+1}"
            )
            self.threads.append(thread)
            thread.start()
            self.logger.info(f"스트림 {idx+1} 스레드 시작됨")

        try:
            # 모든 스레드가 종료될 때까지 대기
            for idx, thread in enumerate(self.threads):
                thread.join()
                self.logger.info(f"스트림 {idx+1} 스레드 종료됨")
        except KeyboardInterrupt:
            self.logger.info("사용자에 의해 중단되었습니다.")
            self.stop()
        finally:
            # VLLM 서버 종료
            self.logger.info("VLLM 서버 종료 중...")
            try:
                self.shared_ocr_client.stop()
                self.logger.info("VLLM 서버 종료 완료")
            except Exception as e:
                self.logger.error(f"VLLM 서버 종료 오류: {e}", exc_info=True)

            self.logger.info("="*80)
            self.logger.info("다중 RTSP 파이프라인 종료")
            self.logger.info("="*80)

    def stop(self):
        """모든 파이프라인 중지"""
        self.logger.info("모든 파이프라인 중지 중...")
        self.stop_event.set()
        for pipeline in self.pipelines:
            pipeline.running = False
            pipeline.stop_event.set()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="RTSP 실시간 차량 데이터셋 구축 파이프라인 (다중 스트림 지원)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예제:
  # 2개의 RTSP 스트림 동시 처리
  python -m datasets.car.pipeline_rtsp \\
      --rtsp_urls rtsp://admin:password@192.168.1.100:554/stream1 \\
                  rtsp://admin:password@192.168.1.101:554/stream2 \\
      --output_dir /path/to/output

  # 단일 RTSP 스트림 처리 (기존 방식)
  python -m datasets.car.pipeline_rtsp \\
      --rtsp_url rtsp://admin:password@192.168.1.100:554/stream \\
      --output_dir /path/to/output

  # Filtering과 Labeling 간격 조정
  python -m datasets.car.pipeline_rtsp \\
      --rtsp_urls rtsp://... rtsp://... \\
      --output_dir /path/to/output \\
      --filtering_interval 30 \\
      --labeling_interval 60
        """
    )

    # RTSP URL 인자 (다중 또는 단일)
    parser.add_argument(
        "--rtsp_url",
        type=str,
        default=None,
        help="단일 RTSP 스트림 URL (단일 스트림 모드)"
    )
    parser.add_argument(
        "--rtsp_urls",
        type=str,
        nargs='+',
        default=None,
        help="여러 RTSP 스트림 URL 리스트 (다중 스트림 모드, 최소 2개)"
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
        default=None,
        help="ROI 파일 경로 (None이면 ROI 필터링 없음)"
    )
    parser.add_argument(
        "--roi_min_iou",
        type=float,
        default=0.2,
        help="ROI 필터링 최소 IoU 임계값 (기본값: 0.7)"
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
        help="GPU 사용률 (기본값: 0.6, 다른 모델을 위해 메모리 일부 예약)"
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

    # 실시간 처리 파라미터
    parser.add_argument(
        "--filtering_interval",
        type=int,
        default=60,
        help="Filtering 실행 간격 (초) (기본값: 60)"
    )
    parser.add_argument(
        "--labeling_interval",
        type=int,
        default=120,
        help="Labeling 실행 간격 (초) (기본값: 120)"
    )
    parser.add_argument(
        "--reconnect_delay",
        type=float,
        default=2.0,
        help="재연결 지연 시간 (초) (기본값: 2.0)"
    )
    parser.add_argument(
        "--show_window",
        action="store_true",
        help="윈도우 표시"
    )
    parser.add_argument(
        "--save_clips",
        action="store_true",
        help="동영상 저장"
    )
    parser.add_argument(
        "--output_video",
        type=str,
        default="rtsp_output.mp4",
        help="저장할 동영상 파일 경로 (기본값: rtsp_output.mp4, 다중 스트림 모드에서는 접두사로 사용)"
    )

    args = parser.parse_args()

    # RTSP URL 검증
    if args.rtsp_urls is not None:
        if len(args.rtsp_urls) < 2:
            parser.error("--rtsp_urls는 최소 2개의 URL이 필요합니다.")
        args.use_multi = True
    elif args.rtsp_url is not None:
        args.use_multi = False
        args.rtsp_urls = [args.rtsp_url]
    else:
        parser.error("--rtsp_url 또는 --rtsp_urls 중 하나는 필수입니다.")

    return args


if __name__ == "__main__":
    args = parse_args()

    if args.use_multi:
        # 다중 RTSP 스트림 모드
        pipeline = MultiRTSPPipeline(
            rtsp_urls=args.rtsp_urls,
            output_base_dir=args.output_dir,
            # Tracking 파라미터
            tracking_model_path=args.tracking_model,
            tracking_conf=args.tracking_conf,
            tracking_iou=args.tracking_iou,
            max_frames_per_track=args.max_frames_per_track,
            roi_file=args.roi_file,
            roi_min_iou=args.roi_min_iou,
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
            # 실시간 처리 파라미터
            filtering_interval=args.filtering_interval,
            labeling_interval=args.labeling_interval,
            reconnect_delay=args.reconnect_delay,
            show_window=args.show_window,
            save_clips=args.save_clips,
            output_video_prefix=args.output_video.replace('.mp4', ''),
        )
    else:
        # 단일 RTSP 스트림 모드 (기존 방식)
        pipeline = RTSPPipeline(
            rtsp_url=args.rtsp_url,
            output_base_dir=args.output_dir,
            # Tracking 파라미터
            tracking_model_path=args.tracking_model,
            tracking_conf=args.tracking_conf,
            tracking_iou=args.tracking_iou,
            max_frames_per_track=args.max_frames_per_track,
            roi_file=args.roi_file,
            roi_min_iou=args.roi_min_iou,
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
            # 실시간 처리 파라미터
            filtering_interval=args.filtering_interval,
            labeling_interval=args.labeling_interval,
            reconnect_delay=args.reconnect_delay,
            show_window=args.show_window,
            save_clips=args.save_clips,
            output_video=args.output_video,
        )

    pipeline.run()

