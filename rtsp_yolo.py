# yolo_rtsp_realtime.py
import time
import cv2
from ultralytics import YOLO

RTSP_URL = "rtsp://admin:cubox2024%21@172.16.150.130:554/onvif/media?profile=M1_Profile1"

# ---- 사용자 설정 ----
MODEL_PATH = "yolo11n.pt"   # 속도 우선. 더 정확히: "yolov8s.pt" / 최신은 "yolov11n.pt"도 가능
CONF_THRES = 0.25
IOU_THRES = 0.45
ALLOWED_CLASSES = [1, 2, 3, 5, 7]  # car, motorcycle, bus, truck
SHOW_WINDOW = False
SAVE_CLIPS = True             # True면 결과 동영상 저장
SAVE_FRAMES_ON_DET = True     # True면 탐지 있을 때 프레임 JPG 저장
OUTPUT_VIDEO = "out_yolo.mp4"  # SAVE_CLIPS=True일 때만 사용
RECONNECT_DELAY = 2.0          # 끊겼을 때 재연결 간격(초)
# ---------------------

def open_capture(url):
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)  # FFMPEG backend 권장
    # 버퍼 지연 최소화 시도
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap

def main():
    model = YOLO(MODEL_PATH)
    # 클래스 이름(참고)
    names = model.model.names if hasattr(model.model, "names") else model.names

    cap = open_capture(RTSP_URL)
    if not cap.isOpened():
        print("[ERROR] RTSP 열기 실패. URL/네트워크 확인 후 재시도합니다.")
    writer = None
    last_save_t = 0
    fps_t = time.time()
    fps = 0.0

    while True:
        if not cap.isOpened():
            time.sleep(RECONNECT_DELAY)
            cap.release()
            cap = open_capture(RTSP_URL)
            continue

        ok, frame = cap.read()
        if not ok or frame is None:
            # 끊김/일시적 오류 → 재시도
            time.sleep(0.05)
            continue

        h, w = frame.shape[:2]

        # 추론 (클래스 필터는 predict의 classes=로 바로 주거나, 후처리로 필터링 가능)
        # stream=False로 한 프레임씩 처리
        results = model.predict(
            frame,
            conf=CONF_THRES,
            iou=IOU_THRES,
            classes=ALLOWED_CLASSES,  # 여기서 필터
            verbose=False
        )

        # 결과 그리기
        det_exist = False
        if results and len(results):
            r = results[0]
            if r.boxes is not None and len(r.boxes) > 0:
                det_exist = True
                for box in r.boxes:
                    cls_id = int(box.cls)
                    if cls_id not in ALLOWED_CLASSES:
                        continue
                    xyxy = box.xyxy[0].tolist()  # [x1,y1,x2,y2]
                    conf = float(box.conf)
                    x1,y1,x2,y2 = map(int, xyxy)
                    label = f"{names.get(cls_id, cls_id)} {conf:.2f}"

                    # 박스 & 라벨
                    cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
                    (tw,th), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    cv2.rectangle(frame, (x1, y1- th - 10), (x1+tw+6, y1), (0,255,0), -1)
                    cv2.putText(frame, label, (x1+3, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2, cv2.LINE_AA)

        # FPS 표시
        now = time.time()
        dt = now - fps_t
        if dt >= 0.5:
            fps = 1.0 / max(1e-6, dt)
            fps_t = now
        cv2.putText(frame, f"FPS ~ {fps:.1f}", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50,200,255), 2, cv2.LINE_AA)

        # 필요 시 저장
        if SAVE_FRAMES_ON_DET and det_exist and (now - last_save_t > 0.5):
            ts = time.strftime("%Y%m%d_%H%M%S")
            cv2.imwrite(f"det_{ts}.jpg", frame)
            last_save_t = now

        if SAVE_CLIPS:
            if writer is None:
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, 20.0, (w, h))
            writer.write(frame)

        if SHOW_WINDOW:
            cv2.imshow("YOLO RTSP (car/motorcycle/bus/truck only)", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # 정리
    if writer is not None:
        writer.release()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
