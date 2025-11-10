import cv2
from datetime import datetime
import time
url = "rtsp://admin:cubox2024%21@172.16.150.130:554/onvif/media?profile=M1_Profile1"
cap = cv2.VideoCapture(url)

while True:
    ret, frame = cap.read()
    if ret:
        path = f"capture_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        cv2.imwrite(path, frame)
        print("Saved:", path)
    time.sleep(1)
