import cv2
from ultralytics import YOLO
import numpy as np


cap = cv2.VideoCapture("/Users/theoy/Movies/GoPro/GX010395.MP4")
# ret, frame = cap.read()
# cv2.imshow("Img", frame)
# cv2.waitKey(0)
model = YOLO("yolov8m.pt")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    results = model(frame)
    result = results[0]
    print(result)
    cv2.imshow("Img", frame)
    key = cv2.waitKey(0)
    if key == 27:
        break
cap.release()
cv2.destroyAllWindows()