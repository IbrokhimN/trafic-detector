# Запускай его как python3 detector.py dataset/main.py

import sys
import cv2
from ultralytics import YOLO

source = sys.argv[1] if len(sys.argv) > 1 else 0

cap = cv2.VideoCapture(source)
if not cap.isOpened():
    print(f"нету {source}")
    sys.exit(1)

model = YOLO("yolov8l.pt")  # m оказался хорош но l будет лучше

CAR_CLASS_ID = 2

CONFIDENCE_THRESHOLD = 0.35

BOX_COLOR = (0, 255, 0) # зеленый
TEXT_COLOR = (255, 255, 255) # текст в выделении
BG_COLOR = (40, 40, 40) # фон
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.6
THICKNESS = 2

while True:
    ret, frame = cap.read()       
    if not ret:                  
        print("Видео закончилось")
        break

    results = model.predict(
        source=frame,
        classes=[CAR_CLASS_ID],
        conf=CONFIDENCE_THRESHOLD,
        verbose=False,
    )

    detections = results[0].boxes

    # кол во машин на кадре
    car_count = len(detections)

    # UI стаф трогать не обязательно
    for box in detections:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])  


        cv2.rectangle(frame, (x1, y1), (x2, y2), BOX_COLOR, THICKNESS)

        label = f"car {conf:.0%}"
        (tw, th), _ = cv2.getTextSize(label, FONT, FONT_SCALE, 1)
        cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), BOX_COLOR, -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 4),
                    FONT, FONT_SCALE, (0, 0, 0), 1, cv2.LINE_AA)

    counter_text = f"Cars: {car_count}"
    (cw, ch), _ = cv2.getTextSize(counter_text, FONT, 1.0, 2)
    cv2.rectangle(frame, (10, 10), (20 + cw, 20 + ch + 10), BG_COLOR, -1)
    cv2.putText(frame, counter_text, (15, 15 + ch),
                FONT, 1.0, TEXT_COLOR, 2, cv2.LINE_AA)

    cv2.imshow("Car Counter", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()           
cv2.destroyAllWindows()  

# may god bless this code to work ( plsss )

