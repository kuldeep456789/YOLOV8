# modules/detection/vehicle_detector.py

import os
from ultralytics import YOLO

class VehicleDetector:
    def __init__(self, model_path="models/yolov8n.pt"):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
        self.model = YOLO(model_path)

    def detect_vehicles(self, frame):
        results = self.model(frame)[0]
        detections = []

        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            detections.append((int(x1), int(y1), int(x2), int(y2), conf))

        return detections
