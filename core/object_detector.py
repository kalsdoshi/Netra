from ultralytics import YOLO

class ObjectDetector:
    def __init__(self, min_confidence=0.4):
        self.model = YOLO("yolov8n.pt")  # lightweight model
        self.min_confidence = min_confidence

    def detect(self, image):
        results = self.model(image, verbose=False)[0]

        objects = []

        for box in results.boxes:
            conf = float(box.conf[0])

            # Skip low-confidence detections to prevent noise in metadata
            if conf < self.min_confidence:
                continue

            cls = int(box.cls[0])
            xyxy = box.xyxy[0].tolist()

            objects.append({
                "label": results.names[cls],
                "confidence": conf,
                "bbox": [int(x) for x in xyxy]
            })

        return objects