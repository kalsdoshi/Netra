from ultralytics import YOLO

class ObjectDetector:
    def __init__(self):
        self.model = YOLO("yolov8n.pt")  # lightweight model

    def detect(self, image):
        results = self.model(image)[0]

        objects = []

        for box in results.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            xyxy = box.xyxy[0].tolist()

            objects.append({
                "label": results.names[cls],
                "confidence": conf,
                "bbox": [int(x) for x in xyxy]
            })

        return objects