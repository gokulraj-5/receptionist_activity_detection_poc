
from ultralytics import YOLO


class PersonDetector:

    def __init__(self, model_path="yolov8n.pt", confidence=0.25, classes=None):

        # Load pretrained YOLO model
        self.model = YOLO(model_path)
        self.confidence = confidence
        self.classes = classes if classes is not None else [0]

    def detect_people(self, frame, conf=None):

        c = conf if conf is not None else self.confidence

        results = self.model.predict(
            source=frame,
            classes=self.classes,
            conf=c,
            verbose=False
        )

        people = []

        for result in results:

            if result.boxes is None:
                continue

            for box in result.boxes:

                # Bounding box coordinates
                x1, y1, x2, y2 = map(
                    int,
                    box.xyxy[0].tolist()
                )

                # Detection confidence
                confidence = float(box.conf[0])

                people.append({
                    "bbox": (x1, y1, x2, y2),
                    "confidence": confidence
                })

        return people
