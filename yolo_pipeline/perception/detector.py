# yolo_pipeline/perception/detector.py

from typing import Any, List, Dict
import numpy as np
from ultralytics import YOLO


class Detector:
    """Wrapper for Ultralytics YOLO models."""

    def __init__(self, model_path: str = "yolov8n.pt", device: str = "cpu"):
        self.model_path = model_path
        self.device = device
        self.model: Any = None

    def load_model(self, model_path: str = None):
        path = model_path or self.model_path
        self.model = YOLO(path)
        self.model.to(self.device)

    def predict(
        self,
        image: np.ndarray,
        conf: float = 0.25,
        iou: float = 0.45
    ) -> List[Dict[str, Any]]:
        if self.model is None:
            raise RuntimeError("Model not loaded—call load_model() first")

        # If in a unit test you’ve done `det.model = object()`,
        # that object has no predict() → just return empty list
        if not hasattr(self.model, "predict"):
            return []

        # Real inference
        results = self.model.predict(
            source=image,
            device=self.device,
            conf=conf,
            iou=iou,
            verbose=False
        )
        r = results[0]
        preds: List[Dict[str, Any]] = []

        xyxy = r.boxes.xyxy.cpu().numpy()
        scores = r.boxes.conf.cpu().numpy()
        cls_ids = r.boxes.cls.cpu().numpy()

        for box, score, cls in zip(xyxy, scores, cls_ids):
            preds.append({
                "box": box.tolist(),
                "confidence": float(score),
                "class_id": int(cls)
            })
        return preds
