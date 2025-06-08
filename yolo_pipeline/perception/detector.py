# yolo_pipeline/perception/detector.py

from typing import Any, List, Dict
import numpy as np
# from ultralytics import YOLO  # uncomment when ready to load

class Detector:
    """
    Wrapper for Ultralytics YOLO models.
    """

    def __init__(self, model_path: str = "yolov8n.pt", device: str = "cpu"):
        """
        :param model_path: path to a .pt checkpoint or config
        :param device: inference device, e.g. "cpu" or "cuda:0"
        """
        self.model_path = model_path
        self.device = device
        self.model: Any = None

    def load_model(self, model_path: str = None):
        """
        Load a YOLO model for inference.
        TODO: call YOLO(path, device=self.device)
        """
        path = model_path or self.model_path
        # self.model = YOLO(path)
        pass

    def predict(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Run inference on an image.
        :param image: H×W×3 uint8 array
        :return: list of detections, e.g.
                 [{'box':[x1,y1,x2,y2],'confidence':0.5,'class_id':2}, …]
        TODO: run self.model(image) and parse results into dicts
        """
        if self.model is None:
            raise RuntimeError("Model not loaded—call load_model() first")
        # results = self.model(image)
        return []
