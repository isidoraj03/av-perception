# tests/test_detector_inference.py

import numpy as np
from yolo_pipeline.perception.detector import Detector

def test_inference_on_blank_image_returns_list():
    det = Detector(model_path="yolov8n.pt", device="cpu")
    det.load_model()
    # Create a blank image (no objects)
    blank = np.zeros((256, 256, 3), dtype=np.uint8)
    preds = det.predict(blank)
    # Should still return a list (possibly empty)
    assert isinstance(preds, list)
    # If detections exist, they must have the correct keys
    for p in preds:
        assert isinstance(p, dict)
        assert {"box", "confidence", "class_id"} <= set(p.keys())
