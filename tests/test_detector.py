# tests/test_detector.py

import pytest
import numpy as np
from yolo_pipeline.perception.detector import Detector

def test_detector_initialization():
    det = Detector(model_path="yolov8n.pt", device="cpu")
    # ensure the attributes exist
    assert hasattr(det, "model_path")
    assert hasattr(det, "device")
    # before loading, model should be None
    assert det.model is None

def test_load_model_does_not_raise():
    det = Detector(model_path="yolov8n.pt", device="cpu")
    # loading the model should not raise
    det.load_model()
    # after loading, model should be an actual YOLO instance
    assert det.model is not None

def test_predict_raises_before_load():
    det = Detector(model_path="yolov8n.pt", device="cpu")
    dummy_img = np.zeros((256, 256, 3), dtype=np.uint8)
    with pytest.raises(RuntimeError):
        _ = det.predict(dummy_img)

def test_predict_returns_list_after_load():
    det = Detector(model_path="yolov8n.pt", device="cpu")
    det.load_model()
    dummy_img = np.zeros((256, 256, 3), dtype=np.uint8)
    preds = det.predict(dummy_img)
    assert isinstance(preds, list)
    # each item, if any, should be a dict with the expected keys
    for p in preds:
        assert isinstance(p, dict)
        assert {"box", "confidence", "class_id"} <= set(p.keys())
