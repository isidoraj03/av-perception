# tests/test_detector.py

import pytest
import numpy as np
from yolo_pipeline.perception.detector import Detector

def test_detector_initialization():
    det = Detector(model_path="yolov8n.pt", device="cpu")
    # ensure the attributes exist
    assert hasattr(det, "model_path")
    assert hasattr(det, "device")
    assert det.model is None

def test_load_model_does_not_raise():
    det = Detector()
    # stubbed load_model() shouldn’t blow up
    det.load_model()
    # model is still None (we haven’t implemented loading yet)
    assert det.model is None

def test_predict_raises_before_load():
    det = Detector()
    dummy_img = np.zeros((256, 256, 3), dtype=np.uint8)
    with pytest.raises(RuntimeError):
        _ = det.predict(dummy_img)

def test_predict_returns_list_when_model_set():
    det = Detector()
    # fake “loaded” model so predict goes past the RuntimeError
    det.model = object()
    dummy_img = np.zeros((256, 256, 3), dtype=np.uint8)
    preds = det.predict(dummy_img)
    assert isinstance(preds, list)
