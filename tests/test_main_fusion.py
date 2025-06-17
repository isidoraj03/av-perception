# tests/test_main_fusion.py

import numpy as np
import pytest
from types import SimpleNamespace
import scripts.main as main_script

class DummyStreamer:
    """Always returns a fixed image + pointcloud, then stops."""
    def __init__(self, *args, **kwargs):
        self.calls = 0

    def load_split(self, *args, **kwargs): pass
    def start(self): pass

    def get_latest_camera_frame(self):
        # return a blank 8×8 RGB frame
        return np.zeros((8,8,3), dtype=np.uint8)

    def get_latest_pointcloud(self):
        # return empty pointcloud
        return np.zeros((0,4), dtype=np.float32)

    def stop(self): pass

class DummyDetector:
    """Always returns exactly one synthetic 2D box."""
    def __init__(self, *args, **kwargs): pass
    def load_model(self): pass
    def predict(self, image):
        return [{"box":[0,0,4,4], "confidence":1.0, "class_id":0}]

class DummyFusion(SimpleNamespace):
    """Drops everything: returns empty list for any input."""
    def __init__(self, *args, **kwargs): super().__init__()
    def fuse(self, dets, pc): return []

@pytest.fixture(autouse=True)
def monkey_all(monkeypatch):
    # patch DataStreamer, Detector, FusionEngine
    monkeypatch.setattr(main_script, "DataStreamer", DummyStreamer)
    monkeypatch.setattr(main_script, "Detector", DummyDetector)
    monkeypatch.setattr(main_script, "FusionEngine",
                        lambda intr, ext, min_points: DummyFusion())

def test_main_fusion_counts(capsys):
    # run only a very short loop
    main_script.main(duration_sec=0.2, interval_sec=0.1)
    out = capsys.readouterr().out

    # should have at least one line with "2D detections: 1 | 3D fused: 0"
    assert "2D detections: 1" in out
    assert "3D fused: 0" in out
