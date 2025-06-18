# tests/test_main_tracker.py

import numpy as np
import pytest
import scripts.main as main_script
from yolo_pipeline.perception.tracker import Detection3D

# Dummy classes to inject predictable behavior
class DummyStreamer:
    def __init__(self, *args, **kwargs): pass
    def load_split(self, *args, **kwargs): pass
    def start(self): pass
    def get_latest_camera_frame(self):
        # Always return a valid 8×8 RGB frame
        return np.zeros((8, 8, 3), dtype=np.uint8)
    def get_latest_pointcloud(self):
        # Always return an (empty) N×4 pointcloud
        return np.zeros((0, 4), dtype=np.float32)
    def stop(self): pass

class DummyDetector:
    def __init__(self, *args, **kwargs): pass
    def load_model(self): pass
    def predict(self, image):
        # One synthetic 2D detection
        return [{"box": [0, 0, 4, 4], "confidence": 0.8, "class_id": 1}]

class DummyFusion:
    def __init__(self, *args, **kwargs): pass
    def fuse(self, dets, pc):
        # One fused 3D detection dict
        return [{
            "box": [0, 0, 4, 4],
            "confidence": 0.8,
            "class_id": 1,
            "num_points": 5,
            "depth": 2.5
        }]

class DummyTracker:
    def __init__(self, *args, **kwargs): pass
    def update(self, fused):
        # Wrap the single fused dict in a Detection3D with stable ID=7
        d = fused[0]
        return [Detection3D(
            track_id=7,
            class_id=d["class_id"],
            confidence=d["confidence"],
            box2d=tuple(d["box"]),
            depth=d["depth"],
            num_points=d["num_points"],
        )]

@pytest.fixture(autouse=True)
def patch_dependencies(monkeypatch):
    # Replace the real classes with our dummies in scripts/main.py
    monkeypatch.setattr(main_script, "DataStreamer", DummyStreamer)
    monkeypatch.setattr(main_script, "Detector", DummyDetector)
    monkeypatch.setattr(main_script, "FusionEngine", lambda *a, **k: DummyFusion())
    monkeypatch.setattr(main_script, "SimpleTracker", lambda *a, **k: DummyTracker())

def test_main_tracker_integration(capsys):
    # Run only a short playback
    main_script.main(duration_sec=0.25, interval_sec=0.1)
    out = capsys.readouterr().out

    # Check the frame header
    assert "Frame 00" in out
    # Check counts for 2D, 3D and tracked
    assert "2D= 1" in out
    assert "3D= 1" in out
    assert "Tracked= 1" in out
    # Check that our dummy track shows up with ID 007 and the right stats
    assert "[ID 007]" in out
    assert "cls=1" in out
    assert "depth=2.50" in out
    assert "pts=5" in out
