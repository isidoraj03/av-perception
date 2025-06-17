# tests/test_fusion_logic.py

import pytest
import numpy as np
from yolo_pipeline.perception.fusion import FusionEngine

def test_fusion_engine_filters_and_rescores():
    intr = {'fx':1.0, 'fy':1.0, 'cx':0.0, 'cy':0.0}
    ext  = {'R': np.eye(3),  'T': [0,0,0]}

    # Engine with min_points=2
    eng = FusionEngine(intr, ext, min_points=2)
    dets = [{'box':[0,0,10,10], 'confidence':0.5}]
    # two points inside
    pc = np.array([[5,5,1,0.],[6,6,1,0.]])
    out = eng.fuse(dets, pc)
    assert len(out) == 1
    od = out[0]
    assert od['num_points'] == 2
    # new confidence = 0.5 * min(1,2/2) == 0.5
    assert pytest.approx(od['confidence'], rel=1e-6) == 0.5

    # Engine with min_points=3 should drop
    eng2 = FusionEngine(intr, ext, min_points=3)
    out2 = eng2.fuse(dets, pc)
    assert out2 == []
