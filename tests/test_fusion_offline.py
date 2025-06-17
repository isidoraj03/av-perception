# tests/test_fusion_offline.py

import subprocess, sys, csv
import numpy as np
from pathlib import Path
import PIL.Image as I

def test_fusion_offline_smoke(tmp_path):
    # 1) create a tiny blank image
    img = I.new("RGB", (8,8), (0,0,0))
    img_path = tmp_path / "img.png"
    img.save(img_path)

    # 2) create a pointcloud with zero points
    pc = np.zeros((0,4), dtype=np.float32)
    pc_path = tmp_path / "pc.npy"
    np.save(pc_path, pc)

    # 3) run the script
    cmd = [
        sys.executable, "scripts/fusion_offline.py",
        "--image",      str(img_path),
        "--pc",         str(pc_path),
        "--weights",    "yolov8n.pt",
        "--min-points", "1",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, f"stderr:\n{res.stderr}"
    out = res.stdout

    # 4) should report zero fused detections
    assert "Found 0 fused detections" in out
