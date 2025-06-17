# tests/test_eval.py

import subprocess
import sys
from pathlib import Path

def test_eval_smoke(tmp_path):
    """Run eval.py on the KITTI-50 subset with a tiny model and check for mAP/FPS."""
    outdir = tmp_path / "val_out"
    cmd = [
        sys.executable,
        "scripts/eval.py",
        "--weights", "yolov8n.pt",
        "--data", "datasets/kitti50.yaml",
        "--device", "cpu",
        "--batch-size", "2",
        "--output-dir", str(outdir),
    ]

    res = subprocess.run(cmd, capture_output=True, text=True)
    # should exit cleanly
    assert res.returncode == 0, f"stderr:\n{res.stderr}"
    stdout = res.stdout

    # must mention mAP and FPS
    assert "mAP" in stdout, stdout
    assert "FPS" in stdout, stdout

    # output dir exists
    assert outdir.exists(), f"{outdir} not created"
