# tests/test_eval_summary.py

import subprocess, csv, sys
from pathlib import Path

def test_eval_summary(tmp_path):
    outdir = tmp_path / "val_out"
    cmd = [
        sys.executable, "scripts/eval.py",
        "--weights", "yolov8n.pt",
        "--data", "datasets/kitti50.yaml",
        "--device", "cpu",
        "--batch-size", "2",
        "--output-dir", str(outdir),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, f"stderr:\n{res.stderr}"

    csv_file = outdir / "summary.csv"
    assert csv_file.exists(), "summary.csv not found"

    rows = list(csv.reader(csv_file.open()))
    # First row is header
    assert rows[0] == ["metric", "value"]
    data = {r[0]: float(r[1]) for r in rows[1:]}
    # Must contain these keys
    for key in ("precision", "recall", "map", "fps"):
        assert key in data, f"{key} missing from summary.csv"
        # value should parse as float ≥0
        assert data[key] >= 0
