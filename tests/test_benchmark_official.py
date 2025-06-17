# tests/test_benchmark_official.py

import subprocess
import sys
import csv
from pathlib import Path

def test_benchmark_official(tmp_path):
    out_base = tmp_path / "runs"
    cmd = [
        sys.executable, "scripts/benchmark_official.py",
        "--data", "datasets/kitti50.yaml",
        "--models", "yolov8s",
        "--batch-size", "2",
        "--output-base", str(out_base)
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, res.stderr

    # CPU dir must exist
    cpu_dir = out_base / "eval_official_yolov8s_cpu"
    assert cpu_dir.exists(), f"{cpu_dir} not created"
    summary = cpu_dir / "summary.csv"
    assert summary.exists(), f"{summary} missing"
    rows = list(csv.reader(summary.open()))
    assert rows[0] == ["metric", "value"]
    data = {r[0]: float(r[1]) for r in rows[1:]}
    for key in ("precision", "recall", "map", "fps"):
        assert key in data, f"{key} missing in {summary}"
        assert data[key] >= 0

    # GPU dir is optional: if it exists, its summary must also be valid
    gpu_dir = out_base / "eval_official_yolov8s_gpu"
    if gpu_dir.exists():
        summary = gpu_dir / "summary.csv"
        assert summary.exists(), f"{summary} missing"
        rows = list(csv.reader(summary.open()))
        data = {r[0]: float(r[1]) for r in rows[1:]}
        for key in ("precision", "recall", "map", "fps"):
            assert key in data, f"{key} missing in {summary}"
            assert data[key] >= 0
