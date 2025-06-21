import subprocess
import sys
import pytest

@pytest.mark.parametrize("fmt", ["onnx"])
def test_export_dispatch(fmt, tmp_path):
    out = tmp_path / "out"
    args = [
        sys.executable, "scripts/export.py",
        "--weights", "yolov8n.pt",
        "--format", fmt,
        "--output-dir", str(out),
    ]
    res = subprocess.run(args, capture_output=True, text=True)
    assert res.returncode == 0, res.stderr
    # The stub prints should mention "exporting to onnx"
    assert "exporting to onnx" in res.stdout.lower()
