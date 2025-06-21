import subprocess, sys, glob, onnx, pytest

def test_export_and_check_onnx(tmp_path):
    out = tmp_path / "export_onnx"
    weights = "yolov8n.pt"  # <-- use the official pretrained name
    cmd = [
        sys.executable, "scripts/export.py",
        "--weights", weights,
        "--format", "onnx",
        "--dynamic",
        "--output-dir", str(out),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, res.stderr

    onnx_files = glob.glob(str(out / "*.onnx"))
    assert onnx_files, f"No ONNX file in {out}"
    model = onnx.load(onnx_files[0])
    onnx.checker.check_model(model)
