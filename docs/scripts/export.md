# `export.py`

Export a YOLO model to ONNX and verify the result with `onnx.checker`.

## Synopsis
```bash
python scripts/export.py \
  --weights <weights.pt> \
  --format onnx \
  [--dynamic] \
  [--int8] \
  [--output-dir DIR]
```

## Description

1. Loads `YOLO(args.weights)`.
2. Calls `model.export(format="onnx", dynamic=…, int8=…)` into `args.output_dir`.
3. Locates the `.onnx` file (even if nested) and copies it to the root of `output_dir`.
4. Runs `onnx.checker.check_model()` to verify integrity.

## Arguments

* `--weights`, `-w` **(required)**
  Path to `.pt` weights.
* `--format`, `-f` **(required)**
  Export format (only `onnx` supported).
* `--dynamic`
  Enable dynamic input shapes.
* `--int8`
  Enable INT8 quantization.
* `--output-dir`, `-o`
  Directory to save exported files (default: `exports`).

## Example

```bash
# Export yolov8n.pt to ONNX with dynamic shapes:
poetry run python scripts/export.py \
  -w yolov8n.pt \
  -f onnx \
  --dynamic \
  -o exports/yolov8n
```
