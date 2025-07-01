# `eval.py`

Evaluate a YOLO model on a validation set, compute mAP/precision/recall, benchmark FPS, and save a CSV summary.

## Synopsis
```bash
python scripts/eval.py \
  --weights <weights.pt> \
  --data <data_yaml> \
  [--device <device>] \
  [--batch-size B] \
  [--output-dir DIR]
```

## Description

1. Loads model via `YOLO(args.weights)` and moves it to `args.device`.
2. Runs `model.val(...)` to compute detection metrics (mAP, precision, recall).
3. Scans the validation folder for images, runs `model.predict(...)` on each to measure FPS.
4. Writes a `summary.csv` with rows: `metric,value` for `precision`, `recall`, `map`, `fps`.

## Arguments

* `--weights` **(required)**
  Path to `.pt` weights.
* `--data` **(required)**
  Path to dataset YAML (must include `val:`).
* `--device`
  Inference device, e.g. `cpu` or `0` (default: `cpu`).
* `--batch-size`
  Batch size for the validation step (default: 16).
* `--output-dir`
  Directory to save logs and outputs (default: `runs/val`).

## Example

```bash
# Evaluate on KITTI-50 subset:
poetry run python scripts/eval.py \
  --weights yolov8n.pt \
  --data datasets/kitti50.yaml \
  --device cpu \
  --batch-size 4 \
  --output-dir runs/val/kitti50
```
