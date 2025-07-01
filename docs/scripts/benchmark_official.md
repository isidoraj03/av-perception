# `benchmark_official.py`

Run `eval.py` on one or more official YOLO model variants (e.g. `yolov8s`, `yolov11s`) on CPU and GPU (if available), collecting their summaries.

## Synopsis
```bash
python scripts/benchmark_official.py \
  --data <data_yaml> \
  [--models name1 name2 …] \
  [--batch-size B] \
  [--output-base DIR]
```

## Description

* Iterates over each model name (no `.pt` extension).
* For **CPU**: creates `runs/eval_official_<model>_cpu`, invokes `eval.py`.
* For **GPU** (if `torch.cuda.is_available()`): creates `…_gpu`, invokes `eval.py`.
* Prints a warning if GPU is unavailable.

## Arguments

* `--data` **(required)**
  Path to dataset YAML (e.g. `datasets/kitti50.yaml`).
* `--models`
  List of model names (default: `yolov8s yolo11s`).
* `--batch-size`
  Inference batch size (default: 4).
* `--output-base`
  Base directory under which to create `eval_official_*` folders (default: `runs`).

## Example

```bash
# Benchmark yolov8s and yolov11s:
poetry run python scripts/benchmark_official.py \
  --data datasets/kitti50.yaml \
  --models yolov8s yolov11s \
  --batch-size 2 \
  --output-base runs/benchmark
```
