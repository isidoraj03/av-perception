# `train.py`

Train a YOLO model (v8, v11, etc.) on your dataset, with MLflow logging.

## Synopsis
```bash
python scripts/train.py \
  --model <model> \
  --data <data_yaml> \
  [--epochs N] \
  [--batch-size B] \
  [--img-size S] \
  [--output-dir DIR]
```

## Description

1. Resolves `--model` into a `.pt` name (downloads pretrained if needed).
2. Configures MLflow experiment (`yolo_training`) and autologging.
3. Calls `YOLO(model_spec).train(...)`.
4. After training, finds the best checkpoint (`best.pt`) and logs it as an MLflow artifact.

## Arguments

* `--model`
  YOLO model name or path (`yolov8n`, `yolov11s`, or `/path/to/custom.pt`).
* `--data` **(required)**
  Path to dataset YAML (must define `train/val` splits).
* `--epochs`
  Number of training epochs (default: 50).
* `--batch-size`
  Batch size (default: 16).
* `--img-size`
  Input image resolution (default: 640).
* `--output-dir`
  Directory under which to save runs (default: `runs/train`).

## Example

```bash
# Train a tiny YOLOv8 model on KITTI-50 subset for 10 epochs:
poetry run python scripts/train.py \
  --model yolov8n \
  --data datasets/kitti50.yaml \
  --epochs 10 \
  --batch-size 8 \
  --img-size 640 \
  --output-dir runs/train/kitti50
```
