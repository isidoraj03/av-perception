# `create_kitti_full.py`

Convert the **full KITTI training split** into YOLO‑formatted images and labels.

## Synopsis
```bash
python scripts/create_kitti_full.py
```

## Description
`create_kitti_full.py` reads dataset paths from `datasets/config.yaml`, copies every camera frame and converts the corresponding KITTI label files to YOLO txt format.

Output directory layout:

```
datasets/kitti_full/
├── images/train/
└── labels/train/
```

A `kitti_full.yaml` file is generated next to `config.yaml` for immediate use with Ultralytics.

## Arguments
*(none)* – all paths are resolved from `datasets/config.yaml`.

## Example
```bash
# From project root
poetry run python scripts/create_kitti_full.py
# Train on the full set
poetry run python scripts/train.py --model yolov8n --data datasets/kitti_full.yaml
```
