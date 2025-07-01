# `fusion_offline.py`

Run YOLO detection + FusionEngine on a single image + pointcloud, printing fused results.

## Synopsis
```bash
python scripts/fusion_offline.py \
  --image <frame.png> \
  --pc <points.npy> \
  --weights <weights.pt> \
  [--min-points N]
```

## Description

1. Loads an RGB image (`.png`/`.jpg`) into an array.
2. Loads a LiDAR pointcloud (`.npy`, shape `N×4`).
3. Runs `YOLO(args.weights).predict(...)` to get 2D detections.
4. Instantiates `FusionEngine` with identity calibration and `min_points`.
5. Calls `fused = engine.fuse(detections, pc)`.
6. Prints the number of fused detections and each dict.

## Arguments

* `--image` **(required)**
  Path to camera frame image.
* `--pc` **(required)**
  Path to NumPy `.npy` pointcloud file.
* `--weights` **(required)**
  Path to YOLO weights.
* `--min-points`
  Minimum LiDAR points per box to keep (default: 3).

## Example

```bash
poetry run python scripts/fusion_offline.py \
  --image tests/assets/000000.png \
  --pc tests/assets/000000.npy \
  --weights yolov8n.pt \
  --min-points 1
```
