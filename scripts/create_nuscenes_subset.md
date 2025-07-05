# `create_nuscenes_subset.py`

Generate a compact YOLO‑formatted subset (front‑camera only) from **nuScenes v1.0‑trainval**.

## Synopsis
```bash
python scripts/create_nuscenes_subset.py --num-images 1500
```

## Description
The script:

1. Opens the official `v1.0-trainval` database (path comes from `datasets/config.yaml`).
2. Iterates through **`CAM_FRONT`** samples in timestamp order.
3. Copies the first *N* images and builds YOLO labels by projecting each 3‑D box onto the image plane.
4. Writes everything to:

   ```
   datasets/nuscenes_subset/
   ├── images/
   └── labels/
   datasets/nuscenes_subset.yaml
   ```

### Supported classes
The ten official nuScenes object classes are mapped 1‑to‑1.

## Arguments
* `-n, --num-images` – number of images to include (default **1500**).

## Example
```bash
# Generate a smaller 500‑frame subset
poetry run python scripts/create_nuscenes_subset.py -n 500
# Benchmark a model on it
poetry run python scripts/eval.py --weights yolov8n.pt --data datasets/nuscenes_subset.yaml
```
