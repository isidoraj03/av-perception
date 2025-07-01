# `main.py`

Live playback demo: stream KITTI or nuScenes, run 2D detection, fuse to 3D, track objects, and display.

## Synopsis
```bash
python scripts/main.py
```

## Description

* Uses `DataStreamer` to stream camera frames + LiDAR at ~10 Hz.
* Loads a YOLO detector and a `FusionEngine` (identity calibration).
* Applies `SimpleTracker` for per-frame tracking.
* Displays an OpenCV window `Overlay` with annotated boxes, IDs, depth, point counts, and FPS.
* Prints per-frame stats to stdout (used by tests).

> **Note:** `main.py` is currently hard-coded to `DATASET_NAME = "nuscenes"`.
> To use KITTI, edit that constant at the top of the script.

## Configuration

* **Adjust duration & rate** by modifying the defaults in `main(duration_sec, interval_sec)`.
* **Exit early**: focus the window and press **q**.

## Example

```bash
# Default run (10 s, 0.05 s interval):
poetry run python scripts/main.py

# For a longer run, edit:
#   main(duration_sec=30.0, interval_sec=0.05)
```
