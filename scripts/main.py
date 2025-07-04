#!/usr/bin/env python3
import os
import time

import cv2
import numpy as np

from yolo_pipeline.io.data_streamer import DataStreamer
from yolo_pipeline.perception.detector import Detector
from yolo_pipeline.perception.fusion import FusionEngine
from yolo_pipeline.perception.tracker import SimpleTracker
from scripts.overlay import draw_overlay

# detect headless/CI environments
HEADLESS = os.getenv("CI") is not None

# Switch between mini vs full nuScenes here:
#   "nuscenes"      → v1.0-mini
#   "nuscenes-full" → v1.0-full
# DATASET_NAME = "nuscenes"        # mini by default
DATASET_NAME = "nuscenes-full" # uncomment to run on full dataset

def main(duration_sec: float = 10.0, interval_sec: float = 0.05):
    # 0) set up DataStreamer
    ds = DataStreamer(config_path="datasets/config.yaml")
    ds.load_split(DATASET_NAME, split="train", shuffle=False)
    ds.start()

    # 1) load detector
    det = Detector(model_path="yolov8n.pt", device="cpu")
    det.load_model()

    # 2) set up fusion
    intrinsics = {"fx": 1.0, "fy": 1.0, "cx": 0.0, "cy": 0.0}
    extrinsics = {"R": np.eye(3), "T": [0, 0, 0]}
    fus = FusionEngine(intrinsics, extrinsics, min_points=3)

    # 3) set up tracker
    tracker = SimpleTracker()

    if not HEADLESS:
        cv2.namedWindow("Overlay", cv2.WINDOW_NORMAL)

    start_time = time.time()
    frame_idx = 0

    while True:
        now = time.time()
        if now - start_time > duration_sec:
            break

        cam = ds.get_latest_camera_frame()
        pc  = ds.get_latest_pointcloud()
        if cam is None or pc is None:
            time.sleep(interval_sec)
            continue

        t0 = time.time()
        dets2d = det.predict(cam)
        fused  = fus.fuse(dets2d, pc)
        tracks = tracker.update(fused)
        t1 = time.time()

        fps = 1.0 / (t1 - t0) if (t1 - t0) > 0 else 0.0

        # pytest‐style printouts
        print(f"2D detections: {len(dets2d)} | 3D fused: {len(fused)}")
        print(f"Frame {frame_idx:02d}: 2D= {len(dets2d)} | 3D= {len(fused)} | Tracked= {len(tracks)}")
        for t in tracks:
            print(f"  [ID {t.track_id:03d}] cls={t.class_id} depth={t.depth:.2f} pts={t.num_points}")

        annotated = draw_overlay(cam, tracks, fps)
        if not HEADLESS:
            cv2.imshow("Overlay", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        frame_idx += 1
        time.sleep(interval_sec)

    ds.stop()
    if not HEADLESS:
        cv2.destroyAllWindows()
    print("Playback done.")

if __name__ == "__main__":
    main()
