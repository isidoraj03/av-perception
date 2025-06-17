#!/usr/bin/env python3
# scripts/main.py

import time
import sys
import numpy as np
from yolo_pipeline.io.data_streamer import DataStreamer
from yolo_pipeline.perception.detector import Detector
from yolo_pipeline.perception.fusion import FusionEngine

def main(duration_sec: float = 3.0, interval_sec: float = 0.1):
    # 0) set up streamer
    ds = DataStreamer(config_path="datasets/config.yaml")
    ds.load_split("kitti", split="train", shuffle=False)
    ds.start()

    # 1) load YOLO detector
    det = Detector(model_path="yolov8n.pt", device="cpu")
    det.load_model()

    # 2) set up simple identity calibration for fusion
    intrinsics = {"fx": 1.0, "fy": 1.0, "cx": 0.0, "cy": 0.0}
    extrinsics = {"R": np.eye(3), "T": [0, 0, 0]}
    fus = FusionEngine(intrinsics, extrinsics, min_points=3)

    t_start = time.time()
    while time.time() - t_start < duration_sec:
        cam = ds.get_latest_camera_frame()
        pc  = ds.get_latest_pointcloud()

        if cam is None or pc is None:
            # still warming up
            print(f"{time.time()-t_start:>4.2f}s | waiting for data…")
        else:
            # 3) 2D detection
            dets2d = det.predict(cam)
            n2d = len(dets2d)

            # 4) fusion → 3D‐enhanced
            fused = fus.fuse(dets2d, pc)
            n3d = len(fused)

            print(f"{time.time()-t_start:>4.2f}s | 2D detections: {n2d} | 3D fused: {n3d}")

        time.sleep(interval_sec)

    ds.stop()
    print("\nDone.")

if __name__ == "__main__":
    main()
