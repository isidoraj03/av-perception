#!/usr/bin/env python3
# scripts/main.py

import time
import numpy as np
from yolo_pipeline.io.data_streamer import DataStreamer
from yolo_pipeline.perception.detector import Detector
from yolo_pipeline.perception.fusion import FusionEngine
from yolo_pipeline.perception.tracker import SimpleTracker

def main(duration_sec: float = 3.0, interval_sec: float = 0.1):
    # 0) set up streamer
    ds = DataStreamer(config_path="datasets/config.yaml")
    ds.load_split("kitti", split="train", shuffle=False)
    ds.start()

    # 1) load YOLO detector
    det = Detector(model_path="yolov8n.pt", device="cpu")
    det.load_model()

    # 2) set up fusion
    intrinsics = {"fx": 1.0, "fy": 1.0, "cx": 0.0, "cy": 0.0}
    extrinsics = {"R": np.eye(3), "T": [0, 0, 0]}
    fus = FusionEngine(intrinsics, extrinsics, min_points=3)

    # 3) set up tracker
    tracker = SimpleTracker()

    t_start = time.time()
    frame_idx = 0

    while time.time() - t_start < duration_sec:
        cam = ds.get_latest_camera_frame()
        pc  = ds.get_latest_pointcloud()

        if cam is None or pc is None:
            print(f"{time.time() - t_start:>5.2f}s | warming up…")
        else:
            # 4) 2D detection
            dets2d = det.predict(cam)
            n2d = len(dets2d)

            # 5) fusion → 3D-enhanced dicts
            fused = fus.fuse(dets2d, pc)
            n3d = len(fused)

            # 6) tracking → Detection3D objects
            tracks = tracker.update(fused)
            n_tr = len(tracks)

            # 7) log summary & per-track info
            print(
                f"{time.time() - t_start:>5.2f}s | "
                f"Frame {frame_idx:02d} | "
                f"2D={n2d:2d} | 3D={n3d:2d} | Tracked={n_tr:2d}"
            )
            for t in tracks:
                # compute box center for display
                x1, y1, x2, y2 = t.box2d
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                print(
                    f"    [ID {t.track_id:03d}] cls={t.class_id} "
                    f"centroid=({cx:.1f},{cy:.1f}) "
                    f"depth={t.depth:.2f} pts={t.num_points}"
                )
            print()

            frame_idx += 1

        time.sleep(interval_sec)

    ds.stop()
    print("Playback done.")

if __name__ == "__main__":
    main()
