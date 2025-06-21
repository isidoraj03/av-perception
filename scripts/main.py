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

# Change this to 'kitti' to use the KITTI dataset instead
# or to 'nuscenes' to use the nuScenes v1.0-mini dataset.
DATASET_NAME = "nuscenes"


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

    # only create a window if we have a display
    if not HEADLESS:
        cv2.namedWindow("Overlay", cv2.WINDOW_NORMAL)

    start_time = time.time()
    frame_idx = 0

    while True:
        now = time.time()
        if now - start_time > duration_sec:
            break

        cam = ds.get_latest_camera_frame()
        pc = ds.get_latest_pointcloud()

        if cam is None or pc is None:
            # still streaming initial frame
            time.sleep(interval_sec)
            continue

        # measure processing time
        t0 = time.time()

        # 4) 2D detect
        dets2d = det.predict(cam)
        # 5) fuse → 3D detections
        fused = fus.fuse(dets2d, pc)
        # 6) track
        tracks = tracker.update(fused)

        # --- print per-frame stats for pytest validation ---
        print(f"2D detections: {len(dets2d)} | 3D fused: {len(fused)}")
        print(
            f"Frame {frame_idx:02d}: 2D= {len(dets2d)} | "
            f"3D= {len(fused)} | Tracked= {len(tracks)}"
        )
        for t in tracks:
            print(
                f"  [ID {t.track_id:03d}] cls={t.class_id} "
                f"depth={t.depth:.2f} pts={t.num_points}"
            )

        t1 = time.time()
        fps = 1.0 / (t1 - t0) if (t1 - t0) > 0 else 0.0

        # 7) annotate & display
        annotated = draw_overlay(cam, tracks, fps)

        if not HEADLESS:
            cv2.imshow("Overlay", annotated)
            # quit on 'q'
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
