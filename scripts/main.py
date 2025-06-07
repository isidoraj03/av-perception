# scripts/main.py

import time
import sys
from yolo_pipeline.io.data_streamer import DataStreamer

def main(duration_sec: float = 3.0, interval_sec: float = 0.1):
    ds = DataStreamer(config_path="datasets/config.yaml")
    ds.load_split("kitti", split="train", shuffle=False)
    ds.start()

    cam_count = 0
    pc_count  = 0
    start_time = time.time()

    while time.time() - start_time < duration_sec:
        cam = ds.get_latest_camera_frame()
        pc  = ds.get_latest_pointcloud()

        if cam is not None:
            cam_count += 1
        if pc is not None:
            pc_count += 1

        print(f"{time.time() - start_time:>4.2f}s | "
              f"cam: {None if cam is None else cam.shape} | "
              f"lidar: {None if pc  is None else pc.shape}")

        time.sleep(interval_sec)

    ds.stop()

    print(f"\nSummary: received {cam_count} camera frames and {pc_count} pointclouds in {duration_sec:.1f}s.")
    if cam_count == 0 or pc_count == 0:
        print("ERROR: Streaming failed to produce data on one or both queues.")
        sys.exit(1)

    print("✅ Smoke test passed: KITTI RGB + LiDAR streaming verified.")

if __name__ == "__main__":
    main()
