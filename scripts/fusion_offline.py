#!/usr/bin/env python3
# scripts/fusion_offline.py

import argparse
import numpy as np
from PIL import Image
from ultralytics import YOLO
from yolo_pipeline.perception.fusion import FusionEngine

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run YOLO + FusionEngine on one image + pointcloud"
    )
    parser.add_argument("--image",      type=str, required=True,
                        help="Path to camera frame (PNG/JPG)")
    parser.add_argument("--pc",         type=str, required=True,
                        help="Path to LiDAR pointcloud (.npy file, N×4 floats)")
    parser.add_argument("--weights",    type=str, required=True,
                        help="YOLO weights file (e.g. yolov8n.pt)")
    parser.add_argument("--min-points", type=int, default=3,
                        help="Minimum lidar‐points per box to keep it")
    return parser.parse_args()

def main():
    args = parse_args()

    # 1) Load data
    img = np.array(Image.open(args.image).convert("RGB"))
    pc  = np.load(args.pc)  # expecting shape (N,4)

    # 2) Run YOLO
    model = YOLO(args.weights)
    results = model.predict(source=img, device="cpu", verbose=False)
    r = results[0]
    detections = []
    for box, conf, cls in zip(r.boxes.xyxy.cpu().numpy(),
                              r.boxes.conf.cpu().numpy(),
                              r.boxes.cls.cpu().numpy()):
        detections.append({
            "box":         box.tolist(),
            "confidence":  float(conf),
            "class_id":    int(cls),
        })

    # 3) Set up a simple identity calibration
    intrinsics = {"fx":1.0, "fy":1.0, "cx":0.0, "cy":0.0}
    extrinsics = {"R": np.eye(3), "T": [0,0,0]}

    # 4) Fuse
    engine = FusionEngine(intrinsics, extrinsics, min_points=args.min_points)
    fused = engine.fuse(detections, pc)

    # 5) Report
    print(f"Found {len(fused)} fused detections (min_points={args.min_points}):")
    for det in fused:
        print(det)

if __name__ == "__main__":
    main()
