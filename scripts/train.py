#!/usr/bin/env python3
# scripts/train.py

import argparse
import os
from ultralytics import YOLO

def parse_args():
    parser = argparse.ArgumentParser(description="Train a YOLO model on your dataset")
    parser.add_argument(
        "--model", type=str, default="yolov8n.pt",
        help="Path to a YOLO .pt file or model name (e.g. yolov8n.pt)"
    )
    parser.add_argument(
        "--data", type=str, required=True,
        help="Path to dataset config (YAML) or directory"
    )
    parser.add_argument(
        "--epochs", type=int, default=50,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size", type=int, default=16,
        help="Batch size"
    )
    parser.add_argument(
        "--img-size", type=int, default=640,
        help="Input image size (pixels)"
    )
    parser.add_argument(
        "--output-dir", type=str, default="runs/train",
        help="Root directory to save training runs"
    )
    return parser.parse_args()

def main():
    args = parse_args()

    # 1. Load model
    print(f"Loading model from {args.model} on device default...")
    model = YOLO(args.model)

    # 2. Train
    print(f"Starting training:\n"
          f"  data={args.data}\n"
          f"  epochs={args.epochs}\n"
          f"  batch_size={args.batch_size}\n"
          f"  img_size={args.img_size}\n"
          f"  output_dir={args.output_dir}")
    model.train(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch_size,
        imgsz=args.img_size,
        project=args.output_dir,
        exist_ok=True  # overwrite if rerunning
    )

    # 3. Save best checkpoint (Ultralytics auto-saves best.pt under project/name/weights/)
    #    We assume default run name "exp" or "expN"; for more control you can add --name flag.
    default_run = os.path.join(args.output_dir, os.listdir(args.output_dir)[-1])
    best_ckpt = os.path.join(default_run, "weights", "best.pt")
    print(f"\n✅ Training complete. Best checkpoint at:\n    {best_ckpt}")

if __name__ == "__main__":
    main()
