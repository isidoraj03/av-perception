#!/usr/bin/env python3
# scripts/train.py

import argparse
import os
import mlflow
import mlflow.pytorch
from ultralytics import YOLO, settings

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a YOLOv8 model on your dataset"
    )
    parser.add_argument(
        "--model", type=str, default="yolov8n.pt",
        help=(
            "Which YOLOv8 variant to train: "
            "`yolov8n`, `yolov8s`, `yolov8m`, `yolov8l`, or `yolov8x` "
            "(you can append `.pt` if you like)."
        )
    )
    parser.add_argument(
        "--data", type=str, required=True,
        help="Path to dataset config (YAML) or image directory"
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

    # 0) MLflow setup
    mlflow.set_experiment("yolo_training")
    mlflow.pytorch.autolog()
    # disable Ultralytics' built-in MLflow hooks
    settings.update(mlflow=False)

    with mlflow.start_run() as run:
        # 1) Log hyperparameters
        mlflow.log_params({
            "model":      args.model,
            "data":       args.data,
            "epochs":     args.epochs,
            "batch_size": args.batch_size,
            "img_size":   args.img_size,
            "output_dir": args.output_dir,
        })

        # 2) Load YOLOv8
        print(f"Loading YOLOv8 model `{args.model}`…")
        model = YOLO(args.model)

        # 3) Train
        print("Starting training with parameters:")
        print(
            f"  data       : {args.data}\n"
            f"  epochs     : {args.epochs}\n"
            f"  batch size : {args.batch_size}\n"
            f"  img size   : {args.img_size}\n"
            f"  output dir : {args.output_dir}"
        )
        model.train(
            data=args.data,
            epochs=args.epochs,
            batch=args.batch_size,
            imgsz=args.img_size,
            project=args.output_dir,
            exist_ok=True
        )

        # 4) Locate & log the best checkpoint
        last_run = sorted(os.listdir(args.output_dir))[-1]
        best_ckpt = os.path.join(args.output_dir, last_run, "weights", "best.pt")
        print(f"\n✅ Training complete. Best checkpoint at:\n    {best_ckpt}")
        mlflow.log_artifact(best_ckpt, artifact_path="checkpoints")

        print(f"MLflow run completed: run_id={run.info.run_id}")

if __name__ == "__main__":
    main()
