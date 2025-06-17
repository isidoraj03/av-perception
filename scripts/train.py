#!/usr/bin/env python3
# scripts/train.py

import argparse
import os
import re
import mlflow
import mlflow.pytorch
from ultralytics import YOLO, settings

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a YOLO model (v8, v11, etc.) on your dataset"
    )
    parser.add_argument(
        "--model", type=str, default="yolov8n",
        help=(
            "YOLO model to use: e.g. 'yolov8n','yolov11n', etc. "
            "You can also pass a path to your own .pt file."
        )
    )
    parser.add_argument(
        "--data", type=str, required=True,
        help="Path to dataset YAML (e.g. coco128.yaml or datasets/kitti50.yaml)"
    )
    parser.add_argument("--epochs",     type=int, default=50, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--img-size",   type=int, default=640, help="Input image size")
    parser.add_argument(
        "--output-dir", type=str, default="runs/train",
        help="Directory to save training runs"
    )
    return parser.parse_args()

def resolve_model(model_arg: str) -> str:
    """
    If model_arg is exactly like 'yolo8n','yolo11s', etc. (no .pt),
    append '.pt' so Ultralytics will download the pretrained weights.
    Otherwise (e.g. a local path), return unchanged.
    """
    base, ext = os.path.splitext(os.path.basename(model_arg))
    # match yolo + digits (8,11,...) + one of n|s|m|l|x
    if re.fullmatch(r"yolo\d+(n|s|m|l|x)", base):
        return base + ".pt"
    return model_arg

def main():
    args = parse_args()

    # 0) MLflow setup
    mlflow.set_experiment("yolo_training")
    mlflow.pytorch.autolog()
    settings.update(mlflow=False)

    # 1) Determine actual model spec
    model_spec = resolve_model(args.model)

    with mlflow.start_run() as run:
        # 2) Log hyperparameters
        mlflow.log_params({
            "model":      args.model,
            "data":       args.data,
            "epochs":     args.epochs,
            "batch_size": args.batch_size,
            "img_size":   args.img_size,
            "output_dir": args.output_dir,
        })

        # 3) Load and train
        print(f"Loading YOLO model from {model_spec}…")
        model = YOLO(model_spec)

        print("Starting training:")
        print(
            f"  data={args.data}\n"
            f"  epochs={args.epochs}\n"
            f"  batch_size={args.batch_size}\n"
            f"  img_size={args.img_size}\n"
            f"  project={args.output_dir}"
        )
        model.train(
            data=args.data,
            epochs=args.epochs,
            batch=args.batch_size,
            imgsz=args.img_size,
            project=args.output_dir,
            exist_ok=True
        )

        # 4) Find & log the best checkpoint
        last_run = sorted(os.listdir(args.output_dir))[-1]
        best_ckpt = os.path.join(args.output_dir, last_run, "weights", "best.pt")
        print(f"\n✅ Training complete. Best checkpoint: {best_ckpt}")
        mlflow.log_artifact(best_ckpt, artifact_path="checkpoints")
        print(f"MLflow run completed: run_id={run.info.run_id}")

if __name__ == "__main__":
    main()
