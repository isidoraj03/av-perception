#!/usr/bin/env python3
# scripts/train.py

import argparse
import os
import mlflow
import mlflow.pytorch
from ultralytics import YOLO, settings

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a YOLO v8 model on your dataset"
    )
    parser.add_argument(
        "--model", type=str, default="yolov8n",
        help=(
            "YOLO v8 model to use: one of 'yolov8n','yolov8s','yolov8m',"
            "'yolov8l', (with or without '.pt'), or a filesystem path"
        )
    )
    parser.add_argument(
        "--data", type=str, required=True,
        help="Path to dataset YAML (e.g. coco128.yaml or datasets/kitti50.yaml)"
    )
    parser.add_argument("--epochs",    type=int, default=50, help="Number of epochs")
    parser.add_argument("--batch-size",type=int, default=16, help="Batch size")
    parser.add_argument("--img-size",  type=int, default=640, help="Input image size")
    parser.add_argument(
        "--output-dir", type=str, default="runs/train",
        help="Directory to save training runs"
    )
    return parser.parse_args()

def resolve_model(model_arg: str) -> str:
    """
    If the arg is exactly 'yolov8n'|...|'yolov8x', append '.pt' so Ultralytics
    can download it. Otherwise return it unchanged.
    """
    base = os.path.basename(model_arg)
    if base in {"yolov8n","yolov8s","yolov8m","yolov8l"}:
        return base + ".pt"
    return model_arg

def main():
    args = parse_args()

    # 0) MLflow setup
    mlflow.set_experiment("yolo_training")
    mlflow.pytorch.autolog()
    settings.update(mlflow=False)

    # 1) Choose actual model spec
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
        print(f"Loading YOLO v8 model from {model_spec}…")
        model = YOLO(model_spec)

        print("Starting training:")
        print(
            f"  data={args.data}\n"
            f"  epochs={args.epochs}\n"
            f"  batch_size={args.batch_size}\n"
            f"  img_size={args.img_size}\n"
            f"  output_dir={args.output_dir}"
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
