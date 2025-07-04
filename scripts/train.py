#!/usr/bin/env python3
# scripts/train.py

import argparse
import os
import re
import multiprocessing
import mlflow
import mlflow.pytorch
import torch
from ultralytics import YOLO, settings

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a YOLO model (v8, v11, etc.) on your dataset"
    )
    parser.add_argument(
        "--model", type=str, default="yolov8n",
        help="YOLO model to use: e.g. 'yolov8n','yolov11n' or path to .pt'"
    )
    parser.add_argument(
        "--data", type=str, required=True,
        help="Path to dataset YAML (e.g. coco128.yaml or datasets/kitti_full.yaml)"
    )
    parser.add_argument(
        "--epochs", type=int, default=50,
        help="Number of epochs"
    )
    parser.add_argument(
        "--batch-size", type=int, default=16,
        help="Batch size"
    )
    parser.add_argument(
        "--img-size", type=int, default=640,
        help="Input image size"
    )
    parser.add_argument(
        "--device", type=str, default="cpu",
        help="Device to train on (e.g. 'cpu' or 'cuda:0')"
    )
    parser.add_argument(
        "--cache", action="store_true",
        help="Cache images & labels in RAM for faster epoch loops"
    )
    parser.add_argument(
        "--workers", type=int, default=multiprocessing.cpu_count(),
        help="Number of DataLoader workers"
    )
    parser.add_argument(
        "--half", action="store_true",
        help="Use mixed precision (fp16). Not usually beneficial on CPU"
    )
    parser.add_argument(
        "--freeze", type=int, default=0,
        help="Freeze first N layers during training"
    )
    parser.add_argument(
        "--mosaic", type=float, default=1.0,
        help="Mosaic augmentation probability (0 to disable)"
    )
    parser.add_argument(
        "--mixup", type=float, default=0.0,
        help="MixUp augmentation probability (0 to disable)"
    )
    parser.add_argument(
        "--copy-paste", type=float, default=0.0, dest="copy_paste",
        help="Copy-Paste augmentation probability (0 to disable)"
    )
    parser.add_argument(
        "--auto-augment", type=str, default="none",
        help="Auto-augment strategy ('none', 'multiscale', 'randaugment', etc.)"
    )
    parser.add_argument(
        "--output-dir", type=str, default="runs/train",
        help="Directory to save training runs"
    )
    return parser.parse_args()

def resolve_model(model_arg: str) -> str:
    base, ext = os.path.splitext(os.path.basename(model_arg))
    if re.fullmatch(r"yolo\d+(n|s|m|l|x)", base):
        return base + ".pt"
    return model_arg

def main():
    args = parse_args()

    # maximize CPU threading
    n = multiprocessing.cpu_count()
    os.environ["OMP_NUM_THREADS"] = str(n)
    os.environ["MKL_NUM_THREADS"] = str(n)
    torch.set_num_threads(n)
    torch.set_num_interop_threads(n)

    # MLflow setup (fail safe)
    mlflow.set_experiment("yolo_training")
    try:
        mlflow.pytorch.autolog()
    except Exception:
        print("⚠️  MLflow autolog failed; continuing without it")
    settings.update(mlflow=False)

    model_spec = resolve_model(args.model)
    with mlflow.start_run() as run:
        mlflow.log_params(vars(args))

        print(f"Loading YOLO model from {model_spec}…")
        model = YOLO(model_spec)

        print("Starting training with settings:")
        print(f"  data={args.data}")
        print(f"  epochs={args.epochs}, batch={args.batch_size}, imgsz={args.img_size}")
        print(f"  device={args.device}, cache={args.cache}, workers={args.workers}")
        print(f"  half={args.half}, freeze={args.freeze}")
        print(f"  mosaic={args.mosaic}, mixup={args.mixup}, copy_paste={args.copy_paste}")
        print(f"  auto_augment={args.auto_augment}")
        print(f"  project={args.output_dir}")

        model.train(
            data=args.data,
            epochs=args.epochs,
            batch=args.batch_size,
            imgsz=args.img_size,
            device=args.device,
            cache=args.cache,
            workers=args.workers,
            half=args.half,
            freeze=args.freeze,
            mosaic=args.mosaic,
            mixup=args.mixup,
            copy_paste=args.copy_paste,
            auto_augment=args.auto_augment,
            project=args.output_dir,
            exist_ok=True
        )

        # find & log best checkpoint
        last = sorted(os.listdir(args.output_dir))[-1]
        best_ckpt = os.path.join(args.output_dir, last, "weights", "best.pt")
        print(f"\n✅ Training complete. Best checkpoint:\n    {best_ckpt}")
        mlflow.log_artifact(best_ckpt, artifact_path="checkpoints")
        print(f"MLflow run completed: run_id={run.info.run_id}")

if __name__ == "__main__":
    main()
