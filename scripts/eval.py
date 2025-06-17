#!/usr/bin/env python3
# scripts/eval.py

import argparse
import os
import time
import yaml
import glob
from ultralytics import YOLO

def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a YOLO model and benchmark inference speed"
    )
    parser.add_argument(
        "--weights", type=str, required=True,
        help="Path to model weights (e.g. yolov11n.pt or runs/train/.../best.pt)"
    )
    parser.add_argument(
        "--data", type=str, required=True,
        help="Path to dataset YAML (must have a 'val:' key)"
    )
    parser.add_argument(
        "--device", type=str, default="cpu",
        help="Device to run on (e.g. 'cpu' or '0' for GPU 0)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=16,
        help="Batch size for the validation step"
    )
    parser.add_argument(
        "--output-dir", type=str, default="runs/val",
        help="Directory to save any validation outputs/logs"
    )
    return parser.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"→ Loading model from {args.weights!r} on device {args.device!r}…")
    model = YOLO(args.weights)
    model.to(args.device)

    # 1) run val() to compute mAP
    print("→ Running validation to compute mAP…")
    t0 = time.time()
    val_res = model.val(
        data=args.data,
        batch=args.batch_size,
        device=args.device,
        project=args.output_dir,
        exist_ok=True
    )
    t_val = time.time() - t0
    print(f"  Validation time: {t_val:.2f}s")

    #  Extract the metrics dictionary from DetMetrics
    if hasattr(val_res, "results_dict"):
        metrics = val_res.results_dict
    elif isinstance(val_res, (list, tuple)) and hasattr(val_res[0], "metrics"):
        metrics = val_res[0].metrics
    else:
        metrics = None

    # Print the first metric containing "map", with a cleaned-up name
    if isinstance(metrics, dict):
        for key, val in metrics.items():
            if "map" in key.lower():
                # strip any prefix and suffix, e.g. "metrics/mAP50-95(B)" → "mAP50-95"
                name = key.split("/")[-1].split("(")[0]
                print(f"  ▶ {name} = {val:.3f}")
                break
        else:
            print(f"  ▶ no 'map' key found; available metrics: {list(metrics.keys())}")
    else:
        print("  ▶ no metrics dict returned; inspect `val_res` for available fields")

    # 2) benchmark inference FPS on val images
    print("→ Measuring inference speed on validation set images…")
    with open(args.data, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    val_path = cfg.get("val")
    if not val_path or not os.path.isdir(val_path):
        print(f"  ✖ 'val:' path {val_path!r} not found, skipping FPS measurement.")
        return

    img_files = []
    for ext in ("*.png", "*.jpg", "*.jpeg"):
        img_files.extend(glob.glob(os.path.join(val_path, ext)))
    img_files = sorted(img_files)
    if not img_files:
        print(f"  ✖ No images under {val_path!r}, skipping FPS measurement.")
        return

    n = len(img_files)
    t0 = time.time()
    for img in img_files:
        _ = model.predict(source=img, device=args.device, batch=1, verbose=False)
    t_inf = time.time() - t0
    fps = n / t_inf if t_inf > 0 else float("inf")
    print(f"  Ran inference on {n} images in {t_inf:.2f}s → {fps:.2f} FPS")

if __name__ == "__main__":
    main()
