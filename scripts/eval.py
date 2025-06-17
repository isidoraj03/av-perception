#!/usr/bin/env python3
# scripts/eval.py

import argparse
import os
import time
import yaml
import glob
import csv
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

    # 1) run val() to compute mAP, precision, recall
    print("→ Running validation to compute mAP, precision, recall…")
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

    # extract metrics dict (new API: DetMetrics.results_dict)
    if hasattr(val_res, "results_dict"):
        metrics = val_res.results_dict
    elif isinstance(val_res, (list, tuple)) and hasattr(val_res[0], "metrics"):
        metrics = val_res[0].metrics
    else:
        metrics = {}

    # pick out precision, recall, first “map” key
    p = next((v for k, v in metrics.items() if "precision" in k.lower()), None)
    r = next((v for k, v in metrics.items() if "recall"    in k.lower()), None)
    mp = next((v for k, v in metrics.items() if "map"       in k.lower()), None)

    if p is not None:
        print(f"  ▶ Precision = {p:.3f}")
    if r is not None:
        print(f"  ▶ Recall    = {r:.3f}")
    if mp is not None:
        print(f"  ▶ mAP       = {mp:.3f}")

    # 2) benchmark inference FPS on val images
    print("→ Measuring inference speed on validation set images…")
    with open(args.data, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    val_path = cfg.get("val")
    img_files = []
    for ext in ("*.png", "*.jpg", "*.jpeg"):
        img_files.extend(glob.glob(os.path.join(val_path or "", ext)))
    img_files = sorted(img_files)
    n = len(img_files)
    if n == 0:
        print(f"  ✖ No images under {val_path!r}, skipping FPS measurement.")
        fps = 0.0
    else:
        t0 = time.time()
        for img in img_files:
            _ = model.predict(source=img, device=args.device, batch=1, verbose=False)
        t_inf = time.time() - t0
        fps = n / t_inf if t_inf > 0 else float("inf")
        print(f"  Ran inference on {n} images in {t_inf:.2f}s → {fps:.2f} FPS")

    # 3) save summary.csv
    summary_rows = [("metric", "value")]
    if p is not None:  summary_rows.append(("precision", f"{p:.3f}"))
    if r is not None:  summary_rows.append(("recall",    f"{r:.3f}"))
    if mp is not None: summary_rows.append(("map",       f"{mp:.3f}"))
    summary_rows.append(("fps",       f"{fps:.2f}"))

    csv_path = os.path.join(args.output_dir, "summary.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(summary_rows)
    print(f"▶ Summary saved to {csv_path}")

if __name__ == "__main__":
    main()
