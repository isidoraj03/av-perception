#!/usr/bin/env python3
# scripts/benchmark_official.py

import argparse
import subprocess
import sys
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run eval.py on official YOLO models and record their summaries."
    )
    parser.add_argument(
        "--data", type=str, required=True,
        help="Path to dataset YAML (e.g. datasets/kitti50.yaml)"
    )
    parser.add_argument(
        "--models", nargs="+", default=["yolov8s", "yolo11s"],
        help="List of model names (without .pt) to benchmark"
    )
    parser.add_argument(
        "--batch-size", type=int, default=4,
        help="Batch size to use for inference"
    )
    parser.add_argument(
        "--output-base", type=str, default="runs",
        help="Base directory under which to create eval_official_<model>_<device> folders"
    )
    return parser.parse_args()

def run_eval(model: str, device: str, data: str, batch: int, outdir: Path):
    cmd = [
        sys.executable, "scripts/eval.py",
        "--weights", f"{model}.pt",
        "--data", data,
        "--device", device,
        "--batch-size", str(batch),
        "--output-dir", str(outdir)
    ]
    print("->", " ".join(cmd))
    subprocess.run(cmd, check=True)

def main():
    args = parse_args()
    base = Path(args.output_base)
    for model in args.models:
        # CPU run
        cpu_out = base / f"eval_official_{model}_cpu"
        cpu_out.mkdir(parents=True, exist_ok=True)
        run_eval(model, "cpu", args.data, args.batch_size, cpu_out)

        # GPU run if available
        try:
            import torch
            cuda_ok = torch.cuda.is_available()
        except ImportError:
            cuda_ok = False

        if cuda_ok:
            gpu_out = base / f"eval_official_{model}_gpu"
            gpu_out.mkdir(parents=True, exist_ok=True)
            run_eval(model, "cuda:0", args.data, args.batch_size, gpu_out)
        else:
            print(f"WARNING: Skipping GPU benchmark for {model} (no CUDA found)")

if __name__ == "__main__":
    main()
