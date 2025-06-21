#!/usr/bin/env python3
# scripts/export.py

import argparse
import os
import glob
import sys
import shutil

import onnx
from ultralytics import YOLO

def parse_args():
    parser = argparse.ArgumentParser(
        description="Export a YOLO model to ONNX and verify the result"
    )
    parser.add_argument(
        "--weights", "-w", type=str, required=True,
        help="Path to model weights file (e.g. yolov8n.pt or runs/train/.../best.pt)"
    )
    parser.add_argument(
        "--format", "-f", choices=["onnx"], required=True,
        help="Target export format (only 'onnx' is supported)"
    )
    parser.add_argument(
        "--dynamic", action="store_true",
        help="Enable dynamic input shapes (where supported)"
    )
    parser.add_argument(
        "--int8", action="store_true",
        help="Enable INT8 quantization (where supported)"
    )
    parser.add_argument(
        "--output-dir", "-o", type=str, default="exports",
        help="Directory to save exported models"
    )
    return parser.parse_args()


def _find_and_flatten(pattern: str, output_dir: str, ext: str) -> str | None:
    """
    Search recursively under output_dir for any file matching ext,
    copy the first one into the root of output_dir, and return its path.
    """
    matches = glob.glob(pattern, recursive=True)
    if not matches:
        return None
    src = matches[0]
    dst = os.path.join(output_dir, os.path.basename(src))
    if os.path.abspath(src) != os.path.abspath(dst):
        shutil.copy(src, dst)
    return dst


def export_to_onnx(model: YOLO, weights: str, dynamic: bool, int8: bool, output_dir: str):
    print("→ Exporting to ONNX…")
    stem = os.path.splitext(os.path.basename(weights))[0]
    os.makedirs(output_dir, exist_ok=True)

    # If you already have stem.onnx locally, just copy + check
    local = f"{stem}.onnx"
    dst = os.path.join(output_dir, local)
    if os.path.isfile(local):
        print(f"→ Found local ONNX '{local}', copying…")
        shutil.copy(local, dst)
    else:
        model.export(
            format="onnx",
            dynamic=dynamic,
            int8=int8,
            project=output_dir,
            name=stem
        )

    # look for the .onnx under any subfolder
    pattern = os.path.join(output_dir, "**", f"{stem}.onnx")
    final = _find_and_flatten(pattern, output_dir, ".onnx")
    if final is None:
        print(f"✖ No ONNX file found under '{output_dir}'", file=sys.stderr)
        sys.exit(1)

    print(f"→ ONNX model saved to '{final}'")
    print("→ Running ONNX checker…")
    m = onnx.load(final)
    onnx.checker.check_model(m)
    print("✅ ONNX model passed basic check")


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"→ Loading YOLO model from {args.weights!r}…")
    model = YOLO(args.weights)

    # Only ONNX is supported
    export_to_onnx(model, args.weights, args.dynamic, args.int8, args.output_dir)

    print(f"✅ Export complete. Check files under '{args.output_dir}'")
