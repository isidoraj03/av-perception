# tests/test_tracker.py

import os
import yaml
import numpy as np
from PIL import Image
from pathlib import Path

from yolo_pipeline.perception.detector import Detector
from yolo_pipeline.perception.fusion import FusionEngine
from yolo_pipeline.perception.tracker import SimpleTracker

def offline_tracker_run(num_frames: int = 5):
    """
    Load num_frames KITTI train frames + pointclouds,
    run detection→fusion→tracking and print a summary.
    """
    # 1) locate config and data folders
    repo_root = Path(__file__).parent.parent
    cfg_path  = repo_root / "datasets" / "config.yaml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    kf        = cfg["kitti"]
    root_cfg  = kf["root"]
    root_dir  = Path(root_cfg) if os.path.isabs(root_cfg) else (cfg_path.parent / root_cfg)
    split     = kf["splits"]["train"]
    img_sub   = split["images"] if isinstance(split, dict) else split
    lidar_sub = split["lidar"]  if isinstance(split, dict) else split

    img_dir = root_dir / img_sub / "image_2"
    pc_dir  = root_dir / lidar_sub / "velodyne"

    img_files = sorted(img_dir.glob("*.png"))[:num_frames]
    if not img_files:
        raise RuntimeError(f"No frames found under {img_dir}")

    # 2) init detector, fusion, tracker
    det     = Detector(model_path="yolov8n.pt", device="cpu")
    det.load_model()
    intr    = {"fx": 1.0, "fy": 1.0, "cx": 0.0, "cy": 0.0}
    extr    = {"R": np.eye(3), "T": [0, 0, 0]}
    fus     = FusionEngine(intr, extr, min_points=3)
    tracker = SimpleTracker()

    history: dict[int, list[int]] = {}

    # 3) process each frame
    for idx, img_path in enumerate(img_files):
        pc_path = pc_dir / f"{img_path.stem}.bin"
        # load data
        img = np.array(Image.open(img_path).convert("RGB"))
        pc  = np.fromfile(str(pc_path), dtype=np.float32).reshape(-1, 4)

        det2d  = det.predict(img)
        fused  = fus.fuse(det2d, pc)
        tracks = tracker.update(fused)

        print(f"Frame {idx:02d}: {len(tracks)} tracks")
        for t in tracks:
            print(f"  ID={t.track_id:03d} cls={t.class_id} pts={t.num_points} depth={t.depth:.2f}")
            history.setdefault(t.track_id, []).append(idx)
        print()

    # 4) summary
    print("=== Track ID appearances ===")
    for tid, frames in sorted(history.items()):
        print(f"  ID {tid:03d} → frames {frames}")

def test_tracker_offline_smoke(capfd):
    # run with a small number of frames for speed
    offline_tracker_run(num_frames=3)
    captured = capfd.readouterr()
    # should have printed at least Frame 00 and the summary header
    assert "Frame 00" in captured.out
    assert "=== Track ID appearances ===" in captured.out
