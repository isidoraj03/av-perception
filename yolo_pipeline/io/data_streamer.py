# yolo_pipeline/io/data_streamer.py

import os
import threading
import time
from typing import Optional

import yaml
import numpy as np
from PIL import Image


class DataStreamer:
    """
    Plays back KITTI as if it were a live sensor (RGB + LiDAR).
    """

    def __init__(self, config_path: str = "datasets/config.yaml"):
        # Remember where config.yaml lives
        self._config_dir = os.path.dirname(config_path)

        # Load YAML (UTF-8)
        with open(config_path, "r", encoding="utf-8") as f:
            self.cfg = yaml.safe_load(f)

        self.current_dataset: Optional[str] = None
        self.current_split:   Optional[str] = None
        self.shuffle:         bool          = False

        # Lists of image paths and LiDAR paths
        self._frame_list: list[str] = []
        self._pc_list:    list[str] = []
        self._idx:        int       = 0

        self._latest_camera: Optional[np.ndarray] = None
        self._latest_lidar:  Optional[np.ndarray] = None

        self._stop_evt = threading.Event()
        self._thread   = None

    def load_split(self, dataset_name: str, split: str = "train", shuffle: bool = False):
        """
        Select dataset & split. Builds lists of image and LiDAR file paths.
        """
        if dataset_name not in self.cfg:
            raise ValueError(f"Unknown dataset '{dataset_name}'")
        split_cfg = self.cfg[dataset_name]["splits"].get(split)
        if split_cfg is None:
            raise ValueError(f"Split '{split}' not in {dataset_name}")

        # Parse image vs lidar subpaths
        if isinstance(split_cfg, str):
            img_split = split_cfg
            lidar_split = split_cfg
        else:
            img_split   = split_cfg["images"]
            lidar_split = split_cfg["lidar"]

        self.current_dataset = dataset_name
        self.current_split   = split
        self.shuffle         = shuffle

        # Resolve root_dir (absolute or relative to config)
        cfg_root = self.cfg[dataset_name]["root"]
        root_dir = cfg_root if os.path.isabs(cfg_root) else os.path.join(self._config_dir, cfg_root)

        # 1) Build image directory (images are under <...>/image_2/)
        img_dir = os.path.join(root_dir, img_split, "image_2")
        if not os.path.isdir(img_dir):
            # fallback if you extracted differently
            img_dir = os.path.join(root_dir, img_split)
            if not os.path.isdir(img_dir):
                raise FileNotFoundError(f"Could not find image directory under {root_dir}/{img_split}")

        # 2) Build LiDAR directory (point clouds under <...>/velodyne/)
        pc_dir = os.path.join(root_dir, lidar_split, "velodyne")
        if not os.path.isdir(pc_dir):
            raise FileNotFoundError(f"Could not find LiDAR directory under {root_dir}/{lidar_split}/velodyne")

        # Gather and sort image files
        img_files = sorted([f for f in os.listdir(img_dir) if f.lower().endswith(".png")])
        if not img_files:
            raise RuntimeError(f"No PNGs in {img_dir}")

        # Build full image paths
        self._frame_list = [os.path.join(img_dir, f) for f in img_files]

        # Build matching LiDAR paths
        self._pc_list = []
        for img_path in self._frame_list:
            base = os.path.basename(img_path).replace(".png", ".bin")
            pc_path = os.path.join(pc_dir, base)
            if not os.path.isfile(pc_path):
                raise FileNotFoundError(f"Missing .bin for {img_path}: {pc_path}")
            self._pc_list.append(pc_path)

        self._idx = 0

    def _load_once(self):
        """
        Preload the very first frame (frame 0) synchronously.
        """
        # Load camera frame
        with Image.open(self._frame_list[self._idx]) as img:
            self._latest_camera = np.array(img.convert("RGB"))

        # Load LiDAR point cloud
        pc = np.fromfile(self._pc_list[self._idx], dtype=np.float32)
        self._latest_lidar = pc.reshape(-1, 4)

    def _loop(self):
        """
        Background loop: reads each image + LiDAR scan in sequence at ~10 Hz.
        """
        while not self._stop_evt.is_set() and self._idx < len(self._frame_list):
            # Load camera frame
            with Image.open(self._frame_list[self._idx]) as img:
                self._latest_camera = np.array(img.convert("RGB"))

            # Load LiDAR point cloud
            pc = np.fromfile(self._pc_list[self._idx], dtype=np.float32)
            self._latest_lidar = pc.reshape(-1, 4)

            self._idx += 1
            time.sleep(0.1)

    def start(self):
        """Spawn the streaming thread (no-op if already running)."""
        if self._thread and self._thread.is_alive():
            return
        self._stop_evt.clear()
        self._idx = 0

        # Preload frame 0 so get_latest_*() is non-None immediately
        if self._frame_list:
            self._load_once()

        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        """Signal thread to stop and wait for it."""
        if not self._thread:
            return
        self._stop_evt.set()
        self._thread.join()
        self._thread = None

    def get_latest_camera_frame(self) -> Optional[np.ndarray]:
        """Return the most recent camera frame (H×W×3) or None."""
        return self._latest_camera

    def get_latest_pointcloud(self) -> Optional[np.ndarray]:
        """Return the most recent LiDAR scan (N×4) or None."""
        return self._latest_lidar
