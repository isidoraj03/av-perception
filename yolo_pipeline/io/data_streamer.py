# File: yolo_pipeline/io/data_streamer.py

from __future__ import annotations
import os
import threading
import time
from pathlib import Path
from typing import Optional, List, Dict, Any

import numpy as np
import yaml
from PIL import Image


class DataStreamer:
    """
    Plays back KITTI or nuScenes as a live RGB-camera + LiDAR stream.

    After calling `start()`, the latest frame and point-cloud are available via
    `get_latest_camera_frame()` and `get_latest_pointcloud()`.
    """

    def __init__(self, config_path: str = "datasets/config.yaml"):
        # Save config directory for resolving relative paths
        self._config_dir = os.path.dirname(config_path)
        # Load YAML config
        with open(config_path, "r", encoding="utf-8") as f:
            self.cfg = yaml.safe_load(f)

        # State
        self.current_dataset: Optional[str] = None
        self.current_split:   Optional[str] = None
        self.shuffle:         bool          = False

        self._frame_list: List[str] = []
        self._pc_list:    List[str] = []
        self._idx:        int       = 0

        self._latest_camera: Optional[np.ndarray] = None
        self._latest_lidar:  Optional[np.ndarray] = None

        self._stop_evt = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def load_split(
        self,
        dataset_name: str,
        split: str = "train",
        shuffle: bool = False
    ) -> None:
        """
        Build internal lists of (image, lidar) pairs for the requested dataset.

        Supported dataset_name values:
          - "kitti"          : KITTI raw
          - "nuscenes"       : nuScenes v1.0-mini
          - "nuscenes-full"  : nuScenes v1.0-trainval / v1.0-test (unfiltered)
          - any other config key that has a `visibility_filters` list:
            treated as a filtered nuScenes-full subset.
        """
        self.current_dataset = dataset_name
        self.current_split   = split
        self.shuffle         = shuffle

        # -- 1) nuScenes-mini ----------------------------
        if dataset_name.lower() == "nuscenes":
            from nuscenes.nuscenes import NuScenes

            root_cfg = self.cfg["nuscenes"]["root"]
            root_dir = Path(root_cfg) if Path(root_cfg).is_absolute() \
                       else Path(self._config_dir) / root_cfg

            nusc = NuScenes(
                version="v1.0-mini",
                dataroot=str(root_dir),
                verbose=False
            )

            self._build_nuscenes_lists(nusc, root_dir,
                cam_channel="CAM_FRONT",
                lidar_channel="LIDAR_TOP"
            )
            return

        # -- 2) filtered nuScenes-full subsets ------------
        # any config entry with `visibility_filters`
        cfg_entry: Dict[str, Any] = self.cfg.get(dataset_name, {})
        if isinstance(cfg_entry, dict) and "visibility_filters" in cfg_entry:
            from nuscenes.nuscenes import NuScenes

            version = "v1.0-test" if split == "test" else "v1.0-trainval"
            root_cfg = cfg_entry["root"]
            root_dir = Path(root_cfg) if Path(root_cfg).is_absolute() \
                       else Path(self._config_dir) / root_cfg

            nusc = NuScenes(version=version, dataroot=str(root_dir), verbose=False)

            filters = [f.lower() for f in cfg_entry["visibility_filters"]]
            self._frame_list = []
            self._pc_list    = []
            for sample in nusc.sample:
                sd_cam   = nusc.get("sample_data", sample["data"]["CAM_FRONT"])
                sd_lidar = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])

                # Check the scene's description for any of our tags
                scene = nusc.get("scene", sample["scene_token"])
                desc  = scene["description"].lower()
                if not any(tag in desc for tag in filters):
                    continue

                self._frame_list.append(root_dir / sd_cam["filename"])
                self._pc_list.append(  root_dir / sd_lidar["filename"])

            self._filter_missing()
            self._idx = 0
            return

        # -- 3) unfiltered nuScenes-full -----------------
        if dataset_name.lower() == "nuscenes-full":
            from nuscenes.nuscenes import NuScenes

            version = "v1.0-test" if split == "test" else "v1.0-trainval"
            root_cfg = self.cfg["nuscenes-full"]["root"]
            root_dir = Path(root_cfg) if Path(root_cfg).is_absolute() \
                       else Path(self._config_dir) / root_cfg

            nusc = NuScenes(version=version, dataroot=str(root_dir), verbose=False)

            self._build_nuscenes_lists(nusc, root_dir,
                cam_channel="CAM_FRONT",
                lidar_channel="LIDAR_TOP"
            )
            return

        # -- 4) KITTI -------------------------------------
        if dataset_name not in self.cfg:
            raise ValueError(f"Unknown dataset '{dataset_name}'")

        split_cfg = self.cfg[dataset_name]["splits"].get(split)
        if split_cfg is None:
            raise ValueError(f"Split '{split}' not configured for {dataset_name}")

        # resolve image / lidar splits
        if isinstance(split_cfg, str):
            img_split, lidar_split = split_cfg, split_cfg
        else:
            img_split   = split_cfg["images"]
            lidar_split = split_cfg["lidar"]

        root_cfg = self.cfg[dataset_name]["root"]
        root_dir = Path(root_cfg) if Path(root_cfg).is_absolute() \
                   else Path(self._config_dir) / root_cfg

        img_dir = root_dir / img_split / "image_2"
        if not img_dir.is_dir():
            img_dir = root_dir / img_split
            if not img_dir.is_dir():
                raise FileNotFoundError(f"Cannot find image dir under {img_dir}")

        pc_dir = root_dir / lidar_split / "velodyne"
        if not pc_dir.is_dir():
            raise FileNotFoundError(f"Cannot find LiDAR dir under {pc_dir}")

        img_files = sorted(f for f in img_dir.iterdir() if f.suffix.lower() == ".png")
        if not img_files:
            raise RuntimeError(f"No *.png images found in {img_dir}")

        self._frame_list = [str(f) for f in img_files]
        self._pc_list    = []
        for img_path in img_files:
            pc_path = pc_dir / f"{img_path.stem}.bin"
            if not pc_path.is_file():
                raise FileNotFoundError(f"Missing point-cloud for {img_path.name}: {pc_path}")
            self._pc_list.append(str(pc_path))

        self._idx = 0

    def _build_nuscenes_lists(
        self,
        nusc,
        root_dir: Path,
        cam_channel: str,
        lidar_channel: str
    ) -> None:
        """
        Populate _frame_list and _pc_list for a nuScenes object without filtering.
        """
        self._frame_list = []
        self._pc_list    = []
        for sample in nusc.sample:
            sd_cam   = nusc.get("sample_data", sample["data"][cam_channel])
            sd_lidar = nusc.get("sample_data", sample["data"][lidar_channel])
            self._frame_list.append(root_dir / sd_cam["filename"])
            self._pc_list.append(  root_dir / sd_lidar["filename"])

        self._filter_missing()
        self._idx = 0

    def _filter_missing(self) -> None:
        """
        Keep only (image, lidar) pairs where both files exist on disk.
        """
        pairs = [
            (str(cam), str(lidar))
            for cam, lidar in zip(self._frame_list, self._pc_list)
            if Path(cam).is_file() and Path(lidar).is_file()
        ]
        if not pairs:
            raise RuntimeError(
                "No usable nuScenes samples found – did you download the "
                "‘samples/…/CAM_FRONT’ and ‘sweeps/LIDAR_TOP’ files?"
            )
        self._frame_list, self._pc_list = map(list, zip(*pairs))

    def _load_once(self) -> None:
        """Load the very first frame and pointcloud synchronously."""
        with Image.open(self._frame_list[self._idx]) as img:
            self._latest_camera = np.array(img.convert("RGB"))
        pc = np.fromfile(self._pc_list[self._idx], dtype=np.float32)
        self._latest_lidar = pc.reshape(-1, 4)

    def _loop(self) -> None:
        """Background thread that reads the next frame every 100 ms."""
        while (not self._stop_evt.is_set()) and (self._idx < len(self._frame_list)):
            with Image.open(self._frame_list[self._idx]) as img:
                self._latest_camera = np.array(img.convert("RGB"))
            pc = np.fromfile(self._pc_list[self._idx], dtype=np.float32)
            self._latest_lidar = pc.reshape(-1, 4)
            self._idx += 1
            time.sleep(0.1)

    def start(self) -> None:
        """Start the streaming thread (idempotent)."""
        if self._thread and self._thread.is_alive():
            return
        self._stop_evt.clear()
        self._idx = 0
        if self._frame_list:
            self._load_once()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop streaming thread and wait for it to finish."""
        if not self._thread:
            return
        self._stop_evt.set()
        self._thread.join()
        self._thread = None

    def get_latest_camera_frame(self) -> Optional[np.ndarray]:
        """Return the most recent RGB frame, or None if not yet loaded."""
        return self._latest_camera

    def get_latest_pointcloud(self) -> Optional[np.ndarray]:
        """Return the most recent LiDAR scan (N×4), or None if not yet loaded."""
        return self._latest_lidar
