# tests/test_data_streamer.py

import time
import numpy as np
from yolo_pipeline.io.data_streamer import DataStreamer

def test_get_latest_camera_frame_is_array():
    ds = DataStreamer(config_path="datasets/config.yaml")
    ds.load_split("kitti", split="train", shuffle=False)

    # There must be at least one frame
    assert len(ds._frame_list) > 0

    ds.start()
    time.sleep(0.2)  # let the loader read at least one frame
    frame = ds.get_latest_camera_frame()
    ds.stop()

    # Check it’s a NumPy array of shape (H, W, 3)
    assert isinstance(frame, np.ndarray)
    assert frame.ndim == 3
    assert frame.shape[2] == 3




def test_get_latest_pointcloud_is_array():
    ds = DataStreamer(config_path="datasets/config.yaml")
    ds.load_split("kitti", split="train", shuffle=False)

    ds.start()
    time.sleep(0.2)
    pc = ds.get_latest_pointcloud()
    ds.stop()

    assert isinstance(pc, np.ndarray)
    assert pc.ndim == 2 and pc.shape[1] == 4
