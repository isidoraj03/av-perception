import time
import numpy as np
import pytest
from yolo_pipeline.io.data_streamer import DataStreamer

def test_nuscenes_streamer_loads_frames_and_pointclouds():
    ds = DataStreamer(config_path="datasets/config.yaml")
    ds.load_split("nuscenes", split="train", shuffle=False)

    # must find at least one sample
    assert len(ds._frame_list) > 0
    assert len(ds._pc_list) > 0

    ds.start()
    time.sleep(0.2)  # wait for first frame
    frame = ds.get_latest_camera_frame()
    pc    = ds.get_latest_pointcloud()
    ds.stop()

    # validate types and shapes
    assert isinstance(frame, np.ndarray)
    assert frame.ndim == 3 and frame.shape[2] == 3

    assert isinstance(pc, np.ndarray)
    assert pc.ndim == 2 and pc.shape[1] == 4
