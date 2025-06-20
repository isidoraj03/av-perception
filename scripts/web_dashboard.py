#!/usr/bin/env python3
# scripts/web_dashboard.py

import time
import cv2
import numpy as np
import streamlit as st

from yolo_pipeline.io.data_streamer import DataStreamer
from yolo_pipeline.perception.detector import Detector
from yolo_pipeline.perception.fusion import FusionEngine
from yolo_pipeline.perception.tracker import SimpleTracker
from scripts.overlay import draw_overlay

st.set_page_config(page_title="Real-time Perception Dashboard", layout="wide")

# -- singletons for the heavy objects --
@st.cache_resource
def init_streamer(config_path="datasets/config.yaml"):
    ds = DataStreamer(config_path=config_path)
    ds.load_split("kitti", split="train", shuffle=False)
    ds.start()
    return ds

@st.cache_resource
def init_detector():
    det = Detector(model_path="yolov8n.pt", device="cpu")
    det.load_model()
    return det

@st.cache_resource
def init_fusion_engine():
    intrinsics = {"fx": 1.0, "fy": 1.0, "cx": 0.0, "cy": 0.0}
    extrinsics = {"R": np.eye(3), "T": [0, 0, 0]}
    return FusionEngine(intrinsics, extrinsics, min_points=3)

@st.cache_resource
def init_tracker():
    return SimpleTracker()

def main():
    st.title("🚀 Real-time Perception Dashboard")

    # Sidebar controls
    fusion_on = st.sidebar.checkbox("Enable LiDAR fusion", True)
    run_duration = st.sidebar.number_input("Run duration (s)", min_value=1.0, max_value=120.0, value=30.0)
    interval_ms  = st.sidebar.slider("Refresh interval (ms)", 50, 500, 100)

    # Initialize modules
    ds      = init_streamer()
    det     = init_detector()
    fus_eng = init_fusion_engine()
    tracker = init_tracker()

    # Placeholders for image + metrics
    img_ph    = st.empty()
    c1, c2, c3 = st.columns(3)
    m2d = c1.metric("2D detections", 0)
    m3d = c2.metric("3D fused",     0)
    mtr = c3.metric("Tracks",       0)

    start = time.time()
    while time.time() - start < run_duration:
        frame = ds.get_latest_camera_frame()
        pc    = ds.get_latest_pointcloud()
        if frame is None or pc is None:
            time.sleep(interval_ms / 1000.0)
            continue

        t0 = time.time()
        dets2d = det.predict(frame)

        if fusion_on:
            fused = fus_eng.fuse(dets2d, pc)
            tracks = tracker.update(fused)
        else:
            fused  = []
            tracks = []

        t1  = time.time()
        fps = 1.0 / (t1 - t0) if t1 > t0 else 0.0

        # annotate and show
        annotated = draw_overlay(frame, tracks, fps)
        # convert BGR → RGB for Streamlit
        annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        img_ph.image(annotated, use_container_width=True)

        # update metrics
        m2d.metric("2D detections", len(dets2d))
        m3d.metric("3D fused",     len(fused))
        mtr.metric("Tracks",       len(tracks))

        time.sleep(interval_ms / 1000.0)

    ds.stop()
    st.write("### ▶️ Streaming finished.")

if __name__ == "__main__":
    main()
