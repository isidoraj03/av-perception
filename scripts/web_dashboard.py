#!/usr/bin/env python3
import time
import cv2
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

from yolo_pipeline.io.data_streamer import DataStreamer
from yolo_pipeline.perception.detector import Detector
from yolo_pipeline.perception.fusion import FusionEngine
from yolo_pipeline.perception.tracker import SimpleTracker
from scripts.overlay import draw_overlay

st.set_page_config(page_title="Real-time Perception Dashboard", layout="wide")

# --- Sidebar controls and sequence selector ---
SEQUENCES = {
    "KITTI Clear-Day":        "kitti",
    "nuScenes (mini v1.0)":   "nuscenes",
    "nuScenes (full v1.0)":   "nuscenes-full",
    "nuScenes (night)":       "nuscenes-full-night",
    "nuScenes (rain)":        "nuscenes-full-rain",
    "nuScenes (combined)":    "nuscenes-full-adverse",
}
selected     = st.sidebar.selectbox("Select Sequence", list(SEQUENCES.keys()))
DATASET_NAME = SEQUENCES[selected]

fusion_on    = st.sidebar.checkbox("Enable LiDAR Fusion", True)
run_duration = st.sidebar.number_input("Run Duration (s)", min_value=1.0, max_value=120.0, value=30.0)
interval_ms  = st.sidebar.slider("Refresh Interval (ms)", 50, 500, 100)

@st.cache_resource
def init_streamer(config_path: str, dataset: str):
    ds = DataStreamer(config_path=config_path)
    ds.load_split(dataset, split="train", shuffle=False)
    ds.start()
    return ds

@st.cache_resource
def init_detector(dataset_name: str):
    # choose different weights depending on dataset
    if dataset_name == "kitti":
        model_path = "runs/train_quick_cpu/train/weights/best.pt"
    else:
        model_path = "runs/train_nuscenes_sub/train/weights/best.pt"
    det = Detector(model_path=model_path, device="cpu")
    det.load_model()
    return det

@st.cache_resource
def init_fusion_engine():
    intrinsics = {"fx":1.0, "fy":1.0, "cx":0.0, "cy":0.0}
    extrinsics = {"R": np.eye(3), "T":[0,0,0]}
    return FusionEngine(intrinsics, extrinsics, min_points=3)

@st.cache_resource
def init_tracker():
    return SimpleTracker()

# initialize components
ds      = init_streamer("datasets/config.yaml", DATASET_NAME)
det     = init_detector(DATASET_NAME)
fus_eng = init_fusion_engine()
tracker = init_tracker()

st.title("🚀 Real-Time Perception Dashboard")
c1, c2, c3 = st.columns(3)
img_ph     = st.empty()
m2d        = c1.metric("2D Detections", 0)
m3d        = c2.metric("3D Fused",      0)
mtr        = c3.metric("Tracks",        0)
lidar_ph   = st.empty()

start_time = time.time()
while time.time() - start_time < run_duration:
    frame = ds.get_latest_camera_frame()
    pc    = ds.get_latest_pointcloud()
    if frame is None or pc is None:
        time.sleep(interval_ms / 1000.0)
        continue

    t0 = time.time()
    dets2d = det.predict(frame)
    if fusion_on:
        fused  = fus_eng.fuse(dets2d, pc)
        tracks = tracker.update(fused)
    else:
        fused, tracks = [], []
    t1 = time.time()
    fps = 1.0 / (t1 - t0) if (t1 - t0) > 0 else 0.0

    annotated = draw_overlay(frame, tracks, fps)
    annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    img_ph.image(annotated, use_container_width=True)

    m2d.metric("2D Detections", len(dets2d))
    m3d.metric("3D Fused",      len(fused))
    mtr.metric("Tracks",        len(tracks))

    if fusion_on and pc.size and pc.shape[0] > 0:
        xs, ys = pc[:,0], pc[:,1]
        fig, ax = plt.subplots()
        ax.scatter(xs, ys, s=1)
        ax.set_aspect("equal")
        ax.set_title("LiDAR Top-Down View")
        lidar_ph.pyplot(fig)
        plt.close(fig)
    else:
        lidar_ph.empty()

    time.sleep(interval_ms / 1000.0)

ds.stop()
st.write("### ▶️ Streaming finished.")
