#!/usr/bin/env python3
# scripts/overlay.py

import cv2
import numpy as np
from typing import List
from yolo_pipeline.perception.tracker import Detection3D

def draw_overlay(frame: np.ndarray, tracks: List[Detection3D], fps: float) -> np.ndarray:
    """
    Annotate the frame with bounding boxes, track IDs, depth, number of points, and FPS.

    Args:
        frame: H×W×3 RGB image as a NumPy array.
        tracks: List of Detection3D objects for this frame.
        fps: Current processing frames-per-second.

    Returns:
        Annotated BGR image ready for cv2.imshow.
    """
    # Convert RGB→BGR for OpenCV
    img = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    # Draw FPS in top-left
    cv2.putText(img, f"FPS: {fps:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

    for det in tracks:
        x1, y1, x2, y2 = map(int, det.box2d)
        color = (255, 0, 0)  # blue box
        # box
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        # label above box
        label = f"ID {det.track_id:03d} D:{det.depth:.1f}m pts:{det.num_points}"
        cv2.putText(img, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    return img
