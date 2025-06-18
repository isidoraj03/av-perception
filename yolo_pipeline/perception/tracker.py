# yolo_pipeline/perception/tracker.py

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple
import numpy as np

@dataclass
class Detection3D:
    """
    A 3D detection / tracklet.
    """
    track_id: int                              # unique ID for this track
    class_id: int                              # object class (e.g. 0=Car, 1=Pedestrian, …)
    confidence: float                          # re‐scored confidence
    box2d: Tuple[float, float, float, float]   # (x_min, y_min, x_max, y_max)
    depth: float                               # median (or mean) depth of points in box
    num_points: int                            # number of LiDAR points in the box

class Sort:
    """
    A minimal SORT tracker skeleton.
    Note: placeholder—assigns a fresh ID each frame.
    """

    def __init__(
        self,
        max_age: int = 3,
        min_hits: int = 1,
        iou_threshold: float = 0.3
    ):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold

        self._next_id = 1
        self._tracks: List[Detection3D] = []

    def update(self, detections: List[Detection3D]) -> List[Detection3D]:
        """
        Assigns each Detection3D a new track_id (ignores history).
        Replace this with full Kalman‐/Hungarian‐based logic later.
        """
        updated: List[Detection3D] = []
        for det in detections:
            det.track_id = self._next_id
            self._next_id += 1
            updated.append(det)

        self._tracks = updated
        return self._tracks

# alias so you can swap in a real ByteTrack later
ByteTrack = Sort

class SimpleTracker:
    """
    Wraps a 3D tracker (Sort/ByteTrack) to consume
    fused‐detection dicts and emit Detection3D with stable IDs.
    """

    def __init__(self, tracker: Any = None):
        """
        Args:
            tracker: instance of Sort/ByteTrack. If None, creates a Sort().
        """
        self._tracker = tracker if tracker is not None else Sort()

    def update(self, fused_dets: List[Dict[str, Any]]) -> List[Detection3D]:
        """
        Args:
            fused_dets: list of dicts, each must contain:
                - 'box': [x_min, y_min, x_max, y_max]
                - 'class_id': int
                - 'confidence': float
                - 'num_points': int
                - optionally 'depth': float
        Returns:
            List of Detection3D with consistent track_id.
        """
        det3d_list: List[Detection3D] = []
        for d in fused_dets:
            x1, y1, x2, y2 = d['box']
            det3d = Detection3D(
                track_id=0,
                class_id=int(d['class_id']),
                confidence=float(d['confidence']),
                box2d=(x1, y1, x2, y2),
                depth=float(d.get('depth', 0.0)),
                num_points=int(d.get('num_points', 0))
            )
            det3d_list.append(det3d)

        # pass through underlying tracker
        tracks = self._tracker.update(det3d_list)
        return tracks
