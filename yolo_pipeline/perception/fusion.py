"""
Sensor fusion: combine 2D detections with 3D LiDAR point clouds.
"""

from typing import List, Dict, Any
import numpy as np

class FusionEngine:
    """
    FusionEngine projects 2D detections into 3D using LiDAR point clouds.
    """

    def __init__(
        self,
        camera_intrinsics: Dict[str, Any],
        extrinsics: Dict[str, Any],
        min_points: int = 3
    ):
        """
        Initialize FusionEngine with calibration data.

        Args:
            camera_intrinsics (dict): { 'fx', 'fy', 'cx', 'cy' }
            extrinsics (dict): { 'R': 3×3 rotation np.ndarray,
                                 'T': length-3 translation np.ndarray }
            min_points (int): minimum number of LiDAR points inside a box to keep it
        """
        self.fx = camera_intrinsics['fx']
        self.fy = camera_intrinsics['fy']
        self.cx = camera_intrinsics['cx']
        self.cy = camera_intrinsics['cy']
        self.R  = extrinsics['R']
        self.T  = np.array(extrinsics['T'])
        self.min_points = min_points

    def fuse(
        self,
        detections: List[Dict[str, Any]],
        pointcloud: np.ndarray
    ) -> List[Dict[str, Any]]:
        """
        Fuse 2D detections with pointcloud to produce 3D-enhanced detections.

        - Projects each LiDAR point into the image plane.
        - Counts points per 2D box.
        - Drops boxes with fewer than min_points.
        - Re-scores retained boxes: new_conf = orig_conf * min(1, count/min_points)

        Args:
            detections: list of { 'box':[x_min,y_min,x_max,y_max], 'confidence':float, ... }
            pointcloud: N×4 array of LiDAR [X,Y,Z,reflectance]

        Returns:
            List of detections with added keys:
              - 'num_points': int
              - 'confidence': updated float
        """
        # 1) transform LiDAR→camera coords
        xyz = pointcloud[:, :3]                          # N×3
        cam_xyz = (self.R @ xyz.T).T + self.T             # N×3

        # 2) only points in front of camera
        z = cam_xyz[:, 2]
        valid = z > 0
        cam_xyz = cam_xyz[valid]
        z = z[valid]

        # 3) project to image
        u = self.fx * (cam_xyz[:, 0] / z) + self.cx
        v = self.fy * (cam_xyz[:, 1] / z) + self.cy

        outputs: List[Dict[str, Any]] = []
        for det in detections:
            x_min, y_min, x_max, y_max = det['box']
            # 4) count points inside box
            inside = (
                (u >= x_min) & (u <= x_max) &
                (v >= y_min) & (v <= y_max)
            )
            count = int(np.sum(inside))

            # 5) filter by threshold
            if count < self.min_points:
                continue

            # 6) re-score
            orig_conf = det.get('confidence', 1.0)
            factor   = min(1.0, count / self.min_points)
            new_conf = orig_conf * factor

            # 7) build output
            out_det = det.copy()
            out_det['num_points']  = count
            out_det['confidence']  = new_conf
            outputs.append(out_det)

        return outputs
