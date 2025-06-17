"""
Calibration utilities for sensor fusion.
"""

def load_camera_intrinsics(yaml_path: str) -> dict:
    """
    Load camera intrinsics from a YAML or calibration file.

    Args:
        yaml_path (str): Path to camera intrinsics file.

    Returns:
        dict: Dictionary containing focal lengths, principal point, distortion coefficients, etc.
    """
    raise NotImplementedError("Camera intrinsics loading not yet implemented.")


def load_lidar_to_camera_extrinsics(txt_path: str) -> dict:
    """
    Load LiDAR→camera extrinsic calibration.

    Args:
        txt_path (str): Path to LiDAR→camera extrinsics file.

    Returns:
        dict: Dictionary containing rotation matrix and translation vector.
    """
    raise NotImplementedError("LiDAR to camera extrinsics loading not yet implemented.")
