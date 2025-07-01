# `yolo_pipeline.perception.fusion`

Sensor fusion: project 2D boxes into 3D using LiDAR.

## Module overview

`FusionEngine` projects 2D detections onto a 3D point cloud, scores and filters them.

## Class `FusionEngine`

### `__init__(camera_intrinsics: Dict[str,Any], extrinsics: Dict[str,Any], min_points: int = 3)`

Initializes with camera intrinsics and extrinsics, stores min_points threshold.

### `fuse(detections: List[Dict], pointcloud: np.ndarray) -> List[Dict]`

Projects LiDAR points to image space and matches against 2D boxes. Returns filtered and rescored detections.

## Example

```python
from yolo_pipeline.perception.fusion import FusionEngine
import numpy as np

intr = {"fx":1,"fy":1,"cx":0,"cy":0}
ext = {"R": np.eye(3), "T": [0,0,0]}
engine = FusionEngine(intr, ext, min_points=2)

dets = [{"box":[10,20,100,120], "confidence":0.8, "class_id":0}]
pc = np.array([[50,60,5,0.],[15,30,2,0.],[200,200,1,0.]], dtype=float)
fused = engine.fuse(dets, pc)
print(fused)
```
