# `yolo_pipeline.io.data_streamer`

Play back KITTI or nuScenes as a live RGB+LiDAR stream.

## Module overview

Reads dataset images and pointclouds in a background thread, exposes latest frame via getters.

## Example

```python
from yolo_pipeline.io.data_streamer import DataStreamer
import time

ds = DataStreamer("datasets/config.yaml")
ds.load_split("kitti", split="train", shuffle=False)
ds.start()
time.sleep(0.2)

frame = ds.get_latest_camera_frame()
pc = ds.get_latest_pointcloud()
print(frame.shape, pc.shape)

ds.stop()
```
