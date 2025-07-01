# `yolo_pipeline.perception.tracker`

Simple 3D tracker skeleton and `Detection3D` dataclass.

## Module overview

- `Detection3D`: Data class for detection with ID and metadata.
- `Sort`: Dummy tracker assigning new ID every frame.
- `SimpleTracker`: Wrapper that emits `Detection3D` from fused detection dicts.

## Example

```python
from yolo_pipeline.perception.tracker import SimpleTracker

fused_dets = [{"box":[0,0,10,10],"class_id":0,"confidence":0.5,"num_points":5,"depth":2.1}]
tracker = SimpleTracker()
tracks = tracker.update(fused_dets)

for t in tracks:
    print(f"ID={t.track_id}, class={t.class_id}, depth={t.depth}, pts={t.num_points}")
```
