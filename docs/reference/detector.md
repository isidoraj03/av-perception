# `yolo_pipeline.perception.detector`

Wrapper around Ultralytics YOLO for easy 2D detection.

## Module overview

Provides a simple `Detector` class that:
- Loads a YOLO model by path or name.
- Moves it to a specified device.
- Runs inference on NumPy images, returning plain Python dicts.

```python
from yolo_pipeline.perception.detector import Detector
import numpy as np

det = Detector(model_path="yolov8n.pt", device="cpu")
det.load_model()
img = np.zeros((640,640,3), dtype=np.uint8)
preds = det.predict(img, conf=0.3, iou=0.5)
print(preds)
```

## Class `Detector`

### `__init__(model_path: str = "yolov8n.pt", device: str = "cpu")`

* **model_path**: YOLO model name or path to a `.pt` file.
* **device**: Inference device string (e.g. `"cpu"`, `"0"`).

### `load_model(model_path: str = None) -> None`

Loads YOLO model and moves it to the specified device.

### `predict(image: np.ndarray, conf: float = 0.25, iou: float = 0.45) -> List[Dict[str,Any]]`

Runs YOLO detection and returns list of detections with bounding box, confidence, and class ID.

## Example

```python
img = np.zeros((256,256,3), dtype=np.uint8)
det = Detector("yolov8n.pt", "cpu")
det.load_model()
preds = det.predict(img, conf=0.1, iou=0.5)
```
