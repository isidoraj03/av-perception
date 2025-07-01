# `overlay.py`

Helper function to draw detection & tracking overlays on an image.

## Function

```python
draw_overlay(frame: np.ndarray, tracks: List[Detection3D], fps: float) -> np.ndarray
```

### Arguments

* `frame`
  H×W×3 RGB image (`np.ndarray`).
* `tracks`
  List of `Detection3D` instances, each with:

  * `track_id`
  * `class_id`
  * `confidence`
  * `box2d` (x_min, y_min, x_max, y_max)
  * `depth`
  * `num_points`
* `fps`
  Current processing frames-per-second.

### Returns

* Annotated **BGR** image (`np.ndarray`) ready for `cv2.imshow()`:

  * Puts `FPS` in the top-left corner.
  * Draws blue bounding boxes and labels with `ID`, `D:<depth>m`, `pts:<num_points>`.

## Example

```python
from scripts.overlay import draw_overlay
annotated = draw_overlay(rgb_frame, tracks, fps)
cv2.imshow("Overlay", annotated)
```
