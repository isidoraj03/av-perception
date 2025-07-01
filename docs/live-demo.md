# Live Playback Demo

Streams camera and LiDAR data, performs 2D detection, fuses to 3D, tracks objects, and displays results in real-time.

---

## Usage

```bash
# Inside the Poetry shell
python scripts/main.py [--duration <seconds>] [--interval <sec>]
# Or without the shell
poetry run python scripts/main.py --duration 30 --interval 0.05
```

### Options

- **`--duration <seconds>`**: Total run time (default: 10.0).  
- **`--interval <sec>`**: Sleep between frames, controls frame rate (default: 0.05, ~20 Hz).  
- **`DATASET_NAME`**: Set via environment variable ("kitti" or "nuscenes", default: "kitti").

### What Happens

1. **DataStreamer**: Loads frames and point clouds.  
2. **Detector**: Runs YOLO for 2D detection.  
3. **FusionEngine**: Projects LiDAR to refine detections.  
4. **SimpleTracker**: Assigns track IDs.  
5. **Overlay**: Shows camera view with boxes, IDs, and FPS.  
6. **Console**: Prints frame stats.  

- **Note:** Press **q** to exit early.