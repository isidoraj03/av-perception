# Web Dashboard

A Streamlit-based dashboard for real-time perception metrics and visualization.

---

## Launch

1. **Ensure dependencies are installed**  
   ```bash
   poetry add streamlit matplotlib opencv-python
   ```

2. **Run the app**  
   ```bash
   poetry run streamlit run scripts/web_dashboard.py
   ```

3. **Open in browser**  
   - Visit [http://localhost:8501](http://localhost:8501) (default Streamlit port).

---

## Features

- **Sequence Selector**: Choose KITTI Clear-Day or nuScenes Rain-Night.  
- **Enable LiDAR Fusion**: Toggle fusion on/off.  
- **Run Duration (s)**: Set demo length.  
- **Refresh Interval (ms)**: Control update frequency.  
- **2D / 3D Metrics**: Real-time monitoring.  
- **Overlay**: Camera view with boxes and tracks.  
- **Top-down LiDAR Scatter**: LiDAR point cloud plot.

---

## Code Highlights

```python
# In scripts/web_dashboard.py
@st.cache_resource
def init_streamer(...): ...
@st.cache_resource
def init_detector(...): ...
...
c1, c2, c3 = st.columns(3)
m2d = c1.metric("2D Detections", 0)
...
```

- **Note:** `@st.cache_resource` caches resources for performance.