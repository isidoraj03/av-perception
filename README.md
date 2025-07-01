Here’s a cleaner, more professional version of your documentation, with improved structure, formatting, and readability:

AV-Perception
=============

**Real-time 2D/3D perception demo using Ultralytics YOLO and KITTI/nuScenes “mini” subsets**

🚀 Quick Start
--------------

### 1\. Clone the Repository


`   git clone https://github.com//av-perception.git  cd av-perception   `

### 2\. Install Dependencies (using Poetry)

`   poetry install  # Optional: enter Poetry shell  poetry shell   `
📦 Dataset Setup
----------------

### KITTI "Mini" Subset

1.  Download the raw KITTI data (images + Velodyne):
    
    *   Dataset: [https://www.cvlibs.net/datasets/kitti/](https://www.cvlibs.net/datasets/kitti/)
        
    *   Raw data: [https://www.cvlibs.net/datasets/kitti/raw\_data.php](https://www.cvlibs.net/datasets/kitti/raw_data.php)
        
2.  Extract the archive anywhere (must include image\_2/ and velodyne/ folders).
    
3.  kitti: root: /path/to/your/KITTI
    
4.  poetry run python scripts/create\_kitti\_subset.py 50
    
5.  datasets/ └── kitti50/ ├── images/ ├── labels/ └── kitti50.yaml
    

### nuScenes v1.0-mini

1.  Download the split from:
    
    *   Website: [https://www.nuscenes.org/download](https://www.nuscenes.org/download)
        
    *   wget https://www.nuscenes.org/data/v1.0-mini.tgztar xzf v1.0-mini.tgz
        
2.  datasets/v1.0-mini/ ├── samples/ ├── sweeps/ └── \*.json files
    
3.  nuscenes: root: datasets/v1.0-mini
    

▶️ Running the Live Demo
------------------------

From a Poetry shell:

`   python scripts/main.py --duration 30 --interval 0.1   `

Or without the shell:

`   poetry run python scripts/main.py --duration 30 --interval 0.1   `

*   Press q in the OpenCV window to quit early.
    
*   DATASET\_NAME=kitti poetry run python scripts/main.py
    

🧪 Other Scripts
----------------

*   poetry run python scripts/train.py --data datasets/kitti50.yaml --epochs 50
    
*   poetry run python scripts/eval.py --weights yolov8n.pt --data datasets/kitti50.yaml --device cpu --batch-size 4 --output-dir runs/val
    
*   poetry run python scripts/benchmark\_official.py --data datasets/kitti50.yaml --models yolov8s yolov11s --batch-size 4
    
*   poetry run python scripts/export.py --weights yolov8n.pt --format onnx --dynamic --output-dir exports
    
*   poetry run python scripts/fusion\_offline.py --image path/to/img.png --pc path/to/pc.npy --weights yolov8n.pt --min-points 3
    

🌐 Web Dashboard
----------------

1.  poetry add streamlit matplotlib opencv-python
    
2.  poetry run streamlit run scripts/web\_dashboard.py
    
3.  Open in browser: [http://localhost:8501](http://localhost:8501)
    

📚 MkDocs Documentation
-----------------------

1.  poetry add --dev mkdocs mkdocs-material
    
2.  poetry run mkdocs serve
    
3.  Open in browser: [http://127.0.0.1:8000](http://127.0.0.1:8000)
    

📜 License & Citation
---------------------

_(Include your preferred license and how to cite this project.)_

🖼️ Recommended Screenshots for README
--------------------------------------

1.  Dataset directory structure showing kitti50/ and v1.0-mini/
    
2.  Annotated live playback frame (bounding boxes, track IDs, FPS)
    
3.  Rendered Quick-Start page from the MkDocs site (with sidebar)
    
4.  Sample from summary.csv produced by evaluation scripts