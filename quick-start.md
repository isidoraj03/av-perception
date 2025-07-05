---
title: Quick Start
---

# Quick Start

Follow these steps to get the AV-Perception demo up and running quickly.

---

1. **Clone the repository**  
   ```bash
   git clone https://github.com/isidoraj03/av-perception.git
   cd av-perception
   ```

2. **Install dependencies with Poetry**  
   ```bash
   poetry install
   # (Optional) Spawn a shell:
   poetry shell
   ```

3. **Download datasets**  
   - See [Data Setup](data-setup.html) for full instructions on preparing the KITTI and nuScenes datasets.  
   - **Note:** The default dataset is KITTI.

4. **Run the live playback demo**  
   ```bash
   # Inside the Poetry shell
   python scripts/main.py --duration 30 --interval 0.1
   # Or without the shell
   poetry run python scripts/main.py --duration 30 --interval 0.1
   ```  
   - `--duration`: Total run time in seconds.  
   - `--interval`: Sleep time between frames (controls frame rate).  
   - **Note:** Press **q** in the OpenCV window to quit early.  
   - To switch datasets (e.g., to nuScenes):  
     ```bash
     DATASET_NAME=nuscenes poetry run python scripts/main.py
     ```

5. **(Optional) Launch the web dashboard**  
   ```bash
   poetry run streamlit run scripts/web_dashboard.py
   ```  
   - Open [http://localhost:8501](http://localhost:8501) in your browser.
