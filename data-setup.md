# Data Setup

This project supports **KITTI “mini” subset** (50 images) and **nuScenes v1.0-mini** (official mini split).

---

## KITTI “mini” Subset

1. **Download raw KITTI data**  
   - Visit [KITTI dataset page](https://www.cvlibs.net/datasets/kitti/).  
   - Download images and Velodyne point clouds from [raw data](https://www.cvlibs.net/datasets/kitti/raw_data.php).

2. **Extract and organize**  
   - Unpack to this structure:  
     ```
     <KITTI_ROOT>/
     ├── image_2/
     └── velodyne/
     ```  
   - Update `datasets/config.yaml`:  
     ```yaml
     kitti:
       root: /absolute/path/to/<KITTI_ROOT>
     ```

3. **Generate the 50-image subset**  
   - Run:  
     ```bash
     poetry run python scripts/create_kitti_subset.py 50
     ```  
   - Creates under `datasets/`:  
     - `kitti50/images/`  
     - `kitti50/labels/`  
     - `kitti50.yaml`  
   - **Note:** The script subsamples the raw KITTI data.

---

## nuScenes v1.0-mini

1. **Download the mini split**  
   - Visit [nuScenes download page](https://www.nuscenes.org/download).  
   - Or use:  
     ```bash
     wget https://www.nuscenes.org/data/v1.0-mini.tgz
     tar xzf v1.0-mini.tgz
     ```

2. **Place the folder**  
   - Move `v1.0-mini/` to:  
     ```
     av-perception/datasets/v1.0-mini/
     ```

3. **Update the config**  
   - In `datasets/config.yaml`:  
     ```yaml
     nuscenes:
       root: v1.0-mini
     ```