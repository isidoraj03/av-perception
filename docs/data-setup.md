---
title: Data Setup
---

# Data Setup

This project supports the official **nuScenes v1.0‑mini**, the 50‑frame **KITTI “mini” subset**, the entire **KITTI full training split**, and arbitrary **nuScenes subsets** generated with the helper scripts below.

---

## KITTI “mini” Subset *(50 frames)*

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

3. **Generate the 50‑image subset**  
   - Run:  
     ```bash
     poetry run python scripts/create_kitti_subset.py 50
     ```  
   - Creates under `datasets/`:  
     - `kitti50/images/`  
     - `kitti50/labels/`  
     - `kitti50.yaml`  

---

## KITTI Full Training Split

> **New!** Use [`create_kitti_full.py`](scripts/create_kitti_full.html) to convert the **entire KITTI training split** into YOLO format.

```bash
poetry run python scripts/create_kitti_full.py
```

This writes:

```
datasets/kitti_full/
├── images/train/
└── labels/train/
datasets/kitti_full.yaml
```

You can now train / evaluate with `datasets/kitti_full.yaml`.

---

## nuScenes v1.0‑mini

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

---

## Custom nuScenes Subset from v1.0‑trainval

If you have the full nuScenes *trainval* set and want a lightweight subset, run [`create_nuscenes_subset.py`](scripts/create_nuscenes_subset.html):

```bash
# 1 500 front‑camera frames by default
poetry run python scripts/create_nuscenes_subset.py
# or a smaller subset
poetry run python scripts/create_nuscenes_subset.py --num-images 500
```

This outputs:

```
datasets/nuscenes_subset/
├── images/
└── labels/
datasets/nuscenes_subset.yaml
```

You can immediately train / evaluate with `datasets/nuscenes_subset.yaml`.
