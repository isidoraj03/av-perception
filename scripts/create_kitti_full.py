#!/usr/bin/env python3
# scripts/create_kitti_full.py

import os
import shutil
import yaml
from PIL import Image

def main():
    # 1) locate config
    script_dir   = os.path.dirname(os.path.realpath(__file__))
    project_root = os.path.dirname(script_dir)
    cfg_path     = os.path.join(project_root, "datasets", "config.yaml")
    config_dir   = os.path.dirname(cfg_path)
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # 2) resolve KITTI root exactly as DataStreamer does
    kf       = cfg["kitti"]
    root_cfg = kf["root"]
    if os.path.isabs(root_cfg):
        root_dir = root_cfg
    else:
        root_dir = os.path.normpath(os.path.join(config_dir, root_cfg))

    # 3) get the train-split image & label bases
    split_cfg  = kf["splits"]["train"]
    img_base   = split_cfg if isinstance(split_cfg, str) else split_cfg["images"]
    lbl_base   = img_base.replace("data_object_image_2", "data_object_label_2")

    src_img_dir = os.path.normpath(os.path.join(root_dir, img_base, "image_2"))
    src_lbl_dir = os.path.normpath(os.path.join(root_dir, lbl_base, "label_2"))

    # sanity-check
    if not os.path.isdir(src_img_dir):
        raise FileNotFoundError(f"Image folder not found: {src_img_dir}")
    if not os.path.isdir(src_lbl_dir):
        raise FileNotFoundError(f"Label folder not found: {src_lbl_dir}")

    # 4) prepare destination
    dst_base       = os.path.join(config_dir, "kitti_full")
    dst_img_train  = os.path.join(dst_base, "images", "train")
    dst_lbl_train  = os.path.join(dst_base, "labels", "train")
    os.makedirs(dst_img_train, exist_ok=True)
    os.makedirs(dst_lbl_train, exist_ok=True)

    # 5) class mapping and iterate _all_ images
    classes = ["Car", "Pedestrian", "Cyclist"]
    cls2id  = {c: i for i, c in enumerate(classes)}
    all_imgs = sorted(f for f in os.listdir(src_img_dir) if f.lower().endswith(".png"))

    for img_name in all_imgs:
        # copy image
        shutil.copy(
            os.path.join(src_img_dir, img_name),
            os.path.join(dst_img_train, img_name)
        )

        # open to get size
        w, h = Image.open(os.path.join(src_img_dir, img_name)).size

        # read KITTI label
        src_label = os.path.join(src_lbl_dir, os.path.splitext(img_name)[0] + ".txt")
        dst_label = os.path.join(dst_lbl_train, os.path.splitext(img_name)[0] + ".txt")
        yolo_lines = []

        if os.path.isfile(src_label):
            with open(src_label, "r") as lf:
                for line in lf:
                    parts = line.strip().split()
                    cls   = parts[0]
                    # keep only our 3 classes
                    if cls not in cls2id:
                        continue
                    # KITTI bbox fields 4–7
                    x_min, y_min, x_max, y_max = map(float, parts[4:8])
                    x_c = (x_min + x_max) / 2  / w
                    y_c = (y_min + y_max) / 2  / h
                    bw  = (x_max - x_min)      / w
                    bh  = (y_max - y_min)      / h
                    yolo_lines.append(
                        f"{cls2id[cls]} {x_c:.6f} {y_c:.6f} {bw:.6f} {bh:.6f}"
                    )

        # write YOLO‐style txt (empty if no objects)
        with open(dst_label, "w") as outf:
            outf.write("\n".join(yolo_lines))

    # 6) write the full YAML
    yaml_txt = f"""\
train: {dst_img_train}
val:   {dst_img_train}
nc:    {len(classes)}
names: {classes}
"""
    yaml_out = os.path.join(config_dir, "kitti_full.yaml")
    with open(yaml_out, "w", encoding="utf-8") as yf:
        yf.write(yaml_txt)

    print(f"✅ Full KITTI dataset created at '{dst_base}' with {len(all_imgs)} images")
    print(f"YAML config written to '{yaml_out}'")

if __name__ == "__main__":
    main()
