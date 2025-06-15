#!/usr/bin/env python3
# scripts/create_kitti_subset.py

import os
import shutil
import yaml
from PIL import Image

def main(num_images: int = 50):
    # 1) locate config
    script_dir  = os.path.dirname(os.path.realpath(__file__))
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

    # 3) get the train-split image base
    split_cfg = kf["splits"]["train"]
    img_base  = split_cfg if isinstance(split_cfg, str) else split_cfg["images"]

    # 4) full image / label paths
    src_img_dir   = os.path.normpath(os.path.join(root_dir, img_base,   "image_2"))
    src_lbl_dir   = os.path.normpath(os.path.join(
                        root_dir,
                        img_base.replace("data_object_image_2", "data_object_label_2"),
                        "label_2"
                    ))

    # sanity-check
    if not os.path.isdir(src_img_dir):
        raise FileNotFoundError(f"Image folder not found: {src_img_dir}")
    if not os.path.isdir(src_lbl_dir):
        raise FileNotFoundError(f"Label folder not found: {src_lbl_dir}")

    # 5) prepare destination
    dst_base   = os.path.join(config_dir, "kitti50")
    dst_imgs   = os.path.join(dst_base, "images")
    dst_lbls   = os.path.join(dst_base, "labels")
    os.makedirs(dst_imgs, exist_ok=True)
    os.makedirs(dst_lbls, exist_ok=True)

    # 6) class mapping and iterate images
    classes = ["Car","Pedestrian","Cyclist"]
    cls2id   = {c:i for i,c in enumerate(classes)}
    all_imgs = sorted(f for f in os.listdir(src_img_dir) if f.lower().endswith(".png"))[:num_images]

    for img_name in all_imgs:
        # copy image
        shutil.copy(
            os.path.join(src_img_dir, img_name),
            os.path.join(dst_imgs, img_name)
        )

        # open to get size
        w,h = Image.open(os.path.join(src_img_dir, img_name)).size

        # read KITTI label
        src_label = os.path.join(src_lbl_dir, os.path.splitext(img_name)[0] + ".txt")
        dst_label = os.path.join(dst_lbls,   os.path.splitext(img_name)[0] + ".txt")
        yolo_lines = []

        if os.path.isfile(src_label):
            with open(src_label, "r") as lf:
                for line in lf:
                    parts = line.strip().split()
                    cls   = parts[0]
                    if cls not in cls2id:
                        continue
                    # KITTI bbox fields 4–7
                    x_min,y_min,x_max,y_max = map(float, parts[4:8])
                    x_c = (x_min + x_max) / 2 / w
                    y_c = (y_min + y_max) / 2 / h
                    bw  = (x_max - x_min) / w
                    bh  = (y_max - y_min) / h
                    yolo_lines.append(
                        f"{cls2id[cls]} {x_c:.6f} {y_c:.6f} {bw:.6f} {bh:.6f}"
                    )

        # write out YOLO‐style txt (empty if no objects)
        with open(dst_label, "w") as outf:
            outf.write("\n".join(yolo_lines))

    # 7) write the subset YAML
    yaml_txt = f"""\
train: {dst_imgs}
val:   {dst_imgs}
nc:    {len(classes)}
names: {classes}
"""
    yaml_out = os.path.join(config_dir, "kitti50.yaml")
    with open(yaml_out, "w", encoding="utf-8") as yf:
        yf.write(yaml_txt)

    print(f"✅ Subset created at '{dst_base}' with {len(all_imgs)} images")
    print(f"YAML config written to '{yaml_out}'")


if __name__ == "__main__":
    main()
