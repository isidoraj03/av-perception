#!/usr/bin/env python3
# scripts/create_nuscenes_subset.py

import os
import argparse
import shutil
import yaml
import numpy as np
from PIL import Image
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import transform_matrix
from pyquaternion import Quaternion

def get_3d_box(translation, size, rotation_matrix):
    """
    Build the 8 corners of a 3D box.
    - translation: [x,y,z] center in world coords
    - size: [l,w,h]
    - rotation_matrix: 3×3 rotation matrix
    Returns 3×8 array of corner points.
    """
    l, w, h = size
    x_c = np.array([ l/2,  l/2, -l/2, -l/2,  l/2,  l/2, -l/2, -l/2])
    y_c = np.array([ w/2, -w/2, -w/2,  w/2,  w/2, -w/2, -w/2,  w/2])
    z_c = np.array([ h/2,  h/2,  h/2,  h/2, -h/2, -h/2, -h/2, -h/2])
    corners = np.vstack((x_c, y_c, z_c))           # 3×8
    corners = rotation_matrix @ corners
    corners += np.array(translation).reshape(3, 1)
    return corners

def main(num_images: int):
    # 1) locate config
    script_dir   = os.path.dirname(os.path.realpath(__file__))
    project_root = os.path.dirname(script_dir)
    cfg_path     = os.path.join(project_root, "datasets", "config.yaml")
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # 2) resolve nuScenes root (handle nested v1.0-full/v1.0-full)
    ncfg     = cfg["nuscenes-full"]
    root_cfg = ncfg["root"]
    if os.path.isabs(root_cfg):
        data_root = root_cfg
    else:
        data_root = os.path.normpath(os.path.join(os.path.dirname(cfg_path), root_cfg))
    nested = os.path.join(data_root, root_cfg)
    if os.path.isdir(nested):
        data_root = nested

    # 3) init NuScenes
    nusc = NuScenes(version="v1.0-trainval", dataroot=data_root, verbose=False)

    # 4) collect all front‐camera sample_data
    front_all = [sd for sd in nusc.sample_data if sd["channel"] == "CAM_FRONT"]
    front_all.sort(key=lambda sd: sd["timestamp"])

    # 5) pick first N whose image exists on disk
    selected = []
    for sd in front_all:
        img_rel = sd["filename"]
        img_abs = os.path.join(data_root, *img_rel.split("/"))
        if os.path.isfile(img_abs):
            selected.append(sd)
            if len(selected) >= num_images:
                break
    if len(selected) < num_images:
        raise RuntimeError(f"Found only {len(selected)} valid CAM_FRONT images, need {num_images}")

    # 6) prepare output dirs
    dst_base = os.path.join(os.path.dirname(cfg_path), "nuscenes_subset")
    dst_imgs = os.path.join(dst_base, "images")
    dst_lbls = os.path.join(dst_base, "labels")
    os.makedirs(dst_imgs, exist_ok=True)
    os.makedirs(dst_lbls, exist_ok=True)

    # 7) define your YOLO classes + mapping
    classes = [
        "car", "truck", "construction_vehicle", "bus", "trailer",
        "barrier", "motorcycle", "bicycle", "pedestrian", "traffic_cone"
    ]
    cls2id = {c: i for i, c in enumerate(classes)}

    # 8) process each selected frame
    for sd in selected:
        # copy image
        img_rel  = sd["filename"]
        img_src  = os.path.join(data_root, *img_rel.split("/"))
        img_name = os.path.basename(img_rel)
        shutil.copy(img_src, os.path.join(dst_imgs, img_name))

        # get image dimensions
        w, h = Image.open(img_src).size

        # build transforms:
        # 1) ego→world, then invert to get world→ego
        ep = nusc.get("ego_pose", sd["ego_pose_token"])
        e2w = transform_matrix(ep["translation"], Quaternion(ep["rotation"]))
        w2e = np.linalg.inv(e2w)

        # 2) sensor→ego, then invert to get ego→sensor
        cs = nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])
        s2e = np.linalg.inv(transform_matrix(cs["translation"], Quaternion(cs["rotation"])))

        # camera intrinsics
        K = np.array(cs["camera_intrinsic"])  # 3×3

        yolo_lines = []
        # grab all annotations for this sample
        ann_tokens = nusc.get("sample", sd["sample_token"])["anns"]
        for at in ann_tokens:
            ann = nusc.get("sample_annotation", at)
            full_cat = ann["category_name"]
            comp = next((p for p in full_cat.split(".") if p in cls2id), None)
            if comp is None:
                continue

            # compute 3D corners in world coords
            corners = get_3d_box(
                translation     = ann["translation"],
                size            = ann["size"],
                rotation_matrix = Quaternion(ann["rotation"]).rotation_matrix
            )

            # project to sensor frame: world→ego→sensor
            corners_h = np.vstack((corners, np.ones((1, 8))))  # 4×8
            cam_s     = s2e @ (w2e @ corners_h)
            pts_cam   = cam_s[:3, :]
            mask      = pts_cam[2, :] > 0
            if not mask.any():
                continue
            pts_cam = pts_cam[:, mask]

            # project to image plane
            pts_img = K @ pts_cam
            xs = pts_img[0] / pts_img[2]
            ys = pts_img[1] / pts_img[2]

            # clip and compute 2D box
            x_min, x_max = np.clip(xs.min(), 0, w), np.clip(xs.max(), 0, w)
            y_min, y_max = np.clip(ys.min(), 0, h), np.clip(ys.max(), 0, h)
            if x_max <= x_min or y_max <= y_min:
                continue

            # normalize for YOLO
            x_c = ((x_min + x_max) / 2) / w
            y_c = ((y_min + y_max) / 2) / h
            bw  = (x_max - x_min) / w
            bh  = (y_max - y_min) / h

            cls_id = cls2id[comp]
            yolo_lines.append(f"{cls_id} {x_c:.6f} {y_c:.6f} {bw:.6f} {bh:.6f}")

        # write label file (may be empty if no valid boxes)
        lbl_path = os.path.join(dst_lbls, os.path.splitext(img_name)[0] + ".txt")
        with open(lbl_path, "w") as outf:
            outf.write("\n".join(yolo_lines))

    # 9) write the dataset YAML
    yaml_txt = f"""\
train: {dst_imgs}
val:   {dst_imgs}
nc:    {len(classes)}
names: {classes}
"""
    yaml_out = os.path.join(os.path.dirname(cfg_path), "nuscenes_subset.yaml")
    with open(yaml_out, "w", encoding="utf-8") as yf:
        yf.write(yaml_txt)

    print(f"✅ Created nuScenes subset ({num_images} samples):\n  {dst_base}\nYAML config → {yaml_out}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a YOLO-formatted nuScenes subset")
    parser.add_argument("-n", "--num-images", type=int, default=1500,
                        help="Number of front-camera samples to include")
    args = parser.parse_args()
    main(args.num_images)
