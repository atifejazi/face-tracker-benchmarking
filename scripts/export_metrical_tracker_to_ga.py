#!/usr/bin/env python3
"""Export MICA/metrical-tracker output to GaussianAvatars Dynamic NeRF format."""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import shutil
from pathlib import Path

import cv2
import numpy as np
import torch
from pytorch3d.transforms import matrix_to_axis_angle, rotation_6d_to_matrix
from tqdm import tqdm


def rot6d_to_aa(rot6d: np.ndarray) -> np.ndarray:
    rot6d = np.asarray(rot6d, dtype=np.float32)
    if rot6d.shape[-1] == 6:
        mat = rotation_6d_to_matrix(torch.from_numpy(rot6d).reshape(-1, 6))
        aa = matrix_to_axis_angle(mat).cpu().numpy().reshape(-1, 3)
        return aa
    raise ValueError(f"Expected rot6d with last dim 6, got {rot6d.shape}")


def opencv_to_ga_transform_matrix(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Match FlashAvatar/metrical-tracker cameras inside GA's transform_matrix loader."""
    w2c = np.eye(4, dtype=np.float64)
    w2c[:3, :3] = np.asarray(R, dtype=np.float64).reshape(3, 3)
    w2c[:3, 3] = np.asarray(t, dtype=np.float64).reshape(3)
    c2w = np.linalg.inv(w2c)
    # readCamerasFromTransforms flips Y/Z columns after loading JSON.
    c2w[:3, 1:3] *= -1
    return c2w


def load_frame(path: str) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


def build_flame_npz(frame: dict, shape: np.ndarray, num_verts: int = 5143) -> dict:
    flame = frame["flame"]
    jaw = rot6d_to_aa(flame["jaw"].reshape(1, 6))
    eyes = rot6d_to_aa(flame["eyes"].reshape(2, 6)).reshape(1, 6)
    expr = np.asarray(flame["exp"], dtype=np.float32).reshape(1, -1)
    if expr.shape[1] < 100:
        expr = np.pad(expr, ((0, 0), (0, 100 - expr.shape[1])))

    return {
        "translation": np.zeros((1, 3), dtype=np.float32),
        "rotation": np.zeros((1, 3), dtype=np.float32),
        "neck_pose": np.zeros((1, 3), dtype=np.float32),
        "jaw_pose": jaw.reshape(1, 3),
        "eyes_pose": eyes.reshape(1, 6),
        "shape": np.asarray(shape, dtype=np.float32).reshape(-1),
        "expr": expr[:, :100],
        "static_offset": np.zeros((1, num_verts, 3), dtype=np.float32),
    }


def make_frame_entry(
    timestep: int,
    c2w: np.ndarray,
    K: np.ndarray,
    h: int,
    w: int,
) -> dict:
    fl_x = float(K[0, 0])
    fl_y = float(K[1, 1])
    cx = float(K[0, 2])
    cy = float(K[1, 2])
    angle_x = math.atan(w / (fl_x * 2)) * 2
    angle_y = math.atan(h / (fl_y * 2)) * 2
    return {
        "timestep_index": timestep,
        "timestep_index_original": timestep,
        "timestep_id": f"frame_{timestep:05d}",
        "camera_index": 0,
        "camera_id": "cam_00",
        "cx": cx,
        "cy": cy,
        "fl_x": fl_x,
        "fl_y": fl_y,
        "h": h,
        "w": w,
        "camera_angle_x": angle_x,
        "camera_angle_y": angle_y,
        "transform_matrix": c2w.tolist(),
        "file_path": f"images/{timestep:05d}_00.png",
        "fg_mask_path": f"fg_masks/{timestep:05d}_00.png",
        "flame_param_path": f"flame_param/{timestep:05d}.npz",
    }


def write_transforms(frames: list[dict], out_path: Path, shared: dict):
    db = dict(shared)
    db["frames"] = frames
    db["timestep_indices"] = sorted({f["timestep_index"] for f in frames})
    db["camera_indices"] = [0]
    with open(out_path, "w") as f:
        json.dump(db, f, indent=4)


def export_dataset(
    track_out: Path,
    imgs_dir: Path,
    alpha_dir: Path,
    tgt_dir: Path,
    checkpoint_subdir: str = "checkpoint",
    train_ratio: float = 0.85,
    val_ratio: float = 0.07,
):
    ckpt_dir = track_out / checkpoint_subdir
    frame_paths = sorted(glob.glob(str(ckpt_dir / "*.frame")))
    if not frame_paths:
        raise FileNotFoundError(f"No .frame files in {ckpt_dir}")

    tgt_dir.mkdir(parents=True, exist_ok=True)
    images_out = tgt_dir / "images"
    masks_out = tgt_dir / "fg_masks"
    flame_out = tgt_dir / "flame_param"
    for d in (images_out, masks_out, flame_out):
        d.mkdir(parents=True, exist_ok=True)

    frames_data = []
    shape = None
    for i, fp in enumerate(tqdm(frame_paths, desc="export")):
        frame = load_frame(fp)
        fid = Path(fp).stem
        img_src = imgs_dir / f"{fid}.png"
        alpha_src = alpha_dir / f"{fid}.jpg"
        if not img_src.exists():
            img_src = track_out / "input" / f"{fid}.png"
        if not alpha_src.exists():
            alpha_src = alpha_dir / f"{fid}.png"
        if not img_src.exists():
            raise FileNotFoundError(f"Missing image for frame {fid}: {img_src}")

        img = cv2.imread(str(img_src))
        h, w = img.shape[:2]
        if alpha_src.exists():
            mask = cv2.imread(str(alpha_src), cv2.IMREAD_GRAYSCALE)
            if mask.shape[:2] != (h, w):
                mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)
        else:
            mask = np.full((h, w), 255, np.uint8)

        alpha = mask.astype(np.float32) / 255.0
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
        img_white = img_rgb * alpha[..., None] + 255.0 * (1.0 - alpha[..., None])
        img_out = cv2.cvtColor(img_white.astype(np.uint8), cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(images_out / f"{i:05d}_00.png"), img_out)
        cv2.imwrite(str(masks_out / f"{i:05d}_00.png"), mask)

        if shape is None:
            shape = np.asarray(frame["flame"]["shape"], dtype=np.float32).reshape(-1)
        flame_npz = build_flame_npz(frame, shape)
        np.savez(tgt_dir / "flame_param" / f"{i:05d}.npz", **flame_npz)

        R = frame["opencv"]["R"][0]
        t = frame["opencv"]["t"][0]
        K = frame["opencv"]["K"][0]
        c2w = opencv_to_ga_transform_matrix(R, t)
        frames_data.append(make_frame_entry(i, c2w, K, h, w))

    shared = {
        k: frames_data[0][k]
        for k in ("cx", "cy", "fl_x", "fl_y", "h", "w", "camera_angle_x", "camera_angle_y")
    }

    n = len(frames_data)
    n_train = max(1, int(n * train_ratio))
    n_val = max(1, int(n * val_ratio))
    n_test = max(1, n - n_train - n_val)
    if n_train + n_val + n_test > n:
        n_test = n - n_train - n_val

    train_frames = frames_data[:n_train]
    val_frames = frames_data[n_train : n_train + n_val]
    test_frames = frames_data[n_train + n_val : n_train + n_val + n_test]

    write_transforms(train_frames, tgt_dir / "transforms_train.json", shared)
    write_transforms(val_frames, tgt_dir / "transforms_val.json", shared)
    write_transforms(test_frames, tgt_dir / "transforms_test.json", shared)

    canonical = {
        "translation": np.zeros((1, 3), dtype=np.float32),
        "rotation": np.zeros((1, 3), dtype=np.float32),
        "neck_pose": np.zeros((1, 3), dtype=np.float32),
        "jaw_pose": np.array([[0.3, 0.0, 0.0]], dtype=np.float32),
        "eyes_pose": np.zeros((1, 6), dtype=np.float32),
        "shape": shape,
        "expr": np.zeros((1, 100), dtype=np.float32),
        "static_offset": np.zeros((1, 5143, 3), dtype=np.float32),
    }
    np.savez(tgt_dir / "canonical_flame_param.npz", **canonical)
    print(f"Exported {n} frames -> {tgt_dir}")
    print(f"  train={len(train_frames)} val={len(val_frames)} test={len(test_frames)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--track-out", type=Path, required=True)
    parser.add_argument("--imgs-dir", type=Path, required=True)
    parser.add_argument("--alpha-dir", type=Path, required=True)
    parser.add_argument("--tgt-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-subdir", default="checkpoint")
    args = parser.parse_args()
    export_dataset(
        args.track_out,
        args.imgs_dir,
        args.alpha_dir,
        args.tgt_dir,
        checkpoint_subdir=args.checkpoint_subdir,
    )


if __name__ == "__main__":
    main()
