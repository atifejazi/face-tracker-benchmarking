#!/usr/bin/env python3
"""Export SMIRK folder tracking to GaussianAvatars' dynamic NeRF format."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
import torch


def fixed_camera_c2w(camera_z: float) -> np.ndarray:
    """JSON c2w which GA loads as OpenCV R=I, T=[0, 0, camera_z]."""
    w2c = np.eye(4, dtype=np.float64)
    w2c[2, 3] = camera_z
    c2w = np.linalg.inv(w2c)
    c2w[:3, 1:3] *= -1
    return c2w


def smirk_translation(
    weak_cam: np.ndarray,
    image_to_crop: np.ndarray,
    focal: float,
    cx: float,
    cy: float,
    camera_z: float,
) -> np.ndarray:
    """Convert SMIRK's crop-space weak camera to approximate perspective translation."""
    scale, tx, ty = np.asarray(weak_cam, dtype=np.float64)
    crop_to_image = np.linalg.inv(np.asarray(image_to_crop, dtype=np.float64))
    crop_center = np.array([(scale * tx + 1.0) * 112.0, (scale * ty + 1.0) * 112.0, 1.0])
    image_center = crop_to_image @ crop_center
    image_center = image_center[:2] / image_center[2]
    inverse_crop_scale = np.linalg.norm(crop_to_image[:2, 0])
    pixels_per_unit = max(inverse_crop_scale * scale * 112.0, 1e-6)
    depth = focal / pixels_per_unit
    return np.array(
        [
            (image_center[0] - cx) * depth / focal,
            (image_center[1] - cy) * depth / focal,
            depth - camera_z,
        ],
        dtype=np.float32,
    )


def link_directory(source: Path, target: Path):
    if target.exists() or target.is_symlink():
        return
    target.symlink_to(source.resolve(), target_is_directory=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--track-params", type=Path, required=True)
    parser.add_argument("--canonical-dataset", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--focal", type=float, default=1200.0)
    parser.add_argument("--camera-z", type=float, default=1.0)
    args = parser.parse_args()

    if args.target.exists():
        raise FileExistsError(f"Refusing to overwrite {args.target}")
    args.target.mkdir(parents=True)
    flame_dir = args.target / "flame_param"
    flame_dir.mkdir()
    link_directory(args.canonical_dataset / "images", args.target / "images")
    link_directory(args.canonical_dataset / "fg_masks", args.target / "fg_masks")

    params = torch.load(args.track_params, map_location="cpu", weights_only=False)
    arrays = {
        key: value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)
        for key, value in params.items()
        if key != "frame_names"
    }
    n_frames = arrays["exp"].shape[0]

    shape = arrays["shape"].mean(axis=0).astype(np.float32)
    static_offset = np.zeros((1, 5143, 3), dtype=np.float32)
    width = height = 512
    cx = cy = 256.0
    c2w = fixed_camera_c2w(args.camera_z)

    for index in range(n_frames):
        expr = np.zeros((1, 100), dtype=np.float32)
        expr[0, : arrays["exp"].shape[1]] = arrays["exp"][index]
        translation = smirk_translation(
            arrays["cam"][index],
            arrays["image_to_crop"][index],
            args.focal,
            cx,
            cy,
            args.camera_z,
        )[None]
        np.savez(
            flame_dir / f"{index:05d}.npz",
            translation=translation,
            rotation=arrays["pose"][index : index + 1].astype(np.float32),
            neck_pose=np.zeros((1, 3), dtype=np.float32),
            jaw_pose=arrays["jaw"][index : index + 1].astype(np.float32),
            eyes_pose=np.zeros((1, 6), dtype=np.float32),
            shape=shape,
            expr=expr,
            static_offset=static_offset,
        )

    np.savez(
        args.target / "canonical_flame_param.npz",
        translation=np.zeros((1, 3), dtype=np.float32),
        rotation=np.zeros((1, 3), dtype=np.float32),
        neck_pose=np.zeros((1, 3), dtype=np.float32),
        jaw_pose=np.array([[0.3, 0.0, 0.0]], dtype=np.float32),
        eyes_pose=np.zeros((1, 6), dtype=np.float32),
        shape=shape,
        expr=np.zeros((1, 100), dtype=np.float32),
        static_offset=static_offset,
    )

    angle = 2 * math.atan(width / (2 * args.focal))
    for split in ("train", "val", "test"):
        source_json = args.canonical_dataset / f"transforms_{split}.json"
        database = json.loads(source_json.read_text())
        for frame in database["frames"]:
            index = int(frame["timestep_index"])
            frame.update(
                {
                    "cx": cx,
                    "cy": cy,
                    "fl_x": args.focal,
                    "fl_y": args.focal,
                    "h": height,
                    "w": width,
                    "camera_angle_x": angle,
                    "camera_angle_y": angle,
                    "transform_matrix": c2w.tolist(),
                    "file_path": f"images/{index:05d}_00.png",
                    "fg_mask_path": f"fg_masks/{index:05d}_00.png",
                    "flame_param_path": f"flame_param/{index:05d}.npz",
                }
            )
        database.update(
            {
                "cx": cx,
                "cy": cy,
                "fl_x": args.focal,
                "fl_y": args.focal,
                "h": height,
                "w": width,
                "camera_angle_x": angle,
                "camera_angle_y": angle,
            }
        )
        (args.target / f"transforms_{split}.json").write_text(json.dumps(database, indent=4))

    print(f"Exported {n_frames} SMIRK frames to {args.target}")


if __name__ == "__main__":
    main()
