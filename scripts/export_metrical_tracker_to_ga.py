#!/usr/bin/env python3
"""Export MICA/metrical-tracker output to GaussianAvatars Dynamic NeRF format.

Default (WSL / proven NeRSemble monocular recipe — legacy moving-camera):
  - FLAME ``rotation`` / ``translation`` = zeros every frame.
  - Per-frame cameras from metrical-tracker OpenCV ``R``, ``t`` (and real K).

Mentor opt-in (``--cancel-camera-movement``):
  - Bake OpenCV ``R, t`` into per-frame FLAME pose.
  - Fixed OpenCV camera at ``camera_z`` (default: median tracker ``t_z``).
  - Warning: baking into FLAME LBS root pose is not equivalent to moving-camera
    export; keep this opt-in until mesh-overlay + PSNR A/B beat legacy.

Diagnostic after export:
  python scripts/diagnose_ga_camera_affix.py --transforms <tgt>/transforms_train.json
"""

from __future__ import annotations

import argparse
import glob
import json
import math
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


def rotmat_to_aa(R: np.ndarray) -> np.ndarray:
    """3x3 rotation matrix -> axis-angle (1, 3)."""
    mat = torch.from_numpy(np.asarray(R, dtype=np.float32).reshape(1, 3, 3))
    aa = matrix_to_axis_angle(mat).cpu().numpy().reshape(1, 3)
    return aa.astype(np.float32)


def opencv_to_ga_transform_matrix(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Match FlashAvatar/metrical-tracker cameras inside GA's transform_matrix loader."""
    w2c = np.eye(4, dtype=np.float64)
    w2c[:3, :3] = np.asarray(R, dtype=np.float64).reshape(3, 3)
    w2c[:3, 3] = np.asarray(t, dtype=np.float64).reshape(3)
    c2w = np.linalg.inv(w2c)
    # readCamerasFromTransforms flips Y/Z columns after loading JSON.
    c2w[:3, 1:3] *= -1
    return c2w


def fixed_camera_c2w(camera_z: float) -> np.ndarray:
    """JSON c2w which GA loads as OpenCV R=I, T=[0, 0, camera_z]."""
    return opencv_to_ga_transform_matrix(np.eye(3), np.array([0.0, 0.0, camera_z]))


def cancel_camera_to_flame(
    R: np.ndarray,
    t: np.ndarray,
    camera_z: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Move OpenCV w2c (R,t) onto FLAME pose; camera becomes fixed at z=camera_z.

    Projection equivalence with fixed cam R=I, T=[0,0,camera_z]:
      R @ v + t  ==  R @ v + (t - [0,0,z]) + [0,0,z]
    Note: GA applies FLAME rotation via LBS, so this is only approximate.
    """
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    t = np.asarray(t, dtype=np.float64).reshape(3)
    rotation = rotmat_to_aa(R)
    translation = (t - np.array([0.0, 0.0, camera_z], dtype=np.float64)).astype(np.float32)
    return rotation, translation.reshape(1, 3)


def load_frame(path: Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


def build_flame_npz(
    frame: dict,
    shape: np.ndarray,
    rotation: np.ndarray,
    translation: np.ndarray,
) -> dict:
    flame = frame["flame"]
    expr = np.asarray(flame["exp"], dtype=np.float32).reshape(1, -1)
    if expr.shape[1] < 100:
        expr = np.concatenate(
            [expr, np.zeros((1, 100 - expr.shape[1]), dtype=np.float32)], axis=1
        )
    elif expr.shape[1] > 100:
        expr = expr[:, :100]

    jaw = rot6d_to_aa(np.asarray(flame["jaw"]).reshape(-1)).astype(np.float32)
    if jaw.ndim == 1:
        jaw = jaw.reshape(1, 3)
    eyes = np.zeros((1, 6), dtype=np.float32)
    if "eyes" in flame:
        eyes = rot6d_to_aa(np.asarray(flame["eyes"]).reshape(-1, 6)).astype(np.float32)
        if eyes.shape[0] == 1 and eyes.size == 6:
            eyes = eyes.reshape(1, 6)
        elif eyes.size >= 6:
            eyes = eyes.reshape(-1)[:6].reshape(1, 6)

    return {
        "translation": np.asarray(translation, dtype=np.float32).reshape(1, 3),
        "rotation": np.asarray(rotation, dtype=np.float32).reshape(1, 3),
        "neck_pose": np.zeros((1, 3), dtype=np.float32),
        "jaw_pose": jaw.reshape(1, 3),
        "eyes_pose": eyes.reshape(1, 6),
        "shape": np.asarray(shape, dtype=np.float32).reshape(-1),
        "expr": expr,
        "static_offset": np.zeros((1, 5143, 3), dtype=np.float32),
    }


def make_frame_entry(i: int, c2w: np.ndarray, K: np.ndarray, h: int, w: int) -> dict:
    fl_x = float(K[0, 0])
    fl_y = float(K[1, 1])
    cx = float(K[0, 2])
    cy = float(K[1, 2])
    return {
        "timestep_index": i,
        "timestep_index_original": i,
        "timestep_id": f"frame_{i:05d}",
        "camera_index": 0,
        "camera_id": "cam_00",
        "cx": cx,
        "cy": cy,
        "fl_x": fl_x,
        "fl_y": fl_y,
        "h": h,
        "w": w,
        "camera_angle_x": 2 * math.atan(w / (2 * fl_x)),
        "camera_angle_y": 2 * math.atan(h / (2 * fl_y)),
        "transform_matrix": c2w.tolist(),
        "file_path": f"images/{i:05d}_00.png",
        "fg_mask_path": f"fg_masks/{i:05d}_00.png",
        "flame_param_path": f"flame_param/{i:05d}.npz",
    }


def write_transforms(frames: list, path: Path, shared: dict):
    payload = {
        **shared,
        "frames": frames,
        "timestep_indices": [f["timestep_index"] for f in frames],
        "camera_indices": [0],
    }
    path.write_text(json.dumps(payload, indent=2))


def export_dataset(
    track_out: Path,
    imgs_dir: Path,
    alpha_dir: Path,
    tgt_dir: Path,
    checkpoint_subdir: str = "checkpoint",
    train_ratio: float = 0.85,
    val_ratio: float = 0.07,
    cancel_camera_movement: bool = False,
    camera_z: float | None = None,
):
    ckpt_dir = track_out / checkpoint_subdir
    frame_paths = sorted(glob.glob(str(ckpt_dir / "*.frame")))
    if not frame_paths:
        raise FileNotFoundError(f"No .frame files in {ckpt_dir}")

    # First pass: choose camera_z = median tracker depth when canceling movement.
    depths = []
    if cancel_camera_movement:
        for fp in frame_paths:
            frame = load_frame(fp)
            depths.append(float(np.asarray(frame["opencv"]["t"][0]).reshape(3)[2]))
        if camera_z is None:
            camera_z = float(np.median(np.asarray(depths, dtype=np.float64)))
        print(
            f"Cancel camera movement: fixed OpenCV camera at z={camera_z:.6f} "
            f"(median tracker t_z={float(np.median(depths)):.6f})"
        )
    elif camera_z is None:
        camera_z = 1.0

    tgt_dir.mkdir(parents=True, exist_ok=True)
    images_out = tgt_dir / "images"
    masks_out = tgt_dir / "fg_masks"
    flame_out = tgt_dir / "flame_param"
    for d in (images_out, masks_out, flame_out):
        d.mkdir(parents=True, exist_ok=True)

    fixed_c2w = fixed_camera_c2w(camera_z)
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

        R = frame["opencv"]["R"][0]
        t = frame["opencv"]["t"][0]
        K = frame["opencv"]["K"][0]

        if cancel_camera_movement:
            rotation, translation = cancel_camera_to_flame(R, t, camera_z)
            c2w = fixed_c2w
        else:
            rotation = np.zeros((1, 3), dtype=np.float32)
            translation = np.zeros((1, 3), dtype=np.float32)
            c2w = opencv_to_ga_transform_matrix(R, t)

        flame_npz = build_flame_npz(
            frame, shape, rotation=rotation, translation=translation
        )
        np.savez(tgt_dir / "flame_param" / f"{i:05d}.npz", **flame_npz)
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
    mode = (
        "fixed-camera (canceled movement)"
        if cancel_camera_movement
        else "legacy moving-camera"
    )
    print(f"Exported {n} frames -> {tgt_dir} [{mode}]")
    print(f"  train={len(train_frames)} val={len(val_frames)} test={len(test_frames)}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Export metrical-tracker → GA. Default = legacy moving-camera "
            "(WSL NeRSemble parity)."
        )
    )
    parser.add_argument("--track-out", type=Path, required=True)
    parser.add_argument("--imgs-dir", type=Path, required=True)
    parser.add_argument("--alpha-dir", type=Path, required=True)
    parser.add_argument("--tgt-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-subdir", default="checkpoint")
    parser.add_argument(
        "--cancel-camera-movement",
        action="store_true",
        help=(
            "Mentor opt-in: bake OpenCV R,t into FLAME; fixed camera at median t_z. "
            "Default is legacy moving-camera."
        ),
    )
    parser.add_argument(
        "--legacy-moving-camera",
        action="store_true",
        help="Deprecated alias: default is already legacy moving-camera.",
    )
    parser.add_argument(
        "--camera-z",
        type=float,
        default=None,
        help="Fixed camera depth when --cancel-camera-movement (default: median t_z).",
    )
    args = parser.parse_args()
    export_dataset(
        args.track_out,
        args.imgs_dir,
        args.alpha_dir,
        args.tgt_dir,
        checkpoint_subdir=args.checkpoint_subdir,
        cancel_camera_movement=args.cancel_camera_movement,
        camera_z=args.camera_z,
    )


if __name__ == "__main__":
    main()
