#!/usr/bin/env python3
"""Export MICA/metrical-tracker output to GaussianAvatars Dynamic NeRF format.

Camera modes (``--camera-mode``; default ``legacy``):

  legacy
      FLAME R/t = 0; per-frame OpenCV cameras from metrical R,t.

  relative_cancel  (alias: ``--cancel-camera-movement``)
      Fixed camera from frame-0 OpenCV extrinsics (GA convention kept on camera).
      FLAME gets R_rel plus **J0-pivot-compensated** t_rel, then mean-T center.
      Projection-equivalent to legacy under FLAME LBS (see
      mica_camera_cancel_and_centering_fix_spec.md).

  relative_no_center
      Same J0-compensated relative split, without mean-T recenter.

  t_only_cancel / cam_canonicalize
      Kept for A/B only; known-incorrect for MICA (discard rotation / wrong SE3).

Never bake absolute OpenCV R (~diag(1,-1,-1) / ~π) into FLAME root.

Diagnostic:
  python scripts/diagnose_ga_camera_affix.py --transforms <tgt>/transforms_train.json
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import pickle
from pathlib import Path

import cv2
import numpy as np
import torch
from pytorch3d.transforms import matrix_to_axis_angle, rotation_6d_to_matrix
from tqdm import tqdm

CAMERA_MODES = (
    "legacy",
    "relative_cancel",
    "relative_no_center",
    "t_only_cancel",
    "cam_canonicalize",
)


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


def _ga_w2c_tz(c2w: np.ndarray) -> float:
    c2w_undo = np.asarray(c2w, dtype=np.float64).copy()
    c2w_undo[:3, 1:3] *= -1
    w2c = np.linalg.inv(c2w_undo)
    return float(w2c[2, 3])


def resolve_flame_pkl(flame_pkl_path: Path | None = None) -> Path:
    if flame_pkl_path is not None and Path(flame_pkl_path).is_file():
        return Path(flame_pkl_path)
    candidates: list[Path] = []
    if "GA_ROOT" in os.environ:
        ga_root = Path(os.environ["GA_ROOT"])
        candidates.extend(
            [
                ga_root / "flame_model" / "assets" / "flame" / "generic_model.pkl",
                ga_root / "flame_model" / "assets" / "flame" / "flame2023.pkl",
            ]
        )
        candidates.extend(ga_root.glob("**/generic_model.pkl"))
    if "MICA_ROOT" in os.environ:
        mica_root = Path(os.environ["MICA_ROOT"])
        candidates.append(mica_root / "data" / "FLAME2020" / "generic_model.pkl")
        candidates.extend(mica_root.glob("**/generic_model.pkl"))
    for c in candidates:
        if c.is_file():
            return c
    raise FileNotFoundError(
        "Could not locate FLAME model .pkl to compute root joint J_0(beta). "
        "Pass --flame-pkl or set GA_ROOT / MICA_ROOT."
    )


def compute_flame_root_joint(
    shape: np.ndarray,
    expr: np.ndarray | None = None,
    flame_pkl_path: Path | None = None,
    n_shape: int = 300,
    n_expr: int = 100,
) -> np.ndarray:
    """Canonical FLAME root joint J_0 from shaped template (matches FlameHead LBS)."""
    flame_pkl_path = resolve_flame_pkl(flame_pkl_path)
    with open(flame_pkl_path, "rb") as f:
        flame_data = pickle.load(f, encoding="latin1")

    v_template = np.asarray(flame_data["v_template"], dtype=np.float64)
    shapedirs = np.asarray(flame_data["shapedirs"], dtype=np.float64)  # V x 3 x n
    J_regressor = flame_data["J_regressor"]

    beta = np.asarray(shape, dtype=np.float64).reshape(-1)
    ns = min(int(n_shape), beta.shape[0], shapedirs.shape[-1])
    v_shaped = v_template + np.einsum("vcd,d->vc", shapedirs[:, :, :ns], beta[:ns])

    if expr is not None and shapedirs.shape[-1] > 300:
        e = np.asarray(expr, dtype=np.float64).reshape(-1)
        ne = min(int(n_expr), e.shape[0], shapedirs.shape[-1] - 300)
        if ne > 0:
            v_shaped = v_shaped + np.einsum(
                "vcd,d->vc", shapedirs[:, :, 300 : 300 + ne], e[:ne]
            )

    if hasattr(J_regressor, "dot"):
        J = J_regressor.dot(v_shaped)
    else:
        J = np.asarray(J_regressor, dtype=np.float64) @ v_shaped
    return np.asarray(J[0], dtype=np.float64).reshape(3)


def compute_camera_mode_poses(
    mode: str,
    Rs: list[np.ndarray],
    ts: list[np.ndarray],
    root_joints: list[np.ndarray] | np.ndarray | None = None,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], str]:
    """Return per-frame (c2w, flame_rot aa 1x3, flame_t 1x3) and a log label.

    For relative_* modes, ``root_joints`` must be J_0(beta[, expr]) per frame (or one
    shared J_0). FLAME LBS rotates about J_0, so
        t_rel = R_ref.T @ (t_i - t_ref) - (I - R_rel) @ J_0
    is required for projection equivalence with legacy.
    """
    n = len(Rs)
    R_ref = np.asarray(Rs[0], dtype=np.float64).reshape(3, 3)
    t_ref = np.asarray(ts[0], dtype=np.float64).reshape(3)
    zero_r = np.zeros((1, 3), dtype=np.float32)
    zero_t = np.zeros((1, 3), dtype=np.float32)
    I3 = np.eye(3, dtype=np.float64)

    if mode == "legacy":
        c2ws = [opencv_to_ga_transform_matrix(R, t) for R, t in zip(Rs, ts)]
        return c2ws, [zero_r] * n, [zero_t] * n, "legacy moving-camera"

    if mode == "cam_canonicalize":
        c2ws, rots, trans = [], [], []
        aa_norms = []
        for R_i, t_i in zip(Rs, ts):
            R_i = np.asarray(R_i, dtype=np.float64).reshape(3, 3)
            t_i = np.asarray(t_i, dtype=np.float64).reshape(3)
            R_p = R_i @ R_ref.T
            t_p = t_i - R_i @ R_ref.T @ t_ref
            c2ws.append(opencv_to_ga_transform_matrix(R_p, t_p))
            rots.append(zero_r)
            trans.append(zero_t)
            aa_norms.append(float(np.linalg.norm(rotmat_to_aa(R_p).reshape(3))))
        label = (
            f"cam_canonicalize (KNOWN BAD for MICA); "
            f"|aa(R')| mean={float(np.mean(aa_norms)):.6f} max={float(np.max(aa_norms)):.6f}"
        )
        return c2ws, rots, trans, label

    need_j0 = mode in ("relative_cancel", "relative_no_center")
    if need_j0 and root_joints is None:
        raise ValueError(f"root_joints J_0(beta) required for camera mode '{mode}'")

    if root_joints is None:
        J0_list = [np.zeros(3, dtype=np.float64)] * n
    elif isinstance(root_joints, np.ndarray) and root_joints.ndim == 1:
        J0_list = [np.asarray(root_joints, dtype=np.float64).reshape(3)] * n
    else:
        J0_list = [np.asarray(j, dtype=np.float64).reshape(3) for j in root_joints]
        if len(J0_list) != n:
            raise ValueError(f"root_joints length {len(J0_list)} != n_frames {n}")

    R_rels_aa: list[np.ndarray] = []
    t_rels: list[np.ndarray] = []
    aa_norms: list[float] = []
    pivot_shifts: list[float] = []

    for R_i, t_i, J0 in zip(Rs, ts, J0_list):
        R_i = np.asarray(R_i, dtype=np.float64).reshape(3, 3)
        t_i = np.asarray(t_i, dtype=np.float64).reshape(3)
        R_rel = R_ref.T @ R_i
        pivot_comp = (I3 - R_rel) @ J0
        if need_j0:
            t_rel = R_ref.T @ (t_i - t_ref) - pivot_comp
        else:
            # t_only_cancel: no rotation into FLAME; uncompensated relative t only.
            t_rel = R_ref.T @ (t_i - t_ref)
        aa = rotmat_to_aa(R_rel)
        R_rels_aa.append(aa)
        t_rels.append(t_rel.astype(np.float64))
        aa_norms.append(float(np.linalg.norm(aa.reshape(3))))
        pivot_shifts.append(float(np.linalg.norm(pivot_comp)))

    if mode == "relative_no_center":
        fixed_c2w = opencv_to_ga_transform_matrix(R_ref, t_ref)
        c2ws = [fixed_c2w] * n
        trans = [t.astype(np.float32).reshape(1, 3) for t in t_rels]
        label = (
            f"relative_no_center (J0-compensated); |aa(R_rel)| mean={float(np.mean(aa_norms)):.6f} "
            f"max={float(np.max(aa_norms)):.6f}; |J0_comp| max={float(np.max(pivot_shifts)):.6f}; "
            f"fixed t_z={_ga_w2c_tz(fixed_c2w):.6f}"
        )
        return c2ws, R_rels_aa, trans, label

    if mode == "relative_cancel":
        t_stack = np.stack(t_rels, axis=0)
        t_mean = t_stack.mean(axis=0)
        trans = [(t - t_mean).astype(np.float32).reshape(1, 3) for t in t_stack]
        # Absorb mean into OpenCV camera: t_cam = t_ref + R_ref @ t_mean
        # (== Transl(-t_mean) @ c2w_ref in world/c2w space).
        t_cam_fixed = t_ref + R_ref @ t_mean
        fixed_c2w = opencv_to_ga_transform_matrix(R_ref, t_cam_fixed)
        c2ws = [fixed_c2w] * n
        label = (
            f"relative_cancel+center (J0-compensated); |aa(R_rel)| mean={float(np.mean(aa_norms)):.6f} "
            f"max={float(np.max(aa_norms)):.6f}; |J0_comp| max={float(np.max(pivot_shifts)):.6f}; "
            f"t_mean_z={float(t_mean[2]):.6f}; fixed t_z={_ga_w2c_tz(fixed_c2w):.6f}"
        )
        return c2ws, R_rels_aa, trans, label

    if mode == "t_only_cancel":
        t_stack = np.stack(t_rels, axis=0)
        t_mean = t_stack.mean(axis=0)
        trans = [(t - t_mean).astype(np.float32).reshape(1, 3) for t in t_stack]
        t_cam_fixed = t_ref + R_ref @ t_mean
        fixed_c2w = opencv_to_ga_transform_matrix(R_ref, t_cam_fixed)
        c2ws = [fixed_c2w] * n
        label = (
            f"t_only_cancel+center (KNOWN BAD for MICA; drops R_rel); "
            f"t_mean_z={float(t_mean[2]):.6f}; fixed t_z={_ga_w2c_tz(fixed_c2w):.6f}"
        )
        return c2ws, [zero_r] * n, trans, label

    raise ValueError(f"Unknown camera mode: {mode}")


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
    camera_mode: str = "legacy",
    cancel_camera_movement: bool = False,
    camera_z: float | None = None,
    flame_pkl_path: Path | None = None,
):
    if cancel_camera_movement:
        camera_mode = "relative_cancel"
    if camera_mode not in CAMERA_MODES:
        raise ValueError(f"camera_mode must be one of {CAMERA_MODES}, got {camera_mode}")
    if camera_z is not None and camera_mode != "legacy":
        print(f"Warning: --camera-z is ignored for camera-mode={camera_mode}")

    ckpt_dir = track_out / checkpoint_subdir
    frame_paths = sorted(glob.glob(str(ckpt_dir / "*.frame")))
    if not frame_paths:
        raise FileNotFoundError(f"No .frame files in {ckpt_dir}")

    Rs, ts, Ks = [], [], []
    exprs: list[np.ndarray] = []
    shape = None
    for fp in frame_paths:
        frame = load_frame(fp)
        Rs.append(np.asarray(frame["opencv"]["R"][0], dtype=np.float64))
        ts.append(np.asarray(frame["opencv"]["t"][0], dtype=np.float64).reshape(3))
        Ks.append(np.asarray(frame["opencv"]["K"][0], dtype=np.float64))
        if shape is None:
            shape = np.asarray(frame["flame"]["shape"], dtype=np.float32).reshape(-1)
        expr = np.asarray(frame["flame"]["exp"], dtype=np.float64).reshape(-1)
        exprs.append(expr)

    root_joints = None
    if camera_mode in ("relative_cancel", "relative_no_center"):
        flame_pkl = resolve_flame_pkl(flame_pkl_path)
        print(f"FLAME pkl for J_0: {flame_pkl}")
        root_joints = [
            compute_flame_root_joint(shape, expr=e, flame_pkl_path=flame_pkl)
            for e in exprs
        ]
        print(f"FLAME root joint J_0(frame0): {root_joints[0]}")

    c2ws, flame_rots, flame_ts, mode_label = compute_camera_mode_poses(
        camera_mode, Rs, ts, root_joints=root_joints
    )
    print(f"Camera mode: {mode_label}")

    tgt_dir.mkdir(parents=True, exist_ok=True)
    images_out = tgt_dir / "images"
    masks_out = tgt_dir / "fg_masks"
    flame_out = tgt_dir / "flame_param"
    for d in (images_out, masks_out, flame_out):
        d.mkdir(parents=True, exist_ok=True)

    frames_data = []
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

        flame_npz = build_flame_npz(
            frame,
            shape,
            rotation=flame_rots[i],
            translation=flame_ts[i],
        )
        np.savez(tgt_dir / "flame_param" / f"{i:05d}.npz", **flame_npz)
        frames_data.append(make_frame_entry(i, c2ws[i], Ks[i], h, w))

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
    print(f"Exported {n} frames -> {tgt_dir} [{camera_mode}]")
    print(f"  train={len(train_frames)} val={len(val_frames)} test={len(test_frames)}")


def main():
    parser = argparse.ArgumentParser(
        description="Export metrical-tracker → GA. Default camera-mode=legacy."
    )
    parser.add_argument("--track-out", type=Path, required=True)
    parser.add_argument("--imgs-dir", type=Path, required=True)
    parser.add_argument("--alpha-dir", type=Path, required=True)
    parser.add_argument("--tgt-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-subdir", default="checkpoint")
    parser.add_argument(
        "--camera-mode",
        choices=CAMERA_MODES,
        default="legacy",
        help="Isolated camera/FLAME export recipe (default: legacy).",
    )
    parser.add_argument(
        "--cancel-camera-movement",
        action="store_true",
        help="Alias for --camera-mode relative_cancel (J0-compensated).",
    )
    parser.add_argument(
        "--legacy-moving-camera",
        action="store_true",
        help="Deprecated alias: default is already legacy.",
    )
    parser.add_argument(
        "--camera-z",
        type=float,
        default=None,
        help="Ignored for non-legacy modes (kept for old CLIs).",
    )
    parser.add_argument(
        "--flame-pkl",
        type=Path,
        default=None,
        help="FLAME .pkl for J_0(beta); default searches GA_ROOT/MICA_ROOT.",
    )
    args = parser.parse_args()
    export_dataset(
        args.track_out,
        args.imgs_dir,
        args.alpha_dir,
        args.tgt_dir,
        checkpoint_subdir=args.checkpoint_subdir,
        camera_mode=args.camera_mode,
        cancel_camera_movement=args.cancel_camera_movement,
        camera_z=args.camera_z,
        flame_pkl_path=args.flame_pkl,
    )


if __name__ == "__main__":
    main()
