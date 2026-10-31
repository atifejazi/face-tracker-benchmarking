#!/usr/bin/env python3
"""Export SMIRK tracking to GA with a projection consistent with SMIRK.

SMIRK projects (X,Y) to crop pixels 112*(1+s*(X+tx), 1-s*(Y+ty)).
Use a fixed OpenCV w2c rotation diag(1,-1,-1), keep FLAME pose unchanged,
and map camera translation back into world axes. No rotation is baked into LBS.
The weak-to-perspective conversion is approximate away from the Z=0 plane.

--projection-mode legacy reproduces the previous upside-down export; the old
flip flags are available only in that mode for reproducing historical runs.
Existing datasets/checkpoints must be re-exported/retrained separately.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch


# OpenCV (Y-down) vs FLAME/SMIRK (Y-up): 180 deg about X.
_RX_PI = np.array(
    [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]],
    dtype=np.float64,
)


def fixed_camera_c2w(camera_z: float, y_flip: bool = False) -> np.ndarray:
    """JSON c2w which GA loads as OpenCV R=I (or Rx(π)), T=[0, 0, camera_z]."""
    w2c = np.eye(4, dtype=np.float64)
    if y_flip:
        w2c[:3, :3] = _RX_PI
    w2c[2, 3] = camera_z
    c2w = np.linalg.inv(w2c)
    c2w[:3, 1:3] *= -1
    return c2w


def _rodrigues(aa: np.ndarray) -> np.ndarray:
    """Axis-angle (3,) -> rotation matrix 3x3."""
    aa = np.asarray(aa, dtype=np.float64).reshape(3)
    theta = np.linalg.norm(aa)
    if theta < 1e-12:
        return np.eye(3, dtype=np.float64)
    k = aa / theta
    K = np.array(
        [[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]],
        dtype=np.float64,
    )
    return np.eye(3) + math.sin(theta) * K + (1.0 - math.cos(theta)) * (K @ K)


def _rotmat_to_aa(R: np.ndarray) -> np.ndarray:
    """Rotation matrix 3x3 -> axis-angle (3,)."""
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    cos_theta = (np.trace(R) - 1.0) * 0.5
    cos_theta = float(np.clip(cos_theta, -1.0, 1.0))
    theta = math.acos(cos_theta)
    if theta < 1e-12:
        return np.zeros(3, dtype=np.float32)
    if abs(theta - math.pi) < 1e-4:
        xx, yy, zz = (R[0, 0] + 1) * 0.5, (R[1, 1] + 1) * 0.5, (R[2, 2] + 1) * 0.5
        xy, xz, yz = R[0, 1] + R[1, 0], R[0, 2] + R[2, 0], R[1, 2] + R[2, 1]
        if xx > yy and xx > zz:
            x = math.sqrt(max(xx, 0.0))
            y = xy / (4 * x + 1e-12)
            z = xz / (4 * x + 1e-12)
        elif yy > zz:
            y = math.sqrt(max(yy, 0.0))
            x = xy / (4 * y + 1e-12)
            z = yz / (4 * y + 1e-12)
        else:
            z = math.sqrt(max(zz, 0.0))
            x = xz / (4 * z + 1e-12)
            y = yz / (4 * z + 1e-12)
        axis = np.array([x, y, z], dtype=np.float64)
        axis /= np.linalg.norm(axis) + 1e-12
        return (axis * theta).astype(np.float32)
    axis = np.array(
        [R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]],
        dtype=np.float64,
    )
    axis /= 2.0 * math.sin(theta)
    return (axis * theta).astype(np.float32)


def apply_flame_y_flip(rotation_aa: np.ndarray, translation: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Legacy: bake Rx(π) into FLAME rotation+translation (LBS-unsafe)."""
    R = _RX_PI @ _rodrigues(rotation_aa)
    t = (_RX_PI @ np.asarray(translation, dtype=np.float64).reshape(3)).astype(np.float32)
    return _rotmat_to_aa(R).reshape(1, 3), t.reshape(1, 3)


def smirk_translation(
    weak_cam: np.ndarray,
    image_to_crop: np.ndarray,
    focal: float,
    cx: float,
    cy: float,
    camera_z: float,
) -> np.ndarray:
    """Convert SMIRK's crop-space weak camera to approximate perspective translation.

    Returned ``t`` is in the OpenCV camera frame for ``R=I`` (Y-down image axes):
    world point ≈ camera origin + R^T @ (0,0,0) with offset ``t`` under w2c=I,t_z=z
    is stored as FLAME translation relative to the fixed camera.
    """
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


def smirk_perspective_translation(weak_cam, image_to_crop, focal, cx, cy, camera_z):
    """World translation for w2c=[diag(1,-1,-1) | (0,0,camera_z)].

    At FLAME Z=0, GA's perspective projection exactly matches SMIRK's
    weak projection followed by inverse crop. crop_face produces an unrotated,
    positive uniform similarity; refuse unsupported transforms rather than
    silently losing their rotation/shear. Units are those of the FLAME model.
    """
    scale, tx, ty = np.asarray(weak_cam, dtype=np.float64)
    transform = np.asarray(image_to_crop, dtype=np.float64)
    if not np.isfinite([scale, tx, ty, focal, cx, cy, camera_z]).all() or scale <= 0 or focal <= 0:
        raise ValueError("SMIRK camera must be finite with positive scale/focal")
    if (transform.shape != (3, 3) or not np.isfinite(transform).all()
            or transform[0, 0] <= 0
            or not np.allclose(transform[:2, :2], np.eye(2)*transform[0, 0], atol=1e-6)
            or not np.allclose(transform[2], [0, 0, 1], atol=1e-6)):
        raise ValueError("Expected the positive, unrotated uniform crop from crop_face")
    inverse = np.linalg.inv(transform)
    center = inverse @ [112*(1+scale*tx), 112*(1-scale*ty), 1]
    pixels_per_unit = inverse[0, 0]*112*scale
    depth = focal/pixels_per_unit
    # camera-space translation: (x, y, depth-camera_z); R.T maps it to world.
    return np.array([(center[0]-cx)/pixels_per_unit,
                     -(center[1]-cy)/pixels_per_unit,
                     camera_z-depth], dtype=np.float32)


def link_directory(source: Path, target: Path):
    if target.exists() or target.is_symlink():
        return
    target.symlink_to(source.resolve(), target_is_directory=True)


def inherit_from_canonical(canonical: Path) -> dict:
    """Pull intrinsics + camera_z from MICA transforms_train.json."""
    transforms = json.loads((canonical / "transforms_train.json").read_text())
    frame0 = transforms["frames"][0]
    fl_x = float(transforms["fl_x"] if "fl_x" in transforms else frame0["fl_x"])
    fl_y = float(transforms.get("fl_y", frame0.get("fl_y", fl_x)))
    cx = float(transforms["cx"] if "cx" in transforms else frame0["cx"])
    cy = float(transforms["cy"] if "cy" in transforms else frame0["cy"])
    w = int(transforms["w"] if "w" in transforms else frame0["w"])
    h = int(transforms["h"] if "h" in transforms else frame0["h"])

    c2w = np.asarray(frame0["transform_matrix"], dtype=np.float64)
    c2w_undo = c2w.copy()
    c2w_undo[:3, 1:3] *= -1
    w2c = np.linalg.inv(c2w_undo)
    camera_z = float(w2c[2, 3])
    return {
        "fl_x": fl_x,
        "fl_y": fl_y,
        "cx": cx,
        "cy": cy,
        "w": w,
        "h": h,
        "camera_z": camera_z,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--projection-mode", choices=("smirk", "legacy"), default="smirk",
                        help="smirk: native camera conversion; legacy: reproduce old exports")
    parser.add_argument("--track-params", type=Path, required=True)
    parser.add_argument("--canonical-dataset", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument(
        "--focal",
        type=float,
        default=None,
        help="Override fl_x/fl_y (default: inherit from canonical transforms).",
    )
    parser.add_argument(
        "--camera-z",
        type=float,
        default=None,
        help="Override fixed-camera z (default: inherit from canonical w2c).",
    )
    parser.add_argument(
        "--opencv-y-flip",
        action="store_true",
        help="Historical camera-only flip; requires --projection-mode legacy.",
    )
    parser.add_argument(
        "--flame-y-flip",
        action="store_true",
        help="Legacy: bake Rx(pi) into FLAME rotation+translation (LBS; usually wrong).",
    )
    parser.add_argument(
        "--no-opencv-flip",
        action="store_true",
        help="Historical no-flip alias; requires --projection-mode legacy.",
    )
    args = parser.parse_args()

    if args.projection_mode != "legacy" and (args.opencv_y_flip or args.flame_y_flip or args.no_opencv_flip):
        parser.error("Historical flip flags require --projection-mode legacy")

    if args.target.exists():
        raise FileExistsError(f"Refusing to overwrite {args.target}")
    intr = inherit_from_canonical(args.canonical_dataset)
    focal = float(args.focal) if args.focal is not None else float(intr["fl_x"])
    camera_z = float(args.camera_z) if args.camera_z is not None else float(intr["camera_z"])
    width, height = int(intr["w"]), int(intr["h"])
    cx, cy = float(intr["cx"]), float(intr["cy"])
    if args.projection_mode == "smirk":
        # GA readCamerasFromTransforms/getProjectionMatrix use symmetric FOV,
        # square pixels and the image center; cx/cy/fl_y metadata are ignored.
        cx, cy = (width-1)/2, (height-1)/2
    print(
        f"Inherited from canonical: fl_x={focal:.3f} cx={cx:.3f} cy={cy:.3f} "
        f"w={width} h={height} camera_z={camera_z:.6f}"
    )

    params = torch.load(args.track_params, map_location="cpu", weights_only=False)
    arrays = {
        key: value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)
        for key, value in params.items()
        if key != "frame_names"
    }
    n_frames = arrays["exp"].shape[0]

    shape = arrays["shape"].mean(axis=0).astype(np.float32)
    static_offset = np.zeros((1, 5143, 3), dtype=np.float32)

    cam_flip = bool(args.opencv_y_flip) and not bool(args.no_opencv_flip)
    flame_flip = bool(args.flame_y_flip) and not bool(args.no_opencv_flip)
    if cam_flip and flame_flip:
        raise SystemExit("Pass only one of --opencv-y-flip (camera) or --flame-y-flip (FLAME bake)")

    corrected = args.projection_mode == "smirk"
    if corrected:
        # Validate every camera before creating the output dataset.
        for i in range(n_frames):
            smirk_perspective_translation(arrays["cam"][i], arrays["image_to_crop"][i],
                                          focal, cx, cy, camera_z)
    for split in ("train", "val", "test"):
        db = json.loads((args.canonical_dataset / f"transforms_{split}.json").read_text())
        for fr in db["frames"]:
            i = int(fr["timestep_index"])
            if i < 0 or i >= n_frames:
                raise ValueError(f"Canonical timestep {i} has no SMIRK parameters")
            if "frame_names" in params and params["frame_names"][i] != Path(fr["file_path"]).stem:
                raise ValueError(f"SMIRK/canonical frame mismatch at timestep {i}")

    args.target.mkdir(parents=True)
    flame_dir = args.target / "flame_param"
    flame_dir.mkdir()
    link_directory(args.canonical_dataset / "images", args.target / "images")
    link_directory(args.canonical_dataset / "fg_masks", args.target / "fg_masks")

    c2w = fixed_camera_c2w(camera_z, y_flip=corrected or cam_flip)

    for index in range(n_frames):
        expr = np.zeros((1, 100), dtype=np.float32)
        expr[0, : arrays["exp"].shape[1]] = arrays["exp"][index]
        translation_fn = smirk_perspective_translation if corrected else smirk_translation
        t_cv = translation_fn(
            arrays["cam"][index],
            arrays["image_to_crop"][index],
            focal,
            cx,
            cy,
            camera_z,
        )
        pose_aa = arrays["pose"][index].astype(np.float32)
        rotation = pose_aa.reshape(1, 3)
        translation = t_cv.reshape(1, 3)

        if flame_flip:
            rotation, translation = apply_flame_y_flip(pose_aa, t_cv)

        np.savez(
            flame_dir / f"{index:05d}.npz",
            translation=translation,
            rotation=rotation,
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

    angle_x = 2 * math.atan(width / (2 * focal))
    angle_y = 2 * math.atan(height / (2 * float(intr["fl_y"] if args.focal is None and not corrected else focal)))
    for split in ("train", "val", "test"):
        source_json = args.canonical_dataset / f"transforms_{split}.json"
        database = json.loads(source_json.read_text())
        for frame in database["frames"]:
            index = int(frame["timestep_index"])
            frame.update(
                {
                    "cx": cx,
                    "cy": cy,
                    "fl_x": focal,
                    "fl_y": float(intr["fl_y"] if args.focal is None and not corrected else focal),
                    "h": height,
                    "w": width,
                    "camera_angle_x": angle_x,
                    "camera_angle_y": angle_y,
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
                "fl_x": focal,
                "fl_y": float(intr["fl_y"] if args.focal is None and not corrected else focal),
                "h": height,
                "w": width,
                "camera_angle_x": angle_x,
                "camera_angle_y": angle_y,
            }
        )
        (args.target / f"transforms_{split}.json").write_text(json.dumps(database, indent=4))

    print(f"Exported {n_frames} SMIRK frames to {args.target}")
    (args.target / "smirk_export.json").write_text(json.dumps({
        "projection_mode": args.projection_mode,
        "camera_flip": corrected or cam_flip,
        "flame_flip": flame_flip,
        "track_params": str(args.track_params.resolve()),
        "canonical_dataset": str(args.canonical_dataset.resolve()),
        "note": "Weak-to-perspective approximation; eyelid offsets are not exported.",
    }, indent=2))
    if corrected:
        print("SMIRK projection: camera Rx(pi), native FLAME pose, world-space translation.")
    elif cam_flip:
        print("Y-down via fixed-camera Rx(pi) (--opencv-y-flip); raw FLAME pose.")
    elif flame_flip:
        print("Applied Rx(pi) to FLAME rotation+translation (--flame-y-flip).")
    else:
        print("No Rx(pi) (raw SMIRK pose, R=I camera).")


if __name__ == "__main__":
    main()
