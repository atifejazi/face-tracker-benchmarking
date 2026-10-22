#!/usr/bin/env python3
"""Diagnose FLAME mesh uprightness for a GA export (SMIRK/MICA).

Checks camera-space chin vs forehead after applying flame_param rotation +
translation with the OpenCV w2c implied by transforms_*.json (same Y/Z flip
as GaussianAvatars' readCamerasFromTransforms). Optionally draws a 2D
landmark overlay on the training image.

Examples:
  python scripts/diagnose_ga_mesh_overlay.py \\
    --dataset "$GA_ROOT/data/ns024_SEN01_smirk" --frames 0,5,10 \\
    --out_dir "$REPO_ROOT/runs/smirk_mesh_diag"
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import sys
from pathlib import Path

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None


def rodrigues(aa: np.ndarray) -> np.ndarray:
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


def load_flame_template(flame_pkl: Path) -> np.ndarray:
    """Load v_template without importing real chumpy (breaks on modern numpy)."""
    import sys
    import types

    class _DummyModule(types.ModuleType):
        def __getattr__(self, item):
            class _Dummy:
                def __setstate__(self, state):
                    self._state = state

                def __array__(self, dtype=None):
                    s = getattr(self, "_state", None)
                    if isinstance(s, tuple) and s:
                        return np.asarray(s[0], dtype=dtype)
                    return np.asarray(s, dtype=dtype)

                @property
                def r(self):
                    return np.asarray(self)

            return _Dummy

    for name in ("chumpy", "chumpy.ch", "chumpy.re", "chumpy.api"):
        sys.modules.setdefault(name, _DummyModule(name))

    with open(flame_pkl, "rb") as f:
        data = pickle.load(f, encoding="latin1")
    vt = data["v_template"]
    if hasattr(vt, "r"):
        return np.asarray(vt.r, dtype=np.float64)
    return np.asarray(vt, dtype=np.float64)


def ga_w2c_from_transform_matrix(c2w_json: np.ndarray) -> np.ndarray:
    """Match GaussianAvatars scene/dataset_readers.readCamerasFromTransforms."""
    c2w = np.asarray(c2w_json, dtype=np.float64).copy()
    c2w[:3, 1:3] *= -1
    return np.linalg.inv(c2w)


def project(K: np.ndarray, X_cam: np.ndarray) -> np.ndarray:
    z = np.clip(X_cam[:, 2:3], 1e-6, None)
    uv = (K @ (X_cam / z).T).T
    return uv[:, :2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True, help="GA data/<id> folder")
    parser.add_argument("--frames", default="0,1,2", help="Comma-separated timestep indices")
    parser.add_argument("--flame_pkl", type=Path, default=None)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument(
        "--forehead_idx",
        type=int,
        default=336,
        help="FLAME vertex near forehead (heuristic; override if needed)",
    )
    parser.add_argument(
        "--chin_idx",
        type=int,
        default=3611,
        help="FLAME vertex near chin (heuristic; override if needed)",
    )
    args = parser.parse_args()

    ds = args.dataset
    transforms_path = ds / "transforms_train.json"
    if not transforms_path.exists():
        raise SystemExit(f"Missing {transforms_path}")

    db = json.loads(transforms_path.read_text())
    frames_by_t = {int(f["timestep_index"]): f for f in db["frames"]}

    flame_pkl = args.flame_pkl
    if flame_pkl is None:
        candidates = [
            Path(__file__).resolve().parents[2] / "GaussianAvatars/flame_model/assets/flame/generic_model.pkl",
        ]
        import os

        ga = os.environ.get("GA_ROOT")
        smirk = os.environ.get("SMIRK_ROOT")
        if ga:
            candidates.insert(0, Path(ga) / "flame_model/assets/flame/generic_model.pkl")
        if smirk:
            candidates.insert(0, Path(smirk) / "assets/FLAME2020/generic_model.pkl")
        for c in candidates:
            if c.exists():
                flame_pkl = c
                break
    if flame_pkl is None or not flame_pkl.exists():
        raise SystemExit("Could not find FLAME generic_model.pkl; pass --flame_pkl")

    try:
        v_template = load_flame_template(flame_pkl)
    except Exception as e:
        raise SystemExit(f"Failed to load FLAME template from {flame_pkl}: {e}") from e

    # Prefer extreme-Y vertices if default indices look wrong for this topology size.
    if args.forehead_idx >= len(v_template) or args.chin_idx >= len(v_template):
        args.forehead_idx = int(np.argmax(v_template[:, 1]))
        args.chin_idx = int(np.argmin(v_template[:, 1]))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    report = []

    for token in args.frames.split(","):
        t = int(token.strip())
        if t not in frames_by_t:
            print(f"skip missing timestep {t}")
            continue
        frame = frames_by_t[t]
        npz_path = ds / frame["flame_param_path"]
        flame = np.load(npz_path)
        R = rodrigues(flame["rotation"][0])
        transl = np.asarray(flame["translation"][0], dtype=np.float64)

        # Shape/expr omitted: uprightness is dominated by global R + camera.
        verts_w = (R @ v_template.T).T + transl
        w2c = ga_w2c_from_transform_matrix(frame["transform_matrix"])
        R_w2c, t_w2c = w2c[:3, :3], w2c[:3, 3]
        verts_c = (R_w2c @ verts_w.T).T + t_w2c

        y_fore = float(verts_c[args.forehead_idx, 1])
        y_chin = float(verts_c[args.chin_idx, 1])
        # OpenCV camera: Y down ⇒ forehead should have smaller Y than chin when upright.
        upright = y_fore < y_chin
        status = "UPRIGHT" if upright else "UPSIDE_DOWN"
        line = {
            "timestep": t,
            "status": status,
            "y_forehead_cam": y_fore,
            "y_chin_cam": y_chin,
            "rotation_aa": flame["rotation"][0].tolist(),
            "translation": transl.tolist(),
        }
        report.append(line)
        print(f"frame {t:05d}: {status}  y_fore={y_fore:.4f} y_chin={y_chin:.4f}")

        if cv2 is not None:
            img_path = ds / frame["file_path"]
            if img_path.exists():
                img = cv2.imread(str(img_path))
                K = np.array(
                    [
                        [frame["fl_x"], 0, frame["cx"]],
                        [0, frame["fl_y"], frame["cy"]],
                        [0, 0, 1],
                    ],
                    dtype=np.float64,
                )
                # Subsample verts for overlay + mark forehead/chin.
                idx = np.linspace(0, len(verts_c) - 1, num=min(800, len(verts_c)), dtype=int)
                uv = project(K, verts_c[idx])
                for u, v in uv.astype(int):
                    if 0 <= u < img.shape[1] and 0 <= v < img.shape[0]:
                        cv2.circle(img, (u, v), 1, (0, 255, 0), -1)
                for vid, color in (
                    (args.forehead_idx, (255, 0, 0)),
                    (args.chin_idx, (0, 0, 255)),
                ):
                    u, v = project(K, verts_c[vid : vid + 1])[0].astype(int)
                    cv2.circle(img, (int(u), int(v)), 5, color, -1)
                    cv2.putText(
                        img,
                        "F" if vid == args.forehead_idx else "C",
                        (int(u) + 6, int(v)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        color,
                        1,
                        cv2.LINE_AA,
                    )
                cv2.putText(
                    img,
                    status,
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 255, 0) if upright else (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )
                out_img = args.out_dir / f"overlay_{t:05d}.png"
                cv2.imwrite(str(out_img), img)

    out_json = args.out_dir / "mesh_upright_report.json"
    out_json.write_text(json.dumps(report, indent=2))
    n_bad = sum(1 for r in report if r["status"] != "UPRIGHT")
    print(f"Wrote {out_json} ({n_bad}/{len(report)} upside-down)")
    sys.exit(1 if n_bad else 0)


if __name__ == "__main__":
    main()
