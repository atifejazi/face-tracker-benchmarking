#!/usr/bin/env python3
"""Gate 1: camera-space vertex equivalence between legacy and relative_cancel exports.

Loads FlameHead from GaussianAvatars, evaluates both datasets on selected frames,
transforms world verts into OpenCV camera space via each dataset's transform_matrix,
and reports max ||x_rel - x_legacy||_2. Pass if max < 1e-5 m.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch


def ga_w2c(c2w_json) -> np.ndarray:
    c2w = np.asarray(c2w_json, dtype=np.float64).copy()
    c2w[:3, 1:3] *= -1
    return np.linalg.inv(c2w)


def load_dataset_frame(ds: Path, i: int):
    train = json.loads((ds / "transforms_train.json").read_text())
    # Prefer matching timestep in train; fall back to any transforms_*.json
    frames = {f["timestep_index"]: f for f in train["frames"]}
    if i not in frames:
        for name in ("transforms_val.json", "transforms_test.json"):
            p = ds / name
            if p.exists():
                for f in json.loads(p.read_text())["frames"]:
                    frames[f["timestep_index"]] = f
    if i not in frames:
        raise KeyError(f"timestep {i} not in {ds}")
    fr = frames[i]
    flame = dict(np.load(ds / fr["flame_param_path"]))
    w2c = ga_w2c(fr["transform_matrix"])
    return flame, w2c


def flame_forward(flame_model, flame: dict, device: torch.device):
    def t(key, shape=None):
        x = torch.from_numpy(np.asarray(flame[key], dtype=np.float32))
        if shape is not None:
            x = x.reshape(*shape)
        return x.to(device)

    shape = t("shape").reshape(-1)
    # FlameHead expects shape [B, n_shape]
    n_shape = flame_model.n_shape_params
    if shape.numel() < n_shape:
        shape = torch.cat([shape, torch.zeros(n_shape - shape.numel(), device=device)])
    shape = shape[:n_shape][None]
    expr = t("expr", (1, -1))
    if expr.shape[-1] < flame_model.n_expr_params:
        expr = torch.cat(
            [
                expr,
                torch.zeros(
                    1, flame_model.n_expr_params - expr.shape[-1], device=device
                ),
            ],
            dim=-1,
        )
    expr = expr[:, : flame_model.n_expr_params]
    static = t("static_offset")
    if static.ndim == 2:
        static = static[None]
    # FlameHead may have more verts than 5143 if add_teeth; pad static_offset
    nv = flame_model.v_template.shape[0]
    if static.shape[1] < nv:
        pad = torch.zeros(1, nv - static.shape[1], 3, device=device, dtype=static.dtype)
        static = torch.cat([static, pad], dim=1)
    elif static.shape[1] > nv:
        static = static[:, :nv]

    verts = flame_model(
        shape,
        expr,
        t("rotation", (1, 3)),
        t("neck_pose", (1, 3)),
        t("jaw_pose", (1, 3)),
        t("eyes_pose", (1, 6)),
        t("translation", (1, 3)),
        zero_centered_at_root_node=False,
        return_landmarks=False,
        return_verts_cano=False,
        static_offset=static,
    )
    return verts[0].detach().cpu().numpy().astype(np.float64)


def to_cam(verts_w: np.ndarray, w2c: np.ndarray) -> np.ndarray:
    R, t = w2c[:3, :3], w2c[:3, 3]
    return (R @ verts_w.T).T + t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--legacy", type=Path, required=True)
    ap.add_argument("--relative", type=Path, required=True)
    ap.add_argument("--ga-root", type=Path, required=True)
    ap.add_argument("--frames", default="0,10,30,50", help="comma-separated timesteps")
    ap.add_argument("--atol", type=float, default=1e-5)
    args = ap.parse_args()

    sys.path.insert(0, str(args.ga_root))
    from flame_model.flame import FlameHead

    device = torch.device("cpu")
    flame_model = FlameHead(shape_params=300, expr_params=100).to(device)
    flame_model.eval()

    frame_ids = [int(x) for x in args.frames.split(",") if x.strip() != ""]
    max_err = 0.0
    per = []
    for i in frame_ids:
        fl_leg, w2c_leg = load_dataset_frame(args.legacy, i)
        fl_rel, w2c_rel = load_dataset_frame(args.relative, i)
        v_leg = flame_forward(flame_model, fl_leg, device)
        v_rel = flame_forward(flame_model, fl_rel, device)
        # Compare face verts only (exclude teeth if counts differ)
        n = min(v_leg.shape[0], v_rel.shape[0], 5023)  # FLAME face verts
        x_leg = to_cam(v_leg[:n], w2c_leg)
        x_rel = to_cam(v_rel[:n], w2c_rel)
        err = float(np.linalg.norm(x_rel - x_leg, axis=1).max())
        mean_err = float(np.linalg.norm(x_rel - x_leg, axis=1).mean())
        per.append({"frame": i, "max_err_m": err, "mean_err_m": mean_err})
        max_err = max(max_err, err)
        print(f"frame {i:05d}: max_err={err:.3e} m  mean_err={mean_err:.3e} m")

    passed = max_err < args.atol
    summary = {
        "max_err_m": max_err,
        "atol_m": args.atol,
        "passed": passed,
        "frames": per,
        "legacy": str(args.legacy),
        "relative": str(args.relative),
    }
    print(json.dumps(summary, indent=2))
    if not passed:
        raise SystemExit(
            f"FAIL: max camera-space vertex error {max_err:.3e} m >= {args.atol}"
        )
    print(f"PASS: max error {max_err:.3e} m < {args.atol}")


if __name__ == "__main__":
    main()
