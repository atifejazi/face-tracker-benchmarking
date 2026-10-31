#!/usr/bin/env python3
"""Report whether GA camera rotation is affixed (constant) or moves per frame.

Reads transforms_train.json and summarizes OpenCV w2c rotation after the same
Y/Z column flip GaussianAvatars applies in readCamerasFromTransforms.

Legacy moving-camera MICA export: moving camera (R_constant false).
Relative cancel+center (``--cancel-camera-movement``): affixed / R_constant true
with R0_is_identity **false** (VHAP-like ~diag(1,-1,-1) convention on camera).
The old absolute-bake cancel that forced R=I is gone — do not expect identity R.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def ga_w2c(c2w_json) -> np.ndarray:
    c2w = np.asarray(c2w_json, dtype=np.float64).copy()
    c2w[:3, 1:3] *= -1
    return np.linalg.inv(c2w)


def save_camera_path_png(Ts: np.ndarray, out_path: Path, title: str):
    """Simple camera-center path plot without matplotlib."""
    from PIL import Image, ImageDraw

    W, H = 640, 480
    img = Image.new("RGB", (W, H), (24, 24, 24))
    draw = ImageDraw.Draw(img)
    draw.text((10, 8), title, fill=(240, 240, 240))
    # Plot T_x vs frame index (top) and T_z vs frame (bottom-ish)
    n = len(Ts)
    pad = 50
    xs = np.linspace(pad, W - pad, n)

    def series(vals, y0, y1, color, label):
        vmin, vmax = float(vals.min()), float(vals.max())
        if abs(vmax - vmin) < 1e-9:
            vmax = vmin + 1.0
        ys = y1 - (vals - vmin) / (vmax - vmin) * (y1 - y0)
        pts = list(zip(xs.tolist(), ys.tolist()))
        if len(pts) >= 2:
            draw.line(pts, fill=color, width=2)
        for x, y in pts[:: max(1, n // 12)]:
            draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=color)
        draw.text((pad, y0 - 16), f"{label}  [{vmin:.3f}, {vmax:.3f}]", fill=color)

    series(Ts[:, 0], 60, 200, (255, 120, 120), "camera T_x")
    series(Ts[:, 1], 230, 370, (120, 255, 120), "camera T_y")
    series(Ts[:, 2], 400, 460, (120, 160, 255), "camera T_z (depth)")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--transforms", type=Path, required=True)
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument(
        "--plot",
        type=Path,
        default=None,
        help="Optional PNG path for camera translation-vs-frame plot",
    )
    args = parser.parse_args()

    db = json.loads(args.transforms.read_text())
    frames = db["frames"]
    if not frames:
        raise SystemExit("no frames")

    Rs = []
    Ts = []
    for f in frames:
        w2c = ga_w2c(f["transform_matrix"])
        Rs.append(w2c[:3, :3])
        Ts.append(w2c[:3, 3])
    Rs = np.stack(Rs)
    Ts = np.stack(Ts)

    R0 = Rs[0]
    T0 = Ts[0]
    R_delta = np.linalg.norm(Rs - R0[None], axis=(1, 2))
    T_delta = np.linalg.norm(Ts - T0[None], axis=1)
    R_const = bool(np.all(R_delta < args.atol))
    T_const = bool(np.all(T_delta < args.atol))
    R_identity = bool(np.allclose(R0, np.eye(3), atol=args.atol))

    summary = {
        "n_frames": len(frames),
        "R_constant": R_const,
        "T_constant": T_const,
        "R0_is_identity": R_identity,
        "R_delta_max": float(R_delta.max()),
        "T_delta_max": float(T_delta.max()),
        "R0": R0.tolist(),
        "T0": T0.tolist(),
        "interpretation": (
            "fixed/affixed camera"
            if R_const and T_const
            else ("constant rotation, moving translation" if R_const else "moving camera (not affixed)")
        ),
    }
    print(json.dumps(summary, indent=2))

    if args.plot is not None:
        save_camera_path_png(Ts, args.plot, summary["interpretation"])
        print(f"Wrote camera path plot: {args.plot}")


if __name__ == "__main__":
    main()
