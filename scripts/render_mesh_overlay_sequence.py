#!/usr/bin/env python3
"""Render FLAME wireframe overlays for every frame of a GA dataset.

Uses GaussianAvatars FlameHead (LBS) + the same OpenCV camera convention as
``readCamerasFromTransforms``. Writes ``overlay_%05d.png`` under --out_dir.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw


def ga_w2c_from_transform_matrix(c2w_json: np.ndarray) -> np.ndarray:
    c2w = np.asarray(c2w_json, dtype=np.float64).copy()
    c2w[:3, 1:3] *= -1
    return np.linalg.inv(c2w)


def load_frames_by_timestep(dataset: Path) -> dict[int, dict]:
    frames: dict[int, dict] = {}
    for split in ("train", "val", "test"):
        p = dataset / f"transforms_{split}.json"
        if not p.exists():
            continue
        db = json.loads(p.read_text())
        for frame in db["frames"]:
            frames[int(frame["timestep_index"])] = frame
    return frames


def unique_edges(faces: np.ndarray) -> np.ndarray:
    e = np.concatenate(
        [faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]],
        axis=0,
    )
    e = np.sort(e, axis=1)
    return np.unique(e, axis=0)


def blend(base: Image.Image, overlay: Image.Image, alpha: float) -> Image.Image:
    return Image.blend(base.convert("RGB"), overlay.convert("RGB"), alpha)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--ga_root", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--edge_stride", type=int, default=2, help="Keep every Nth edge for speed/clarity")
    parser.add_argument("--color", default="0,220,0", help="RGB wireframe color")
    parser.add_argument("--alpha", type=float, default=0.55)
    args = parser.parse_args()

    sys.path.insert(0, str(args.ga_root))
    from flame_model.flame import FlameHead

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    flame = FlameHead(shape_params=300, expr_params=100).to(device).eval()
    faces = flame.faces.detach().cpu().numpy().astype(np.int32)
    edges = unique_edges(faces)[:: max(1, args.edge_stride)]

    frames = load_frames_by_timestep(args.dataset)
    if not frames:
        raise SystemExit(f"No transforms_*.json frames in {args.dataset}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    color = tuple(int(x) for x in args.color.split(","))
    timesteps = sorted(frames.keys())

    with torch.no_grad():
        for t in timesteps:
            frame = frames[t]
            npz_path = args.dataset / frame["flame_param_path"]
            img_path = args.dataset / frame["file_path"]
            if not npz_path.exists() or not img_path.exists():
                print(f"skip missing {t}")
                continue

            d = np.load(npz_path)
            shape = torch.from_numpy(np.asarray(d["shape"], dtype=np.float32).reshape(1, -1)).to(device)
            if shape.shape[1] < 300:
                shape = torch.nn.functional.pad(shape, (0, 300 - shape.shape[1]))
            expr = torch.from_numpy(np.asarray(d["expr"], dtype=np.float32).reshape(1, -1)).to(device)
            if expr.shape[1] < 100:
                expr = torch.nn.functional.pad(expr, (0, 100 - expr.shape[1]))
            rotation = torch.from_numpy(np.asarray(d["rotation"], dtype=np.float32).reshape(1, 3)).to(device)
            neck = torch.from_numpy(np.asarray(d["neck_pose"], dtype=np.float32).reshape(1, 3)).to(device)
            jaw = torch.from_numpy(np.asarray(d["jaw_pose"], dtype=np.float32).reshape(1, 3)).to(device)
            eyes = torch.from_numpy(np.asarray(d["eyes_pose"], dtype=np.float32).reshape(1, 6)).to(device)
            transl = torch.from_numpy(np.asarray(d["translation"], dtype=np.float32).reshape(1, 3)).to(device)
            static = None
            if "static_offset" in d.files:
                so = np.asarray(d["static_offset"], dtype=np.float32)
                if so.size and so.shape[-1] == 3:
                    static = torch.from_numpy(so.reshape(1, -1, 3)).to(device)
                    # FlameHead may have extra teeth verts; pad zeros if needed.
                    n_expected = flame.v_template.shape[0]
                    if static.shape[1] < n_expected:
                        pad = torch.zeros(1, n_expected - static.shape[1], 3, device=device, dtype=static.dtype)
                        static = torch.cat([static, pad], dim=1)
                    elif static.shape[1] > n_expected:
                        static = static[:, :n_expected]

            verts = flame(
                shape,
                expr,
                rotation,
                neck,
                jaw,
                eyes,
                transl,
                return_landmarks=False,
                static_offset=static,
            )[0].detach().cpu().numpy()

            w2c = ga_w2c_from_transform_matrix(frame["transform_matrix"])
            verts_c = (w2c[:3, :3] @ verts.T).T + w2c[:3, 3]
            K = np.array(
                [
                    [frame["fl_x"], 0.0, frame["cx"]],
                    [0.0, frame["fl_y"], frame["cy"]],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float64,
            )
            z = np.clip(verts_c[:, 2:3], 1e-6, None)
            uv = (K @ (verts_c / z).T).T[:, :2]

            base = Image.open(img_path).convert("RGB")
            wire = base.copy()
            draw = ImageDraw.Draw(wire)
            w, h = base.size
            for i0, i1 in edges:
                if verts_c[i0, 2] <= 0.05 or verts_c[i1, 2] <= 0.05:
                    continue
                u0, v0 = uv[i0]
                u1, v1 = uv[i1]
                if not np.isfinite([u0, v0, u1, v1]).all():
                    continue
                if (
                    (u0 < -50 and u1 < -50)
                    or (u0 > w + 50 and u1 > w + 50)
                    or (v0 < -50 and v1 < -50)
                    or (v0 > h + 50 and v1 > h + 50)
                ):
                    continue
                draw.line([(u0, v0), (u1, v1)], fill=color, width=1)

            out = blend(base, wire, args.alpha)
            out.save(args.out_dir / f"overlay_{t:05d}.png")
            print(f"\r{t:05d}/{timesteps[-1]:05d}", end="", flush=True)

    print(f"\nWrote overlays to {args.out_dir}")


if __name__ == "__main__":
    main()
