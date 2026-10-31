#!/usr/bin/env python3
"""Render FLAME wireframe overlays for every frame of a GA dataset.

Uses GaussianAvatars FlameHead (LBS) + the same OpenCV camera convention as
``readCamerasFromTransforms``. Writes ``overlay_%05d.png`` under --out_dir.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from ga_mesh_utils import (decode_flame, draw_landmarks, draw_wire, load_flame,
                           load_frames_by_timestep, project_vertices, unique_edges)


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

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    flame = load_flame(args.ga_root, device)
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

            with np.load(npz_path) as data:
                verts, landmarks = decode_flame(flame, data, device)
            base = Image.open(img_path).convert("RGB")
            uv, depth = project_vertices(verts, frame, base.size)
            landmark_uv, _ = project_vertices(landmarks, frame, base.size)
            out = draw_landmarks(draw_wire(base, uv, depth, edges, color, args.alpha), landmark_uv)
            out.save(args.out_dir / f"overlay_{t:05d}.png")
            print(f"\r{t:05d}/{timesteps[-1]:05d}", end="", flush=True)

    print(f"\nWrote overlays to {args.out_dir}")


if __name__ == "__main__":
    main()
