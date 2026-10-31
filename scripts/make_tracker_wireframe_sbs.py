#!/usr/bin/env python3
"""Side-by-side exported LBS mesh overlays on GA renders for MICA | SMIRK | VHAP."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from ga_mesh_utils import (decode_flame, draw_landmarks, draw_wire, load_flame,
                           load_frames_by_timestep, project_vertices, unique_edges)


def letterbox(im: Image.Image, size: int, fill=(255, 255, 255)) -> Image.Image:
    w, h = im.size
    scale = size / max(w, h)
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    im = im.resize((nw, nh), Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (size, size), fill)
    canvas.paste(im, ((size - nw) // 2, (size - nh) // 2))
    return canvas


def _font(size: int = 18):
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size)
    except Exception:
        return ImageFont.load_default()


def draw_label(im: Image.Image, text: str, *, corner: str = "tl") -> Image.Image:
    out = im.copy()
    draw = ImageDraw.Draw(out)
    font = _font(18)
    pad = 6
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    w, h = out.size
    if corner == "br":
        x0, y0 = w - tw - 2 * pad, h - th - 2 * pad
    else:
        x0, y0 = 0, 0
    draw.rectangle([x0, y0, x0 + tw + 2 * pad, y0 + th + 2 * pad], fill=(0, 0, 0))
    draw.text((x0 + pad, y0 + pad), text, fill=(255, 255, 255), font=font)
    return out


def banner(width: int, text: str, height: int = 28) -> Image.Image:
    bar = Image.new("RGB", (width, height), (20, 20, 20))
    draw = ImageDraw.Draw(bar)
    font = _font(16)
    draw.text((8, 5), text, fill=(255, 255, 255), font=font)
    return bar


def project_wire(
    flame,
    edges: np.ndarray,
    flame_npz: Path,
    frame: dict,
    base: Image.Image,
    color: tuple[int, int, int],
    alpha: float,
    device: torch.device,
) -> Image.Image:
    with np.load(flame_npz) as data:
        verts, landmarks = decode_flame(flame, data, device)
    uv, depth = project_vertices(verts, frame, base.size)
    landmark_uv, _ = project_vertices(landmarks, frame, base.size)
    return draw_landmarks(draw_wire(base, uv, depth, edges, color, alpha), landmark_uv)


def find_render_dir(ga_root: Path, data_id: str) -> Path:
    cands = sorted(ga_root.glob(f"output/{data_id}_256*_25k/train/ours_*"))
    if not cands:
        raise SystemExit(f"No GA renders for {data_id}")
    return cands[-1] / "renders"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ga-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--panel", type=int, default=256)
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--edge-stride", type=int, default=2)
    ap.add_argument("--alpha", type=float, default=0.55)
    ap.add_argument("--color", default="0,220,0")
    ap.add_argument("--mica-id", required=True)
    ap.add_argument("--smirk-id", required=True)
    ap.add_argument("--vhap-id", required=True)
    ap.add_argument("--title", default="")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    flame = load_flame(args.ga_root, device)
    faces = flame.faces.detach().cpu().numpy().astype(np.int32)
    edges = unique_edges(faces)[:: max(1, args.edge_stride)]
    color = tuple(int(x) for x in args.color.split(","))

    trackers = [
        ("MICA", args.mica_id),
        ("SMIRK", args.smirk_id),
        ("VHAP", args.vhap_id),
    ]
    meta = []
    for name, data_id in trackers:
        ds = args.ga_root / "data" / data_id
        frames = load_frames_by_timestep(ds)
        renders = find_render_dir(args.ga_root, data_id)
        render_imgs = sorted(renders.glob("*.png"))
        # Prefer matching by sorted timesteps; GA train renders follow train+val order.
        # Use shared timestep intersection later; map timestep -> flame + frame.
        flame_dir = ds / "flame_param"
        meta.append(
            {
                "name": name,
                "ds": ds,
                "frames": frames,
                "render_imgs": render_imgs,
                "flame_dir": flame_dir,
            }
        )

    # Shared timesteps present in all three
    shared = sorted(set.intersection(*[set(m["frames"].keys()) for m in meta]))
    if not shared:
        raise SystemExit("No shared timesteps across trackers")

    # Build per-tracker list of (timestep, render_path) by index order of shared timesteps.
    # Renders are usually ordered by increasing timestep within train/val concat.
    # Match render index to timestep by sorting frames as stored in transforms_train then val.
    def ordered_timesteps(ds: Path) -> list[int]:
        out = []
        for split in ("train", "val", "test"):
            p = ds / f"transforms_{split}.json"
            if not p.exists():
                continue
            for fr in json.loads(p.read_text())["frames"]:
                out.append(int(fr["timestep_index"]))
        return out

    for m in meta:
        order = ordered_timesteps(m["ds"])
        t2r = {}
        for i, t in enumerate(order):
            if i < len(m["render_imgs"]):
                t2r[t] = m["render_imgs"][i]
        m["t2r"] = t2r

    shared = [t for t in shared if all(t in m["t2r"] for m in meta)]
    if not shared:
        raise SystemExit("No shared timesteps with renders")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.parent / (args.out.stem + "_frames")
    tmp.mkdir(parents=True, exist_ok=True)

    for fi, t in enumerate(shared):
        panels = []
        for m in meta:
            frame = m["frames"][t]
            # inject w/h from transforms if missing on frame
            if "w" not in frame or "h" not in frame:
                for split in ("train", "val", "test"):
                    p = m["ds"] / f"transforms_{split}.json"
                    if p.exists():
                        db = json.loads(p.read_text())
                        frame = {**{"w": db.get("w", 512), "h": db.get("h", 512)}, **frame}
                        break
            # Project onto native GA render, then letterbox so aspect (VHAP) stays correct.
            base = Image.open(m["t2r"][t]).convert("RGB")
            rel = frame.get("flame_param_path")
            flame_npz = (m["ds"] / rel) if rel else (m["flame_dir"] / f"{t:05d}.npz")
            if not flame_npz.exists():
                raise SystemExit(f"Missing flame for {m['name']} t={t}: {flame_npz}")
            wired = project_wire(
                flame, edges, flame_npz, frame, base, color, args.alpha, device
            )
            panels.append(draw_label(letterbox(wired, args.panel), m["name"]))

        row = Image.new("RGB", (args.panel * 3, args.panel), (255, 255, 255))
        for i, p in enumerate(panels):
            row.paste(p, (i * args.panel, 0))
        if args.title:
            bar = banner(row.width, f"{args.title}  |  MICA   SMIRK   VHAP  |  t={t:05d}")
            sbs = Image.new("RGB", (row.width, bar.height + row.height), (20, 20, 20))
            sbs.paste(bar, (0, 0))
            sbs.paste(row, (0, bar.height))
        else:
            sbs = row
        sbs.save(tmp / f"{fi:05d}.png")
        print(f"\r{fi+1}/{len(shared)}", end="", flush=True)

    print()
    # encode
    import subprocess

    ffmpeg = "ffmpeg"
    for cand in (
        Path.home() / "miniconda3/envs/tracker/bin/ffmpeg",
        Path("/usr/bin/ffmpeg"),
    ):
        if cand.exists():
            ffmpeg = str(cand)
            break
    cmd = [
        ffmpeg,
        "-y",
        "-framerate",
        str(args.fps),
        "-i",
        str(tmp / "%05d.png"),
        "-vf",
        "scale=trunc(iw/2)*2:trunc(ih/2)*2",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(args.out),
    ]
    subprocess.check_call(cmd)
    print(f"Wrote {args.out} ({len(shared)} frames)")


if __name__ == "__main__":
    main()
