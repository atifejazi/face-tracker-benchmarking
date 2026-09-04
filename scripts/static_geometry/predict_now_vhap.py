#!/usr/bin/env python3
"""Predict NoW validation meshes with native VHAP (1-frame monocular track).

Each NoW image is treated as a 1-frame video: NoW face crop → RVM preprocess →
VHAP ``track.py`` with **default (native) stage budgets** (not MultiREX lightB).

Writes predicted_meshes/<subject>/<challenge>/<IMG>.{ply,npy} in millimeters.
Run inside the ``VHAP`` conda env.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import trimesh
from tqdm import tqdm

VHAP_ROOT = Path(os.environ.get("VHAP_ROOT", ""))
if not VHAP_ROOT:
    raise SystemExit("Set VHAP_ROOT (see config/paths.env)")
sys.path.insert(0, str(VHAP_ROOT))

from vhap.model.flame import FlameHead  # noqa: E402


def now_seven_from_68(lmk68: np.ndarray) -> np.ndarray:
    landmark_51 = lmk68[17:]
    return landmark_51[[19, 22, 25, 28, 16, 31, 37]]


def crop_face_bgr(image_bgr: np.ndarray, bbx: dict, scale: float = 1.6) -> np.ndarray:
    h, w = image_bgr.shape[:2]
    left, right, top, bottom = bbx["left"], bbx["right"], bbx["top"], bbx["bottom"]
    old_size = (right - left + bottom - top) / 2.0
    cx = right - (right - left) / 2.0
    cy = bottom - (bottom - top) / 2.0
    size = int(old_size * scale)
    x0 = max(0, int(cx - size / 2))
    y0 = max(0, int(cy - size / 2))
    x1 = min(w, int(cx + size / 2))
    y1 = min(h, int(cy + size / 2))
    return image_bgr[y0:y1, x0:x1]


def resize_long_side(image_bgr: np.ndarray, long_side: int) -> np.ndarray:
    h, w = image_bgr.shape[:2]
    m = max(h, w)
    if m <= long_side:
        return image_bgr
    s = long_side / m
    return cv2.resize(image_bgr, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)


def write_one_frame_mp4(image_bgr: np.ndarray, out_mp4: Path, fps: int = 30) -> None:
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    if out_mp4.exists():
        out_mp4.unlink()
    h, w = image_bgr.shape[:2]
    writer = cv2.VideoWriter(
        str(out_mp4),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Cannot open VideoWriter for {out_mp4}")
    writer.write(image_bgr)
    writer.release()


def run_logged(cmd: list[str], log_path: Path, cwd: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w") as log:
        subprocess.run(cmd, cwd=str(cwd), check=True, stdout=log, stderr=subprocess.STDOUT)


def latest_tracked_params(output_root: Path) -> Path:
    candidates = list(output_root.rglob("tracked_flame_params_*.npz"))
    if not candidates:
        raise FileNotFoundError(f"No tracked_flame_params under {output_root}")

    def epoch(path: Path) -> int:
        return int(path.stem.split("_")[-1])

    return max(candidates, key=epoch)


def decode_mesh_mm(npz_path: Path, flame: FlameHead, device: str):
    data = np.load(npz_path, allow_pickle=True)
    i = 0
    shape = torch.from_numpy(np.asarray(data["shape"])).float().to(device)[None]
    expr = torch.from_numpy(np.asarray(data["expr"][i : i + 1])).float().to(device)
    rotation = torch.from_numpy(np.asarray(data["rotation"][i : i + 1])).float().to(device)
    neck = torch.from_numpy(np.asarray(data["neck_pose"][i : i + 1])).float().to(device)
    jaw = torch.from_numpy(np.asarray(data["jaw_pose"][i : i + 1])).float().to(device)
    eyes = torch.from_numpy(np.asarray(data["eyes_pose"][i : i + 1])).float().to(device)
    translation = torch.from_numpy(np.asarray(data["translation"][i : i + 1])).float().to(device)
    static_offset = None
    if "static_offset" in data.files:
        static_offset = torch.from_numpy(np.asarray(data["static_offset"])).float().to(device)

    with torch.no_grad():
        verts, lmks = flame(
            shape,
            expr,
            rotation,
            neck,
            jaw,
            eyes,
            translation,
            static_offset=static_offset,
        )
    verts_mm = verts[0].detach().cpu().numpy() * 1000.0
    lmk68 = lmks[0, :68].detach().cpu().numpy()
    landmark_7 = now_seven_from_68(lmk68) * 1000.0
    return verts_mm, landmark_7


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset_root",
        type=Path,
        default=Path(os.environ.get("NOW_DATASET", "data/now-dataset/dataset")),
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path(os.environ.get("NOW_RESULTS", "runs/now")) / "vhap/predicted_meshes",
    )
    parser.add_argument(
        "--work_root",
        type=Path,
        default=Path(os.environ.get("NOW_RESULTS", "runs/now")) / "vhap_work",
    )
    parser.add_argument("--image_list", type=Path, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--long_side", type=int, default=1024)
    parser.add_argument("--crop_scale", type=float, default=1.6)
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument(
        "--keep_work",
        action="store_true",
        help="Keep per-image preprocess/track workdirs (default: delete after success)",
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional cap for debugging")
    args = parser.parse_args()

    image_list = args.image_list or (args.dataset_root / "imagepathsvalidation.txt")
    pictures = args.dataset_root / "NoW_Dataset/final_release_version/iphone_pictures"
    detected = args.dataset_root / "NoW_Dataset/final_release_version/detected_face"
    paths = [ln.strip() for ln in image_list.read_text().splitlines() if ln.strip()]
    if args.limit is not None:
        paths = paths[: args.limit]

    # FlameHead + subprocesses resolve assets relative to VHAP root
    os.chdir(VHAP_ROOT)

    data_root = args.work_root / "data"
    track_root = args.work_root / "track"
    log_dir = args.work_root / "logs"
    data_root.mkdir(parents=True, exist_ok=True)
    track_root.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    flame = FlameHead(300, 100, add_teeth=True).to(args.device).eval()
    faces = flame.faces.detach().cpu().numpy() if torch.is_tensor(flame.faces) else np.asarray(flame.faces)

    ok = fail = 0
    for rel in tqdm(paths, desc="VHAP NoW native"):
        subject, challenge, filename = rel.split("/")
        stem = Path(filename).stem
        seq = f"{subject}__{challenge}__{stem}".replace(" ", "_")
        out_dir = args.output_dir / subject / challenge
        out_ply = out_dir / f"{stem}.ply"
        out_npy = out_dir / f"{stem}.npy"
        if args.skip_existing and out_ply.exists() and out_npy.exists():
            ok += 1
            continue

        img_path = pictures / rel
        bbx_path = detected / subject / challenge / f"{stem}.npy"
        if not img_path.exists() or not bbx_path.exists():
            print(f"missing {img_path} or {bbx_path}", flush=True)
            fail += 1
            continue

        seq_data = data_root / seq
        seq_track = track_root / f"{seq}_native_whiteBg_staticOffset"
        mp4 = data_root / f"{seq}.mp4"

        try:
            image = cv2.imread(str(img_path))
            if image is None:
                raise RuntimeError(f"failed to read {img_path}")
            bbx = np.load(bbx_path, allow_pickle=True, encoding="latin1").item()
            crop = crop_face_bgr(image, bbx, scale=args.crop_scale)
            crop = resize_long_side(crop, args.long_side)
            write_one_frame_mp4(crop, mp4)

            # Fresh preprocess dir (preprocess writes alongside mp4 stem)
            pre_dir = data_root / seq
            if pre_dir.exists():
                shutil.rmtree(pre_dir)
            # preprocess_video names folder from mp4 stem; ensure seq name matches
            # Use symlink/copy so sequence name == folder name expected by track
            if mp4.stem != seq:
                pass
            # Rename convention: write mp4 as {seq}.mp4 so folder is data/{seq}/
            run_logged(
                [
                    sys.executable,
                    "vhap/preprocess_video.py",
                    "--input",
                    str(mp4.resolve()),
                    "--matting_method",
                    "robust_video_matting",
                    "--target_fps",
                    "30",
                ],
                log_dir / f"{seq}_preprocess.log",
                VHAP_ROOT,
            )

            if seq_track.exists():
                shutil.rmtree(seq_track)
            seq_track.mkdir(parents=True, exist_ok=True)

            run_logged(
                [
                    sys.executable,
                    "vhap/track.py",
                    "--data.root_folder",
                    str(data_root.resolve()),
                    "--data.sequence",
                    seq,
                    "--exp.output_folder",
                    str(seq_track.resolve()),
                    "--batch_size",
                    "1",
                ],
                log_dir / f"{seq}_track.log",
                VHAP_ROOT,
            )

            params_path = latest_tracked_params(seq_track)
            verts_mm, landmark_7 = decode_mesh_mm(params_path, flame, args.device)

            out_dir.mkdir(parents=True, exist_ok=True)
            trimesh.Trimesh(vertices=verts_mm, faces=faces, process=False).export(str(out_ply))
            np.save(out_npy, landmark_7.astype(np.float32))
            ok += 1
            print(f"[{ok}/{len(paths)}] saved {out_ply.relative_to(args.output_dir)}", flush=True)
        except Exception as exc:
            print(f"FAILED {rel}: {exc}", flush=True)
            fail += 1
        finally:
            if not args.keep_work:
                for p in (mp4, seq_data, seq_track):
                    if p.is_file():
                        p.unlink(missing_ok=True)
                    elif p.is_dir():
                        shutil.rmtree(p, ignore_errors=True)

    print(f"done ok={ok} fail={fail} -> {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
