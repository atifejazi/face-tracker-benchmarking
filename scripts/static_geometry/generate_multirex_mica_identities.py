#!/usr/bin/env python3
"""Generate one MICA identity.npy per MultiREX subject from a frontal frame.

Picks a preferred camera per subject (400291 / 400030 when available), extracts
a few frames, runs MICA demo.py, and averages the resulting identity codes.

Run inside the ``tracker`` conda env (needs insightface + MICA weights).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import os
from pathlib import Path

import cv2
import numpy as np

MICA_ROOT = Path(os.environ.get("MICA_ROOT", ""))
if not MICA_ROOT:
    raise SystemExit("Set MICA_ROOT (see config/paths.env)")
PREF_CAMS = [
    "400291", "400030", "400347", "400436", "400275",
    "400017", "400039", "400018", "400042", "400485",
]


def subject_of(stem: str) -> str:
    return stem.split("--")[3]


def camera_of(stem: str) -> str:
    return stem.split("#")[-1]


def pick_video(videos_dir: Path, subject: str) -> Path:
    candidates = sorted(
        v for v in videos_dir.glob("*.mp4") if f"--{subject}--" in v.name
    )
    if not candidates:
        raise FileNotFoundError(f"No videos for subject {subject}")
    by_cam = {camera_of(v.stem): v for v in candidates}
    for cam in PREF_CAMS:
        if cam in by_cam:
            return by_cam[cam]
    return candidates[0]


def extract_frames(video: Path, out_dir: Path, indices=(0, 200, 400)) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("*.png"):
        old.unlink()
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {video}")
    wanted = set(indices)
    idx = 0
    written = 0
    while wanted:
        ok, frame = cap.read()
        if not ok:
            break
        if idx in wanted:
            cv2.imwrite(str(out_dir / f"f{written:03d}.png"), frame)
            wanted.remove(idx)
            written += 1
        idx += 1
    cap.release()
    if written == 0:
        raise RuntimeError(f"No frames extracted from {video}")


def run_mica(in_dir: Path, out_dir: Path, arc_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    arc_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "demo.py",
        "-i", str(in_dir),
        "-o", str(out_dir),
        "-a", str(arc_dir),
        "-m", "data/pretrained/mica.tar",
    ]
    subprocess.run(cmd, cwd=str(MICA_ROOT), check=True)
    ids = sorted(out_dir.glob("*/identity.npy"))
    if not ids:
        raise RuntimeError(f"MICA produced no identity under {out_dir}")
    stacked = np.stack([np.load(p) for p in ids]).astype(np.float32)
    return stacked.mean(axis=0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--videos_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--subjects", nargs="*", default=None)
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--work_dir", type=Path, default=Path("/tmp/mica_id_multirex"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    videos_dir = args.videos_dir.resolve()

    subjects = args.subjects
    if not subjects:
        subjects = sorted({
            subject_of(v.stem) for v in videos_dir.glob("*.mp4")
        })

    for subject in subjects:
        out = args.output_dir / f"{subject}.npy"
        if args.skip_existing and out.exists():
            print(f"skip {subject} (exists)", flush=True)
            continue
        video = pick_video(videos_dir, subject)
        work = args.work_dir / subject
        in_dir = work / "in"
        print(f"[{subject}] frames from {video.name}", flush=True)
        extract_frames(video, in_dir)
        identity = run_mica(in_dir, work / "out", work / "arc")
        np.save(out, identity.astype(np.float32))
        print(f"[{subject}] saved {out} shape={identity.shape}", flush=True)


if __name__ == "__main__":
    main()
