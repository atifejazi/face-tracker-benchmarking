#!/usr/bin/env python3
"""Extract MultiREX-format FLAME params from videos using the MICA metrical-tracker.

Per video: extract frames -> run the metrical-tracker optimizer -> collect the
per-frame FLAME expression + jaw (+ eyelids) into a MultiREX .npz.

Supports optional ``--frame_stride`` / ``--cameras`` / ``--manifest`` for the
ranking subset protocol. When striding, only selected frames are tracked and
``frame_indices.npy`` is written next to each ``.npz``.

If ``--reuse_full_dir`` contains a full-length npz for a clip, it is sliced to
the stride indices instead of retracking.

Run inside the ``tracker`` conda env.
"""

from __future__ import annotations

import argparse
import json
import pickle
import shutil
import subprocess
import sys
import os
from pathlib import Path

import cv2
import numpy as np
import torch
from pytorch3d.transforms import matrix_to_axis_angle, rotation_6d_to_matrix

N_EXP_COMPONENTS = 100
TRACKER_ROOT = Path(os.environ.get("METRICAL_TRACKER_ROOT", ""))
if not TRACKER_ROOT:
    raise SystemExit("Set METRICAL_TRACKER_ROOT (see config/paths.env)")


def sanitize(stem: str) -> str:
    parts = stem.split("#")
    subject = parts[0].split("--")[3]
    sequence = parts[2] if len(parts) > 2 else "seq"
    camera = parts[-1]
    return f"mrex_{subject}_{sequence}_{camera}"


def camera_of(stem: str) -> str:
    return stem.split("#")[-1]


def rot6d_to_axis_angle(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float32).reshape(-1, 6)
    matrix = rotation_6d_to_matrix(torch.from_numpy(value))
    return matrix_to_axis_angle(matrix).numpy()


def extract_frames_subsampled(
    video_path: Path, source_dir: Path, frame_indices: list[int]
) -> int:
    """Write selected frames as consecutive 00000.png, 00001.png, ..."""
    if source_dir.exists():
        shutil.rmtree(source_dir)
    source_dir.mkdir(parents=True, exist_ok=True)
    wanted = set(frame_indices)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {video_path}")
    index = 0
    written = 0
    by_orig = {orig: i for i, orig in enumerate(frame_indices)}
    while wanted:
        ok, frame = cap.read()
        if not ok:
            break
        if index in wanted:
            out_i = by_orig[index]
            cv2.imwrite(str(source_dir / f"{out_i:05d}.png"), frame)
            wanted.remove(index)
            written += 1
        index += 1
    cap.release()
    if written != len(frame_indices):
        raise RuntimeError(
            f"Extracted {written}/{len(frame_indices)} frames from {video_path.name}"
        )
    return written


def write_config(config_name: str, actor_dir: Path) -> Path:
    config_path = TRACKER_ROOT / "configs" / "actors" / f"{config_name}.yml"
    config_path.write_text(
        f"actor: '{actor_dir.as_posix()}'\n"
        "save_folder: './output/'\n"
        "optimize_shape: true\n"
        "optimize_jaw: true\n"
        "begin_frames: 0\n"
        "keyframes: [ 0, 1 ]\n"
        "fps: 25\n"
    )
    return config_path


def run_tracker(config_path: Path, log_path: Path) -> None:
    with open(log_path, "w") as log:
        subprocess.run(
            [sys.executable, "tracker.py", "--cfg", str(config_path)],
            cwd=str(TRACKER_ROOT),
            check=True,
            stdout=log,
            stderr=subprocess.STDOUT,
        )


def collect_params(checkpoint_dir: Path, n_frames: int) -> dict:
    frames = sorted(checkpoint_dir.glob("*.frame"))
    if len(frames) != n_frames:
        raise RuntimeError(f"{checkpoint_dir}: {len(frames)} frames != expected {n_frames}")
    expressions, jaws, eyelids = [], [], []
    for path in frames:
        flame = torch.load(path, map_location="cpu", weights_only=False)["flame"]
        expressions.append(np.asarray(flame["exp"], dtype=np.float32).reshape(-1))
        jaws.append(rot6d_to_axis_angle(flame["jaw"])[0])
        eyelids.append(np.asarray(flame["eyelids"], dtype=np.float32).reshape(-1))
    expressions = np.stack(expressions)
    if expressions.shape[1] < N_EXP_COMPONENTS:
        pad = np.zeros(
            (expressions.shape[0], N_EXP_COMPONENTS - expressions.shape[1]), np.float32
        )
        expressions = np.concatenate([expressions, pad], axis=1)
    poses = np.concatenate([np.zeros((len(jaws), 3), np.float32), np.stack(jaws)], axis=1)
    return {
        "expressions": expressions.astype(np.float32),
        "poses": poses.astype(np.float32),
        "eyelids": np.stack(eyelids).astype(np.float32),
    }


def slice_full_npz(full_path: Path, frame_indices: list[int]) -> dict:
    data = np.load(full_path)
    n = data["expressions"].shape[0]
    if n < max(frame_indices) + 1:
        raise ValueError(f"{full_path}: length {n} too short for indices")
    idx = np.asarray(frame_indices, dtype=np.int64)
    out = {
        "expressions": np.asarray(data["expressions"][idx], dtype=np.float32),
        "poses": np.asarray(data["poses"][idx], dtype=np.float32),
    }
    if "eyelids" in data.files:
        out["eyelids"] = np.asarray(data["eyelids"][idx], dtype=np.float32)
    return out


def clips_from_args(args, bbox_mapping) -> list[dict]:
    if args.manifest is not None:
        manifest = json.loads(args.manifest.read_text())
        clips = []
        for clip in manifest["clips"]:
            if args.subject_id and clip["subject"] != args.subject_id:
                continue
            clips.append(clip)
        return clips

    videos = sorted(
        v for v in args.videos_dir.glob("*.mp4") if f"--{args.subject_id}--" in v.name
    )
    camera_filter = set(args.cameras) if args.cameras else None
    clips = []
    for video in videos:
        if video.name not in bbox_mapping:
            continue
        cam = camera_of(video.stem)
        if camera_filter is not None and cam not in camera_filter:
            continue
        n_gt = len(bbox_mapping[video.name])
        indices = list(range(0, n_gt, args.frame_stride))
        clips.append(
            {
                "subject": args.subject_id,
                "camera": cam,
                "video_name": video.name,
                "stem": video.stem,
                "n_gt_frames": n_gt,
                "frame_indices": indices,
                "n_subsampled": len(indices),
            }
        )
    return clips


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--videos_dir", required=True, type=Path)
    parser.add_argument("--bbox_pickle", required=True, type=Path)
    parser.add_argument(
        "--identity",
        type=Path,
        default=None,
        help="Single identity.npy, or omit when --identity_dir is set",
    )
    parser.add_argument(
        "--identity_dir",
        type=Path,
        default=None,
        help="Directory of <subject>.npy identities (used with --manifest)",
    )
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--subject_id", default=None)
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--frame_stride", type=int, default=1)
    parser.add_argument("--cameras", nargs="*", default=None)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument(
        "--reuse_full_dir",
        type=Path,
        default=None,
        help="If a full-length npz exists here, slice it instead of tracking",
    )
    args = parser.parse_args()

    with open(args.bbox_pickle, "rb") as handle:
        bbox_mapping = pickle.load(handle)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_dir = args.output_dir / "logs"
    log_dir.mkdir(exist_ok=True)

    clips = clips_from_args(args, bbox_mapping)
    if not clips:
        raise SystemExit("No clips to process")

    for clip in clips:
        stem = clip["stem"]
        video = args.videos_dir / clip["video_name"]
        out_npz = args.output_dir / f"{stem}.npz"
        out_idx = args.output_dir / f"{stem}.frame_indices.npy"
        if args.skip_existing and out_npz.exists() and out_idx.exists():
            print(f"skip {stem} (exists)")
            continue

        frame_indices = list(clip["frame_indices"])
        n_sub = len(frame_indices)

        # Prefer slicing a finished full-length track when available.
        reused = False
        if args.reuse_full_dir is not None:
            full = args.reuse_full_dir / f"{stem}.npz"
            if full.exists():
                try:
                    data = np.load(full)
                    if data["expressions"].shape[0] == clip["n_gt_frames"]:
                        params = slice_full_npz(full, frame_indices)
                        np.savez(out_npz, **params)
                        np.save(out_idx, np.asarray(frame_indices, dtype=np.int64))
                        print(
                            f"[{stem}] sliced full track -> {params['expressions'].shape}",
                            flush=True,
                        )
                        reused = True
                except Exception as exc:
                    print(f"[{stem}] reuse failed ({exc}); will retrack", flush=True)

        if reused:
            continue

        subject = clip["subject"]
        if args.identity_dir is not None:
            identity_path = args.identity_dir / f"{subject}.npy"
        elif args.identity is not None:
            identity_path = args.identity
        else:
            raise SystemExit("Need --identity or --identity_dir")
        identity = np.load(identity_path).astype(np.float32)

        # Distinct actor name so full-length and subsampled runs do not collide.
        config_name = sanitize(stem) + f"_s{args.frame_stride if args.manifest is None else 'rank'}"
        if args.manifest is not None:
            config_name = sanitize(stem) + "_srank"
        actor_dir = TRACKER_ROOT / "input" / config_name
        actor_dir.mkdir(parents=True, exist_ok=True)
        np.save(actor_dir / "identity.npy", identity)

        print(f"[{stem}] extracting {n_sub}/{clip['n_gt_frames']} frames ...", flush=True)
        extract_frames_subsampled(video, actor_dir / "source", frame_indices)

        # Wipe prior checkpoint for this actor so frame count matches.
        ckpt_root = TRACKER_ROOT / "output" / config_name
        if ckpt_root.exists():
            shutil.rmtree(ckpt_root)

        config_path = write_config(config_name, actor_dir)
        print(f"[{stem}] tracking ...", flush=True)
        run_tracker(config_path, log_dir / f"{config_name}.log")

        checkpoint_dir = TRACKER_ROOT / "output" / config_name / "checkpoint"
        params = collect_params(checkpoint_dir, n_sub)
        np.savez(out_npz, **params)
        np.save(out_idx, np.asarray(frame_indices, dtype=np.int64))
        print(
            f"[{stem}] saved {out_npz.name}: exp {params['expressions'].shape}",
            flush=True,
        )


if __name__ == "__main__":
    main()
