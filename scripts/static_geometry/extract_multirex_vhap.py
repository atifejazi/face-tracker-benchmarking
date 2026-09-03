#!/usr/bin/env python3
"""Extract MultiREX-format FLAME params from videos using native VHAP tracking.

Supports ranking subset mode via ``--manifest`` / ``--frame_stride``: builds a
shortened mp4 from selected frames, tracks that clip, and writes a subsampled
``.npz`` plus ``.frame_indices.npy``.

Run inside the ``VHAP`` conda env.
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

VHAP_ROOT = Path(os.environ.get("VHAP_ROOT", ""))
if not VHAP_ROOT:
    raise SystemExit("Set VHAP_ROOT (see config/paths.env)")

N_EXP_COMPONENTS = 100
SOURCE_FPS = 30


def sanitize(stem: str) -> str:
    parts = stem.split("#")
    subject = parts[0].split("--")[3]
    sequence = parts[2] if len(parts) > 2 else "seq"
    camera = parts[-1]
    return f"mrex_{subject}_{sequence}_{camera}"


def camera_of(stem: str) -> str:
    return stem.split("#")[-1]


def run(cmd: list[str], log_path: Path) -> None:
    with open(log_path, "w") as log:
        subprocess.run(cmd, cwd=str(VHAP_ROOT), check=True, stdout=log, stderr=subprocess.STDOUT)


def latest_tracked_params(output_root: Path) -> Path:
    candidates = list(output_root.rglob("tracked_flame_params_*.npz"))
    if not candidates:
        raise FileNotFoundError(f"No tracked_flame_params under {output_root}")

    def epoch(path: Path) -> int:
        return int(path.stem.split("_")[-1])

    return max(candidates, key=epoch)


def align_to_length(params_path: Path, n_out: int) -> dict:
    """Map VHAP timesteps onto a contiguous output of length n_out."""
    data = np.load(params_path, allow_pickle=True)
    expr = np.asarray(data["expr"], dtype=np.float32)
    jaw = np.asarray(data["jaw_pose"], dtype=np.float32)
    neck = np.asarray(data["neck_pose"], dtype=np.float32)
    timestep = [int(str(t)) for t in data["timestep_id"]]

    if expr.shape[1] < N_EXP_COMPONENTS:
        pad = np.zeros((expr.shape[0], N_EXP_COMPONENTS - expr.shape[1]), np.float32)
        expr = np.concatenate([expr, pad], axis=1)
    poses = np.concatenate([neck, jaw], axis=1).astype(np.float32)

    by_index = {t: i for i, t in enumerate(timestep)}
    exp_out = np.zeros((n_out, N_EXP_COMPONENTS), np.float32)
    pose_out = np.zeros((n_out, 6), np.float32)
    last = min(by_index) if by_index else 0
    for gt_idx in range(n_out):
        if gt_idx in by_index:
            last = gt_idx
        src = by_index.get(gt_idx, by_index.get(last, 0))
        exp_out[gt_idx] = expr[src]
        pose_out[gt_idx] = poses[src]
    return {"expressions": exp_out, "poses": pose_out}


def write_short_mp4(video_path: Path, out_mp4: Path, frame_indices: list[int]) -> None:
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    if out_mp4.exists():
        out_mp4.unlink()
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {video_path}")
    ok, first = cap.read()
    if not ok:
        raise RuntimeError(f"Empty video {video_path}")
    h, w = first.shape[:2]
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    writer = cv2.VideoWriter(
        str(out_mp4),
        cv2.VideoWriter_fourcc(*"mp4v"),
        SOURCE_FPS,
        (w, h),
    )
    wanted = set(frame_indices)
    by_order = frame_indices
    index = 0
    buffered = {}
    while len(buffered) < len(frame_indices):
        ok, frame = cap.read()
        if not ok:
            break
        if index in wanted:
            buffered[index] = frame
        index += 1
    cap.release()
    for orig in by_order:
        if orig not in buffered:
            raise RuntimeError(f"Missing frame {orig} in {video_path.name}")
        writer.write(buffered[orig])
    writer.release()


def slice_full_npz(full_path: Path, frame_indices: list[int]) -> dict:
    data = np.load(full_path)
    idx = np.asarray(frame_indices, dtype=np.int64)
    return {
        "expressions": np.asarray(data["expressions"][idx], dtype=np.float32),
        "poses": np.asarray(data["poses"][idx], dtype=np.float32),
    }


def clips_from_args(args, bbox_mapping) -> list[dict]:
    if args.manifest is not None:
        manifest = json.loads(args.manifest.read_text())
        return [
            c
            for c in manifest["clips"]
            if not args.subject_id or c["subject"] == args.subject_id
        ]

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
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--work_dir", required=True, type=Path)
    parser.add_argument("--subject_id", default=None)
    parser.add_argument("--n_downsample", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--light_photo", action="store_true")
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--skip_preprocess_if_ready", action="store_true")
    parser.add_argument("--frame_stride", type=int, default=1)
    parser.add_argument("--cameras", nargs="*", default=None)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--reuse_full_dir", type=Path, default=None)
    args = parser.parse_args()

    with open(args.bbox_pickle, "rb") as handle:
        bbox_mapping = pickle.load(handle)

    args.videos_dir = args.videos_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    args.work_dir = args.work_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_dir = args.output_dir / "logs"
    log_dir.mkdir(exist_ok=True)
    data_root = args.work_dir / "data"
    out_root = args.work_dir / "output"
    shorts_root = args.work_dir / "shorts"
    data_root.mkdir(parents=True, exist_ok=True)
    out_root.mkdir(parents=True, exist_ok=True)
    shorts_root.mkdir(parents=True, exist_ok=True)

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
                            f"[{stem}] sliced full VHAP track -> {params['expressions'].shape}",
                            flush=True,
                        )
                        continue
                except Exception as exc:
                    print(f"[{stem}] reuse failed ({exc}); will retrack", flush=True)

        name = sanitize(stem) + "_srank"
        short_mp4 = shorts_root / f"{name}.mp4"
        print(f"[{stem}] writing short mp4 ({n_sub} frames) ...", flush=True)
        write_short_mp4(video, short_mp4, frame_indices)

        link = data_root / f"{name}.mp4"
        if link.exists() or link.is_symlink():
            link.unlink()
        link.symlink_to(short_mp4.resolve())
        link_abs = str(link.absolute())
        seq_dir = data_root / name

        # Always re-preprocess shortened clips (frame count differs from full).
        if seq_dir.exists():
            shutil.rmtree(seq_dir)
        print(f"[{stem}] preprocess ...", flush=True)
        pre_cmd = [
            sys.executable,
            "vhap/preprocess_video.py",
            "--input",
            link_abs,
            "--target_fps",
            str(SOURCE_FPS),
            "--matting_method",
            "robust_video_matting",
        ]
        if args.n_downsample in (2, 4, 8):
            pre_cmd += ["--downsample_scales", str(args.n_downsample)]
        try:
            run(pre_cmd, log_dir / f"{name}_preprocess.log")

            seq_output = out_root / f"{name}_lightB"
            if seq_output.exists():
                shutil.rmtree(seq_output)
            seq_output.mkdir(parents=True, exist_ok=True)

            print(f"[{stem}] track (light_photo={args.light_photo}) ...", flush=True)
            track_cmd = [
                sys.executable,
                "vhap/track.py",
                "--data.root_folder",
                str(data_root.resolve()),
                "--data.sequence",
                name,
                "--exp.output_folder",
                str(seq_output.resolve()),
            ]
            if args.n_downsample in (2, 4, 8):
                track_cmd += ["--data.n_downsample_rgb", str(args.n_downsample)]
            if args.batch_size is not None:
                track_cmd += ["--batch_size", str(args.batch_size)]
            if args.light_photo:
                track_cmd += [
                    "--pipeline.rgb_init_texture.num_steps",
                    "100",
                    "--pipeline.rgb_init_all.num_steps",
                    "100",
                    "--pipeline.rgb_init_offset.num_steps",
                    "0",
                    "--pipeline.rgb_sequential_tracking.num_steps",
                    "20",
                    "--pipeline.rgb_global_tracking.num_epochs",
                    "5",
                    "--pipeline.lmk_sequential_tracking.num_steps",
                    "30",
                    "--pipeline.lmk_global_tracking.num_epochs",
                    "5",
                ]
            run(track_cmd, log_dir / f"{name}_track_lightB.log")

            params_path = latest_tracked_params(seq_output)
            params = align_to_length(params_path, n_sub)
            np.savez(out_npz, **params)
            np.save(out_idx, np.asarray(frame_indices, dtype=np.int64))
            print(
                f"[{stem}] saved {out_npz.name}: exp {params['expressions'].shape} "
                f"(from {params_path.name})",
                flush=True,
            )
        except Exception as exc:
            print(f"[{stem}] FAILED: {exc}", flush=True)
            continue


if __name__ == "__main__":
    main()
