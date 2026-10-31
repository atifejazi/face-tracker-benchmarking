#!/usr/bin/env python3
"""Temporal jitter metric for tracked FLAME mesh sequences.

Jitter is the RMS magnitude of per-vertex acceleration (second finite
difference across frames). Lower means smoother, less jittery tracking.

It consumes the ``decoded_flame_meshes/*.npy`` produced by the MultiREX
evaluation (one array of shape ``(frames, verts, 3)`` per video), so the
identity/topology is identical across trackers and the metric isolates
temporal stability of the expression + jaw animation.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

# decoded FLAME meshes are stored in meters; report jitter in millimeters
METERS_TO_MM = 1000.0


def sequence_jitter(vertices: np.ndarray) -> np.ndarray:
    """Return per-frame RMS acceleration magnitude (mm / frame^2).

    vertices: (F, V, 3). Output length is F-2 (interior frames only).
    """
    if vertices.ndim != 3 or vertices.shape[0] < 3:
        raise ValueError(f"Need (F>=3, V, 3) vertex array, got {vertices.shape}")
    acceleration = vertices[2:] - 2.0 * vertices[1:-1] + vertices[:-2]
    magnitude = np.linalg.norm(acceleration, axis=-1) * METERS_TO_MM  # (F-2, V)
    return np.sqrt(np.mean(magnitude**2, axis=1))  # RMS over vertices -> (F-2,)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--decoded_dir",
        required=True,
        type=Path,
        help="Folder of per-video decoded vertex sequences (*.npy).",
    )
    parser.add_argument(
        "--output_csv",
        required=True,
        type=Path,
        help="Per-video summary CSV (one row per sequence).",
    )
    parser.add_argument(
        "--per_frame_csv",
        type=Path,
        default=None,
        help="Optional unpooled CSV with one row per frame.",
    )
    parser.add_argument("--tracker", default="unknown", help="Tracker label for CSV rows.")
    args = parser.parse_args()

    sequences = sorted(args.decoded_dir.glob("*.npy"))
    if not sequences:
        raise SystemExit(f"No .npy sequences in {args.decoded_dir}")

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    per_frame_rows = []

    for path in sequences:
        vertices = np.load(path)
        per_frame = sequence_jitter(vertices)
        video = path.stem
        # subject id is the 4th "--" token in the multiface naming scheme
        tokens = video.split("--")
        subject = tokens[3].split("#")[0] if len(tokens) > 3 else ""
        camera = video.split("#")[-1] if "#" in video else ""
        summary_rows.append(
            {
                "tracker": args.tracker,
                "source_video": video,
                "subject": subject,
                "camera": camera,
                "n_frames": int(vertices.shape[0]),
                "jitter_rms_mm": float(np.sqrt(np.mean(per_frame**2))),
                "jitter_mean_mm": float(np.mean(per_frame)),
                "jitter_max_mm": float(np.max(per_frame)),
            }
        )
        if args.per_frame_csv is not None:
            for frame_nb, value in enumerate(per_frame):
                per_frame_rows.append(
                    {
                        "tracker": args.tracker,
                        "source_video": video,
                        "subject": subject,
                        "camera": camera,
                        "frame_nb": frame_nb + 1,  # interior frame index
                        "jitter_mm": float(value),
                    }
                )

    with open(args.output_csv, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    if args.per_frame_csv is not None:
        args.per_frame_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(args.per_frame_csv, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(per_frame_rows[0].keys()))
            writer.writeheader()
            writer.writerows(per_frame_rows)

    overall = np.sqrt(np.mean([row["jitter_rms_mm"] ** 2 for row in summary_rows]))
    print(f"[{args.tracker}] {len(summary_rows)} sequences")
    print(f"  dataset jitter RMS: {overall:.4f} mm/frame^2")
    print(f"  per-video summary : {args.output_csv}")
    if args.per_frame_csv is not None:
        print(f"  per-frame (unpooled): {args.per_frame_csv}")


if __name__ == "__main__":
    main()
