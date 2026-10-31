#!/usr/bin/env python3
"""Build the MultiREX front+angle ranking manifest (stride-subsampled frames)."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

FRONT_PREFERENCE = ["400291", "400030"]
ANGLE_PREFERENCE = ["400347", "400017", "400436", "400039"]


def subject_of(stem: str) -> str:
    return stem.split("--")[3]


def camera_of(stem: str) -> str:
    return stem.split("#")[-1]


def pick_cam(by_cam: dict, preference: list[str]) -> str:
    for cam in preference:
        if cam in by_cam:
            return cam
    raise KeyError(f"No preferred camera in {sorted(by_cam)}; tried {preference}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--videos_dir", type=Path, required=True)
    parser.add_argument("--bbox_pickle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frame_stride", type=int, default=8)
    parser.add_argument(
        "--front_only",
        action="store_true",
        help="Only include the frontal camera per subject (faster ranking)",
    )
    args = parser.parse_args()

    with open(args.bbox_pickle, "rb") as handle:
        bbox = pickle.load(handle)

    videos = sorted(args.videos_dir.glob("*.mp4"))
    by_subject: dict[str, dict[str, Path]] = {}
    for video in videos:
        if video.name not in bbox:
            continue
        by_subject.setdefault(subject_of(video.stem), {})[camera_of(video.stem)] = video

    roles = (("front", FRONT_PREFERENCE),)
    if not args.front_only:
        roles = (("front", FRONT_PREFERENCE), ("angled", ANGLE_PREFERENCE))

    clips = []
    for subject in sorted(by_subject):
        cams = by_subject[subject]
        for role, pref in roles:
            cam = pick_cam(cams, pref)
            video = cams[cam]
            n_gt = len(bbox[video.name])
            indices = list(range(0, n_gt, args.frame_stride))
            clips.append(
                {
                    "subject": subject,
                    "role": role,
                    "camera": cam,
                    "video_name": video.name,
                    "stem": video.stem,
                    "n_gt_frames": n_gt,
                    "frame_indices": indices,
                    "n_subsampled": len(indices),
                }
            )

    payload = {
        "protocol": "multirex_ranking_front_only" if args.front_only else "multirex_ranking_front_angle",
        "frame_stride": args.frame_stride,
        "front_only": bool(args.front_only),
        "n_subjects": len(by_subject),
        "n_clips": len(clips),
        "front_preference": FRONT_PREFERENCE,
        "angle_preference": ANGLE_PREFERENCE if not args.front_only else [],
        "clips": clips,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(
        f"wrote {args.output}: {payload['n_subjects']} subjects, "
        f"{payload['n_clips']} clips, stride={args.frame_stride}"
    )


if __name__ == "__main__":
    main()
