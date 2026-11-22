#!/usr/bin/env python3
"""Verify decoded time for one monocular clip and build its shared GA inputs."""

import argparse
import hashlib
from itertools import groupby
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from export_metrical_tracker_to_ga import export_dataset
from prepare_common_canvas import adapt, native_frames, new_dataset, numbered_images, prepare, write_json


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def decode_video(video, folder, jpeg_size=None):
    folder.mkdir()
    filters = "fps=25"
    output = folder / "%06d.png"
    codec = []
    if jpeg_size:
        filters += f",scale={jpeg_size[0]}:{jpeg_size[1]}"
        output = folder / "%06d.jpg"
        codec = ["-qscale:v", "1"]
    subprocess.run(["ffmpeg", "-v", "error", "-nostdin", "-i", str(video),
                    "-map", "0:v:0", "-vf", filters, *codec,
                    "-start_number", "0", "-threads", "2", str(output)], check=True)
    return numbered_images(folder)


def verify_metrical(actor, track, decoded, begin_frames):
    if digest(actor / "video.mp4") != digest(decoded["video"]):
        raise ValueError("Metrical Tracker used a different source video")
    source_paths = sorted((actor / "source").glob("*.png"))
    frames = decoded["frames"]
    if len(source_paths) != len(frames):
        raise ValueError("Metrical Tracker and reference decode lengths differ")
    for index, path in enumerate(source_paths):
        archived = cv2.imread(str(path))
        if not np.array_equal(archived, cv2.imread(str(frames[index]))):
            raise ValueError(f"Metrical source decode differs at {index} in {actor}")
    x0, x1, y0, y1 = map(int, torch.load(actor / "bbox.pt", map_location="cpu", weights_only=False))
    padding = (max(x1 - x0, y1 - y0) - (x1 - x0)) // 2
    vertical = (max(x1 - x0, y1 - y0) - (y1 - y0)) // 2
    mapping = {}
    for timestep, checkpoint in enumerate(sorted((track / "checkpoint").glob("*.frame"))):
        source_id = int(checkpoint.stem) + begin_frames
        crop = cv2.imread(str(frames[source_id]))[y0:y1, x0:x1]
        crop = cv2.copyMakeBorder(crop, vertical, vertical, padding, padding, cv2.BORDER_CONSTANT)
        fitted_rgb = cv2.imread(str(track / "input" / f"{checkpoint.stem}.png"))
        recreated = cv2.resize(crop, fitted_rgb.shape[1::-1], interpolation=cv2.INTER_CUBIC)
        if not np.array_equal(recreated, fitted_rgb):
            raise ValueError(f"Checkpoint input crop differs for source frame {source_id} in {track}")
        mapping[timestep] = source_id
    if not mapping:
        raise ValueError(f"No Metrical Tracker checkpoints in {track}")
    return mapping, [x0, y0, x1, y1]


def prepare_clip(args):
    video = args.video.resolve(strict=True)
    with new_dataset(args.output) as stage:
        frames = decode_video(video, stage / "decoded_rgb")
        decoded = dict(video=video, frames=frames)
        mica, bbox = verify_metrical(args.actor, args.metrical_track, decoded, args.begin_frames)
        print(f"Verified {len(mica)} Metrical checkpoint inputs against decoded timing")
        # Re-decode the prefit JPEGs byte for byte before accepting their aligned alphas.
        if digest(args.prefit_video) != digest(video):
            raise ValueError("Full-frame matting used a different source video")
        first = cv2.imread(str(frames[0]))
        reconstructed = decode_video(video, stage / "jpeg_check", first.shape[1::-1])
        prefit = numbered_images(args.prefit / "images")
        full_alpha = numbered_images(args.prefit / "alpha_maps")
        if reconstructed.keys() != prefit.keys() or reconstructed.keys() != full_alpha.keys():
            raise ValueError("Decoded JPEGs, prefit images and full-frame alpha counts differ")
        if any(digest(path) != digest(prefit[index]) for index, path in reconstructed.items()):
            raise ValueError("Prefit JPEG replay differs from the recorded prefit images")
        print(f"Verified {len(prefit)} full-frame matting inputs against decoded timing")

        vhap = {}
        if args.vhap:
            for timestep, frame in native_frames(args.vhap).items():
                original = int(frame["timestep_id"])
                if original != frame["timestep_index_original"] or original not in prefit:
                    raise ValueError("VHAP timestep identifiers disagree with the prefit decode")
                vhap[timestep] = original
        support = set(mica.values()) & frames.keys() & full_alpha.keys()
        if args.vhap:
            support &= set(vhap.values())
        # Select the longest contiguous support, with the earlier block breaking ties.
        intervals = [
            [source_id for _, source_id in rows]
            for _, rows in groupby(enumerate(sorted(support)), key=lambda row: row[1] - row[0])
        ]
        selected = max(intervals, key=len, default=[])
        if len(selected) < 25:
            raise ValueError(f"Only {len(selected)} contiguous matched frames are available")
        train, val = int(.85 * len(selected)), max(1, int(.07 * len(selected)))
        write_json(stage / "splits.json", dict(train=selected[:train], val=selected[train:train + val],
                                               test=selected[train + val:]))
        # Keep the native export independent of study staging. Its RGB is never used for GA.
        export_dataset(args.metrical_track, args.metrical_track / "input", args.crop_alpha,
                       stage / "native_mica", flame_pkl_path=args.flame_pkl)
        common = stage / "common"
        prepare(SimpleNamespace(rgb=stage / "decoded_rgb", alpha=args.prefit / "alpha_maps",
                                splits=stage / "splits.json", source=args.source, camera_id=args.camera_id,
                                fps=25., size=512, smirk_focal=None, output=common))
        # Finish adapters after the atomic move so their asset symlinks use stable paths.
        write_json(stage / "mica_frame_map.json", mica)
        if args.vhap:
            write_json(stage / "vhap_frame_map.json", vhap)
        write_json(stage / "lineage.json", dict(
            source_video=str(video), source_video_sha256=digest(video),
            metrical_track=str(args.metrical_track.resolve()), actor=str(args.actor.resolve()),
            begin_frames=args.begin_frames, metrical_bbox_xyxy=bbox, full_frame_prefit=str(args.prefit.resolve()),
            vhap_native=str(args.vhap.resolve()) if args.vhap else None,
            selected_source_frames=selected,
            omitted_common_frames=sorted(support.difference(selected)),
            omitted_decoded_frames=sorted(set(frames).difference(selected)),
            timing_verification="Exact decoded RGB, checkpoint crop and prefit JPEG checks passed",
        ))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("output", "video", "actor", "metrical-track", "crop-alpha",
                   "flame-pkl", "prefit", "prefit-video"):
        parser.add_argument(f"--{option}", required=True, type=Path)
    parser.add_argument("--begin-frames", required=True, type=int)
    parser.add_argument("--vhap", type=Path)
    parser.add_argument("--source", required=True)
    parser.add_argument("--camera-id", required=True)
    args = parser.parse_args()
    cv2.setNumThreads(2)
    args.output = args.output.resolve()
    if not args.output.exists():
        prepare_clip(args)
    lineage = json.loads((args.output / "lineage.json").read_text())
    if not (args.output / "mica").is_dir():
        adapt(SimpleNamespace(common=args.output / "common", native=args.output / "native_mica",
                              layout="metrical", frame_map=args.output / "mica_frame_map.json",
                              metrical_bbox=lineage["metrical_bbox_xyxy"], output=args.output / "mica"))
    if args.vhap and not (args.output / "vhap").is_dir():
        adapt(SimpleNamespace(common=args.output / "common", native=args.vhap, layout="vhap",
                              frame_map=args.output / "vhap_frame_map.json", metrical_bbox=None,
                              output=args.output / "vhap"))


if __name__ == "__main__":
    main()
