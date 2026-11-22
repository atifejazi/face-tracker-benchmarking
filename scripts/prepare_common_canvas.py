#!/usr/bin/env python3
"""Build a shared monocular GA canvas and adapt native tracker exports to it."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import math
from pathlib import Path
import tempfile

import numpy as np
from PIL import Image, ImageOps


SPLITS = ("train", "val", "test")
MANIFEST = "common_canvas.json"
PINHOLE = "pinhole_image_edges"


def read_json(path):
    return json.loads(path.read_text())


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


@contextmanager
def new_dataset(output):
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Refusing to overwrite {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".canvas-", dir=output.parent) as temporary:
        stage = Path(temporary) / "dataset"
        stage.mkdir()
        yield stage
        stage.rename(output)


def numbered_images(directory):
    images = {}
    for path in directory.iterdir():
        if path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
            if not path.stem.isdecimal() or int(path.stem) in images:
                raise ValueError(f"Expected unique numeric image IDs in {directory}: {path.name}")
            images[int(path.stem)] = path
    return images


def intrinsics(fx, fy, cx, cy, width, height):
    if not np.isfinite([fx, fy, cx, cy]).all() or min(fx, fy) <= 0:
        raise ValueError("Nonfinite or nonpositive camera intrinsics")
    return dict(w=width, h=height, fl_x=fx, fl_y=fy, cx=cx, cy=cy,
                camera_angle_x=2 * math.atan(width / (2 * fx)),
                camera_angle_y=2 * math.atan(height / (2 * fy)))


def write_splits(directory, splits):
    for split, frames in splits.items():
        write_json(directory / f"transforms_{split}.json", {
            "frames": frames,
            "timestep_indices": [frame["timestep_index"] for frame in frames],
            "camera_indices": [0] if frames else [],
        })


def prepare(args):
    images, masks = numbered_images(args.rgb), numbered_images(args.alpha)
    splits = read_json(args.splits)
    if set(splits) != set(SPLITS) or not splits["train"] or not splits["test"]:
        raise ValueError("Specify train, val and test lists with nonempty train and test")
    split_by_id = {}
    for split in SPLITS:
        for index in splits[split]:
            if type(index) is not int or index < 0 or index in split_by_id:
                raise ValueError(f"Invalid or repeated decoded frame ID: {index}")
            if index not in images or index not in masks:
                raise ValueError(f"Missing decoded RGB or full-frame alpha for source frame {index}")
            split_by_id[index] = split

    size = args.size
    if size <= 0 or not math.isfinite(args.fps) or args.fps <= 0:
        raise ValueError("Size and frame rate must be positive")
    with Image.open(images[min(split_by_id)]) as image:
        width, height = image.size
    side = max(width, height)
    left, top = (side - width) // 2, (side - height) // 2
    pad = (left, top, side - width - left, side - height - top)
    scale = size / side
    source_to_canvas = np.array([[scale, 0, left * scale], [0, scale, top * scale], [0, 0, 1.]])

    # Used only as the SMIRK weak-to-perspective camera convention.
    focal = args.smirk_focal if args.smirk_focal is not None else 2 * size
    camera = intrinsics(focal, focal, size / 2, size / 2, size, size)
    template_transform = np.eye(4)
    template_transform[2, 3] = 1
    by_split = {split: [] for split in SPLITS}
    provenance = []
    with new_dataset(args.output) as stage:
        for name in ("images", "fg_masks", "tracking_images"):
            (stage / name).mkdir()
        for timestep, index in enumerate(sorted(split_by_id)):
            with Image.open(images[index]) as source, Image.open(masks[index]) as mask:
                if source.size != (width, height) or mask.size != source.size:
                    raise ValueError(f"RGB/alpha/source dimensions disagree at decoded frame {index}")
                rgb = source.convert("RGB")
                alpha = mask.convert("L")

            name = f"{timestep:05d}_00.png"
            for folder, data, border in (
                ("images", Image.composite(rgb, Image.new("RGB", rgb.size, "white"), alpha), 255),
                ("fg_masks", alpha, 0),
                ("tracking_images", rgb, 0),
            ):
                fill = (border,) * 3 if data.mode == "RGB" else border
                padded = ImageOps.expand(data, border=pad, fill=fill)
                padded.resize((size, size), Image.Resampling.LANCZOS).save(stage / folder / name)
            split = split_by_id[index]
            by_split[split].append({
                **camera, "transform_matrix": template_transform.tolist(),
                "timestep_index": timestep, "timestep_index_original": index,
                "camera_index": 0, "camera_id": args.camera_id,
                "file_path": f"images/{name}", "fg_mask_path": f"fg_masks/{name}",
            })
            provenance.append({
                "source_frame": index, "timestep": timestep, "split": split,
                "time_seconds_on_decoded_grid": index / args.fps,
                "rgb_sha256": hashlib.sha256(images[index].read_bytes()).hexdigest(),
                "alpha_sha256": hashlib.sha256(masks[index].read_bytes()).hexdigest(),
            })
        write_splits(stage, by_split)
        write_json(stage / MANIFEST, {
            "source": args.source, "camera_id": args.camera_id, "fps": args.fps,
            "source_raster": [width, height], "canvas_size": size,
            "source_to_canvas_image_edges": source_to_canvas.tolist(),
            "smirk_focal_pixels": focal, "smirk_camera_z": 1, "frames": provenance,
        })
    print(f"Prepared {len(provenance)} shared frames at {size} × {size}")


def source_to_native(layout, source_raster, native_raster, bbox):
    width, height = source_raster
    native_w, native_h = native_raster
    if layout == "vhap":
        return np.diag([native_w / width, native_h / height, 1.])
    if bbox is None:
        raise ValueError("Metrical Tracker requires the recorded crop box")
    x0, y0, x1, y1 = bbox
    if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height and native_w == native_h):
        raise ValueError("Invalid Metrical Tracker crop or nonsquare native raster")
    crop_w, crop_h = x1 - x0, y1 - y0
    side = max(crop_w, crop_h)
    pad_x, pad_y = (side - crop_w) // 2, (side - crop_h) // 2
    sx = native_w / (crop_w + 2 * pad_x)
    sy = native_h / (crop_h + 2 * pad_y)
    return np.array([[sx, 0, (pad_x - x0) * sx], [0, sy, (pad_y - y0) * sy], [0, 0, 1.]])


def native_frames(dataset):
    frames = {}
    for split in SPLITS:
        db = read_json(dataset / f"transforms_{split}.json")
        shared = {key: db[key] for key in ("w", "h", "fl_x", "fl_y", "cx", "cy") if key in db}
        for frame in db["frames"]:
            timestep = frame["timestep_index"]
            if timestep in frames or frame.get("camera_index", 0) != 0:
                raise ValueError("Expected one monocular native frame per timestep")
            frames[timestep] = {**shared, **frame}
    return frames


def adapt(args):
    common = read_json(args.common / MANIFEST)
    mapping = read_json(args.frame_map)
    if not isinstance(mapping, dict) or any(
        not key.isdecimal() or str(int(key)) != key or type(value) is not int or value < 0
        for key, value in mapping.items()
    ):
        raise ValueError("Frame map must map native timestep strings to nonnegative decoded source IDs")
    source_to_timestep = {value: int(key) for key, value in mapping.items()}
    if len(source_to_timestep) != len(mapping) or list(sorted(source_to_timestep.values())) != [
        source_to_timestep[index] for index in sorted(source_to_timestep)
    ]:
        raise ValueError("Frame map must be one-to-one and preserve time order")
    if args.layout == "vhap" and args.metrical_bbox:
        raise ValueError("VHAP adapter expects a full-frame native export")

    frames = native_frames(args.native)
    canvas_affine = np.array(common["source_to_canvas_image_edges"])
    by_split = {split: [] for split in SPLITS}
    lineage = []
    with new_dataset(args.output) as stage:
        for folder in ("images", "fg_masks"):
            (stage / folder).symlink_to((args.common / folder).resolve(strict=True), target_is_directory=True)
        canonical = "canonical_flame_param.npz"
        (stage / canonical).symlink_to((args.native / canonical).resolve(strict=True))
        (stage / "flame_param").mkdir()
        for split in SPLITS:
            for target in read_json(args.common / f"transforms_{split}.json")["frames"]:
                source_index = target["timestep_index_original"]
                native_timestep = source_to_timestep.get(source_index)
                if native_timestep not in frames:
                    raise ValueError(f"No native timestep for decoded frame {source_index}")
                native = frames[native_timestep]
                affine = source_to_native(args.layout, common["source_raster"],
                                          (native["w"], native["h"]), args.metrical_bbox)
                native_to_canvas = canvas_affine @ np.linalg.inv(affine)
                native_k = np.array([[native["fl_x"], 0, native["cx"]],
                                     [0, native["fl_y"], native["cy"]], [0, 0, 1.]])
                k = native_to_canvas @ native_k
                frame = {**target, **intrinsics(k[0, 0], k[1, 1], k[0, 2], k[1, 2],
                                                target["w"], target["h"])}
                frame["camera_model"] = PINHOLE
                frame["transform_matrix"] = native["transform_matrix"]
                frame["flame_param_path"] = f"flame_param/{frame['timestep_index']:05d}.npz"
                native_params = (args.native / native["flame_param_path"]).resolve(strict=True)
                (stage / frame["flame_param_path"]).symlink_to(native_params)
                by_split[split].append(frame)
                lineage.append({
                    "source_frame": source_index, "native_timestep": native_timestep,
                    "canvas_timestep": frame["timestep_index"], "split": split,
                    "native_to_canvas_image_edges": native_to_canvas.tolist(),
                })
        write_splits(stage, by_split)
        write_json(stage / MANIFEST, {
            **common, "common_assets": str(args.common.resolve()),
            "native": str(args.native.resolve()), "native_layout": args.layout,
            "metrical_bbox_xyxy": args.metrical_bbox, "native_camera_lineage": lineage,
        })
    print(f"Adapted {len(lineage)} {args.layout} cameras without changing FLAME or RGB targets")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(required=True)
    shared = commands.add_parser("prepare", help="Share RGB, alpha, splits and the SMIRK camera template")
    shared.add_argument("--rgb", type=Path, required=True, help="Numbered decoded RGB images")
    shared.add_argument("--alpha", type=Path, required=True, help="Matching numbered full-frame alphas")
    shared.add_argument("--splits", type=Path, required=True, help="JSON with train, val, test source IDs")
    shared.add_argument("--source", required=True, help="Stable source video identifier")
    shared.add_argument("--camera-id", required=True, help="Physical camera identifier")
    shared.add_argument("--fps", type=float, required=True, help="Verified decoded rate")
    shared.add_argument("--size", type=int, default=512, help="Square output size, before GA resizes")
    shared.add_argument("--smirk-focal", type=float, help="Fixed SMIRK convention, defaults to twice canvas size")
    shared.add_argument("--output", type=Path, required=True)
    shared.set_defaults(run=prepare)

    native = commands.add_parser("adapt", help="Map a native Metrical Tracker or VHAP export to the common canvas")
    native.add_argument("--common", type=Path, required=True)
    native.add_argument("--native", type=Path, required=True)
    native.add_argument("--layout", choices=("metrical", "vhap"), required=True)
    native.add_argument("--frame-map", type=Path, required=True, help="JSON from native timestep to verified decoded ID")
    native.add_argument("--metrical-bbox", type=int, nargs=4, metavar=("X0", "Y0", "X1", "Y1"))
    native.add_argument("--output", type=Path, required=True)
    native.set_defaults(run=adapt)
    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
