#!/usr/bin/env python3
"""Audit GA inputs and summarize comparable tracking and rendering measurements."""

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean

CLIP = ("dataset", "subject", "sequence", "camera")
FRAME = (*CLIP, "frame")
FIELDS = (*FRAME, "tracker", "protocol", "motion_split", "view_split", "capture",
          "region", "metric", "unit", "value")
LABELS = ("glasses", "facial_hair", "occlusion", "expression_intensity",
          "mouth_motion", "head_pose", "motion_speed")
GROUPS = (*CLIP, *LABELS, "region", "motion_split", "view_split", "capture")


def read_csv(path, required, unique=()):
    with path.open(newline="") as file:
        reader = csv.DictReader(file)
        missing = set(required) - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path}: missing columns {sorted(missing)}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"{path}: no rows")
    if any(not all(row[key] and row[key].strip() for key in required) for row in rows):
        raise ValueError(f"{path}: blank required field")
    keys = {tuple(row[key] for key in unique) for row in rows}
    if unique and len(keys) != len(rows):
        raise ValueError(f"{path}: duplicate key {unique}")
    return rows


def write_csv(path, fields, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", newline="") as file:
        writer = csv.DictWriter(file, fields)
        writer.writeheader()
        writer.writerows(rows)


def import_multirex(args):
    rows = []
    for tracker, filename in args.metrics:
        for row in read_csv(Path(filename), ("source_video", "camera", "region",
                                              "identity", "frame_nb", "value")):
            stem = Path(row["source_video"]).stem
            sequence, separator, camera = stem.rpartition("#")
            if not separator or camera != row["camera"].lstrip("#"):
                raise ValueError(f"{filename}: conflicting camera for {stem}")
            rows.append(dict(
                dataset="MultiREX", subject=row["identity"], sequence=sequence,
                camera=camera, frame=str(int(row["frame_nb"])), tracker=tracker,
                protocol=args.protocol, motion_split="unknown", view_split="unknown",
                capture="monocular", region=row["region"], metric="region_aligned_vertex_error",
                unit="mm", value=row["value"],
            ))
    validate_metrics(rows)
    write_csv(args.output, FIELDS, rows)


def validate_metrics(rows):
    seen = set()
    for row in rows:
        key = tuple(row[field] for field in FIELDS if field != "value")
        if key in seen:
            raise ValueError(f"Duplicate metric row {key}")
        seen.add(key)
        if not math.isfinite(float(row["value"])):
            raise ValueError(f"Nonfinite metric value on {key}")


def summarize(args):
    rows = read_csv(args.metrics, FIELDS)
    validate_metrics(rows)
    label_tables = []
    for path, keys in ((args.clip_labels, CLIP), (args.frame_labels, FRAME)):
        labels = {}
        if path:
            for row in read_csv(path, keys, keys):
                labels[tuple(row[key] for key in keys)] = {
                    field: row[field] for field in LABELS if row.get(field)
                }
        label_tables.append((keys, labels))

    # Always separate datasets, tracker, metric definitions and units.
    group_keys = tuple(dict.fromkeys(("dataset", "tracker", "protocol", "metric", "unit",
                                     "region", "motion_split", "view_split", "capture",
                                     *args.group_by)))
    groups = defaultdict(lambda: defaultdict(list))
    for row in rows:
        tags = {}
        for keys, labels in label_tables:
            tags.update(labels.get(tuple(row[key] for key in keys), {}))
        group = tuple(row.get(key, tags.get(key, "unknown")) for key in group_keys)
        clip = tuple(row[key] for key in CLIP)
        groups[group][clip].append(float(row["value"]))

    summary = []
    for group, clips in sorted(groups.items()):
        subjects = defaultdict(list)
        for clip, values in clips.items():
            subjects[clip[:2]].append(mean(values))
        summary.append(dict(zip(group_keys, group), n_identities=len(subjects),
                            n_clips=len(clips), n_observations=sum(map(len, clips.values())),
                            mean=mean(mean(values) for values in subjects.values())))
    write_csv(args.output, (*group_keys, "n_identities", "n_clips", "n_observations", "mean"),
              summary)


def ga_frames(dataset):
    """Validate GA split inputs and preserve the reader's per-split camera ordering."""
    import numpy as np

    if not (dataset / "canonical_flame_param.npz").is_file():
        raise ValueError(f"{dataset}: missing canonical_flame_param.npz")
    images = set()
    frame_keys = set()
    all_meshes = {}
    result = {}
    for split in ("train", "val", "test"):
        document = json.loads((dataset / f"transforms_{split}.json").read_text())
        frames = []
        for entry in document["frames"]:
            timestep = entry["timestep_index"]
            if type(timestep) is not int or timestep < 0:
                raise ValueError(f"{split}: invalid timestep {timestep}")
            camera = str(entry.get("camera_id", entry.get("camera_index", "")))
            if not camera or (timestep, camera) in frame_keys:
                raise ValueError(f"{split}: missing camera or duplicate frame {(timestep, camera)}")
            frame_keys.add((timestep, camera))
            image = dataset / entry["file_path"]
            if not image.suffix:
                image = image.with_suffix(".png")
            mesh = dataset / entry["flame_param_path"]
            if not image.is_file() or not mesh.is_file():
                raise ValueError(f"{split}: missing image or mesh for {(timestep, camera)}")
            if image.resolve() in images:
                raise ValueError(f"Repeated RGB path across entries: {image}")
            images.add(image.resolve())
            old_mesh = all_meshes.setdefault(timestep, mesh.resolve())
            if old_mesh != mesh.resolve() and old_mesh.read_bytes() != mesh.read_bytes():
                raise ValueError(f"Conflicting mesh records for timestep {timestep}")
            transform = np.asarray(entry["transform_matrix"], dtype=float)
            fov = float(entry.get("camera_angle_x", document.get("camera_angle_x", math.nan)))
            if (transform.shape != (4, 4) or not np.isfinite(transform).all()
                    or not np.allclose(transform[3], [0, 0, 0, 1])
                    or not np.allclose(transform[:3, :3].T @ transform[:3, :3], np.eye(3), atol=1e-4)
                    or np.linalg.det(transform[:3, :3]) <= 0 or not 0 < fov < math.pi):
                raise ValueError(f"{split}: invalid camera transform or FOV for {image}")
            source_index = entry.get("timestep_index_original")
            frames.append((entry, image, mesh, camera, source_index))
        result[split] = frames
    if not result["train"] or 0 not in all_meshes:
        raise ValueError("GA requires nonempty training data and timestep zero")
    return result


def ga_manifest(args):
    splits = ga_frames(args.source)
    rows = []
    for index, (entry, image, _, camera, source_index) in enumerate(splits[args.split]):
        timestep = entry["timestep_index"]
        rows.append(dict(render=f"{index:05d}.png", image=str(image.resolve()),
                         timestep=timestep, export_source_index=source_index, camera=camera,
                         dataset=args.dataset, subject=args.subject, sequence=args.sequence,
                         motion_split=args.motion_split, view_split=args.view_split,
                         capture=args.capture, ga_split=args.split))
    if not rows:
        raise ValueError(f"{args.split} is empty")
    write_csv(args.output, tuple(rows[0]), rows)


def score_rgb(args):
    import numpy as np
    from PIL import Image

    keys = ("render", "timestep", *CLIP, "motion_split", "view_split", "capture")
    frames = read_csv(args.manifest, keys, ("render",))
    expected = {row["render"] for row in frames}
    for folder in (args.renders, args.references):
        found = {file.name for file in folder.glob("*.png")}
        if found != expected:
            raise ValueError(f"{folder}: missing {sorted(expected - found)[:5]}, "
                             f"unexpected {sorted(found - expected)[:5]}")
    rows = []
    for frame in frames:
        def rgb(folder):
            with Image.open(folder / frame["render"]) as image:
                return np.asarray(image.convert("RGB"), dtype=np.float32) / 255

        pred, gt = rgb(args.renders), rgb(args.references)
        if pred.shape != gt.shape:
            raise ValueError(f"RGB size mismatch for {frame['render']}: {pred.shape} vs {gt.shape}")
        mse = float(np.mean((pred - gt) ** 2))
        rows.append(dict({key: frame[key] for key in (*CLIP, "motion_split", "view_split", "capture")},
                         frame=frame["timestep"], tracker=args.tracker, protocol=args.protocol,
                         region="full_frame", metric="psnr", unit="dB",
                         value=min(99, -10 * math.log10(mse)) if mse else 99))
    validate_metrics(rows)
    write_csv(args.output, FIELDS, rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    multirex = commands.add_parser("multirex", help="Normalize region-aligned MultiREX CSVs")
    multirex.add_argument("--metrics", nargs=2, metavar=("TRACKER", "CSV"), action="append", required=True)
    multirex.add_argument("--protocol", required=True)
    multirex.set_defaults(handler=import_multirex)

    group = commands.add_parser("summarize", help="Group metrics with equal weight per identity")
    group.add_argument("--metrics", type=Path, required=True)
    group.add_argument("--group-by", nargs="+", choices=GROUPS, default=[])
    group.add_argument("--clip-labels", type=Path)
    group.add_argument("--frame-labels", type=Path)
    group.set_defaults(handler=summarize)

    manifest = commands.add_parser("ga-manifest", help="Validate a GA export and map render indices")
    manifest.add_argument("--source", type=Path, required=True)
    manifest.add_argument("--split", choices=("train", "val", "test"), required=True)
    for key in ("dataset", "subject", "sequence"):
        manifest.add_argument(f"--{key}", required=True)
    for name in ("motion", "view"):
        manifest.add_argument(f"--{name}-split", choices=("seen", "unseen", "unknown"), required=True)
    manifest.add_argument("--capture", choices=("monocular", "multiview"), required=True)
    manifest.set_defaults(handler=ga_manifest)

    rgb = commands.add_parser("score-rgb", help="Score exact render/GT image pairs by manifest")
    for key in ("manifest", "renders", "references"):
        rgb.add_argument(f"--{key}", type=Path, required=True)
    rgb.add_argument("--tracker", required=True)
    rgb.add_argument("--protocol", required=True)
    rgb.set_defaults(handler=score_rgb)

    for command in (multirex, group, manifest, rgb):
        command.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        args.handler(args)
    except (ValueError, KeyError, OSError) as error:
        parser.exit(1, f"error: {error}\n")


if __name__ == "__main__":
    main()
