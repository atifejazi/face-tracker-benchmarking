#!/usr/bin/env python3
"""Lock, train and render the matched NeRSemble perturbation study."""

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
from PIL import Image
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audit_tracking import ga_frames, write_csv
from ga_mesh_utils import ga_w2c_from_transform_matrix, load_flame, project_vertices
from perturb_avatar import (
    decode, export_training_dataset, load_renderer, lower_face_direction,
    perturb, read_state, region_bounds, state_from_drive, state_from_export,
)
from prepare_common_canvas import new_dataset, read_json, write_json
from run_expanded import execute


SUBJECTS = ("017", "018", "024", "037")
DOSES = {"shape": (0.25, 0.75), "expression": (0.25, 0.75),
         "jaw": (1, 3), "lower_face": (1, 3), "lag": (40, 80)}
RECOVERY = {"chin": "lower_face", "lag": "lag"}
START = 2
RASTER = 256


def path_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def version(repo):
    def git(*args):
        return subprocess.check_output(["git", "-C", str(repo), *args])
    return dict(commit=git("rev-parse", "HEAD").decode().strip(),
                tracked_diff_sha256=hashlib.sha256(git("diff", "HEAD", "--binary")).hexdigest())


def item_key(row):
    split, _, entry, image, camera, source = row
    return dict(split=split, timestep=entry["timestep_index"],
                camera=camera, source_frame=source, reference=str(image.resolve()))


def load_clip(directory):
    splits = ga_frames(directory)
    rows = [(split, i, entry, image, camera, source) for split, records in splits.items()
            for i, (entry, image, _, camera, source) in enumerate(records)]
    first = rows[0][-1]
    if any(row[4] != "222200037" or row[2]["timestep_index"] != i or row[-1] != first + i
           for i, row in enumerate(rows)):
        raise ValueError(f"Invalid or noncontiguous camera/timing support at {directory}")
    for _, _, frame, _, _, _ in rows:
        if (frame["w"], frame["h"]) != (512, 512):
            raise ValueError(f"Expected the 512px shared canvas at {directory}")
        if not (directory / frame["fg_mask_path"]).is_file():
            raise ValueError(f"Missing foreground matte at {directory}")
    return splits, rows


def case_specs(smoke, subjects):
    for subject in ("017",) if smoke else subjects:
        yield subject, "mica", ("SEN-01",) if smoke else ("SEN-01", "SEN-05", "SEN-09"), (0,) if smoke else (0, 1, 2)
    if not smoke:
        for subject in ("017", "018"):
            if subject in subjects:
                for tracker in ("smirk", "vhap"):
                    yield subject, tracker, ("SEN-01", "SEN-09"), (0,)


def state_variants(flame, base):
    direction, chin_support = lower_face_direction(flame, base, 35)
    variants = {"clean": base}
    for family, doses in DOSES.items():
        args = SimpleNamespace(family=family, component=0, fps=25)
        for grade, dose in zip(("low", "high"), doses):
            variants[f"{family}_{grade}"] = perturb(flame, base, args, dose, direction)
    # Validate the timing operation and its unaffected channels before training.
    for grade, shift in (("low", 1), ("high", 2)):
        lag = variants[f"lag_{grade}"]
        np.testing.assert_allclose(lag["expr"][START:], base["expr"][START-shift:-shift], atol=1e-6)
        from scipy.spatial.transform import Rotation
        actual = Rotation.from_rotvec(lag["jaw_pose"][START:])
        expected = Rotation.from_rotvec(base["jaw_pose"][START-shift:-shift])
        if np.max((expected.inv() * actual).magnitude()) > 1e-6:
            raise ValueError("Jaw lag deviates from the intended original frame shift")
        for key in ("shape", "rotation", "neck_pose", "eyes_pose", "static_offset", "dynamic_offset"):
            np.testing.assert_array_equal(lag[key], base[key])
    return variants, chin_support


def protocol_code():
    scripts = Path(__file__).resolve().parents[1]
    files = [Path(__file__), scripts / "perturb_avatar.py", scripts / "prepare_common_canvas.py",
             scripts / "ga_mesh_utils.py", Path(__file__).with_name("prepare_protocol_clip.py"),
             Path(__file__).with_name("score_protocol.py")]
    return {file.name: path_hash(file) for file in files}


def record_projection(clip, rows, vertices, landmarks):
    """Check upright clean-drive geometry and the native pinhole remap independently."""
    from scene.dataset_readers import readCamerasFromTransforms
    from utils.camera_utils import loadCam

    records = {split: readCamerasFromTransforms(str(clip), f"transforms_{split}.json", True)
               for split in {row[0] for row in rows}}
    archive = read_json(clip / "common_canvas.json") if (clip / "common_canvas.json").exists() else None
    lineage = {}
    native = {}
    if archive and "native_camera_lineage" in archive:
        from prepare_common_canvas import native_frames
        lineage = {row["canvas_timestep"]: row for row in archive["native_camera_lineage"]}
        native = native_frames(Path(archive["native"]))
    errors = []
    for (split, index, frame, _, _, _), verts, points in zip(rows, vertices, landmarks):
        uv, depth = project_vertices(points[:68], frame, size=(RASTER, RASTER))
        if not np.isfinite(uv).all() or depth.min() <= .2 or uv[8, 1] <= uv[27, 1]:
            raise ValueError(f"Invalid clean drive face projection at {clip}/{frame['timestep_index']}")
        if (uv < 0).any() or (uv >= RASTER).any():
            raise ValueError(f"Clean face landmarks clipped at {clip}/{frame['timestep_index']}")
        if lineage:
            original = lineage[frame["timestep_index"]]
            source = native[original["native_timestep"]]
            old = ga_w2c_from_transform_matrix(source["transform_matrix"])
            k = np.array([[source["fl_x"], 0, source["cx"]], [0, source["fl_y"], source["cy"]], [0, 0, 1.]])
            xyz = verts @ old[:3, :3].T + old[:3, 3]
            uvw = xyz @ (np.asarray(original["native_to_canvas_image_edges"]) @ k).T
            expected = .5 * uvw[:, :2] / uvw[:, 2:] - .5
            camera = loadCam(SimpleNamespace(resolution=2, data_device="cpu"), index, records[split][index], 1.)
            homog = np.column_stack((verts, np.ones(len(verts)))) @ camera.full_proj_transform.numpy()
            actual = (homog[:, :2] / homog[:, 3:] + 1) * RASTER / 2 - .5
            errors.append(float(np.max(np.linalg.norm(expected - actual, axis=1))))
    if errors and max(errors) > .001:
        raise ValueError(f"GA remap differs from the native projection at {clip}")
    return dict(upright_visible_frames=len(rows), max_native_remap_error_px=max(errors) if errors else None)


def save_geometry(root, variants, flame, rows, chin_support):
    indices = [row[2]["timestep_index"] for row in rows]
    baseline, landmarks = decode(flame, variants["clean"], indices)
    face = flame.mask.v.face.numpy()
    lips = flame.mask.v.lips.numpy()
    output = []
    for treatment, state in variants.items():
        verts, _ = decode(flame, state, indices)
        displacement = np.linalg.norm(verts - baseline, axis=-1) * 1000
        np.savez_compressed(root / f"{treatment}.npz", **state)
        for row, magnitudes in zip(rows, displacement):
            output.append(dict(
                treatment=treatment, timestep=row[2]["timestep_index"], source_frame=row[-1],
                face_rms_mm=float(np.sqrt(np.mean(magnitudes[face] ** 2))),
                face_p95_mm=float(np.quantile(magnitudes[face], .95)),
                lip_rms_mm=float(np.sqrt(np.mean(magnitudes[lips] ** 2))),
                lip_p95_mm=float(np.quantile(magnitudes[lips], .95)),
                chin_rms_mm=float(np.sqrt(np.mean(magnitudes[chin_support] ** 2))),
                chin_p95_mm=float(np.quantile(magnitudes[chin_support], .95))))
    write_csv(root / "geometry.csv", tuple(output[0]), output)
    return baseline, landmarks


def lock(args):
    flame = load_flame(args.ga_root, torch.device("cpu"))
    cases = []
    common_records = {}
    with new_dataset(args.output) as stage:
        for subject, tracker, sequences, seeds in case_specs(args.smoke, args.subjects):
            source = args.inputs / subject / "SEN-01" / tracker
            source_splits, _ = load_clip(source)
            source_state = state_from_export(flame, source_splits)
            relative = Path(tracker) / subject
            directory = stage / relative
            directory.mkdir(parents=True)
            arms = ("clean", "chin", "lag") if tracker == "mica" else ("clean",)
            source_variants, _ = state_variants(flame, source_state)
            for arm in arms:
                state = source_variants["clean" if arm == "clean" else f"{RECOVERY[arm]}_high"]
                transforms = {split: read_json(source / f"transforms_{split}.json") for split in source_splits}
                dataset = directory / "train_inputs" / arm
                export_training_dataset(source, source_splits, transforms, state, START, dataset)
                np.savez_compressed(directory / "train_inputs" / f"{arm}.npz", **state)
            cases.append(dict(subject=subject, tracker=tracker, seeds=seeds, arms=arms, clips=[]))
            for sequence in sequences:
                clip = args.inputs / subject / sequence / tracker
                splits, all_rows = load_clip(clip)
                shared = [item_key(row) for row in all_rows]
                if common_records.setdefault((subject, sequence), shared) != shared:
                    raise ValueError(f"Cross-tracker time, splits or targets differ for {subject} {sequence}")
                if tracker != "mica" and sequence == "SEN-01":
                    continue  # Transfer arm scores only SEN-09.
                drive = source_state if sequence == "SEN-01" else state_from_drive(flame, source_state, splits)
                variants, chin_support = state_variants(flame, drive)
                base = directory / sequence
                base.mkdir()
                split = "val" if args.smoke else "test"
                rows = [row for row in all_rows
                        if row[2]["timestep_index"] >= START and (sequence != "SEN-01" or row[0] == split)]
                if args.smoke:
                    rows = rows[:3]
                vertices, landmarks = save_geometry(base, variants, flame, rows, chin_support)
                check = record_projection(clip, rows, vertices, landmarks)
                references = stage / "reference" / subject / sequence
                references.mkdir(parents=True, exist_ok=True)
                records = []
                for row, verts in zip(rows, vertices):
                    split, index, frame, image, _, original = row
                    filename = f"{original:06d}.png"
                    with Image.open(image) as ref:
                        raster = ref.convert("RGB").resize((RASTER, RASTER))
                    target = references / filename
                    if target.exists():
                        with Image.open(target) as other:
                            if not np.array_equal(np.asarray(other), np.asarray(raster)):
                                raise ValueError("Common reference bytes differ between trackers")
                    else:
                        raster.save(target)
                        with Image.open(clip / frame["fg_mask_path"]) as alpha:
                            alpha.convert("L").resize((RASTER, RASTER)).save(references / f"{original:06d}_alpha.png")
                    records.append({**item_key(row), "record_index": index, "filename": filename,
                                    "reference_sha256": path_hash(target),
                                    "chin_box": region_bounds(verts[chin_support], frame, RASTER, RASTER, original)})
                write_json(base / "frames.json", records)
                cases[-1]["clips"].append(dict(sequence=sequence, dataset=str(clip), projection=check,
                                               frame_count=len(rows)))
                print(f"Locked {tracker} {subject} {sequence} · {len(rows)} evaluation frames", flush=True)
        repo = Path(__file__).resolve().parents[2]
        datasets = sorted({(case["subject"], sequence, case["tracker"])
                           for case in cases
                           for sequence in ("SEN-01", *(clip["sequence"] for clip in case["clips"]))})
        dataset_hashes = {}
        for subject, sequence, tracker in datasets:
            directory = args.inputs / subject / sequence / tracker
            splits = ga_frames(directory)
            dataset_hashes[f"{tracker}/{subject}/{sequence}"] = dict(
                transforms={split: path_hash(directory / f"transforms_{split}.json") for split in splits},
                frames=[dict(source_frame=original, rgb_sha256=path_hash(rgb), flame_sha256=path_hash(mesh),
                             alpha_sha256=path_hash(directory / frame["fg_mask_path"]))
                        for entries in splits.values() for frame, rgb, mesh, _, original in entries])
        write_json(stage / "protocol.json", dict(
            smoke=args.smoke, ga_root=str(args.ga_root), cases=cases, fps=25, lag_trim=START,
            train_steps=2 if args.smoke else 3500, raster=RASTER, scene_extent=.1, doses=DOSES,
            primary_unseen="SEN-09", detector_min_paired_coverage=.8, face_lpips_raster=96,
            train_arm_families=RECOVERY, code=protocol_code(),
            datasets=dataset_hashes, repos={"benchmark": version(repo), "ga": version(args.ga_root)},
            flame_sha256=path_hash(args.ga_root / "flame_model/assets/flame/generic_model.pkl"),
        ))


def checkpoint_path(root, seed, arm, iterations):
    return root / "models" / f"seed_{seed}" / arm / "point_cloud" / f"iteration_{iterations}"


def train(args, config):
    flame = load_flame(args.ga_root, torch.device("cpu"))
    steps = config["train_steps"]
    for case in config["cases"]:
        root = args.output / case["tracker"] / case["subject"]
        for seed in case["seeds"]:
            for arm in case["arms"]:
                checkpoint = checkpoint_path(root, seed, arm, steps)
                model = checkpoint.parents[1]
                if (model / "verified.json").is_file():
                    continue
                model.mkdir(parents=True, exist_ok=False)
                execute(args.ga_root / "train.py", [
                    "-s", root / "train_inputs" / arm, "-m", model, "--eval", "--bind_to_mesh",
                    "--white_background", "--not_finetune_flame_params", "-r", 2, "--sh_degree", 0,
                    "--lambda_scale", 1, "--lambda_xyz", 0.01, "--scaling_lr", .005, "--scene_extent", .1,
                    "--iterations", steps, "--interval", steps, "--test_iterations", steps + 1,
                    "--position_lr_max_steps", steps, "--densify_from_iter", 500,
                    "--densification_interval", 100, "--densify_until_iter", 2500,
                    "--opacity_reset_interval", 3000, "--port", args.port, "--seed", seed,
                ], args.ga_root, model / "train.log")
                expected = read_state(root / "train_inputs" / f"{arm}.npz", flame)
                saved = read_state(checkpoint / "flame_param.npz", flame)
                for key in expected:
                    original = expected[key] if key in ("shape", "static_offset") else expected[key][START:]
                    np.testing.assert_array_equal(original.reshape(saved[key].shape), saved[key])
                parsed = ast.parse((model / "cfg_args").read_text(), mode="eval").body
                settings = {item.arg: ast.literal_eval(item.value) for item in parsed.keywords}
                if settings["seed"] != seed or settings["scene_extent"] != .1 or not settings["not_finetune_flame_params"]:
                    raise ValueError(f"Checkpoint config disagrees with the locked run at {model}")
                write_json(model / "verified.json", dict(seed=seed, arm=arm, steps=steps,
                                                          gaussian_sha256=path_hash(checkpoint / "point_cloud.ply")))
                print(f"Trained {case['tracker']} {case['subject']} seed {seed} {arm}", flush=True)


def render_conditions(models, cameras, pipeline, background, states, images, records, flame, references):
    from gaussian_renderer import render

    by_treatment = {}
    with torch.no_grad():
        for treatment in states:
            folder = images / treatment.stem
            folder.mkdir()
            by_treatment[treatment.stem] = {
                key: torch.as_tensor(arr, dtype=torch.float32, device="cuda")
                for key, arr in read_state(treatment, flame).items()
            }
        for index, metadata in enumerate(records):
            view = cameras[index]
            if view.timestep != metadata["timestep"] or (view.image_width, view.image_height) != (RASTER, RASTER):
                raise ValueError("GA renderer frame or size disagrees with the lock")
            expected = references / metadata["filename"]
            with Image.open(expected) as ref:
                identity = (view.original_image[:3].permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
                if not np.array_equal(identity, np.asarray(ref)):
                    raise ValueError(f"GA renderer uses different target pixels at {expected}")
            for condition, state in by_treatment.items():
                models.flame_param = state
                models.select_mesh_by_timestep(view.timestep)
                result = render(view, models, pipeline, background)["render"]
                if not torch.isfinite(result).all():
                    raise ValueError(f"Nonfinite Gaussian render for {condition}/{metadata['filename']}")
                rgb = result.clamp(0, 1).mul(255).round().to(torch.uint8).permute(1, 2, 0).cpu().numpy()
                Image.fromarray(rgb).save(images / condition / metadata["filename"])


def render_study(args, config):
    from scene.dataset_readers import readCamerasFromTransforms

    flame = load_flame(args.ga_root, torch.device("cpu"))
    setup = SimpleNamespace(ga_root=args.ga_root, sh_degree=0, resolution=2, white_background=True)
    for case in config["cases"]:
        root = args.output / case["tracker"] / case["subject"]
        for clip in case["clips"]:
            sequence = clip["sequence"]
            base = root / sequence
            records = read_json(base / "frames.json")
            references = args.output / "reference" / case["subject"] / sequence
            for record in records:
                if path_hash(references / record["filename"]) != record["reference_sha256"]:
                    raise ValueError("The locked RGB reference changed")
            camera_records = {split: readCamerasFromTransforms(clip["dataset"], f"transforms_{split}.json", True)
                              for split in {record["split"] for record in records}}
            cameras = [camera_records[record["split"]][record["record_index"]] for record in records]
            for seed in case["seeds"]:
                for arm in case["arms"]:
                    output = base / "renders" / f"seed_{seed}" / arm
                    if (output / "complete.json").is_file():
                        continue
                    checkpoint = checkpoint_path(root, seed, arm, config["train_steps"])
                    if not (checkpoint.parents[1] / "verified.json").exists():
                        raise ValueError(f"No verified checkpoint at {checkpoint}")
                    options = ("clean", *(f"{family}_{grade}" for family in DOSES for grade in ("low", "high"))) if arm == "clean" else ("clean", f"{RECOVERY[arm]}_high")
                    gaussians, views, pipeline, background = load_renderer(setup, records, cameras, checkpoint / "point_cloud.ply")
                    with new_dataset(output) as temporary:
                        render_conditions(gaussians, views, pipeline, background, [base / f"{name}.npz" for name in options],
                                          temporary, records, flame, references)
                        write_json(temporary / "complete.json", dict(frames=len(records), conditions=options))
                    print(f"Rendered {case['tracker']} {case['subject']} {sequence} seed {seed} {arm}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("lock", "train", "render"))
    for key in ("inputs", "ga-root", "output"):
        parser.add_argument(f"--{key}", required=True, type=lambda name: Path(name).resolve())
    parser.add_argument("--smoke", action="store_true", help="Two-step check using only SEN-01 validation frames")
    parser.add_argument("--subjects", nargs="+", choices=SUBJECTS, default=SUBJECTS,
                        help="Lock an independent batch of the preregistered identities")
    parser.add_argument("--port", type=int, default=6081)
    args = parser.parse_args()
    if len(set(args.subjects)) != len(args.subjects):
        parser.error("Subject IDs must be unique")
    torch.set_num_threads(2)
    sys.path.insert(0, str(args.ga_root))
    os.chdir(args.ga_root)
    if args.stage == "lock":
        lock(args)
    else:
        config = read_json(args.output / "protocol.json")
        if config["ga_root"] != str(args.ga_root) or config["smoke"] != args.smoke:
            parser.error("Use the GA checkout and mode in the locked protocol")
        if config["code"] != protocol_code():
            parser.error("Study code differs from the locked protocol")
        (train if args.stage == "train" else render_study)(args, config)


if __name__ == "__main__":
    main()
