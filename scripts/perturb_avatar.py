#!/usr/bin/env python3
"""Measure one FLAME perturbation family on a frozen GaussianAvatars checkpoint."""

import argparse
import json
import math
import os
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation, Slerp
import torch

from audit_tracking import FIELDS, ga_frames, write_csv

UNITS = dict(shape="coefficient", expression="coefficient", jaw="degree",
             lower_face="mm", lag="ms")
DYNAMIC = ("expr", "rotation", "neck_pose", "jaw_pose", "eyes_pose", "translation")


def read_state(path, flame):
    with np.load(path, allow_pickle=False) as archive:
        state = {key: archive[key] for key in archive.files}
    return validate_state(state, flame)


def validate_state(state, flame):
    width = len(state["expr"])
    sizes = dict(expr=100, rotation=3, neck_pose=3, jaw_pose=3, eyes_pose=6, translation=3)
    if state["shape"].shape != (300,) or any(
        state[key].shape != (width, size) for key, size in sizes.items()
    ):
        raise ValueError("Expected a GA FLAME checkpoint with 300 shape and 100 expression coefficients")
    vertices = len(flame.v_template)
    if state["static_offset"].shape not in ((vertices, 3), (1, vertices, 3)):
        raise ValueError("Checkpoint static_offset does not match FLAME topology")
    if state["dynamic_offset"].shape != (width, vertices, 3):
        raise ValueError("Checkpoint dynamic_offset dimensions do not match FLAME")
    if any(not np.isfinite(state[key]).all() for key in (*sizes, "shape", "static_offset", "dynamic_offset")):
        raise ValueError("Nonfinite values in the FLAME checkpoint")
    return state


def decode(flame, state, timesteps, neutral=False):
    indices = np.asarray(timesteps)
    shape = np.repeat(state["shape"][None], len(indices), axis=0)
    motion = [state[key][indices] for key in DYNAMIC]
    dynamic_offset = state["dynamic_offset"][indices]
    if neutral:
        motion = [np.zeros_like(values) for values in motion]
        dynamic_offset = np.zeros_like(dynamic_offset)
    with torch.no_grad():
        vertices, landmarks = flame(
            *[torch.as_tensor(array, dtype=torch.float32) for array in (shape, *motion)],
            static_offset=torch.as_tensor(state["static_offset"], dtype=torch.float32),
            dynamic_offset=torch.as_tensor(dynamic_offset, dtype=torch.float32),
        )
    return vertices.numpy(), landmarks.numpy()


def state_from_export(flame, splits):
    """Read one FLAME state from a GA export before avatar fine-tuning."""
    meshes = {entry["timestep_index"]: mesh for rows in splits.values()
              for entry, _, mesh, _, _ in rows}
    if set(meshes) != set(range(len(meshes))):
        raise ValueError("Driving timesteps must be contiguous and start at zero")
    source_identity = None
    motion = {key: [] for key in DYNAMIC}
    for index in range(len(meshes)):
        with np.load(meshes[index], allow_pickle=False) as data:
            identity = (data["shape"].reshape(-1), data["static_offset"].reshape(-1, 3))
            if source_identity is None:
                source_identity = identity
            elif any(not np.array_equal(a, b) for a, b in zip(source_identity, identity)):
                raise ValueError("Driving export identity varies by frame")
            for key in DYNAMIC:
                motion[key].append(data[key].reshape(-1))
    target = {key: np.stack(values).astype(np.float32) for key, values in motion.items()}
    target["shape"], target["static_offset"] = source_identity
    target["dynamic_offset"] = np.zeros((len(meshes), len(flame.v_template), 3), dtype=np.float32)
    return validate_state(target, flame)


def state_from_drive(flame, checkpoint_state, splits):
    """Take target motion and retain the avatar checkpoint's identity."""
    target = state_from_export(flame, splits)
    combined = dict(target, shape=checkpoint_state["shape"],
                    static_offset=checkpoint_state["static_offset"].reshape(-1, 3),
                    translation=target["translation"].copy())
    preserve_root_pivot(flame, target, combined)
    return combined


def export_training_dataset(source, splits, transforms, state, start, output):
    """Write a GA dataset with common RGB and reindexed valid training support."""
    if not any(row[0]["timestep_index"] >= start for row in splits["train"]):
        raise ValueError("Common lag interval leaves no training frames")
    mesh_dir = output / "flame_param"
    mesh_dir.mkdir(parents=True)
    for timestep in range(start, len(state["expr"])):
        fields = {key: state[key][[timestep]] for key in DYNAMIC}
        np.savez(mesh_dir / f"{timestep - start:05d}.npz", **fields,
                 shape=state["shape"], static_offset=state["static_offset"].reshape(-1, 3))
    with np.load(source / "canonical_flame_param.npz", allow_pickle=False) as canonical:
        fields = dict(canonical)
    fields.update(shape=state["shape"], static_offset=state["static_offset"].reshape(-1, 3))
    np.savez(output / "canonical_flame_param.npz", **fields)
    for split, rows in splits.items():
        entries = []
        for entry, image, _, _, _ in rows:
            if entry["timestep_index"] < start:
                continue
            new_index = entry["timestep_index"] - start
            item = dict(entry, file_path=str(image.resolve()), timestep_index=new_index,
                        flame_param_path=f"flame_param/{new_index:05d}.npz")
            if "fg_mask_path" in item:
                item["fg_mask_path"] = str((source / item["fg_mask_path"]).resolve())
            entries.append(item)
        # The GA reader consumes frame timestep indices directly. Keep the
        # optional summary indices consistent for other manifest readers.
        document = dict(transforms[split], frames=entries)
        if "timestep_indices" in document:
            document["timestep_indices"] = sorted({entry["timestep_index"] for entry in entries})
        (output / f"transforms_{split}.json").write_text(json.dumps(document, indent=2) + "\n")


def lower_face_direction(flame, base, radius_mm):
    vertices, landmarks = decode(flame, base, [0], neutral=True)
    vertices, landmarks = vertices[0], landmarks[0]
    allowed = np.zeros(len(vertices), dtype=bool)
    allowed[flame.mask.v.face.numpy()] = True
    allowed[flame.mask.v.lips.numpy()] = False
    allowed[flame.mask.v.boundary.numpy()] = False
    # Compact, smooth radial weighting centered on the neutral chin landmark.
    distance = np.linalg.norm(vertices - landmarks[8], axis=1)
    weights = np.maximum(1 - (distance / (radius_mm / 1000)) ** 2, 0) ** 2
    weights[~allowed] = 0
    support = weights > 0
    if not support.any():
        raise ValueError("Lower-face offset has empty FLAME support")
    direction = np.zeros_like(vertices)
    direction[:, 1] = -weights / np.sqrt(np.mean(weights[support] ** 2))
    return direction, support


def preserve_root_pivot(flame, base, perturbed):
    """Keep the rigid root transform fixed when the FLAME root joint moves."""
    root = flame.J_regressor[0].numpy()
    basis = np.einsum("v,vcd->cd", root, flame.shapedirs.numpy())
    shape_delta = perturbed["shape"] - base["shape"]
    expr_delta = perturbed["expr"] - base["expr"]
    static_delta = (perturbed["static_offset"] - base["static_offset"]).reshape(-1, 3)
    delta = shape_delta @ basis[:, :300].T + expr_delta @ basis[:, 300:].T + root @ static_delta
    rotation = Rotation.from_rotvec(base["rotation"])
    perturbed["translation"] += rotation.apply(delta) - delta


def perturb(flame, base, args, dose, direction):
    state = {key: value.copy() for key, value in base.items()}
    if dose == 0:
        return state
    if args.family == "shape":
        state["shape"][args.component] += dose
    elif args.family == "expression":
        state["expr"][:, args.component] += dose
    elif args.family == "lower_face":
        state["static_offset"] += direction.reshape(state["static_offset"].shape) * (dose / 1000)
    elif args.family == "jaw":
        offset = np.zeros(3)
        offset[args.component] = np.deg2rad(dose)
        state["jaw_pose"] = (Rotation.from_rotvec(base["jaw_pose"]) *
                             Rotation.from_rotvec(offset)).as_rotvec().astype(base["jaw_pose"].dtype)
    elif args.family == "lag":
        grid = np.arange(len(base["expr"]), dtype=float) / args.fps
        query = grid - dose / 1000
        valid = query >= 0
        state["expr"][valid] = np.stack([
            np.interp(query[valid], grid, column) for column in base["expr"].T
        ], axis=1)
        state["jaw_pose"][valid] = Slerp(grid, Rotation.from_rotvec(base["jaw_pose"]))(
            query[valid]).as_rotvec()
    preserve_root_pivot(flame, base, state)
    return state


def aperture(landmarks):
    absolute = np.linalg.norm(landmarks[:, 62] - landmarks[:, 66], axis=1)
    eye_distance = np.linalg.norm(landmarks[:, 36] - landmarks[:, 45], axis=1)
    if np.any(eye_distance <= 0):
        raise ValueError("Invalid interocular distance")
    return absolute / eye_distance


def motion_rows(frames, base_aperture, perturbed_aperture, displacement):
    for index, frame in enumerate(frames):
        yield dict(timestep=frame[0]["timestep_index"],
                   rms_displacement_mm=float(displacement[index]),
                   aperture_ratio=float(perturbed_aperture[index]),
                   aperture_absolute_error=float(abs(perturbed_aperture[index] - base_aperture[index])))


def region_crop(image, points, entry, w, h, stem):
    from ga_mesh_utils import project_vertices

    uv, depth = project_vertices(points, entry, size=(w, h))
    if not np.isfinite(uv).all() or np.any(depth <= 0):
        raise ValueError(f"Invalid baseline region projection for {stem}")
    pad = max(6, 0.25 * float(np.ptp(uv[:, 0])))
    x0, y0 = np.maximum(0, np.floor(uv.min(axis=0) - pad)).astype(int)
    x1, y1 = np.minimum([w, h], np.ceil(uv.max(axis=0) + pad)).astype(int)
    if x0 >= x1 or y0 >= y1:
        raise ValueError(f"Baseline region is outside the reference raster for {stem}")
    return image[:, y0:y1, x0:x1]


def render_dose(args, state, frames, regions, out, renderer):
    from PIL import Image
    from gaussian_renderer import render

    gaussians, cameras, pipeline, background = renderer
    gaussians.flame_param = {
        key: torch.as_tensor(state[key], device="cuda", dtype=torch.float32)
        for key in (*DYNAMIC, "shape", "static_offset", "dynamic_offset")
    }
    render_dir = out / "renders"
    gt_dir = out / "reference"
    render_dir.mkdir()
    gt_dir.mkdir()
    rows = []
    for index, ((entry, _, _, camera, _), view) in enumerate(zip(frames, cameras)):
        stem = f"{index:05d}.png"
        if view.timestep != entry["timestep_index"]:
            raise ValueError(f"Camera timestep mismatch for {stem}")
        with torch.no_grad():
            gaussians.select_mesh_by_timestep(view.timestep)
            prediction = render(view, gaussians, pipeline, background)["render"].clamp(0, 1).cpu()
            reference = view.original_image[:3].cpu()
        if not torch.isfinite(prediction).all():
            raise ValueError(f"Nonfinite render for {stem}")
        for image, directory in ((prediction, render_dir), (reference, gt_dir)):
            rgb = (image.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
            Image.fromarray(rgb).save(directory / stem)
        w, h = view.image_width, view.image_height
        difference = (prediction - reference).square()
        crops = {"full_frame": difference}
        for region, points in regions.items():
            crops[region] = region_crop(difference, points[index], entry, w, h, stem)
        for region, crop in crops.items():
            mse = float(crop.mean())
            rows.append(dict(
                dataset=args.dataset_name, subject=args.subject, sequence=args.sequence,
                camera=camera, frame=view.timestep, tracker=args.tracker,
                protocol=f"frozen_avatar_{args.family}_{out.name}",
                motion_split=args.motion_split, view_split=args.view_split,
                capture=args.capture, region=region, metric="psnr", unit="dB",
                value=min(99, -10 * math.log10(mse)) if mse else 99,
            ))
    return rows


def load_renderer(args, frames, camera_records, ply):
    from argparse import Namespace
    from gaussian_renderer import FlameGaussianModel
    from scene import CameraDataset
    from utils.camera_utils import cameraList_from_camInfos

    if not torch.cuda.is_available():
        raise RuntimeError("GaussianAvatars rendering requires CUDA")
    previous = Path.cwd()
    try:
        os.chdir(args.ga_root)
        gaussians = FlameGaussianModel(args.sh_degree)
    finally:
        os.chdir(previous)
    # This standalone renderer supplies the checkpoint's exact FLAME state per arm.
    gaussians.load_ply(str(ply), has_target=True)
    gaussians.active_sh_degree = args.sh_degree
    camera_options = Namespace(resolution=args.resolution, data_device="cuda", select_camera_id=-1)
    cameras = CameraDataset(cameraList_from_camInfos(camera_records, 1.0, camera_options))
    if len(cameras) != len(frames):
        raise ValueError("Camera count does not match the selected frames")
    pipeline = Namespace(compute_cov3D_python=False, convert_SHs_python=False, debug=False)
    background = torch.full((3,), float(args.white_background), device="cuda")
    return gaussians, cameras, pipeline, background


def run(args):
    from ga_mesh_utils import load_flame

    if (any(not math.isfinite(dose) or dose <= 0 for dose in args.doses)
            or len(set(args.doses)) != len(args.doses) or args.max_frames < 1
            or not math.isfinite(args.radius_mm) or args.radius_mm <= 0):
        raise ValueError("Use unique positive doses, a positive radius and at least one frame")
    if args.family in ("shape", "expression", "jaw"):
        limit = {"shape": 300, "expression": 100, "jaw": 3}[args.family]
        if not 0 <= args.component < limit:
            raise ValueError(f"{args.family} component must be in [0, {limit - 1}]")
    if args.checkpoint is None and (args.render or args.drive_dataset or args.drive_state):
        raise ValueError("Rendering and alternate drives require --checkpoint")
    flame = load_flame(args.ga_root, torch.device("cpu"))
    source_splits = ga_frames(args.dataset.resolve())
    ply = None
    if args.checkpoint:
        ply = args.checkpoint / "point_cloud.ply"
        sidecar = args.checkpoint / "flame_param.npz"
        if not ply.is_file() or not sidecar.is_file():
            raise ValueError("Checkpoint must contain point_cloud.ply and flame_param.npz")
        base = read_state(sidecar, flame)
    else:
        base = state_from_export(flame, source_splits)
    if args.drive_state:
        if args.drive_dataset or args.export_training_datasets:
            raise ValueError("An alternate drive state requires the evaluation export and its camera grid")
        base = read_state(args.drive_state, flame)
    source_timesteps = {row[0]["timestep_index"] for frames in source_splits.values() for row in frames}
    if source_timesteps != set(range(len(base["expr"]))):
        raise ValueError("Source export and FLAME state have different timestep support")
    if args.drive_dataset is not None and args.export_training_datasets:
        raise ValueError("Training dataset export requires the original source sequence")
    evaluation_dataset = (args.drive_dataset or args.dataset).resolve()
    splits = ga_frames(evaluation_dataset) if args.drive_dataset else source_splits
    if args.drive_dataset:
        base = state_from_drive(flame, base, splits)
    frame_count = len(base["expr"])
    all_timesteps = {row[0]["timestep_index"] for frames in splits.values() for row in frames}
    if all_timesteps != set(range(frame_count)):
        raise ValueError("GA export and FLAME sidecar must have the same contiguous timestep support")
    valid_start = 0
    if args.family == "lag" and args.doses:
        if args.fps is None or not math.isfinite(args.fps) or args.fps <= 0 or frame_count < 2:
            raise ValueError("Lag requires a verified constant timestep FPS and at least two frames")
        source_ids = {row[0]["timestep_index"]: row[-1] for frames in splits.values() for row in frames}
        if any(type(source_ids[index]) is not int or source_ids[index] != source_ids[0] + index
               for index in range(frame_count)):
            raise ValueError("Lag requires contiguous original export indices without gaps")
        valid_start = math.ceil(max(args.doses, default=0) * args.fps / 1000)
    candidates = [(split, index, row) for split, frames in splits.items()
                  if args.split in ("all", split) for index, row in enumerate(frames)
                  if row[0]["timestep_index"] >= valid_start]
    if not candidates:
        raise ValueError("No frames on the common valid interval")
    selected = np.linspace(0, len(candidates) - 1, min(len(candidates), args.max_frames), dtype=int)
    selected = [candidates[index] for index in selected]

    # Pass the shared FOV and image dimensions through to mesh projection.
    transforms = {
        split: json.loads((evaluation_dataset / f"transforms_{split}.json").read_text())
        for split in splits
    }
    frames = [({**{key: value for key, value in transforms[split].items() if key != "frames"}, **row[0]}, *row[1:])
              for split, _, row in selected]
    timesteps = [frame[0]["timestep_index"] for frame in frames]
    baseline_vertices, baseline_landmarks = decode(flame, base, timesteps)
    baseline_aperture = aperture(baseline_landmarks)
    direction, support = None, np.ones(len(flame.v_template), dtype=bool)
    if args.family == "lower_face":
        direction, support = lower_face_direction(flame, base, args.radius_mm)
    elif args.family in ("jaw", "expression", "lag"):
        support[:] = False
        support[flame.mask.v.lips.numpy()] = True
    regions = {"mouth": baseline_landmarks[:, 48:68]}
    if args.family == "lower_face":
        regions["lower_face"] = baseline_vertices[:, support]
    renderer = None
    if args.render:
        from scene.dataset_readers import readCamerasFromTransforms

        records = {
            split: readCamerasFromTransforms(str(evaluation_dataset),
                                             f"transforms_{split}.json", args.white_background)
            for split in {split for split, _, _ in selected}
        }
        renderer = load_renderer(args, frames, [records[split][index] for split, index, _ in selected], ply)
    args.output.mkdir(parents=True, exist_ok=False)
    rows = []
    summary = []
    for dose_index, dose in enumerate([0.0, *sorted(args.doses)]):
        state = perturb(flame, base, args, dose, direction)
        vertices, landmarks = decode(flame, state, timesteps)
        displacement = np.sqrt(np.mean(np.sum((vertices[:, support] -
                                               baseline_vertices[:, support]) ** 2, axis=-1), axis=1)) * 1000
        out = args.output / f"dose_{dose_index:02d}"
        out.mkdir()
        frame_rows = list(motion_rows(frames, baseline_aperture, aperture(landmarks), displacement))
        write_csv(out / "geometry.csv", tuple(frame_rows[0]), frame_rows)
        np.savez_compressed(out / "flame_param.npz", **state)
        if args.export_training_datasets:
            export_training_dataset(evaluation_dataset, splits, transforms, state,
                                    valid_start, out / "dataset")
        if renderer is not None:
            rows.extend(render_dose(args, state, frames, regions, out, renderer))
        summary.append(dict(dose=dose, directory=out.name, support_vertices=int(support.sum()),
                            mean_frame_rms_mm=float(displacement.mean())))
    if rows:
        write_csv(args.output / "rgb_metrics.csv", FIELDS, rows)
    parameters = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    manifest = dict(parameters=parameters, dose_unit=UNITS[args.family], timesteps=timesteps,
                    split_frame_keys=[dict(split=split, render=f"{i:05d}.png", camera=row[3],
                                           source_image=str(row[1].resolve()), timestep=row[0]["timestep_index"],
                                           export_source_index=row[-1])
                                      for i, (split, _, row) in enumerate(selected)],
                    common_valid_start=valid_start, doses=summary)
    (args.output / "experiment.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("ga-root", "dataset", "output"):
        parser.add_argument(f"--{key}", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path,
                        help="GA checkpoint directory. Omit to prepare geometry from the source export.")
    parser.add_argument("--drive-dataset", type=Path,
                        help="Optional registered driving export for the same identity")
    parser.add_argument("--drive-state", type=Path,
                        help="Alternate full FLAME sidecar on the same camera/timestamp grid")
    parser.add_argument("--export-training-datasets", action="store_true",
                        help="Write matched GA training datasets with disjoint RGB splits")
    parser.add_argument("--family", choices=UNITS, required=True)
    parser.add_argument("--doses", type=float, nargs="+", default=[])
    parser.add_argument("--component", type=int, default=0)
    parser.add_argument("--radius-mm", type=float, default=35)
    parser.add_argument("--fps", type=float)
    parser.add_argument("--split", choices=("all", "train", "val", "test"), default="all")
    parser.add_argument("--max-frames", type=int, default=24)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--resolution", type=int, default=2)
    parser.add_argument("--sh-degree", type=int, default=0)
    parser.add_argument("--white-background", action="store_true")
    parser.add_argument("--dataset-name", default="NeRSemble")
    for key in ("subject", "sequence", "tracker"):
        parser.add_argument(f"--{key}", required=True)
    for key in ("motion-split", "view-split"):
        parser.add_argument(f"--{key}", choices=("seen", "unseen", "unknown"), required=True)
    parser.add_argument("--capture", choices=("monocular", "multiview"), default="monocular")
    args = parser.parse_args()
    try:
        run(args)
    except (ValueError, KeyError, OSError, RuntimeError) as error:
        parser.exit(1, f"error: {error}\n")


if __name__ == "__main__":
    main()
