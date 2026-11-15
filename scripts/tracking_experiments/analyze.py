"""Check pairing and summarize frozen sensitivity and training effects."""
from collections import defaultdict
import argparse
import csv
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, median
import sys

import numpy as np
from PIL import Image
import torch
from plyfile import PlyData

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from study_files import read_study
from ga_mesh_utils import load_flame
from perturb_avatar import decode, lower_face_direction, read_state, region_crop



@lru_cache(maxsize=None)
def experiment(folder):
    return json.loads((folder / "experiment.json").read_text())


@lru_cache(maxsize=None)
def files(folder, dose):
    frames = experiment(folder)["split_frame_keys"]
    paths = folder / dose
    names = {row["render"] for row in frames}
    for kind in ("reference", "renders"):
        assert {p.name for p in (paths / kind).glob("*.png")} == names, paths / kind
    return {
        (row["camera"], row["timestep"]): (paths / "reference" / row["render"],
                                            paths / "renders" / row["render"])
        for row in frames
    }


@lru_cache(maxsize=None)
def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_pair(clean, other):
    assert clean.keys() == other.keys()
    assert all(digest(clean[key][0]) == digest(other[key][0]) for key in clean)


def write(run, name, rows):
    path = run / name
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sensitivity(run, subjects):
    effects = []
    for subject in subjects:
        for motion in ("seen", "unseen"):
            for family in ("lag", "jaw", "lower_face"):
                folder = run / subject / f"{motion}_{family}"
                metrics = defaultdict(dict)
                with (folder / "rgb_metrics.csv").open() as file:
                    for row in csv.DictReader(file):
                        metrics[row["protocol"][-7:], row["region"]][(row["camera"], int(row["frame"]))] = float(row["value"])
                baseline = files(folder, "dose_00")
                region = "lower_face" if family == "lower_face" else "mouth"
                for dose in experiment(folder)["doses"]:
                    directory = dose["directory"]
                    check_pair(baseline, files(folder, directory))
                    reference = metrics["dose_00", region]
                    pred = metrics[directory, region]
                    assert reference.keys() == pred.keys() == baseline.keys()
                    deltas = [pred[key] - reference[key] for key in pred]
                    effects.append(dict(
                        subject=subject, motion=motion, family=family, dose=dose["dose"],
                        unit=experiment(folder)["dose_unit"], region=region, frames=len(deltas),
                        psnr_db=mean(pred.values()), delta_psnr_db=mean(deltas),
                        mean_vertex_rms_mm=dose["mean_frame_rms_mm"],
                    ))
    write(run, "sensitivity_by_subject.csv", effects)
    groups = defaultdict(list)
    for row in effects:
        groups[row["motion"], row["family"], row["dose"], row["unit"], row["region"]].append(row)
    summary = []
    for (motion, family, dose, unit, region), rows in groups.items():
        vals = [r["delta_psnr_db"] for r in rows]
        summary.append(dict(
            motion=motion, family=family, dose=dose, unit=unit, region=region,
            n_identities=len(rows), total_frames=sum(r["frames"] for r in rows),
            median_psnr_delta_db=median(vals), min_delta_db=min(vals), max_delta_db=max(vals),
            n_decreased=sum(v < -0.1 for v in vals), n_increased=sum(v > 0.1 for v in vals),
        ))
    write(run, "sensitivity_summary.csv", summary)


def ga_metadata(dataset):
    frames = {}
    for split in ("train", "val", "test"):
        doc = json.loads((dataset / f"transforms_{split}.json").read_text())
        defaults = {key: value for key, value in doc.items() if key != "frames"}
        for frame in doc["frames"]:
            cam = str(frame.get("camera_id", frame.get("camera_index")))
            frames[cam, frame["timestep_index"]] = defaults | frame
    return frames


def read_rgb(path):
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.float32).transpose(2, 0, 1) / 255


def compensation(run, subjects, flame):
    effects = []
    per_frame = []
    for subject in subjects:
        base_dir = run / subject
        for regime in ("same_sentence", "unseen"):
            clean_folder = (base_dir / "same_sentence_dose_00_00" if regime == "same_sentence"
                            else base_dir / "unseen_lower_face")
            clean_files = files(clean_folder, "dose_00")
            ex = experiment(clean_folder)
            state = read_state(clean_folder / "dose_00" / "flame_param.npz", flame)
            _, support = lower_face_direction(flame, state, 35)
            timesteps = [timestep for _, timestep in clean_files]
            verts, _ = decode(flame, state, timesteps)
            dataset = Path(ex["parameters"]["drive_dataset"] or ex["parameters"]["dataset"])
            metadata = ga_metadata(dataset)

            groups = defaultdict(dict)
            for trained in (0, 1):
                for driven in (0, 1):
                    if regime == "same_sentence":
                        folder = base_dir / f"same_sentence_dose_{trained:02d}_{driven:02d}"
                        dose = "dose_00"
                    elif trained == 0:
                        folder = base_dir / "unseen_lower_face"
                        dose = f"dose_{2 * driven:02d}"
                    else:
                        folder = base_dir / f"unseen_dose_01_{2 * driven:02d}"
                        dose = "dose_00"
                    pair = files(folder, dose)
                    check_pair(clean_files, pair)
                    for index, key in enumerate(clean_files):
                        ref_file, render_file = pair[key]
                        diff = (read_rgb(ref_file) - read_rgb(render_file)) ** 2
                        _, h, w = diff.shape
                        crop = region_crop(diff, verts[index, support], metadata[key], w, h, key)
                        mse = float(crop.mean())
                        psnr = min(99, -10 * math.log10(mse)) if mse else 99
                        groups[trained, driven][key] = psnr
                        per_frame.append(dict(subject=subject, regime=regime, camera=key[0],
                                              timestep=key[1], trained_mm=3 * trained,
                                              drive_mm=3 * driven, chin_psnr_db=psnr))
            clean_clean = mean(groups[0, 0].values())
            clean_bias = mean(groups[0, 1].values())
            bias_clean = mean(groups[1, 0].values())
            bias_bias = mean(groups[1, 1].values())
            effects.append(dict(
                subject=subject, regime=regime, frames=len(clean_files),
                clean_train_clean_drive_db=clean_clean, clean_train_biased_drive_db=clean_bias,
                biased_train_clean_drive_db=bias_clean, biased_train_biased_drive_db=bias_bias,
                retrain_gain_at_biased_drive_db=bias_bias - clean_bias,
                retrain_gain_at_clean_drive_db=bias_clean - clean_clean,
                train_drive_interaction_db=(bias_bias - clean_bias) - (bias_clean - clean_clean),
            ))
    write(run, "compensation_by_subject.csv", effects)
    write(run, "compensation_by_frame.csv", per_frame)


def local_offsets(run, subjects, iterations, flame):
    """Describe offsets bound to the perturbed area in the local triangle frame."""
    rows = []
    faces = flame.faces.numpy()
    for subject in subjects:
        folder = run / subject
        state = read_state(folder / "training_inputs/dose_00/flame_param.npz", flame)
        _, support = lower_face_direction(flame, state, 35)
        chin_faces = support[faces].sum(axis=1) >= 2
        for trained in (0, 1):
            ck = folder / f"model_dose_{trained:02d}/point_cloud/iteration_{iterations}"
            props = PlyData.read(str(ck / "point_cloud.ply"))["vertex"].data
            mask = chin_faces[props["binding_0"].astype(int)]
            for region, select in (("chin_support", mask), ("all_bindings", np.ones(len(props), bool))):
                offsets = np.stack([props[name][select] for name in ("x", "y", "z")], axis=1)
                normal = np.abs(offsets[:, 1])
                rows.append(dict(subject=subject, train_bias_mm=3 * trained, region=region,
                                 n_splats=len(normal), median_abs_normal_offset=float(np.median(normal)),
                                 p95_abs_normal_offset=float(np.quantile(normal, .95)),
                                 median_centroid_offset=float(np.median(np.linalg.norm(offsets, axis=1)))))
    write(run, "local_splat_offsets.csv", rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path, help="Completed run directory")
    parser.add_argument("--ga-root", type=Path, help="Required for archives without run_config.json")
    args = parser.parse_args()
    run = args.run.resolve()
    subjects, config = read_study(run)
    ga = args.ga_root or config.get("ga_root")
    if not ga:
        parser.error("Supply --ga-root when analyzing an archived run")
    sensitivity(run, subjects)
    flame = load_flame(Path(ga).resolve(), torch.device("cpu"))
    compensation(run, subjects, flame)
    local_offsets(run, subjects, config["iterations"], flame)


if __name__ == "__main__":
    main()
