#!/usr/bin/env python3
"""Score the locked protocol and summarize paired effects by seed and identity."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, median
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audit_tracking import write_csv
from prepare_common_canvas import read_json, write_json


ARTICULATION = ("expression", "jaw", "lag")
PRIMARY = dict(shape="face_lpips", expression="aperture_error_pp",
               jaw="aperture_error_pp", lower_face="chin_psnr_db", lag="aperture_error_pp")
RGB_METRICS = ("face_psnr_db", "face_lpips", "frame_psnr_db", "frame_lpips",
               "foreground_psnr_db", "mouth_psnr_db", "chin_psnr_db")
METRICS = (*RGB_METRICS, "aperture_error_pp")


def jobs(root, config):
    for case in config["cases"]:
        subject, tracker = case["subject"], case["tracker"]
        for clip in case["clips"]:
            sequence = clip["sequence"]
            base = root / tracker / subject / sequence
            reference = root / "reference" / subject / sequence
            records = read_json(base / "frames.json")
            yield case, sequence, base, reference, records


def conditions(case, config):
    for seed in case["seeds"]:
        for arm in case["arms"]:
            names = ["clean"]
            if arm == "clean":
                names.extend(f"{family}_{grade}" for family in config["doses"] for grade in ("low", "high"))
            else:
                names.append(f"{config['train_arm_families'][arm]}_high")
            for name in names:
                yield seed, arm, name


def detect(job):
    import cv2
    import mediapipe as mp

    cv2.setNumThreads(1)
    images, output, frames, reference = job
    if output.is_file():
        return
    result = {}
    mouth_indices = sorted({index for edge in mp.solutions.face_mesh.FACEMESH_LIPS for index in edge})
    with mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1,
                                        min_detection_confidence=.5) as detector:
        for filename in frames:
            bgr = cv2.imread(str(images / filename))
            if bgr is None or bgr.shape != (256, 256, 3):
                raise ValueError(f"Invalid 256px RGB input at {images / filename}")
            found = detector.process(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)).multi_face_landmarks
            record = None
            if found:
                points = np.array([(p.x * 256, p.y * 256) for p in found[0].landmark])
                eye = float(np.linalg.norm(points[33] - points[263]))
                if eye > 0 and np.isfinite(points).all():
                    record = dict(aperture_px=float(np.linalg.norm(points[13] - points[14])),
                                  interocular_px=eye, inner_lips=points[[13, 14]].tolist())
                    if reference:
                        for name, subset, margin in (("face", points, .10),
                                                      ("mouth", points[mouth_indices], .25)):
                            pad = max(6, margin * float(np.ptp(subset[:, 0])))
                            start = np.maximum(0, np.floor(subset.min(axis=0) - pad)).astype(int)
                            end = np.minimum(256, np.ceil(subset.max(axis=0) + pad)).astype(int)
                            record[f"{name}_box"] = [*start.tolist(), *end.tolist()]
            result[filename] = record
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp")
    write_json(temporary, result)
    temporary.replace(output)


def detector_fingerprint():
    import mediapipe as mp
    package = Path(mp.__file__).resolve().parent
    weights = sorted(package.rglob("face_landmark*.tflite")) + sorted(package.rglob("face_detection_short_range.tflite"))
    if not weights:
        raise FileNotFoundError("Could not fingerprint the pinned Face Mesh weights")
    return dict(package_version=mp.__version__, static_image_mode=True, min_detection_confidence=.5,
                assets={str(path.relative_to(package)): hashlib.sha256(path.read_bytes()).hexdigest() for path in weights})


def landmarks_stage(root, config, workers, references_only):
    provenance = root / "detector.json"
    fingerprint = detector_fingerprint()
    if provenance.is_file() and read_json(provenance) != fingerprint:
        raise ValueError("Face Mesh does not match the recorded detector")
    write_json(provenance, fingerprint)
    tasks = []
    seen = set()
    for case, sequence, base, reference, records in jobs(root, config):
        names = [record["filename"] for record in records]
        if reference not in seen:
            tasks.append((reference, reference / "landmarks.json", names, True))
            seen.add(reference)
        if references_only:
            continue
        for seed, arm, name in conditions(case, config):
            if name != "clean" and not any(name.startswith(family) for family in ARTICULATION):
                continue
            renders = base / "renders" / f"seed_{seed}" / arm
            if not (renders / "complete.json").is_file():
                raise ValueError(f"Missing completed render at {renders}")
            tasks.append((renders / name, renders / f"{name}_landmarks.json", names, False))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for number, _ in enumerate(pool.map(detect, tasks), start=1):
            if number % 24 == 0:
                print(f"Detected landmarks for {number}/{len(tasks)} runs", flush=True)
    print(f"Verified landmark files for {len(tasks)} runs", flush=True)


def read_image(path, mode="RGB"):
    with Image.open(path) as image:
        if image.size != (256, 256):
            raise ValueError(f"Unexpected evaluation raster at {path}")
        return np.array(image.convert(mode), dtype=np.float32) / 255


def slice_box(image, box):
    x0, y0, x1, y1 = box
    return image[y0:y1, x0:x1]


def psnr(values):
    return min(99., -10 * math.log10(float(values))) if values > 0 else 99.


def score_rgb(root, config, device):
    import lpips
    import torch
    import torch.nn.functional as F

    torch.set_num_threads(2)
    network = lpips.LPIPS(net="alex", verbose=False).to(device).eval()
    model_asset = Path(lpips.__file__).parent / "weights/v0.1/alex.pth"
    weights = Path(torch.hub.get_dir()) / "checkpoints" / "alexnet-owt-7be5be79.pth"
    face_raster = config["face_lpips_raster"]
    write_json(root / "lpips.json", dict(net="alex", face_crop_raster=face_raster, input_range=[-1, 1],
                                        calibration_sha256=hashlib.sha256(model_asset.read_bytes()).hexdigest(),
                                        alexnet_sha256=hashlib.sha256(weights.read_bytes()).hexdigest()))

    def tensor(images):
        return (torch.from_numpy(np.stack(images)).permute(0, 3, 1, 2).to(device) * 2 - 1).contiguous()

    def face_tensor(image):
        value = tensor([image])
        return F.interpolate(value, size=(face_raster, face_raster), mode="bilinear", align_corners=False, antialias=True)

    for case, sequence, base, reference, records in jobs(root, config):
        landmark_ref = read_json(reference / "landmarks.json")
        expected = [rec["filename"] for rec in records]
        abs_reference = {name: read_image(reference / name) for name in expected}
        alpha = {name: read_image(reference / f"{Path(name).stem}_alpha.png", "L") for name in expected}
        with torch.no_grad():
            for seed, arm, name in conditions(case, config):
                input_dir = base / "renders" / f"seed_{seed}" / arm
                source = input_dir / name
                destination = input_dir / f"{name}_metrics.csv"
                if destination.is_file():
                    continue
                if sorted(p.name for p in source.glob("*.png")) != expected:
                    raise ValueError(f"Mismatched render filenames under {source}")
                path_lip = input_dir / f"{name}_landmarks.json"
                landmarks = read_json(path_lip) if path_lip.is_file() else {}
                if (name == "clean" or any(name.startswith(family) for family in ARTICULATION)) and set(landmarks) != set(expected):
                    raise ValueError("Missing aligned lip-aperture records")
                output = []
                for offset in range(0, len(records), 16):
                    group = records[offset:offset + 16]
                    images = [read_image(source / record["filename"]) for record in group]
                    targets = [abs_reference[record["filename"]] for record in group]
                    differences = np.square(np.stack(images) - np.stack(targets))
                    full = network(tensor(images), tensor(targets)).flatten().cpu().tolist()
                    face_images, face_targets, qualifying = [], [], []
                    for i, (record, error) in enumerate(zip(group, differences)):
                        filename = record["filename"]
                        ref_landmarks = landmark_ref[filename]
                        render_landmarks = landmarks.get(filename)
                        matte = alpha[filename]
                        if matte.sum() <= 0:
                            raise ValueError("Empty common foreground alpha")
                        row = dict(source_frame=record["source_frame"], aperture_error_pp=None,
                                   reference_aperture_pp=None, rendered_aperture_pp=None,
                                   frame_psnr_db=psnr(error.mean()), frame_lpips=full[i],
                                   foreground_psnr_db=psnr((error * matte[..., None]).sum() / (3 * matte.sum())),
                                   chin_psnr_db=psnr(slice_box(error, record["chin_box"]).mean()),
                                   face_psnr_db=None, face_lpips=None, mouth_psnr_db=None)
                        if ref_landmarks:
                            face_box, mouth_box = (ref_landmarks[f"{region}_box"] for region in ("face", "mouth"))
                            row["face_psnr_db"] = psnr(slice_box(error, face_box).mean())
                            row["mouth_psnr_db"] = psnr(slice_box(error, mouth_box).mean())
                            row["reference_aperture_pp"] = ref_landmarks["aperture_px"] / ref_landmarks["interocular_px"] * 100
                            face_images.append(face_tensor(slice_box(images[i], face_box)))
                            face_targets.append(face_tensor(slice_box(targets[i], face_box)))
                            qualifying.append(i)
                            if render_landmarks:
                                row["rendered_aperture_pp"] = render_landmarks["aperture_px"] / ref_landmarks["interocular_px"] * 100
                                row["aperture_error_pp"] = abs(row["rendered_aperture_pp"] - row["reference_aperture_pp"])
                        output.append(row)
                    if qualifying:
                        scores = network(torch.cat(face_images), torch.cat(face_targets)).flatten().cpu().tolist()
                        for index, score in zip(qualifying, scores):
                            output[offset + index]["face_lpips"] = score
                temporary = destination.with_suffix(".tmp")
                write_csv(temporary, tuple(output[0]), output)
                temporary.replace(destination)
        print(f"Scored {case['tracker']} {case['subject']} {sequence}", flush=True)


def paired(metric, conditions_table, keys, minimum):
    arrays = [[row[metric] for row in conditions_table[key]] for key in keys]
    available = np.array([[bool(value) for value in values] for values in arrays]).all(axis=0)
    count = int(available.sum())
    total = len(available)
    valid = count > 0 and (metric != "aperture_error_pp" or count / total >= minimum)
    averages = [mean(float(value) for value, keep in zip(values, available) if keep) for values in arrays] if valid else None
    return count, total, averages


def summarize(root, config):
    response, recovery = [], []
    minimum = config["detector_min_paired_coverage"]
    for case, sequence, base, _, records in jobs(root, config):
        identities = dict(tracker=case["tracker"], subject=case["subject"], sequence=sequence)
        values = {}
        for seed, arm, name in conditions(case, config):
            path = base / "renders" / f"seed_{seed}" / arm / f"{name}_metrics.csv"
            with path.open() as handle:
                rows = list(csv.DictReader(handle))
            if [int(row["source_frame"]) for row in rows] != [rec["source_frame"] for rec in records]:
                raise ValueError(f"Measured rows disagree with the locked manifest at {path}")
            values[seed, arm, name] = rows
        for seed in case["seeds"]:
            for family, doses in config["doses"].items():
                cells = [(seed, "clean", name) for name in ("clean", f"{family}_low", f"{family}_high")]
                metrics = METRICS if family in ARTICULATION else RGB_METRICS
                for metric in metrics:
                    valid, total, means = paired(metric, values, cells, minimum)
                    response.append(dict(**identities, seed=seed, family=family, metric=metric,
                                         primary=metric == PRIMARY[family], paired_frames=valid, total_frames=total,
                                         low_dose=doses[0], high_dose=doses[1], zero_dose=means[0] if means else None,
                                         low_minus_zero=means[1] - means[0] if means else None,
                                         high_minus_zero=means[2] - means[0] if means else None))
            for arm, family in config["train_arm_families"].items():
                if arm not in case["arms"]:
                    continue
                metric = PRIMARY[family]
                cells = [(seed, train_arm, drive) for train_arm in ("clean", arm)
                         for drive in ("clean", f"{family}_high")]
                valid, total, means = paired(metric, values, cells, minimum)
                direction = 1 if metric.endswith("_db") else -1
                clean_gain, biased_gain = (direction * (means[2] - means[0]), direction * (means[3] - means[1])) if means else (None, None)
                recovery.append(dict(**identities, seed=seed, family=family, metric=metric,
                                     paired_frames=valid, total_frames=total, clean_train_clean_drive=means[0] if means else None,
                                     clean_train_biased_drive=means[1] if means else None,
                                     biased_train_clean_drive=means[2] if means else None,
                                     biased_train_biased_drive=means[3] if means else None,
                                     clean_drive_gain=clean_gain, biased_drive_gain=biased_gain,
                                     interaction=biased_gain - clean_gain if means else None))
    return response, recovery


def save_summaries(root, response, recovery):
    write_csv(root / "dose_response_by_seed.csv", tuple(response[0]), response)
    if recovery:
        write_csv(root / "recovery_by_seed.csv", tuple(recovery[0]), recovery)
    for label, rows, columns, keys in (
        ("dose_response", [row for row in response if row["primary"]], ("low_minus_zero", "high_minus_zero"),
         ("tracker", "sequence", "family", "metric")),
        ("recovery", recovery, ("biased_drive_gain", "clean_drive_gain", "interaction"),
         ("tracker", "sequence", "family", "metric")),
    ):
        groups = {}
        for row in rows:
            identity = tuple(row[key] for key in keys)
            groups.setdefault(identity, {}).setdefault(row["subject"], []).append(row)
        output = []
        for identity, subjects in groups.items():
            for column in columns:
                per_subject = {subject: [row[column] for row in entries if row[column] is not None]
                               for subject, entries in subjects.items()}
                scores = {subject: mean(entries) for subject, entries in per_subject.items() if entries}
                output.append({**dict(zip(keys, identity)), "contrast": column, "identities": len(scores),
                               "planned_identities": len(subjects),
                               "subject_seed_counts": {s: len(entries) for s, entries in per_subject.items()},
                               "subject_means": scores, "median": median(scores.values()) if scores else None,
                               "range": [min(scores.values()), max(scores.values())] if scores else None,
                               "leave_one_out_medians": {
                                   subject: median(value for key, value in scores.items() if key != subject)
                                   for subject in scores if len(scores) > 1}})
        write_json(root / f"{label}_summary.json", output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("reference", "landmarks", "rgb", "analyze"))
    parser.add_argument("run", type=lambda path: Path(path).resolve())
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--include", action="append", default=[], type=lambda path: Path(path).resolve(),
                        help="Analyze another completed batch with the same locked protocol")
    parser.add_argument("--results", type=lambda path: Path(path).resolve(),
                        help="New summary directory when analyzing multiple batches")
    args = parser.parse_args()
    config = read_json(args.run / "protocol.json")
    if hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != config["code"][Path(__file__).name]:
        parser.error("Score implementation differs from the locked protocol")
    if args.stage in ("reference", "landmarks"):
        landmarks_stage(args.run, config, args.workers, args.stage == "reference")
    elif args.stage == "rgb":
        score_rgb(args.run, config, args.device)
    else:
        response, recovery = summarize(args.run, config)
        seen = {case["subject"] for case in config["cases"]}
        if args.include and args.results is None:
            parser.error("--include requires a new --results directory")
        for extra in args.include:
            additional = read_json(extra / "protocol.json")
            subjects = {case["subject"] for case in additional["cases"]}
            settings = ("code", "train_steps", "fps", "raster", "lag_trim", "scene_extent",
                        "doses", "train_arm_families", "primary_unseen", "face_lpips_raster", "detector_min_paired_coverage")
            if additional["smoke"] or config["smoke"] or seen & subjects or any(config[k] != additional[k] for k in settings):
                raise ValueError("The additional batch overlaps or differs from the locked protocol")
            for filename in ("detector.json", "lpips.json"):
                if read_json(args.run / filename) != read_json(extra / filename):
                    raise ValueError(f"Metric provenance differs for {extra}/{filename}")
            dose, gains = summarize(extra, additional)
            response.extend(dose)
            recovery.extend(gains)
            seen.update(subjects)
        destination = args.results or args.run
        if args.results:
            destination.mkdir(parents=True, exist_ok=False)
            write_json(destination / "analysis_inputs.json", [str(run) for run in (args.run, *args.include)])
        save_summaries(destination, response, recovery)


if __name__ == "__main__":
    main()
