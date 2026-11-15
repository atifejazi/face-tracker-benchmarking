"""Compare image-based lip aperture against each exact reference frame."""
import argparse
from collections import defaultdict
import csv
from functools import lru_cache
import hashlib
import json
from pathlib import Path
from statistics import mean, median

import cv2
import mediapipe as mp
import numpy as np

from study_files import read_study

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(root, name, rows):
    with (root / name).open("w", newline="") as file:
        writer = csv.DictWriter(file, list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def evaluate(root, subjects, mesh):
    @lru_cache(maxsize=None)
    def aperture(path):
        image = cv2.imread(str(path))
        if image is None:
            raise FileNotFoundError(f"Cannot read image {path}")
        height, width = image.shape[:2]
        result = mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if not result.multi_face_landmarks:
            return None
        landmarks = result.multi_face_landmarks[0].landmark
        points = np.array([(landmarks[i].x * width, landmarks[i].y * height)
                           for i in (13, 14, 33, 263)])
        eye = np.linalg.norm(points[2] - points[3])
        if eye <= 0 or not np.isfinite(points).all():
            return None
        return float(np.linalg.norm(points[0] - points[1])), float(eye)


    rows = []
    observations = []
    for subject in subjects:
        for motion in ("seen", "unseen"):
            for family in ("lag", "jaw"):
                folder = root / subject / f"{motion}_{family}"
                experiment = json.loads((folder / "experiment.json").read_text())
                doses = [value["directory"] for value in experiment["doses"]]
                results = []
                for frame in experiment["split_frame_keys"]:
                    name = frame["render"]
                    input_ref = folder / doses[0] / "reference" / name
                    assert all(digest(input_ref) == digest(folder / dose / "reference" / name) for dose in doses)
                    ref = aperture(input_ref)
                    pred = [aperture(folder / dose / "renders" / name) for dose in doses]
                    valid = ref is not None and all(point is not None for point in pred)
                    errors = [abs(ref[0] - point[0]) / ref[1] for point in pred] if valid else [None] * len(doses)
                    results.append(errors)
                    for dose, point, error in zip(experiment["doses"], pred, errors):
                        observations.append(dict(
                            subject=subject, motion=motion, family=family,
                            timestep=frame["timestep"], dose=dose["dose"],
                            reference_detected=ref is not None, render_detected=point is not None,
                            all_doses_detected=valid, aperture_abs_error=error,
                        ))
                paired = [errors for errors in results if errors[0] is not None]
                for index, dose in enumerate(experiment["doses"]):
                    rows.append(dict(
                        subject=subject, motion=motion, family=family, dose=dose["dose"],
                        paired_frames=len(paired), selected_frames=len(results),
                        mean_aperture_abs_error=mean(errors[index] for errors in paired) if paired else None,
                        mean_delta_from_zero=mean(errors[index] - errors[0] for errors in paired) if paired else None,
                    ))

    write(root, "image_aperture_by_subject.csv", rows)
    write(root, "image_aperture_by_frame.csv", observations)
    groups = defaultdict(list)
    for row in rows:
        if row["mean_delta_from_zero"] is not None:
            groups[row["motion"], row["family"], row["dose"]].append(row)
    summary = []
    for (motion, family, dose), values in groups.items():
        changes = [r["mean_delta_from_zero"] for r in values]
        summary.append(dict(motion=motion, family=family, dose=dose, n_identities=len(changes),
                            median_delta=median(changes), min_delta=min(changes), max_delta=max(changes),
                            valid_frames=sum(r["paired_frames"] for r in values),
                            selected_frames=sum(r["selected_frames"] for r in values)))
    if summary:
        write(root, "image_aperture_summary.csv", summary)
    print(f"Processed {len(observations)} reference/render combinations.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    subjects, _ = read_study(args.run)
    with mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1,
                                          min_detection_confidence=0.5) as mesh:
        evaluate(args.run, subjects, mesh)


if __name__ == "__main__":
    main()
