"""Validate analyzed frame support, split isolation and the injected motion arrays."""
import argparse
import json
from pathlib import Path
from study_files import read_study

import numpy as np
from scipy.spatial.transform import Rotation

def validate(run, subjects, iterations):
    for subject in subjects:
        root = run / subject
        baseline = {}
        for dose in ("dose_00", "dose_01"):
            dataset = root / "training_inputs" / dose / "dataset"
            splits = {}
            for split in ("train", "val", "test"):
                doc = json.loads((dataset / f"transforms_{split}.json").read_text())
                splits[split] = [row["file_path"] for row in doc["frames"]]
            train, val, test = (set(splits[key]) for key in ("train", "val", "test"))
            assert train.isdisjoint(val) and train.isdisjoint(test) and val.isdisjoint(test)
            if not baseline:
                baseline = splits
            assert baseline == splits
            assert (root / f"model_{dose}/point_cloud/iteration_{iterations}/point_cloud.ply").is_file()
        for motion in ("seen", "unseen"):
            folder = root / f"{motion}_lag"
            exp = json.loads((folder / "experiment.json").read_text())
            assert exp["common_valid_start"] == 2
            assert min(exp["timesteps"]) >= 2
            with np.load(folder / "dose_00/flame_param.npz") as original:
                for dose, shift in (("dose_01", 1), ("dose_02", 2)):
                    with np.load(folder / dose / "flame_param.npz") as shifted:
                        for key in ("shape", "rotation", "neck_pose", "eyes_pose", "static_offset"):
                            np.testing.assert_array_equal(shifted[key], original[key])
                        np.testing.assert_allclose(shifted["expr"][2:], original["expr"][2-shift:len(original["expr"])-shift],
                                                   atol=1e-6)
                        goal = Rotation.from_rotvec(original["jaw_pose"][2-shift:len(original["jaw_pose"])-shift])
                        actual = Rotation.from_rotvec(shifted["jaw_pose"][2:])
                        assert np.max((goal.inv() * actual).magnitude()) < 1e-6
        print(subject, "matched splits and exact lag shifts verified")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    subjects, config = read_study(args.run)
    validate(args.run, subjects, config["iterations"])


if __name__ == "__main__":
    main()
