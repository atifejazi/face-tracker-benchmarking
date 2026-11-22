#!/usr/bin/env python3
"""Train and render the matched lower-face bias and articulation study."""

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audit_tracking import ga_frames


def execute(script, arguments, directory, log):
    command = [sys.executable, "-B", str(script), *map(str, arguments)]
    environment = os.environ.copy()
    environment.setdefault("OMP_NUM_THREADS", "2")
    environment.setdefault("OPENBLAS_NUM_THREADS", "2")
    with log.open("x") as file:
        file.write(f"{json.dumps(command)}\n")
        file.flush()
        result = subprocess.run(command, cwd=directory, env=environment,
                                stdout=file, stderr=subprocess.STDOUT, check=False)
    if result.returncode:
        raise RuntimeError(f"{script.name} failed with exit code {result.returncode}. See {log}")


def run_case(args, case):
    subject, source, drive = case["subject"], Path(case["source"]), Path(case["drive"])
    directory = args.output / subject
    directory.mkdir()
    perturb = Path(__file__).resolve().parents[1] / "perturb_avatar.py"
    common = ["--ga-root", args.ga_root, "--subject", subject, "--tracker", "mica",
              "--view-split", "seen", "--white-background", "--resolution", "2",
              "--sh-degree", "0", "--max-frames", str(args.frames)]

    def sweep(name, source_export, sequence, motion, family, extra):
        execute(perturb, [
            *common, "--dataset", source_export, "--sequence", sequence, "--motion-split",
            motion, "--family", family, "--output", directory / name, *extra,
        ], args.ga_root, directory / f"{name}.log")

    sweep("training_inputs", source, "SEN-01", "seen", "lower_face",
          ["--doses", 3, "--export-training-datasets"])
    datasets = {}
    checkpoints = {}
    seed_args = ["--seed", args.seed] if args.seed is not None else []
    for arm in ("dose_00", "dose_01"):
        datasets[arm] = directory / "training_inputs" / arm / "dataset"
        model = directory / f"model_{arm}"
        execute(args.ga_root / "train.py", [
            "-s", datasets[arm], "-m", model, "--eval", "--bind_to_mesh",
            "--white_background", "--not_finetune_flame_params", "-r", 2,
            "--sh_degree", 0, "--lambda_scale", 1, "--lambda_xyz", 0.01,
            "--scaling_lr", 0.005, "--iterations", args.iterations,
            "--interval", args.iterations, "--test_iterations", args.iterations + 1,
            "--position_lr_max_steps", args.iterations, "--densify_from_iter", 500,
            "--densification_interval", 100, "--densify_until_iter", 2500,
            "--opacity_reset_interval", 3000, "--port", args.port, *seed_args,
        ], args.ga_root, directory / f"train_{arm}.log")
        checkpoints[arm] = model / "point_cloud" / f"iteration_{args.iterations}"
        print(f"Trained {subject} {arm}", flush=True)

    clean = datasets["dose_00"]
    fixed_render = ["--checkpoint", checkpoints["dose_00"], "--render"]
    for family, doses in (("lag", (40, 80)), ("jaw", (1, 3)), ("lower_face", (1, 3))):
        conditions = [*fixed_render, "--doses", *doses, "--fps", 25]
        sweep(f"seen_{family}", clean, "SEN-01", "seen", family,
              [*conditions, "--split", "train"])
        sweep(f"unseen_{family}", clean, "SEN-05", "unseen", family,
              [*conditions, "--drive-dataset", drive])
        print(f"Rendered {subject} {family}", flush=True)

    for arm, checkpoint in checkpoints.items():
        for dose in ("00", "01"):
            state = directory / "training_inputs" / f"dose_{dose}" / "flame_param.npz"
            sweep(f"same_sentence_{arm}_{dose}", clean, "SEN-01", "seen", "lower_face",
                  ["--checkpoint", checkpoint, "--drive-state", state, "--split", "test",
                   "--max-frames", case["test_frames"], "--render"])

    # The unseen sweep already provides both clean-model renders.
    for dose in ("00", "02"):
        state = directory / "unseen_lower_face" / f"dose_{dose}" / "flame_param.npz"
        sweep(f"unseen_dose_01_{dose}", drive, "SEN-05", "unseen", "lower_face",
              ["--checkpoint", checkpoints["dose_01"], "--drive-state", state, "--render"])
    print(f"Completed {subject}", flush=True)


def preflight(case):
    source = ga_frames(Path(case["source"]))
    drive = ga_frames(Path(case["drive"]))
    if not source["test"]:
        raise ValueError(f"{case['subject']}: original SEN-01 export needs a test split")
    if any(len({row[3] for frames in dataset.values() for row in frames}) != 1
           for dataset in (source, drive)):
        raise ValueError(f"{case['subject']}: use a monocular export for each sequence")
    source_images = {row[1].resolve() for rows in source.values() for row in rows}
    if any(row[1].resolve() in source_images for rows in drive.values() for row in rows):
        raise ValueError(f"{case['subject']}: training and driving exports share RGB paths")
    for dataset in (source, drive):
        indices = sorted((row[0]["timestep_index"], row[-1])
                         for frames in dataset.values() for row in frames)
        if not all(index == number and type(original) is int and original == indices[0][1] + number
                   for number, (index, original) in enumerate(indices)):
            raise ValueError(f"{case['subject']}: lag requires contiguous original export indices")
    if len(source["train"]) < 3 or sum(map(len, drive.values())) < 3:
        raise ValueError(f"{case['subject']}: lag sweep needs at least three timed observations")
    return len(source["test"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ga-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New run directory")
    parser.add_argument("--case", nargs=3, metavar=("SUBJECT", "SEN01_EXPORT", "SEN05_EXPORT"),
                        action="append", required=True,
                        help="Same identity and physical camera, with verified 25 fps FLAME exports")
    parser.add_argument("--iterations", type=int, default=3500)
    parser.add_argument("--frames", type=int, default=32)
    parser.add_argument("--port", type=int, default=6030)
    parser.add_argument("--seed", type=int,
                        help="GA training seed for both arms. Defaults to GA's original seed 0.")
    args = parser.parse_args()
    args.ga_root, args.output = args.ga_root.resolve(), args.output.resolve()
    if args.iterations < 1 or args.frames < 1 or not 1 <= args.port <= 65535:
        parser.error("Use positive iteration and frame counts and a valid TCP port")
    if args.seed is not None and not 0 <= args.seed <= 2**32 - 1:
        parser.error("--seed must be an unsigned 32-bit integer")
    cases = []
    for subject, source, drive in args.case:
        if not re.fullmatch(r"[A-Za-z0-9_-]+", subject) or any(row["subject"] == subject for row in cases):
            parser.error(f"Use a unique subject identifier containing letters, digits, _ or -: {subject}")
        cases.append(dict(subject=subject, source=str(Path(source).resolve()), drive=str(Path(drive).resolve())))

    try:
        if not (args.ga_root / "train.py").is_file():
            raise ValueError(f"Missing train.py under {args.ga_root}")
        for case in cases:
            case["test_frames"] = preflight(case)
        args.output.mkdir(parents=True, exist_ok=False)
        config = dict(cases=cases, ga_root=str(args.ga_root), iterations=args.iterations,
                      frames=args.frames, fps=25, seed=args.seed or 0, port=args.port)
        (args.output / "run_config.json").write_text(json.dumps(config, indent=2) + "\n")
        for case in cases:
            run_case(args, case)
        (args.output / "complete.json").write_text(json.dumps([case["subject"] for case in cases]) + "\n")
    except (OSError, KeyError, ValueError, RuntimeError) as error:
        parser.exit(1, f"error: {error}\n")


if __name__ == "__main__":
    main()
