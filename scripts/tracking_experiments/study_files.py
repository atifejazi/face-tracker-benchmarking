"""Read the analysis inputs for a completed run or the original 3,500-step archive."""

import json


def read_study(run):
    manifest = run / "run_config.json"
    if manifest.is_file():
        config = json.loads(manifest.read_text())
        subjects = [case["subject"] for case in config["cases"]]
        completion = run / "complete.json"
        if not completion.is_file() or json.loads(completion.read_text()) != subjects:
            raise ValueError(f"Incomplete study under {run}. Finish rendering before analysis.")
    else:
        subjects = sorted({file.parent.parent.name for file in run.glob("*/training_inputs/experiment.json")})
        config = {"iterations": 3500}
    if not subjects:
        raise ValueError(f"No completed study subjects under {run}")
    return subjects, config
