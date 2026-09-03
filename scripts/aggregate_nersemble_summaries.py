#!/usr/bin/env python3
"""Build SUMMARY JSON files from per-run status.json under NERSSEMBLE_RUNS."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def load_rows(runs_root: Path, pattern: str) -> list[dict]:
    rows = []
    for p in sorted(runs_root.glob(pattern)):
        if p.is_file():
            rows.append(json.loads(p.read_text()))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=Path(os.environ.get("NERSSEMBLE_RUNS", "runs/nersemble")),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(os.environ.get("REPO_ROOT", ".")) / "results",
    )
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    all_rows = load_rows(args.runs_root, "ns*/status.json")
    by_tracker = {
        "mica": load_rows(args.runs_root, "ns*_mica/status.json"),
        "smirk": load_rows(args.runs_root, "ns*_smirk/status.json"),
        "vhap": load_rows(args.runs_root, "ns*_vhap/status.json"),
        "gt": load_rows(args.runs_root, "ns*_gt/status.json"),
    }

    (args.out_dir / "nersemble_SUMMARY_ALL_TRACKERS.json").write_text(
        json.dumps(all_rows, indent=2) + "\n"
    )
    for name, rows in by_tracker.items():
        if rows:
            (args.out_dir / f"nersemble_SUMMARY_{name.upper()}.json").write_text(
                json.dumps(rows, indent=2) + "\n"
            )
    print(f"Wrote {len(all_rows)} runs -> {args.out_dir}")


if __name__ == "__main__":
    main()
