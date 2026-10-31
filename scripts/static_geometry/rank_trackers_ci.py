#!/usr/bin/env python3
"""Rank trackers with subject-level means and 95% confidence intervals."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd

REGIONS = ["mouth_region", "cheek_region", "nose_region", "eyes_forehead_region"]


def subject_means(metrics_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(metrics_csv)
    df["identity"] = df["identity"].astype(str)
    # region average per frame then mean over frames/regions -> subject mean
    per = (
        df.groupby(["identity", "source_video", "frame_nb"])["value"]
        .mean()
        .reset_index()
    )
    # also keep per-region subject means
    region = (
        df.groupby(["identity", "region"])["value"].mean().unstack("region")
    )
    region["region_avg"] = region[REGIONS].mean(axis=1)
    # clip-level then subject-level for view roles
    clip = (
        df.groupby(["identity", "source_video"])["value"].mean().reset_index()
    )
    subj = clip.groupby("identity")["value"].mean().rename("clip_mean")
    out = region.join(subj, how="left")
    out.index.name = "identity"
    return out.reset_index()


def jitter_by_subject(jitter_csv: Path) -> pd.Series:
    if not jitter_csv.exists():
        return pd.Series(dtype=float)
    df = pd.read_csv(jitter_csv)
    df["subject"] = df["subject"].astype(str)
    # RMS across clips per subject
    return df.groupby("subject")["jitter_rms_mm"].apply(
        lambda s: float(np.sqrt(np.mean(np.square(s))))
    )


def mean_ci(values: np.ndarray) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    n = len(values)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    mean = float(np.mean(values))
    if n == 1:
        return mean, float("nan"), float("nan")
    sem = float(np.std(values, ddof=1) / np.sqrt(n))
    half = 1.96 * sem
    return mean, mean - half, mean + half


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval_root", type=Path, required=True)
    parser.add_argument("--trackers", nargs="+", required=True)
    parser.add_argument("--output_csv", type=Path, required=True)
    parser.add_argument("--output_md", type=Path, default=None)
    parser.add_argument("--per_subject_csv", type=Path, default=None)
    args = parser.parse_args()

    subject_rows = []
    summary_rows = []

    for tracker in args.trackers:
        metrics = args.eval_root / tracker / "metrics.csv"
        jitter = args.eval_root / tracker / "jitter_by_video.csv"
        if not metrics.exists():
            print(f"[skip] {tracker}: no metrics")
            continue
        subj = subject_means(metrics)
        jit = jitter_by_subject(jitter)
        subj["jitter_rms_mm"] = subj["identity"].map(jit)
        subj["tracker"] = tracker
        subject_rows.append(subj)

        row = {"tracker": tracker, "n_subjects": len(subj)}
        for col in REGIONS + ["region_avg", "jitter_rms_mm"]:
            mean, lo, hi = mean_ci(subj[col].to_numpy() if col in subj else np.array([]))
            row[f"{col}_mean"] = mean
            row[f"{col}_ci95_lo"] = lo
            row[f"{col}_ci95_hi"] = hi
        summary_rows.append(row)

    if not summary_rows:
        raise SystemExit("No tracker metrics found")

    if subject_rows and args.per_subject_csv:
        args.per_subject_csv.parent.mkdir(parents=True, exist_ok=True)
        pd.concat(subject_rows, ignore_index=True).to_csv(args.per_subject_csv, index=False)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = ["tracker", "n_subjects"]
    for col in REGIONS + ["region_avg", "jitter_rms_mm"]:
        fields += [f"{col}_mean", f"{col}_ci95_lo", f"{col}_ci95_hi"]
    with open(args.output_csv, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow({k: row.get(k, "") for k in fields})

    if args.output_md:
        lines = [
            "| Tracker | Region avg mm (95% CI) | Mouth | Cheek | Nose | Eyes/Forehead | Jitter mm/f² (95% CI) | N |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]

        def fmt(row, key):
            m, lo, hi = row[f"{key}_mean"], row[f"{key}_ci95_lo"], row[f"{key}_ci95_hi"]
            if np.isnan(m):
                return "n/a"
            if np.isnan(lo):
                return f"{m:.3f}"
            return f"{m:.3f} [{lo:.3f}, {hi:.3f}]"

        for row in summary_rows:
            lines.append(
                f"| {row['tracker']} | {fmt(row, 'region_avg')} | "
                f"{fmt(row, 'mouth_region')} | {fmt(row, 'cheek_region')} | "
                f"{fmt(row, 'nose_region')} | {fmt(row, 'eyes_forehead_region')} | "
                f"{fmt(row, 'jitter_rms_mm')} | {row['n_subjects']} |"
            )
        args.output_md.write_text("\n".join(lines) + "\n")
        print(args.output_md.read_text())

    print(f"wrote {args.output_csv}")


if __name__ == "__main__":
    main()
