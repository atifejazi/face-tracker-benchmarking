#!/usr/bin/env python3
"""Compare newly tracked FLAME params against past_nersemble_tracked baselines."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


PARAM_KEYS = (
    "expr",
    "jaw_pose",
    "neck_pose",
    "eyes_pose",
    "rotation",
    "translation",
    "shape",
    "static_offset",
)


def mean_abs(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        return float("nan")
    return float(np.mean(np.abs(a - b)))


def max_abs(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        return float("nan")
    return float(np.max(np.abs(a - b)))


def compare_flame_dirs(new_dir: Path, past_dir: Path) -> dict:
    new_files = sorted(new_dir.glob("*.npz"))
    past_files = sorted(past_dir.glob("*.npz"))
    past_map = {p.name: p for p in past_files}
    shared = [p for p in new_files if p.name in past_map]
    out = {
        "n_new": len(new_files),
        "n_past": len(past_files),
        "n_shared": len(shared),
        "keys": {},
    }
    if not shared:
        return out

    accum = {k: [] for k in PARAM_KEYS}
    for nf in shared:
        a = np.load(nf)
        b = np.load(past_map[nf.name])
        for k in PARAM_KEYS:
            if k in a.files and k in b.files:
                accum[k].append(mean_abs(a[k], b[k]))

    for k, vals in accum.items():
        if not vals:
            continue
        arr = np.asarray(vals, dtype=np.float64)
        out["keys"][k] = {
            "mean_frame_mad": float(np.mean(arr)),
            "max_frame_mad": float(np.max(arr)),
            "median_frame_mad": float(np.median(arr)),
            "n": int(arr.size),
        }
    return out


def compare_canonical(new_p: Path, past_p: Path) -> dict | None:
    if not new_p.exists() or not past_p.exists():
        return None
    a = np.load(new_p)
    b = np.load(past_p)
    out = {}
    for k in PARAM_KEYS:
        if k in a.files and k in b.files:
            out[k] = {"mad": mean_abs(a[k], b[k]), "max_abs": max_abs(a[k], b[k])}
    return out


def compare_smirk_pt(new_p: Path, past_p: Path) -> dict | None:
    if not new_p.exists() or not past_p.exists():
        return None
    import torch

    a = torch.load(new_p, map_location="cpu", weights_only=False)
    b = torch.load(past_p, map_location="cpu", weights_only=False)
    out = {}
    for k in ("shape", "exp", "pose", "jaw", "eyelids", "cam"):
        if k in a and k in b:
            av = a[k].detach().cpu().numpy() if hasattr(a[k], "detach") else np.asarray(a[k])
            bv = b[k].detach().cpu().numpy() if hasattr(b[k], "detach") else np.asarray(b[k])
            n = min(len(av), len(bv))
            out[k] = {
                "n_new": int(len(av)),
                "n_past": int(len(bv)),
                "mad": mean_abs(av[:n], bv[:n]),
                "max_abs": max_abs(av[:n], bv[:n]),
            }
    return out


def compare_tracked_npz(new_p: Path, past_p: Path) -> dict | None:
    if not new_p.exists() or not past_p.exists():
        return None
    a = np.load(new_p)
    b = np.load(past_p)
    out = {}
    for k in ("expr", "jaw_pose", "neck_pose", "eyes_pose", "rotation", "translation", "shape", "static_offset"):
        if k in a.files and k in b.files:
            av, bv = a[k], b[k]
            if av.ndim >= 1 and bv.ndim >= 1 and av.shape[0] != bv.shape[0] and av.ndim == bv.ndim:
                n = min(av.shape[0], bv.shape[0])
                av, bv = av[:n], bv[:n]
            out[k] = {"mad": mean_abs(av, bv), "max_abs": max_abs(av, bv), "shape_new": list(a[k].shape), "shape_past": list(b[k].shape)}
    return out


def verdict(tracker: str, flame: dict, extra: dict | None = None) -> str:
    """Heuristic: near-identical vs materially different tracker outputs."""
    keys = flame.get("keys", {})
    if flame.get("n_shared", 0) == 0:
        return "MISSING"

    # Shape + expression + jaw are the main tracker signal (rotation/translation often zeroed).
    expr = keys.get("expr", {}).get("mean_frame_mad", float("nan"))
    jaw = keys.get("jaw_pose", {}).get("mean_frame_mad", float("nan"))
    shape = keys.get("shape", {}).get("mean_frame_mad", float("nan"))

    # Tolerances: identical re-run is ~1e-5..1e-3; different seed/machine often >>0.05
    if np.isnan(expr) or np.isnan(jaw):
        return "UNKNOWN"
    if expr < 1e-3 and jaw < 1e-3 and (np.isnan(shape) or shape < 1e-3):
        return "MATCH"
    if expr < 0.05 and jaw < 0.05 and (np.isnan(shape) or shape < 0.05):
        return "CLOSE"
    return "DIFFERENT"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--new-root", type=Path, required=True)
    ap.add_argument("--past-root", type=Path, required=True)
    ap.add_argument("--subjects", nargs="+", default=["017", "018", "037"])
    ap.add_argument("--sens", nargs="+", default=["SEN-01", "SEN-05", "SEN-09"])
    ap.add_argument("--trackers", nargs="+", default=["mica", "smirk", "vhap"])
    ap.add_argument("--output-json", type=Path, required=True)
    ap.add_argument("--output-md", type=Path, required=True)
    args = ap.parse_args()

    report = {"clips": [], "summary": {}}
    rows = []

    for tracker in args.trackers:
        verdicts = []
        for subject in args.subjects:
            for sen in args.sens:
                new = args.new_root / tracker / subject / sen
                past = args.past_root / tracker / subject / sen
                entry = {
                    "tracker": tracker,
                    "subject": subject,
                    "sen": sen,
                    "new_exists": new.exists(),
                    "past_exists": past.exists(),
                }
                if not new.exists() or not past.exists():
                    entry["verdict"] = "MISSING"
                    report["clips"].append(entry)
                    rows.append(entry)
                    continue

                flame = compare_flame_dirs(new / "flame_param", past / "flame_param")
                entry["flame_param"] = flame
                entry["canonical"] = compare_canonical(
                    new / "canonical_flame_param.npz", past / "canonical_flame_param.npz"
                )
                if tracker == "smirk":
                    entry["track_params"] = compare_smirk_pt(
                        new / "track_params.pt", past / "track_params.pt"
                    )
                if tracker == "vhap":
                    entry["tracked_flame_params"] = compare_tracked_npz(
                        new / "tracked_flame_params.npz", past / "tracked_flame_params.npz"
                    )
                entry["verdict"] = verdict(tracker, flame)
                verdicts.append(entry["verdict"])
                report["clips"].append(entry)
                rows.append(entry)
        report["summary"][tracker] = {
            "n": len(verdicts),
            "MATCH": verdicts.count("MATCH"),
            "CLOSE": verdicts.count("CLOSE"),
            "DIFFERENT": verdicts.count("DIFFERENT"),
            "MISSING": verdicts.count("MISSING"),
            "UNKNOWN": verdicts.count("UNKNOWN"),
        }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2))

    lines = [
        "# Tracker sanity: new vs past_nersemble_tracked",
        "",
        "## Summary",
        "",
        "| Tracker | MATCH | CLOSE | DIFFERENT | MISSING |",
        "|---------|------:|------:|----------:|--------:|",
    ]
    for tracker, s in report["summary"].items():
        lines.append(
            f"| {tracker} | {s['MATCH']} | {s['CLOSE']} | {s['DIFFERENT']} | {s['MISSING']} |"
        )
    lines += [
        "",
        "Verdict heuristic on per-frame `flame_param` MAD for expr/jaw/shape:",
        "- **MATCH**: MAD < 1e-3 (bit-identical / tiny float noise)",
        "- **CLOSE**: MAD < 0.05 (same track, minor numeric drift)",
        "- **DIFFERENT**: larger MAD (tracker outputs materially disagree)",
        "",
        "## Per-clip",
        "",
        "| Tracker | Subj | SEN | Frames new/past/shared | expr MAD | jaw MAD | shape MAD | Verdict |",
        "|---------|------|-----|------------------------|---------:|--------:|----------:|---------|",
    ]
    for e in rows:
        fp = e.get("flame_param") or {}
        keys = fp.get("keys") or {}
        expr = keys.get("expr", {}).get("mean_frame_mad", float("nan"))
        jaw = keys.get("jaw_pose", {}).get("mean_frame_mad", float("nan"))
        shape = keys.get("shape", {}).get("mean_frame_mad", float("nan"))
        n = f"{fp.get('n_new','-')}/{fp.get('n_past','-')}/{fp.get('n_shared','-')}"
        lines.append(
            f"| {e['tracker']} | {e['subject']} | {e['sen']} | {n} | "
            f"{expr:.4g} | {jaw:.4g} | {shape:.4g} | {e['verdict']} |"
        )

    args.output_md.write_text("\n".join(lines) + "\n")
    print(args.output_md.read_text())


if __name__ == "__main__":
    main()
