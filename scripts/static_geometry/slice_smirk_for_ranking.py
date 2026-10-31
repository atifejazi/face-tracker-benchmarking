#!/usr/bin/env python3
"""Slice existing SMIRK MultiREX npz files to ranking frame indices."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--input_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--skip_existing", action="store_true")
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text())
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for clip in manifest["clips"]:
        stem = clip["stem"]
        src = args.input_dir / f"{stem}.npz"
        dst = args.output_dir / f"{stem}.npz"
        dst_idx = args.output_dir / f"{stem}.frame_indices.npy"
        if args.skip_existing and dst.exists() and dst_idx.exists():
            print(f"skip {stem}")
            continue
        if not src.exists():
            print(f"MISSING {src}")
            continue
        data = np.load(src)
        idx = np.asarray(clip["frame_indices"], dtype=np.int64)
        if data["expressions"].shape[0] < idx.max() + 1:
            raise SystemExit(f"{src}: length {data['expressions'].shape[0]} < need {idx.max()+1}")
        out = {
            "expressions": np.asarray(data["expressions"][idx], dtype=np.float32),
            "poses": np.asarray(data["poses"][idx], dtype=np.float32),
        }
        if "eyelids" in data.files:
            out["eyelids"] = np.asarray(data["eyelids"][idx], dtype=np.float32)
        np.savez(dst, **out)
        np.save(dst_idx, idx)
        print(f"[{stem}] sliced {out['expressions'].shape}")


if __name__ == "__main__":
    main()
