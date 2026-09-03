#!/usr/bin/env python3
"""MultiREX geometry eval for stride-subsampled tracker outputs.

Decodes subsampled FLAME params, compares against gt[frame_indices], and writes
unpooled metrics.csv where frame_nb is the **original** GT frame index.
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import trimesh
from tqdm import tqdm

MULTIREX_ROOT = Path(__file__).resolve().parent.parent / "ubisoft-laforge-multirex"
sys.path.insert(0, str(MULTIREX_ROOT / "src"))

from multirex.FLAME.FLAME import FLAME  # noqa: E402
from multirex.multirex_evaluation import (  # noqa: E402
    compute_metrics_mask,
    flame_to_multi_topo_and_scale,
    prepare_masks,
    rigid_align,
)


def decode_one(code_path: Path, neutral_obj: Path, flame: FLAME, device) -> np.ndarray:
    flame_params = np.load(code_path, allow_pickle=True)
    neutral_mesh = trimesh.load(str(neutral_obj), merge_norm=True, merge_tex=True)
    neutral_vertices = neutral_mesh.vertices
    exp = np.asarray(flame_params["expressions"], dtype=np.float32)
    pose = np.asarray(flame_params["poses"], dtype=np.float32).copy()
    eyelid_params = flame_params.get("eyelids", None)
    pose[:, :3] = 0
    neutral_face = np.tile(neutral_vertices, (pose.shape[0], 1, 1))
    input_dict = {
        "expression_params": torch.from_numpy(exp).float().to(device),
        "pose_params": torch.from_numpy(pose[:, :3]).float().to(device),
        "jaw_params": torch.from_numpy(pose[:, 3:]).float().to(device),
        "eyelid_params": (
            torch.from_numpy(np.asarray(eyelid_params, dtype=np.float32)).float().to(device)
            if eyelid_params is not None
            else None
        ),
    }
    output_dict = flame(
        input_dict,
        neutral_meshes=torch.from_numpy(neutral_face).float().to(device),
    )
    return output_dict["vertices"].detach().cpu().numpy()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_folder", required=True, type=Path)
    parser.add_argument("--output_folder", required=True, type=Path)
    parser.add_argument(
        "--assets_path",
        type=Path,
        default=MULTIREX_ROOT / "assets",
    )
    parser.add_argument(
        "--gt_path",
        type=Path,
        default=MULTIREX_ROOT / "assets" / "multiface_gt",
    )
    parser.add_argument("--n_exp_components", type=int, default=100)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    out = args.output_folder
    out.mkdir(parents=True, exist_ok=True)
    decoded_dir = out / "decoded_flame_meshes"
    decoded_dir.mkdir(parents=True, exist_ok=True)
    csv_file = out / "metrics.csv"
    if csv_file.exists() and not args.overwrite:
        raise FileExistsError(f"{csv_file} exists; pass --overwrite")

    with open(args.assets_path / "id_mesh_weights_metadata_flame.pickle", "rb") as handle:
        id_metadata = pickle.load(handle)

    device = torch.device("cpu")
    flame = FLAME(
        flame_model_path=args.assets_path / "FLAME/generic_model.pkl",
        flame_lmk_embedding_path=args.assets_path / "FLAME/landmark_embedding.npy",
        n_shape=300,
        n_exp=args.n_exp_components,
        assets_path=args.assets_path,
    )
    masks_info = prepare_masks(args.assets_path / "regions")
    mask_names, masks, rigid_align_masks, _ = masks_info

    npz_files = sorted(args.input_folder.glob("*.npz"))
    if not npz_files:
        raise SystemExit(f"No npz in {args.input_folder}")

    rows = []
    for code_path in tqdm(npz_files):
        stem = code_path.stem
        subject = stem.split("--")[3]
        cam = "#" + stem.split("#")[-1]
        idx_path = args.input_folder / f"{stem}.frame_indices.npy"
        if not idx_path.exists():
            raise FileNotFoundError(f"Missing frame indices: {idx_path}")
        frame_indices = np.load(idx_path).astype(np.int64)

        if subject not in id_metadata:
            print(f"skip unknown subject {subject}")
            continue
        meta = id_metadata[subject]
        neutral = args.assets_path / "Neutrals_FLAME" / meta["neutral_mesh_flame"]
        gt_path = args.gt_path / meta["gt_sequence_name"]
        weights_path = args.assets_path / "conversion_matrices" / meta["conversion_weights_flame"]
        weights = __import__("scipy").sparse.load_npz(weights_path)

        decoded = decode_one(code_path, neutral, flame, device)
        if decoded.shape[0] != len(frame_indices):
            raise RuntimeError(
                f"{stem}: decoded {decoded.shape[0]} != indices {len(frame_indices)}"
            )
        np.save(decoded_dir / f"{stem}.npy", decoded)

        gt_full = np.load(gt_path)
        gt = gt_full[frame_indices]
        seq_multi = flame_to_multi_topo_and_scale(decoded, None, None, weights)

        for mask_name, mask in zip(mask_names, masks):
            aligned = rigid_align(gt.copy(), seq_multi.copy(), rigid_align_masks[mask_name])
            _, per_frame = compute_metrics_mask(gt, aligned, mask)
            for local_i, value in enumerate(per_frame):
                rows.append(
                    {
                        "source_video": f"{stem}.npy",
                        "camera": cam,
                        "region": mask_name,
                        "identity": str(int(subject)) if subject.isdigit() else subject,
                        "frame_nb": int(frame_indices[local_i]),
                        "value": float(value),
                    }
                )

    pd.DataFrame(rows).to_csv(csv_file, index=False)
    print(f"wrote {csv_file} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
