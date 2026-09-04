#!/usr/bin/env python3
"""Predict NoW validation meshes with SMIRK (single-image FLAME).

Writes predicted_meshes/<subject>/<challenge>/<IMG>.{ply,npy} in millimeters.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import trimesh
from skimage.io import imread
from skimage.transform import estimate_transform, warp
from tqdm import tqdm

SMIRK_ROOT = Path(os.environ.get("SMIRK_ROOT", ""))
if not SMIRK_ROOT:
    raise SystemExit("Set SMIRK_ROOT (see config/paths.env)")
sys.path.insert(0, str(SMIRK_ROOT))

from src.FLAME.FLAME import FLAME  # noqa: E402
from src.smirk_encoder import SmirkEncoder  # noqa: E402


def crop_now(image_rgb: np.ndarray, bbx: dict, scale: float = 1.6, crop_size: int = 224):
    left, right, top, bottom = bbx["left"], bbx["right"], bbx["top"], bbx["bottom"]
    old_size = (right - left + bottom - top) / 2
    center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0])
    size = int(old_size * scale)
    src_pts = np.array(
        [
            [center[0] - size / 2, center[1] - size / 2],
            [center[0] - size / 2, center[1] + size / 2],
            [center[0] + size / 2, center[1] - size / 2],
        ]
    )
    dst_pts = np.array([[0, 0], [0, crop_size - 1], [crop_size - 1, 0]])
    tform = estimate_transform("similarity", src_pts, dst_pts)
    dst = warp(image_rgb / 255.0, tform.inverse, output_shape=(crop_size, crop_size))
    return (dst * 255.0).astype(np.uint8)


def now_seven_from_68(lmk68: np.ndarray) -> np.ndarray:
    """Same 7-landmark selection as MICA/NoW papers (from 68 FAN landmarks)."""
    landmark_51 = lmk68[17:]
    return landmark_51[[19, 22, 25, 28, 16, 31, 37]]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset_root",
        type=Path,
        default=Path(os.environ.get("NOW_DATASET", "data/now-dataset/dataset")),
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path(os.environ.get("NOW_RESULTS", "runs/now")) / "smirk/predicted_meshes",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(os.environ.get("SMIRK_ROOT", ".")) / "pretrained_models/SMIRK_em1.pt",
    )
    parser.add_argument("--image_list", type=Path, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--skip_existing", action="store_true")
    args = parser.parse_args()

    image_list = args.image_list or (args.dataset_root / "imagepathsvalidation.txt")
    pictures = args.dataset_root / "NoW_Dataset/final_release_version/iphone_pictures"
    detected = args.dataset_root / "NoW_Dataset/final_release_version/detected_face"
    paths = [ln.strip() for ln in image_list.read_text().splitlines() if ln.strip()]

    # FLAME() resolves assets relative to CWD
    import os

    os.chdir(SMIRK_ROOT)

    encoder = SmirkEncoder().to(args.device)
    ckpt = torch.load(str(args.checkpoint), map_location=args.device, weights_only=False)
    encoder.load_state_dict(
        {k.replace("smirk_encoder.", ""): v for k, v in ckpt.items() if "smirk_encoder" in k}
    )
    encoder.eval()
    flame = FLAME().to(args.device)
    faces = flame.faces_tensor.cpu().numpy()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    ok = fail = 0

    for rel in tqdm(paths, desc="SMIRK NoW"):
        subject, challenge, filename = rel.split("/")
        stem = Path(filename).stem
        out_dir = args.output_dir / subject / challenge
        out_ply = out_dir / f"{stem}.ply"
        out_npy = out_dir / f"{stem}.npy"
        if args.skip_existing and out_ply.exists() and out_npy.exists():
            ok += 1
            continue

        img_path = pictures / rel
        bbx_path = detected / subject / challenge / f"{stem}.npy"
        if not img_path.exists() or not bbx_path.exists():
            print(f"missing {img_path} or {bbx_path}")
            fail += 1
            continue

        image = imread(str(img_path))[:, :, :3]
        bbx = np.load(bbx_path, allow_pickle=True, encoding="latin1").item()
        crop = crop_now(image, bbx)
        # crop_now already returns RGB uint8 from RGB input
        tensor = (
            torch.from_numpy(crop)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .float()
            .to(args.device)
            / 255.0
        )

        with torch.no_grad():
            outputs = encoder(tensor)
            flame_out = flame.forward(outputs)
            verts = flame_out["vertices"][0].cpu().numpy()
            # Prefer 3D FAN landmarks when available
            if "landmarks_fan_3d" in flame_out:
                lmk68 = flame_out["landmarks_fan_3d"][0].cpu().numpy()
            else:
                lmk68 = flame_out["landmarks_fan"][0].cpu().numpy()
                if lmk68.shape[-1] == 2:
                    # 2D only — lift with corresponding vertices via seletec_3d68
                    _, _, lmk3d = flame.get_landmarks(flame_out["vertices"])
                    lmk68 = lmk3d[0].cpu().numpy()
            landmark_7 = now_seven_from_68(lmk68)

        # SMIRK/FLAME verts are in meters-ish FLAME units; NoW expects mm like MICA (*1000)
        verts_mm = verts * 1000.0
        lmk_mm = landmark_7 * 1000.0

        out_dir.mkdir(parents=True, exist_ok=True)
        trimesh.Trimesh(vertices=verts_mm, faces=faces, process=False).export(str(out_ply))
        np.save(out_npy, lmk_mm.astype(np.float32))
        ok += 1

    print(f"done ok={ok} fail={fail} -> {args.output_dir}")


if __name__ == "__main__":
    main()
