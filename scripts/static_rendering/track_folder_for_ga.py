#!/usr/bin/env python3
"""Run SMIRK on an ordered image folder and save parameters for GA export.

Requires SMIRK on PYTHONPATH (run with: cd "$SMIRK_ROOT" and PYTHONPATH=.
or PYTHONPATH="$SMIRK_ROOT"). See README section E2.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from skimage.transform import estimate_transform, warp

from src.smirk_encoder import SmirkEncoder
from utils.mediapipe_utils import run_mediapipe


def crop_face(
    frame: np.ndarray, landmarks: np.ndarray, scale: float = 1.4
) -> tuple[np.ndarray, np.ndarray]:
    left, right = landmarks[:, 0].min(), landmarks[:, 0].max()
    top, bottom = landmarks[:, 1].min(), landmarks[:, 1].max()
    old_size = (right - left + bottom - top) / 2
    center = np.array([(right + left) / 2, (bottom + top) / 2])
    size = int(old_size * scale)
    src = np.array(
        [
            [center[0] - size / 2, center[1] - size / 2],
            [center[0] - size / 2, center[1] + size / 2],
            [center[0] + size / 2, center[1] - size / 2],
        ]
    )
    dst = np.array([[0, 0], [0, 223], [223, 0]])
    transform = estimate_transform("similarity", src, dst)
    crop = warp(
        frame,
        transform.inverse,
        output_shape=(224, 224),
        preserve_range=True,
    ).astype(np.uint8)
    return crop, transform.params.astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=Path("pretrained_models/SMIRK_em1.pt"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    image_paths = sorted(
        p for p in args.image_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}
    )
    if not image_paths:
        raise FileNotFoundError(f"No images in {args.image_dir}")

    encoder = SmirkEncoder().to(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    state = {
        key.replace("smirk_encoder.", ""): value
        for key, value in checkpoint.items()
        if "smirk_encoder" in key
    }
    encoder.load_state_dict(state)
    encoder.eval()

    collected = {
        "shape": [],
        "exp": [],
        "pose": [],
        "jaw": [],
        "eyelids": [],
        "cam": [],
        "image_to_crop": [],
    }

    with torch.no_grad():
        for index, path in enumerate(image_paths):
            frame = cv2.imread(str(path))
            landmarks = run_mediapipe(frame)
            if landmarks is None:
                raise RuntimeError(f"MediaPipe failed on frame {path.name}")
            crop, image_to_crop = crop_face(frame, landmarks[..., :2])
            crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            tensor = (
                torch.from_numpy(crop)
                .permute(2, 0, 1)
                .unsqueeze(0)
                .float()
                .to(args.device)
                / 255.0
            )
            output = encoder(tensor)
            collected["shape"].append(output["shape_params"][0].cpu())
            collected["exp"].append(output["expression_params"][0].cpu())
            collected["pose"].append(output["pose_params"][0].cpu())
            collected["jaw"].append(output["jaw_params"][0].cpu())
            collected["eyelids"].append(output["eyelid_params"][0].cpu())
            collected["cam"].append(output["cam"][0].cpu())
            collected["image_to_crop"].append(torch.from_numpy(image_to_crop))
            print(f"\rSMIRK {index + 1}/{len(image_paths)}", end="", flush=True)

    result = {key: torch.stack(value) for key, value in collected.items()}
    # Identity must be constant over the sequence.
    result["shape"] = result["shape"].mean(dim=0, keepdim=True).expand(len(image_paths), -1).clone()
    result["frame_names"] = [path.stem for path in image_paths]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, args.output)
    print(f"\nSaved {len(image_paths)} frames to {args.output}")


if __name__ == "__main__":
    main()
