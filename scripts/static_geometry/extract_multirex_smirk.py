#!/usr/bin/env python3
"""Extract MultiREX-format FLAME parameters from videos using SMIRK."""

import argparse
import os
import pickle
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
SMIRK_ROOT = Path(os.environ.get("SMIRK_ROOT", SCRIPT_DIR.parent.parent / "other" / "smirk"))
MULTIREX_ROOT = Path(os.environ.get("MULTIREX_ROOT", "data/multirex/ubisoft-laforge-multirex"))

sys.path.insert(0, str(SMIRK_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

from multirex_crop import crop_frame_with_bbox  # noqa: E402
from src.smirk_encoder import SmirkEncoder  # noqa: E402

N_EXP_COMPONENTS = 100
CROP_SCALE = 1.4


def resolve_device(requested):
    if requested != "cuda" or not torch.cuda.is_available():
        return "cpu"
    try:
        torch.zeros(1, device="cuda").item()
    except Exception:
        return "cpu"
    return "cuda"


def outputs_to_multirex(outputs_list):
    expressions = np.stack([o["expression_params"].squeeze(0) for o in outputs_list], axis=0)
    pose_params = np.stack([o["pose_params"].squeeze(0) for o in outputs_list], axis=0)
    jaw_params = np.stack([o["jaw_params"].squeeze(0) for o in outputs_list], axis=0)
    eyelids = np.stack([o["eyelid_params"].squeeze(0) for o in outputs_list], axis=0)

    if expressions.shape[1] < N_EXP_COMPONENTS:
        pad = np.zeros((expressions.shape[0], N_EXP_COMPONENTS - expressions.shape[1]), dtype=np.float32)
        expressions = np.concatenate([expressions, pad], axis=1)

    poses = np.concatenate([pose_params, jaw_params], axis=1).astype(np.float32)
    return {
        "expressions": expressions.astype(np.float32),
        "poses": poses,
        "eyelids": eyelids.astype(np.float32),
    }


def process_video(video_path, bbox_sequence, encoder, device, max_frames=None):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    outputs_list = []
    frame_idx = 0
    n_bboxes = len(bbox_sequence)

    with torch.no_grad():
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if max_frames is not None and frame_idx >= max_frames:
                break
            if frame_idx >= n_bboxes:
                break

            bbox = bbox_sequence[frame_idx]
            cropped, _ = crop_frame_with_bbox(
                frame, bbox, scale=CROP_SCALE, image_size=224, deca_style=False
            )
            cropped_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
            cropped_rgb = cv2.resize(cropped_rgb, (224, 224))
            image = torch.tensor(cropped_rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            image = image.to(device)

            outputs = encoder(image)
            outputs_list.append({k: v.detach().cpu() for k, v in outputs.items()})
            frame_idx += 1

    cap.release()
    if frame_idx == 0:
        raise RuntimeError(f"No frames processed for {video_path}")
    if frame_idx != n_bboxes and max_frames is None:
        print(f"Warning: processed {frame_idx} frames but bbox sequence has {n_bboxes}")
    return outputs_to_multirex(outputs_list)


def main():
    parser = argparse.ArgumentParser(description="Extract SMIRK FLAME params for MultiREX evaluation")
    parser.add_argument(
        "--videos_dir",
        type=str,
        default=str(MULTIREX_ROOT / "videos_gamma_corrected"),
    )
    parser.add_argument(
        "--bbox_pickle",
        type=str,
        default=str(MULTIREX_ROOT / "assets" / "video_bbox_mapping.pickle"),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(Path(os.environ.get("MULTIREX_RESULTS", "runs/multirex_ranking_s8")) / "params" / "smirk_full"),
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(SMIRK_ROOT / "pretrained_models" / "SMIRK_em1.pt"),
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max_frames", type=int, default=None, help="Limit frames (smoke test)")
    parser.add_argument("--video", type=str, default=None, help="Process a single video filename")
    parser.add_argument(
        "--subject_id",
        type=str,
        default=None,
        help="Process every benchmark video for one subject ID",
    )
    parser.add_argument("--skip_existing", action="store_true", help="Skip videos with existing .npz output")
    args = parser.parse_args()
    if args.video and args.subject_id:
        parser.error("--video and --subject_id cannot be used together")

    device = resolve_device(args.device)
    print(f"Using device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.bbox_pickle, "rb") as f:
        bbox_mapping = pickle.load(f)

    encoder = SmirkEncoder().to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    checkpoint_encoder = {
        k.replace("smirk_encoder.", ""): v
        for k, v in checkpoint.items()
        if "smirk_encoder" in k
    }
    encoder.load_state_dict(checkpoint_encoder)
    encoder.eval()

    videos_dir = Path(args.videos_dir)
    video_paths = sorted(videos_dir.glob("*.mp4"))
    if args.video:
        video_paths = [videos_dir / args.video]
    elif args.subject_id:
        subject_token = f"--{args.subject_id}--"
        video_paths = [path for path in video_paths if subject_token in path.name]
        if not video_paths:
            raise FileNotFoundError(
                f"No videos found for subject {args.subject_id} in {videos_dir}"
            )

    for video_path in video_paths:
        video_name = video_path.name
        if video_name not in bbox_mapping:
            print(f"Skipping {video_name}: no bbox sequence found")
            continue

        out_path = Path(args.output_dir) / f"{video_path.stem}.npz"
        if args.skip_existing and out_path.exists():
            print(f"Skipping {video_name} (already exists)")
            continue

        print(f"Processing {video_name}...")
        flame_params = process_video(
            video_path,
            bbox_mapping[video_name],
            encoder,
            device,
            max_frames=args.max_frames,
        )
        np.savez(out_path, **flame_params)
        print(
            f"  Saved {out_path.name}: "
            f"expressions {flame_params['expressions'].shape}, poses {flame_params['poses'].shape}"
        )


if __name__ == "__main__":
    main()
