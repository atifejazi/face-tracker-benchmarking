#!/usr/bin/env python3
"""Predict NoW validation meshes with MICA (single-image shape).

Writes predicted_meshes/<subject>/<challenge>/<IMG>.{ply,npy} with vertices
and 7 landmarks in millimeters, matching now_evaluation layout.
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
from insightface.app import FaceAnalysis
from insightface.app.common import Face
from insightface.utils import face_align
from skimage.io import imread
from skimage.transform import estimate_transform, warp
from tqdm import tqdm

import os
from pathlib import Path

MICA_ROOT = Path(os.environ.get("MICA_ROOT", ""))
if not MICA_ROOT:
    raise SystemExit("Set MICA_ROOT (see config/paths.env)")
sys.path.insert(0, str(MICA_ROOT))

from configs.config import get_cfg_defaults  # noqa: E402
from utils import util  # noqa: E402

INPUT_MEAN = 127.5
INPUT_STD = 127.5


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
    return dst.astype(np.float32)


def arcface_blob_from_crop_bgr(crop_bgr: np.ndarray, app: FaceAnalysis) -> np.ndarray:
    """Match MICA tester.process_image: detect on crop, ArcFace-norm, blob."""
    bboxes, kpss = app.det_model.detect(crop_bgr, max_num=0, metric="default")
    if bboxes.shape[0] < 1:
        aimg = cv2.resize(crop_bgr, (112, 112))
        return cv2.dnn.blobFromImages(
            [aimg],
            1.0 / INPUT_STD,
            (112, 112),
            (INPUT_MEAN, INPUT_MEAN, INPUT_MEAN),
            swapRB=True,
        )[0]
    i = 0
    face = Face(bbox=bboxes[i, 0:4], kps=kpss[i] if kpss is not None else None, det_score=bboxes[i, 4])
    aimg = face_align.norm_crop(crop_bgr, landmark=face.kps)
    return cv2.dnn.blobFromImages(
        [aimg],
        1.0 / INPUT_STD,
        (112, 112),
        (INPUT_MEAN, INPUT_MEAN, INPUT_MEAN),
        swapRB=True,
    )[0]


def load_mica(checkpoint: Path, device: str):
    cfg = get_cfg_defaults()
    cfg.model.testing = True
    mica = util.find_model_using_name(model_dir="micalib.models", model_name=cfg.model.name)(
        cfg, device
    )
    ckpt = torch.load(str(checkpoint), map_location=device, weights_only=False)
    if "arcface" in ckpt:
        mica.arcface.load_state_dict(ckpt["arcface"])
    if "flameModel" in ckpt:
        mica.flameModel.load_state_dict(ckpt["flameModel"])
    mica.eval()
    faces = mica.flameModel.generator.faces_tensor.cpu().numpy()
    return mica, faces


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
        default=Path(os.environ.get("NOW_RESULTS", "runs/now")) / "mica/predicted_meshes",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(os.environ.get("MICA_ROOT", ".")) / "data/pretrained/mica.tar",
    )
    parser.add_argument("--image_list", type=Path, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--skip_existing", action="store_true")
    args = parser.parse_args()

    os.chdir(MICA_ROOT)
    image_list = args.image_list or (args.dataset_root / "imagepathsvalidation.txt")
    pictures = args.dataset_root / "NoW_Dataset/final_release_version/iphone_pictures"
    detected = args.dataset_root / "NoW_Dataset/final_release_version/detected_face"

    paths = [ln.strip() for ln in image_list.read_text().splitlines() if ln.strip()]
    mica, faces = load_mica(args.checkpoint, args.device)

    app = FaceAnalysis(name="antelopev2", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(224, 224))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    ok = fail = 0

    for rel in tqdm(paths, desc="MICA NoW"):
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
        crop_bgr = cv2.cvtColor((crop * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        blob = arcface_blob_from_crop_bgr(crop_bgr, app)

        image_t = torch.from_numpy(crop.transpose(2, 0, 1)).float()[None].to(args.device)
        arcface = torch.from_numpy(np.asarray(blob)).float()[None].to(args.device)

        with torch.no_grad():
            codedict = mica.encode(image_t, arcface)
            opdict = mica.decode(codedict)
            meshes = opdict["pred_canonical_shape_vertices"]
            lmk = mica.flame.compute_landmarks(meshes)
            mesh = meshes[0]
            landmark_51 = lmk[0, 17:]
            landmark_7 = landmark_51[[19, 22, 25, 28, 16, 31, 37]].cpu().numpy() * 1000.0

        out_dir.mkdir(parents=True, exist_ok=True)
        verts = (mesh.cpu().numpy() * 1000.0).astype(np.float64)
        trimesh.Trimesh(vertices=verts, faces=faces, process=False).export(str(out_ply))
        np.save(out_npy, landmark_7.astype(np.float32))
        ok += 1

    print(f"done ok={ok} fail={fail} -> {args.output_dir}")


if __name__ == "__main__":
    main()
