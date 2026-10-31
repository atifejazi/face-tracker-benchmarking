"""Shared LBS/projection helpers for inspecting the mesh actually used by GA."""
from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw


def load_flame(ga_root, device, flame_pkl=None):
    ga_root = Path(ga_root).resolve()
    sys.path.insert(0, str(ga_root))
    from flame_model.flame import FlameHead
    # FlameHead/FlameMask load additional assets relative to GA's cwd.
    flame_pkl = Path(flame_pkl).resolve() if flame_pkl is not None else None
    previous = Path.cwd()
    try:
        os.chdir(ga_root)
        kwargs = {} if flame_pkl is None else {"flame_model_path": str(Path(flame_pkl).resolve())}
        return FlameHead(300, 100, **kwargs).to(device).eval()
    finally:
        os.chdir(previous)


def load_frames_by_timestep(dataset):
    frames = {}
    for split in ("train", "val", "test"):
        path = Path(dataset)/f"transforms_{split}.json"
        if not path.exists():
            continue
        db = json.loads(path.read_text())
        shared = {k: v for k, v in db.items() if k != "frames"}
        for frame in db["frames"]:
            frames[int(frame["timestep_index"])] = {**shared, **frame}
    return frames


def ga_w2c_from_transform_matrix(c2w_json):
    c2w = np.asarray(c2w_json, dtype=np.float64).copy()
    c2w[:3, 1:3] *= -1
    return np.linalg.inv(c2w)


def ga_intrinsics(frame, size=None):
    """Match GA's symmetric FOV projection, including its pixel-center mapping.

    The reader ignores cx/cy/fl_y and derives fy from horizontal FOV and aspect.
    CUDA rasterizer ndc2Pix(v,S)=((v+1)*S-1)/2 puts the center at (S-1)/2.
    """
    w, h = size or (int(frame['w']), int(frame['h']))
    source_w, source_h = float(frame['w']), float(frame['h'])
    focal = source_w/(2*math.tan(float(frame['camera_angle_x'])/2))
    fx, fy = focal*w/source_w, focal*h/source_h
    return np.array([[fx, 0, (w-1)/2], [0, fy, (h-1)/2], [0, 0, 1.]])


def project_vertices(vertices, frame, size=None):
    w2c = ga_w2c_from_transform_matrix(frame['transform_matrix'])
    camera = np.asarray(vertices) @ w2c[:3, :3].T + w2c[:3, 3]
    with np.errstate(divide='ignore', invalid='ignore'):
        uv = (camera @ ga_intrinsics(frame, size).T)[:, :2]/camera[:, 2:3]
    return uv, camera[:, 2]


def decode_flame(flame, data, device):
    def tensor(key, count):
        a = np.asarray(data[key], dtype=np.float32).reshape(1, -1)
        if a.shape[1] > count:
            raise ValueError(f'{key}: {a.shape[1]} coefficients exceed {count}')
        return torch.nn.functional.pad(torch.from_numpy(a).to(device), (0, count-a.shape[1]))
    static = None
    if 'static_offset' in data:
        a = np.asarray(data['static_offset'], dtype=np.float32).reshape(1, -1, 3)
        n = len(flame.v_template)
        if a.shape[1] > n:
            raise ValueError('static_offset has more vertices than GA FLAME')
        static = torch.nn.functional.pad(torch.from_numpy(a).to(device), (0, 0, 0, n-a.shape[1]))
    with torch.no_grad():
        verts, landmarks = flame(
            tensor('shape', 300), tensor('expr', 100), tensor('rotation', 3),
            tensor('neck_pose', 3), tensor('jaw_pose', 3), tensor('eyes_pose', 6),
            tensor('translation', 3), zero_centered_at_root_node=False,
            return_landmarks=True, static_offset=static)
    return verts[0].cpu().numpy(), landmarks[0].cpu().numpy()


def unique_edges(faces):
    return np.unique(np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]],
                                            faces[:, [2, 0]]]), axis=1), axis=0)


def draw_wire(base, uv, depth, edges, color=(0, 220, 0), alpha=0.55):
    wire = base.convert('RGB').copy()
    draw = ImageDraw.Draw(wire)
    w, h = base.size
    for i, j in edges:
        a, b = uv[i], uv[j]
        if min(depth[i], depth[j]) <= 0.01 or not np.isfinite([a, b]).all():
            continue
        if ((a[0] < -50 and b[0] < -50) or (a[0] > w+50 and b[0] > w+50)
                or (a[1] < -50 and b[1] < -50) or (a[1] > h+50 and b[1] > h+50)):
            continue
        draw.line([tuple(a), tuple(b)], fill=color, width=1)
    return Image.blend(base.convert('RGB'), wire, alpha)


def draw_landmarks(image, uv):
    draw = ImageDraw.Draw(image)
    for index, label in ((8, 'chin'), (27, 'brow'), (30, 'nose'), (36, 'eye'), (45, 'eye')):
        x, y = uv[index]
        if np.isfinite([x, y]).all() and 0 <= x < image.width and 0 <= y < image.height:
            draw.ellipse((x-3, y-3, x+3, y+3), fill=(255, 70, 40))
            draw.text((x+4, y), label, fill=(255, 70, 40))
    return image
