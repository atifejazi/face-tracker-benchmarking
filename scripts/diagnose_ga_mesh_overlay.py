#!/usr/bin/env python3
"""Inspect GA's full posed LBS mesh; optionally compare SMIRK's native projection.

Green wire/labels are the exported GA mesh. With --track-params, the second
panel is the same FLAME geometry under SMIRK's native weak projection + inverse
crop (eyelid offsets excluded in both). This tests export fidelity, not accuracy
against ground truth. Exit 1 for an inverted/hidden face or excessive reference
error; never modifies the dataset. Works on CPU without the CUDA rasterizer.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from ga_mesh_utils import (decode_flame, draw_landmarks, draw_wire, load_flame,
                           load_frames_by_timestep, project_vertices, unique_edges)


def smirk_native_pixels(vertices, weak_cam, image_to_crop):
    """Renderer.forward/batch_orth_proj followed by SMIRK's crop pixel mapping."""
    scale, tx, ty = np.asarray(weak_cam, dtype=np.float64)
    crop = np.column_stack((112*(1+scale*(vertices[:, 0]+tx)),
                            112*(1-scale*(vertices[:, 1]+ty)), np.ones(len(vertices))))
    image = crop @ np.linalg.inv(image_to_crop).T
    return image[:, :2]/image[:, 2:3]


def orientation(landmark_uv, landmark_depth):
    # Standard FLAME 68 landmarks: chin=8, nose bridge/brow=27. Avoid guessed
    # vertex indices and rigid rotations about the origin (FLAME rotates at J0).
    if not np.isfinite(landmark_uv[:68]).all() or np.any(landmark_depth[:68] <= 0.01):
        return 'NOT_VISIBLE'
    return 'UPRIGHT' if landmark_uv[8, 1] > landmark_uv[27, 1] else 'UPSIDE_DOWN'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--frames', default='0,1,2', help='Comma-separated timesteps, or all')
    parser.add_argument('--ga-root', type=Path, default=Path(os.environ.get('GA_ROOT', Path(__file__).resolve().parents[2]/'GaussianAvatars')))
    parser.add_argument('--flame_pkl', type=Path, default=None, help='Optional GA FLAME model override')
    parser.add_argument('--out_dir', type=Path, required=True)
    parser.add_argument('--track-params', type=Path, help='Original SMIRK track_params.pt')
    parser.add_argument('--max-mean-error', type=float, default=8., help='Reference error gate in dataset pixels')
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    torch.set_num_threads(2)
    device = torch.device(args.device)
    flame = load_flame(args.ga_root, device, args.flame_pkl)
    edges = unique_edges(flame.faces.cpu().numpy())[::3]
    frames = load_frames_by_timestep(args.dataset)
    selected = sorted(frames) if args.frames == 'all' else [int(x) for x in args.frames.split(',')]
    if not selected or any(t not in frames for t in selected):
        raise SystemExit('No frames selected or requested timesteps missing from transforms')
    params = None
    if args.track_params:
        params = torch.load(args.track_params, map_location='cpu', weights_only=False)
        params = {k: v.numpy() if torch.is_tensor(v) else v for k, v in params.items()}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    report = []
    for t in selected:
        frame = frames[t]
        with np.load(args.dataset/frame['flame_param_path']) as data:
            verts, landmarks = decode_flame(flame, data, device)
        uv, depth = project_vertices(verts, frame)
        l_uv, l_depth = project_vertices(landmarks, frame)
        status = orientation(l_uv, l_depth)
        row = {'timestep': t, 'status': status, 'method': 'GA FlameHead LBS + GA rasterizer projection',
               'chin_xy': l_uv[8].tolist(), 'brow_xy': l_uv[27].tolist(),
               'minimum_face_depth': float(l_depth[:68].min())}
        base = Image.open(args.dataset/frame['file_path']).convert('RGB')
        if base.size != (int(frame['w']), int(frame['h'])):
            raise ValueError(f'Image/transforms size mismatch at timestep {t}')
        panel = draw_landmarks(draw_wire(base, uv, depth, edges), l_uv)
        panels = [panel]
        titles = [f'GA LBS mesh: {status} | t={t}']
        if params is not None:
            if params['frame_names'][t] != Path(frame['file_path']).stem:
                raise ValueError(f'SMIRK/canonical frame mismatch at timestep {t}')
            source = dict(shape=params['shape'].mean(axis=0), expr=params['exp'][t],
                          rotation=params['pose'][t], neck_pose=np.zeros(3),
                          jaw_pose=params['jaw'][t], eyes_pose=np.zeros(6), translation=np.zeros(3))
            native_v, native_l = decode_flame(flame, source, device)
            native_uv = smirk_native_pixels(native_v, params['cam'][t], params['image_to_crop'][t])
            native_luv = smirk_native_pixels(native_l, params['cam'][t], params['image_to_crop'][t])
            errors = np.linalg.norm(l_uv[:68]-native_luv[:68], axis=1)
            row['native_reference_mean_error_px'] = float(errors.mean())
            row['native_reference_max_error_px'] = float(errors.max())
            row['reference_pass'] = bool(np.isfinite(errors).all() and errors.mean() <= args.max_mean_error)
            panels.append(draw_landmarks(draw_wire(base, native_uv, np.ones(len(native_v)), edges), native_luv))
            titles.append(f'SMIRK native reference | mean difference {errors.mean():.2f}px')
        out = Image.new('RGB', (base.width*len(panels), base.height+25), (20, 20, 20))
        draw = ImageDraw.Draw(out)
        for i, (im, title) in enumerate(zip(panels, titles)):
            out.paste(im, (i*base.width, 25))
            draw.text((i*base.width+5, 6), title, fill='white')
        out.save(args.out_dir/f'overlay_{t:05d}.png')
        report.append(row)
        print(f'{t:05d}: {status}' + (f' native mean={errors.mean():.3f}px' if params is not None else ''))
    (args.out_dir/'mesh_upright_report.json').write_text(json.dumps(report, indent=2))
    bad = sum(r['status'] != 'UPRIGHT' or not r.get('reference_pass', True) for r in report)
    print(f'{len(report)} frames checked; {bad} failed. Wrote {args.out_dir}')
    raise SystemExit(1 if bad else 0)


if __name__ == '__main__':
    main()
