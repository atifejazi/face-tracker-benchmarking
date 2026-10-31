#!/usr/bin/env python3
"""Read-only CPU audit of a trained mesh-bound GaussianAvatar.

Uses the GA environment (torch, roma, plyfile). Optional software renders follow
the CUDA forward equations, but are NOT bit-exact GPU reproductions: operation
ordering and fused multiply-add differ. Never use these images as benchmark scores.
No model, dataset, or training code is modified.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import roma
from PIL import Image
from plyfile import PlyData


def frames_at(path):
    frames = {}
    for split in ("train", "val", "test"):
        p = path / f"transforms_{split}.json"
        if p.exists():
            for f in json.loads(p.read_text())["frames"]:
                frames[f["timestep_index"]] = f
    return frames


def world_to_camera(frame):
    c2w = np.array(frame["transform_matrix"], dtype=np.float64)
    c2w[:3, 1:3] *= -1
    return np.linalg.inv(c2w)


def verify_gauge(mesh_model, legacy, relative):
    from verify_mica_camera_equiv import flame_forward, to_cam
    from utils.graphics_utils import compute_face_orientation
    old, new = frames_at(legacy), frames_at(relative)
    if old.keys() != new.keys():
        raise ValueError("Legacy and relative datasets must contain the same timesteps")
    max_error = 0.
    max_cov_error = 0.
    centers, effective, naive = [], [], []
    for i in sorted(old):
        p0 = dict(np.load(legacy / old[i]["flame_param_path"]))
        p1 = dict(np.load(relative / new[i]["flame_param_path"]))
        w0, w1 = world_to_camera(old[i]), world_to_camera(new[i])
        v0 = flame_forward(mesh_model, p0, torch.device("cpu"))
        v1 = flame_forward(mesh_model, p1, torch.device("cpu"))
        max_error = max(max_error, float(np.linalg.norm(to_cam(v0, w0)-to_cam(v1, w1), axis=1).max()))
        covs = []
        # A normal-sized anisotropic Gaussian on every face, including teeth.
        for verts, w in ((v0, w0), (v1, w1)):
            o, s = compute_face_orientation(torch.tensor(verts, dtype=torch.float32), mesh_model.faces, True)
            q = roma.rotvec_to_unitquat(torch.tensor([[.3, .6, -.1]])).expand(len(o), -1)
            fq = torch.nn.functional.normalize(roma.rotmat_to_unitquat(o), dim=-1)
            r = roma.unitquat_to_rotmat(roma.quat_product(fq, q)).numpy()
            a = (w[:3, :3] @ r) * (s.numpy()*np.array([.4, .7, 1.1]))[:, None, :]
            covs.append(a @ a.transpose(0, 2, 1))
        max_cov_error = max(max_cov_error, float(np.abs(covs[0]-covs[1]).max()))
        rr = roma.rotvec_to_rotmat(torch.tensor(p1["rotation"]).reshape(1, 3)).numpy()[0].astype(float)
        t = p1["translation"].reshape(3).astype(float)
        beta = torch.tensor(np.concatenate([p1["shape"].reshape(-1), p1["expr"].reshape(-1)]))
        shaped = mesh_model.v_template + torch.einsum("d,vcd->vc", beta, mesh_model.shapedirs)
        shaped += torch.tensor(p1["static_offset"]).reshape(-1, 3)
        joint = (mesh_model.J_regressor @ shaped)[0].numpy().astype(float)
        c = np.linalg.inv(w1)[:3, 3]
        centers.append(np.linalg.inv(w0)[:3, 3])
        naive.append(rr.T @ (c-t))
        effective.append(rr.T @ (c-t-(np.eye(3)-rr)@joint))
    def radius(p):
        p = np.array(p)
        return float(1.1*np.linalg.norm(p-p.mean(0), axis=1).max())
    return dict(frames=len(old), vertices_per_frame=len(v0),
                max_vertex_error_m=max_error, max_camera_covariance_error_m2=max_cov_error,
                radius_legacy_m=radius(centers), radius_without_pivot_m=radius(naive),
                radius_with_pivot_m=radius(effective),
                max_effective_camera_center_error_m=float(np.linalg.norm(np.array(effective)-centers, axis=1).max()))


def project_splats(centers, scales, rotations, w2c, fx, fy, width, height,
                   dtype=np.float32, factor_first=False):
    """CUDA-style world covariance, or equivalent projected-factor control."""
    cam = (centers @ w2c[:3, :3].T + w2c[:3, 3]).astype(dtype)
    z = cam[:, 2]
    jac = np.zeros((len(z), 2, 3), dtype=dtype)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        jac[:, 0, 0], jac[:, 1, 1] = fx / z, fy / z
        jac[:, 0, 2] = -fx * np.clip(cam[:, 0] / z, -1.3 * width / (2 * fx), 1.3 * width / (2 * fx)) / z
        jac[:, 1, 2] = -fy * np.clip(cam[:, 1] / z, -1.3 * height / (2 * fy), 1.3 * height / (2 * fy)) / z
        factor = rotations.astype(dtype) * scales.astype(dtype)[:, None, :]
        jw = jac @ w2c[:3, :3].astype(dtype)
        if factor_first:
            projected = jw @ factor
            cov = projected @ projected.transpose(0, 2, 1)
        else:
            world_cov = factor @ factor.transpose(0, 2, 1)
            cov = jw @ world_cov @ jw.transpose(0, 2, 1)
        cov += np.eye(2, dtype=dtype)[None] * .3
        a, b, c = cov[:, 0, 0], cov[:, 0, 1], cov[:, 1, 1]
        det = a * c - b * b
        conic = np.stack([c, -b, a], axis=1) / det[:, None]
        mid = (a + c) * .5
        radius = np.ceil(3 * np.sqrt(np.maximum(0, mid + np.sqrt(np.maximum(.1, mid * mid - det)))))
        xy = cam[:, :2] / (z[:, None] + 1e-7) * np.array([fx, fy], dtype=dtype)
        xy += np.array([(width - 1) / 2, (height - 1) / 2], dtype=dtype)
    return dict(z=z, xy=xy, det=det, conic=conic, radius=radius)


def render_cpu(proj, colors, opacity, width, height, keep):
    """Small diagnostic SH0 renderer; center-depth sorting, white background."""
    dtype = proj["xy"].dtype
    accum = np.zeros((height, width, 3), dtype=dtype)
    trans = np.ones((height, width), dtype=dtype)
    for i in np.argsort(proj["z"]):
        if not keep[i] or proj["z"][i] <= .2 or proj["det"][i] == 0:
            continue
        r = proj["radius"][i]
        if not np.isfinite(r):
            continue
        x, y = proj["xy"][i]
        x0, x1 = max(0, int((x-r)/16)*16), min(width, int((x+r+15)/16)*16)
        y0, y1 = max(0, int((y-r)/16)*16), min(height, int((y+r+15)/16)*16)
        if x1 <= x0 or y1 <= y0:
            continue
        dx = x - np.arange(x0, x1, dtype=dtype)[None, :]
        dy = y - np.arange(y0, y1, dtype=dtype)[:, None]
        q = proj["conic"][i]
        power = -.5 * (q[0]*dx*dx + q[2]*dy*dy) - q[1]*dx*dy
        alpha = np.minimum(.99, opacity[i] * np.exp(np.minimum(power, 0)))
        t = trans[y0:y1, x0:x1]
        valid = (power <= 0) & (alpha >= 1/255) & (t >= .0001) & (t*(1-alpha) >= .0001)
        alpha = np.where(valid, alpha, 0)
        accum[y0:y1, x0:x1] += (alpha*t)[:, :, None] * colors[i]
        t *= 1-alpha
    return np.clip(accum + trans[:, :, None], 0, 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ga-root", required=True, type=Path)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--legacy", type=Path, help="Also verify every exported frame against this legacy dataset")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--iteration", type=int, default=25000)
    parser.add_argument("--frame", type=int, default=89, help="Dataset timestep, not shuffled render index")
    parser.add_argument("--resolution", type=int, default=2, help="Image downsample factor")
    parser.add_argument("--render-cpu", action="store_true")
    parser.add_argument("--ablate-above-m", type=float, default=1., help="Diagnostic only, not a recommended training threshold")
    args = parser.parse_args()
    for key in ("ga_root", "dataset", "model", "output"):
        setattr(args, key, getattr(args, key).resolve())
    if args.legacy is not None:
        args.legacy = args.legacy.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    os.chdir(args.ga_root)  # FlameHead resolves its assets against the GA root.
    sys.path.insert(0, str(args.ga_root))
    from flame_model.flame import FlameHead
    from utils.graphics_utils import compute_face_orientation

    torch.set_num_threads(2)
    mesh_model = FlameHead(300, 100).cpu().eval()
    point_dir = args.model / "point_cloud" / f"iteration_{args.iteration}"
    params = dict(np.load(point_dir / "flame_param.npz"))
    v = PlyData.read(str(point_dir / "point_cloud.ply"))["vertex"].data
    stack = lambda names: np.stack([v[k] for k in names], axis=1)
    local = stack(["x", "y", "z"])
    local_scales = np.exp(stack([f"scale_{j}" for j in range(3)]))
    binding = v["binding_0"].astype(int)
    def t(k):
        return torch.from_numpy(params[k].astype(np.float32))
    with torch.no_grad():
        verts = mesh_model(t("shape")[None], t("expr")[[args.frame]],
                           t("rotation")[[args.frame]], t("neck_pose")[[args.frame]],
                           t("jaw_pose")[[args.frame]], t("eyes_pose")[[args.frame]],
                           t("translation")[[args.frame]], return_landmarks=False,
                           static_offset=t("static_offset"))[0]
        orientation, face_scale = compute_face_orientation(verts, mesh_model.faces, True)
        center = verts[mesh_model.faces].mean(1).numpy()[binding]
        centers = center + np.einsum("nij,nj->ni", orientation.numpy()[binding], local) * face_scale.numpy()[binding]
        scales = local_scales * face_scale.numpy()[binding]
        face_q = torch.nn.functional.normalize(roma.rotmat_to_unitquat(orientation)[torch.tensor(binding)], dim=-1)
        local_q = torch.nn.functional.normalize(torch.tensor(stack([f"rot_{j}" for j in range(4)])), dim=-1)
        rotations = roma.unitquat_to_rotmat(roma.quat_product(face_q, roma.quat_wxyz_to_xyzw(local_q))).numpy()
    frame = frames_at(args.dataset)[args.frame]
    w2c = world_to_camera(frame)
    width, height = round(frame["w"]/args.resolution), round(frame["h"]/args.resolution)
    fx, fy = frame["fl_x"]*width/frame["w"], frame["fl_y"]*height/frame["h"]
    proj_args = (centers, scales, rotations, w2c, fx, fy, width, height)
    projections = {
        "world_float32": project_splats(*proj_args),
        "world_float64": project_splats(*proj_args, dtype=np.float64),
        "factor_float32": project_splats(*proj_args, factor_first=True),
    }
    opacity = 1 / (1 + np.exp(-v["opacity"]))
    colors = np.maximum(0, .5 + .28209479177387814 * stack([f"f_dc_{j}" for j in range(3)]))
    maxima = scales.max(1)
    report = dict(model=str(args.model), dataset=str(args.dataset), frame=args.frame,
                  n_splats=len(v), max_world_scale_m=float(maxima.max()),
                  splats_above_ablation_threshold=int((maxima > args.ablate_above_m).sum()),
                  ablation_threshold_m=args.ablate_above_m,
                  local_scale_quantiles=np.quantile(local_scales.max(1), [.5, .99, 1]).tolist(),
                  world_scale_quantiles_m=np.quantile(maxima, [.5, .99, 1]).tolist())
    report["projection_controls"] = {
        k: dict(visible_nonpositive_determinants=int(((p["det"] <= 0) & (p["z"] > .2)).sum()))
        for k, p in projections.items()
    }
    bad = set(np.flatnonzero((projections["world_float32"]["det"] <= 0) & (projections["world_float32"]["z"] > .2)).tolist())
    bad.update(np.argsort(maxima)[-12:].tolist())
    report["suspect_splats"] = [dict(index=i, face=int(binding[i]),
        scales_m=scales[i].tolist(), opacity=float(opacity[i]), rgb=colors[i].tolist(),
        determinants={k: float(p["det"][i]) for k, p in projections.items()}) for i in sorted(bad)]
    report["checkpoint_history"] = []
    for path in sorted(args.model.glob("chkpnt*.pth"), key=lambda p: int(p.stem[6:])):
        # These are the user's local training checkpoints, not downloaded data.
        ck, iteration = torch.load(path, map_location="cpu", weights_only=False)
        report["checkpoint_history"].append(dict(iteration=iteration, n_splats=len(ck[1]),
            max_local_scale=float(ck[4].detach().exp().max()), max_local_offset=float(ck[1].detach().norm(dim=1).max()),
            spatial_lr_scale=float(ck[13]), optimizer_lrs={g["name"]: g["lr"] for g in ck[12]["param_groups"]}))
    if args.legacy is not None:
        with torch.no_grad():
            report["all_frame_gauge_check"] = verify_gauge(mesh_model, args.legacy, args.dataset)
    if args.render_cpu:
        if any(k.startswith("f_rest_") for k in v.dtype.names):
            raise ValueError("CPU image controls currently support only SH degree zero")
        report["cpu_render_note"] = "Diagnostic approximation, not pixel-exact CUDA output or benchmark scores"
        for name, proj in projections.items():
            image = render_cpu(proj, colors, opacity, width, height, np.ones(len(v), bool))
            Image.fromarray((image*255+.5).astype(np.uint8)).save(args.output / f"{name}.png")
        image = render_cpu(projections["world_float32"], colors, opacity, width, height, maxima <= args.ablate_above_m)
        Image.fromarray((image*255+.5).astype(np.uint8)).save(args.output / "ablate_extreme_splats.png")
    (args.output / "splat_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("n_splats", "max_world_scale_m", "projection_controls")}, indent=2))
    print(f"Evidence written to {args.output}")


if __name__ == "__main__":
    main()
