#!/usr/bin/env python3
"""Compute PSNR and LPIPS between paired pred/gt image folders."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def load_rgb(path: Path) -> torch.Tensor:
    """Return float tensor CHW in [0, 1]."""
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img).astype(np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)


def psnr(pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor | None = None) -> float:
    if mask is not None:
        # mask: 1xHxW
        m = mask.expand_as(pred)
        mse = ((pred - gt) ** 2 * m).sum() / m.sum().clamp_min(1.0)
    else:
        mse = F.mse_loss(pred, gt)
    mse = float(mse.item())
    if mse <= 1e-10:
        return 99.0
    return -10.0 * np.log10(mse)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_dir", type=Path, required=True)
    parser.add_argument("--gt_dir", type=Path, required=True)
    parser.add_argument("--mask_dir", type=Path, default=None, help="Optional alpha/fg masks")
    parser.add_argument("--name", type=str, default="run")
    parser.add_argument("--out_json", "--output_json", type=Path, default=None, dest="out_json")
    parser.add_argument("--out_csv", type=Path, default=None)
    parser.add_argument("--lpips_net", type=str, default="alex", choices=["alex", "vgg", "squeeze"])
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--eval_size", type=int, default=None, help="If set, resize both to this square")
    args = parser.parse_args()

    pred_paths = sorted(args.pred_dir.glob("*.png"))
    if not pred_paths:
        raise SystemExit(f"No PNGs in {args.pred_dir}")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    import lpips

    loss_fn = lpips.LPIPS(net=args.lpips_net).to(device).eval()

    rows = []
    with torch.no_grad():
        for pp in pred_paths:
            gp = args.gt_dir / pp.name
            if not gp.exists():
                # try stem without camera suffix e.g. 00000_00.png -> 00000.png
                alt = args.gt_dir / f"{pp.stem.split('_')[0]}.png"
                if alt.exists():
                    gp = alt
                else:
                    print(f"skip missing gt for {pp.name}")
                    continue
            pred = load_rgb(pp)
            gt = load_rgb(gp)
            if pred.shape[-2:] != gt.shape[-2:]:
                pred = F.interpolate(pred[None], size=gt.shape[-2:], mode="bilinear", align_corners=False)[0]

            if args.eval_size is not None:
                size = (args.eval_size, args.eval_size)
                pred = F.interpolate(pred[None], size=size, mode="bilinear", align_corners=False)[0]
                gt = F.interpolate(gt[None], size=size, mode="bilinear", align_corners=False)[0]

            mask = None
            if args.mask_dir is not None:
                mp = args.mask_dir / pp.name
                if not mp.exists():
                    mp = args.mask_dir / f"{pp.stem.split('_')[0]}.png"
                if not mp.exists():
                    mp = args.mask_dir / f"{pp.stem.split('_')[0]}.jpg"
                if mp.exists():
                    m = np.asarray(Image.open(mp).convert("L")).astype(np.float32) / 255.0
                    mask = torch.from_numpy(m)[None]
                    if mask.shape[-2:] != gt.shape[-2:]:
                        mask = F.interpolate(mask[None], size=gt.shape[-2:], mode="nearest")[0]

            ps = psnr(pred, gt, mask=None)
            # LPIPS expects [-1,1]
            p_in = pred[None].to(device) * 2 - 1
            g_in = gt[None].to(device) * 2 - 1
            lp = float(loss_fn(p_in, g_in).item())
            if mask is not None:
                # approximate masked LPIPS via masked images (common practical approach)
                m = mask.to(device)
                p_m = (pred.to(device) * m)[None] * 2 - 1
                g_m = (gt.to(device) * m)[None] * 2 - 1
                lp_m = float(loss_fn(p_m, g_m).item())
                ps_m = psnr(pred, gt, mask)
            else:
                lp_m = None
                ps_m = None

            rows.append(
                {
                    "frame": pp.name,
                    "psnr": ps,
                    "lpips": lp,
                    "lpips_masked": lp_m,
                    "psnr_masked": ps_m,
                }
            )

    if not rows:
        raise SystemExit("No paired frames evaluated")

    summary = {
        "name": args.name,
        "n_frames": len(rows),
        "psnr_mean": float(np.mean([r["psnr"] for r in rows])),
        "psnr_std": float(np.std([r["psnr"] for r in rows])),
        "lpips_mean": float(np.mean([r["lpips"] for r in rows])),
        "lpips_std": float(np.std([r["lpips"] for r in rows])),
        "lpips_net": args.lpips_net,
        "eval_size": args.eval_size,
    }
    masked = [r["psnr_masked"] for r in rows if r["psnr_masked"] is not None]
    if masked:
        summary["psnr_masked_mean"] = float(np.mean(masked))
        summary["lpips_masked_mean"] = float(np.mean([r["lpips_masked"] for r in rows if r["lpips_masked"] is not None]))

    print(json.dumps(summary, indent=2))

    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_json, "w") as f:
            json.dump({"summary": summary, "frames": rows}, f, indent=2)
    if args.out_csv:
        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)


if __name__ == "__main__":
    main()
