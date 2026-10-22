# ns017 SEN-01 — commands / results notes

**Broken Oct-4 Ubuntu run (do not treat as golden):** cancel-camera MICA + SMIRK `--focal 1200 --camera-z 1.0` + Rx on rotation only → MICA PSNR 20.92 / LPIPS 0.475; SMIRK 18.70 / 0.491; VHAP 29.56 / 0.238.

**Corrected recipe (after `AGENT_HANDOFF_NS017_REGRESSION.md`):**

- MICA: `export_metrical_tracker_to_ga.py` **default** = legacy moving-camera (no `--cancel-camera-movement`).
- SMIRK: inherit MICA K/z; **no** `--opencv-y-flip` by default. Rigid mesh_diag UPSIDE_DOWN is expected for OpenGL cam — do not auto-flip (see `runs/smirk_export_ab_ns017/RESULTS.md`).

**Fixed re-run (2026-10-04/05)** then **SMIRK export A/B (2026-10-09):**

| Tracker | PSNR | LPIPS (alex) | Notes |
|---------|-----:|-------------:|-------|
| MICA | **26.06** | **0.267** | legacy moving-camera (was 20.92 / 0.475) |
| SMIRK inherit/no-Rx | **25.93** | **0.273** | A/B winner; prior cam-Rx/flame-bake were ~15–17 |
| VHAP | 29.56 | 0.238 | unchanged |

Diagnose: MICA cameras moving (`R_constant=false`). SMIRK: trust LBS + PSNR, not rigid mesh_diag.

Renders:

- `GaussianAvatars/output/ns017_SEN01_mica_256_25k/train/ours_25000/renders.mp4`
- `GaussianAvatars/output/ns017_SEN01_smirk_256_25k/train/ours_25000/renders.mp4`

Metrics JSON: `runs/nersemble_017_SEN01/`
