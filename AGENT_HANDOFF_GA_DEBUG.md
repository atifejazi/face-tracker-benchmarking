# Handoff: pipeline architecture, deps, and current GA render issues

**Date:** 2026-10-07  
**Audience:** next agent (debugging phase)  
**Repo:** `face-tracker-benchmarking` under `$ROOTDIR`  
**Load paths:** `set -a && source config/paths.env && set +a` then `cd "$REPO_ROOT"`

Do **not** assume `/home/atifl/...`. Use `$GA_ROOT`, `$NERSSEMBLE_RUNS`, etc. from `config/paths.env`.

---

## 1. What this repo is for

Benchmark **monocular face trackers** (MICA/metrical, SMIRK, VHAP) by feeding their FLAME tracks into **Gaussian Avatars (GA)**, then scoring:

- **PSNR / LPIPS** on GA train-split renders vs GT images (`scripts/static_rendering/eval_psnr_lpips.py`)
- **SyncNet** lip-sync (optional, after ffmpeg + audio)
- Separate tracks: MultiREX ranking, NoW geometry (not the focus of the current GA render bug)

Primary demo sequence for the current mess: **NeRSemble subject 017, SEN-01** (`cam_222200037`). Subject **024** has an earlier “full test” proof where VHAP looked strong.

---

## 2. Architecture (NeRSemble → GA)

```text
NeRSemble MP4 (monocular cam)
        │
        ├─► MICA identity + metrical-tracker ──► export_metrical_tracker_to_ga.py ──► GA data/<id>_mica
        │         alphas via RobustVideoMatting (standalone inference.py)
        │
        ├─► SMIRK track_folder_for_ga.py ──────► export_smirk_to_ga.py ─────────────► GA data/<id>_smirk
        │         (reuses MICA images/fg_masks as canonical canvas)
        │
        └─► VHAP preprocess → track → export_as_nerf_dataset.py ───────────────────► GA data/<id>_vhap
                  (own canvas; white bg + static offset)

Each GA dataset ──► train.py (--bind_to_mesh, 25k) ──► render.py (--skip_val --skip_test)
                 ──► eval_psnr_lpips.py on train/ours_25000/renders
```

**One-shot driver:** `scripts/run_nersemble_sentence.sh`  
`SUBJECT=017 SEN=SEN-01-cramp_small_danger bash scripts/run_nersemble_sentence.sh [all|mica|smirk|vhap]`

**Manual recipes:** README sections **E1** (MICA), **E2** (SMIRK), **E3** (VHAP).

### Per-arm details

| Arm | Track env | Export | GA train flags | Notes |
|-----|-----------|--------|----------------|-------|
| **MICA** | `tracker` | `scripts/export_metrical_tracker_to_ga.py` | `-r 2 --sh_degree 0 --lambda_scale 0 --lambda_xyz 0 --iterations 25000` | Default export = **legacy moving-camera** (FLAME R/t = 0; per-frame OpenCV cams). Mentor “affix” = opt-in `--cancel-camera-movement` (known regression if used as default). |
| **SMIRK** | `smirk` then `gaussian-avatars` for export | `scripts/export_smirk_to_ga.py` | same as MICA | Fixed cam; inherits MICA `fl_x/cx/cy/w/h/camera_z`. Shares MICA `images` / `fg_masks` via symlink. |
| **VHAP** | `VHAP` | upstream `vhap/export_as_nerf_dataset.py` | `-r 256` (native), same 25k / sh0 | Own preprocess/track/export tree under `$VHAP_ROOT`. |

**Important:** PSNR/LPIPS in this pipeline is always on **train-split** renders (`render.py --skip_val --skip_test`), not held-out val/test.

### Key scripts in this repo

| Script | Role |
|--------|------|
| `scripts/export_metrical_tracker_to_ga.py` | Metrical → GA NeRF JSON + `flame_param/` |
| `scripts/export_smirk_to_ga.py` | SMIRK `track_params.pt` → GA (weak-cam → approx perspective `t`) |
| `scripts/static_rendering/track_folder_for_ga.py` | SMIRK folder tracker |
| `scripts/static_rendering/eval_psnr_lpips.py` | PSNR + LPIPS (alex) |
| `scripts/diagnose_ga_camera_affix.py` | Moving vs affixed camera check |
| `scripts/diagnose_ga_mesh_overlay.py` | **Rigid** upright heuristic (can **lie** vs GA LBS) |
| `scripts/render_mesh_overlay_sequence.py` | **LBS** FLAME wireframe overlays (trust this over rigid diag) |
| `scripts/run_nersemble_sentence.sh` | End-to-end driver |

---

## 3. Dependencies / environments

Configured in `SETUP.md` + `config/paths.env`.

### Layout (`$ROOTDIR`)

```text
ROOTDIR/
├── face-tracker-benchmarking/   # this repo
├── GaussianAvatars/
├── VHAP/
├── syncnet_python/
├── other/{MICA,metrical-tracker,smirk,RobustVideoMatting,now_evaluation}/
└── data/{nersemble_dset/multiview,multirex/...,now-dataset/...}
```

### Conda envs (expected names)

| Env | Python | PyTorch (this machine) | Used for |
|-----|--------|----------------------|----------|
| `tracker` | 3.9 | 2.8.0+cu128 | MICA, metrical-tracker, RVM alphas, SyncNet, ffmpeg helpers |
| `smirk` | 3.9 | 2.8.0+cu128 | SMIRK (+ pytorch3d, trimesh, `numpy<2`, mediapipe) |
| `VHAP` | 3.10 | 2.11.0+cu128 | VHAP preprocess/track/export |
| `gaussian-avatars` | 3.10 | 2.11.0+cu128 | GA train/render; needs `nvdiffrast`, `diff-gaussian-rasterization`, `simple-knn`, FLAME assets |
| `multirex` | 3.8 | older CUDA | MultiREX ranking only |

Also: `ffmpeg` in those envs; RTX 50-series → **cu128** wheels.

### External projects (clone separately)

| Component | Upstream |
|-----------|----------|
| MICA | https://github.com/Zielon/MICA |
| metrical-tracker | https://github.com/Zielon/metrical-tracker |
| SMIRK | https://github.com/georgeretsi/smirk |
| RobustVideoMatting | https://github.com/PeterL1n/RobustVideoMatting |
| Gaussian Avatars | https://github.com/ShenhanQian/GaussianAvatars |
| VHAP | https://github.com/ShenhanQian/VHAP |
| SyncNet | https://github.com/joonson/syncnet_python |

### Local VHAP patch (already applied on this machine)

In `$VHAP_ROOT/vhap/preprocess_video.py`:

- RVM runs under `model.eval()`
- `--rvm-downsample-ratio` default **0.125** (was hardcoded 0.5) — better mattes on ~2–3k NeRSemble
- Not the same as `--downsample-scales` (that only writes `images_2` for tracking)

FLAME path may point at FLAME **2020** `generic_model.pkl` (not 2023).

### GA runtime notes

- Export helpers live in **this** repo, not upstream GA.
- Odd-height renders can break `yuv420p` mp4s → runner `fix_ga_mp4s` / even-dim ffmpeg rebuild.
- May need `LD_LIBRARY_PATH=$CONDA_PREFIX/lib` for torch libs under `gaussian-avatars`.

---

## 4. Exact issue with Gaussian Avatar renders right now

### Symptom

On **ns017 SEN-01**, side-by-side GA renders and LBS wireframes show:

| Tracker | Train PSNR / LPIPS | Visual |
|---------|-------------------:|--------|
| **VHAP** | **~29.56 / 0.238** | Good (best of three). Left alone during MICA/SMIRK churn. |
| **MICA** | **~26.06 / 0.267** | Better than cancel-cam (~20.9), still soft / ghosty vs VHAP; crop can look off-center. |
| **SMIRK** | **~25.93 / 0.273** (inherit/no-Rx A/B) | Matches MICA when export skips `--opencv-y-flip`. Prior cam-Rx/flame-bake (~15–17) were wrong. |

Subject **024** full-test reference (same recipe family, train PSNR): VHAP **35.66**, MICA ~22.4 (cancel-cam era), SMIRK ~21.4 — see `runs/full_test_20261004/RESULTS_nersemble.md`.

### Root cause class (not “tracking randomly failed”)

Problems are in **GA export conventions** (camera / FLAME pose / weak-cam depth), especially:

1. **FLAME `rotation` goes through LBS**, not a rigid vertex transform. Baking `Rx(π)` or camera cancel into FLAME root pose is **not** equivalent to a rigid OpenCV map. Rigid `diagnose_ga_mesh_overlay.py` can say UPRIGHT while LBS wireframes are wrong — prefer `render_mesh_overlay_sequence.py`.

2. **Mentor “affix / cancel camera” for MICA** (bake OpenCV `R,t` into FLAME; fixed cam) was made default → ns017 MICA ~21 PSNR. **Reverted default to legacy moving-camera** → ~26 PSNR. Affix remains opt-in `--cancel-camera-movement` and should **not** be default without a bake that respects LBS (or a non-LBS path).

3. **SMIRK** converts weak-perspective crop cam → approximate perspective `translation` (`smirk_translation` in `export_smirk_to_ga.py`). That approximation + Y-up FLAME vs OpenCV Y-down is the open wound.

### SMIRK export A/B (resolved — 2026-10-09)

Controlled A/B on ns017 SEN-01 (same past `track_params.pt`, MICA canvas, 25k): see `runs/smirk_export_ab_ns017/RESULTS.md`.

| Export | Rigid mesh_diag | GA train PSNR | Notes |
|--------|-----------------|--------------:|-------|
| **Inherit MICA K/z, no Rx (OpenGL `R=diag(1,-1,-1)`, `t_z<0`)** | falsely UPSIDE_DOWN | **25.925** | **winner**; ≈ MICA ~26.06 |
| WSL `--focal 1200 --camera-z 1.0`, no Rx | falsely UPSIDE_DOWN | 25.785 | tied; prefer inherit |
| Inherit + `--opencv-y-flip` (`R=I`, `t_z>0`) | UPRIGHT | 24.349 | looks “fixed” to rigid diag; worse PSNR |
| WSL 1200 + `--opencv-y-flip` | UPRIGHT | 16.647 | bad |
| Prior flame-bake / cam-Rx on disk | — | ~17.22 / ~15.01 | superseded |

**Do not** auto-re-export with `--opencv-y-flip` when rigid `mesh_diag` says UPSIDE_DOWN (`run_nersemble_sentence.sh` now warns only). Trust LBS overlays + PSNR. Canonical on-disk `data/ns017_SEN01_smirk` may still be an older cam-Rx export — re-export with inherit/no-Rx before comparing to VHAP.

### MICA status (deferred)

- **Not** camera-affixed right now (by design after regression).
- Mentor still wants affixed camera eventually; needs a correct formulation, not the broken LBS bake.
- Wireframe alignment is OK-ish; quality gap vs VHAP remains.

### VHAP status

Healthy. Only intentional change: RVM `--rvm-downsample-ratio 0.125`. Do not “fix” VHAP while debugging MICA/SMIRK.

---

## 5. Artifacts to open first

```text
# Metrics
runs/nersemble_017_SEN01/ns017_SEN01_mica_psnr_lpips.json          # ~26.06
runs/nersemble_017_SEN01/ns017_SEN01_smirk_psnr_lpips.json          # ~17.22 old flame-bake
runs/nersemble_017_SEN01/ns017_SEN01_smirk_camrx_psnr_lpips.json    # ~15.01 cam-Rx
runs/full_test_20261004_ns017/ns017_SEN01_vhap_psnr_lpips.json     # ~29.56
runs/full_test_20261004/RESULTS_nersemble.md                       # ns024 table (VHAP 35.66)

# Videos
runs/nersemble_017_SEN01/ns017_SEN01_mica_smirk_vhap_wireframe_side_by_side.mp4
runs/nersemble_017_SEN01/ns017_SEN01_mica_smirkcamrx_vhap_side_by_side.mp4
runs/full_test_20261004/see_four_tasks/ns024_SEN01_mica_smirk_vhap_side_by_side.mp4

# Datasets
$GA_ROOT/data/ns017_SEN01_{mica,smirk,vhap}/
$GA_ROOT/output/ns017_SEN01_mica_256_25k/
$GA_ROOT/output/ns017_SEN01_smirk_256_25k/          # old
$GA_ROOT/output/ns017_SEN01_smirk_camrx_256_25k/    # new
$GA_ROOT/output/ns017_SEN01_vhap_256w_25k/
```

Prior writeups: `AGENT_HANDOFF_NS017_REGRESSION.md`, `REGRESSION_NS017.md` (cancel-cam focus; SMIRK section partly superseded by cam-Rx findings above).

---

## 6. Suggested debugging focus (for the next agent)

1. **SMIRK first:** improve weak-cam → GA depth/intrinsics/translation so LBS wireframe stays upright **and** PSNR beats ~17–19 (toward VHAP). Do not trust rigid-only upright diag. Avoid baking Rx into FLAME LBS root.
2. **MICA second:** optional A/B for true camera affix that is LBS-correct; do not restore cancel-as-default without overlay + PSNR evidence.
3. Leave VHAP / RVM / GA λ flags alone for fair export A/Bs (README MICA/SMIRK λ=0).
4. Prefer new `output/...` names when retraining; do not overwrite good baselines without asking.

### Success criteria (practical)

- SMIRK LBS wireframe locked to face (eyes/mouth) across the clip.
- SMIRK train PSNR clearly above broken ~18.7 **and** above cam-Rx ~15; directionally toward MICA/VHAP. **Met on ns017 SEN-01: inherit/no-Rx ~25.9.**
- Visual GA renders not mush / inverted / sliding off the head.
- Document which export flags produced the winning run (`runs/smirk_export_ab_ns017/RESULTS.md`).

### Next agent focus (post A/B)

1. Re-export canonical `ns017_SEN01_smirk` with inherit/no-Rx; refresh side-by-side vs MICA/VHAP.
2. Spot-check another subject/sentence with the same flags (no auto cam-Rx).
3. MICA affix remains deferred; leave VHAP alone.
