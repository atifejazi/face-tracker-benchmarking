# README / SETUP errata and missing details

> **Status:** The actionable items below were folded into `README.md`, `SETUP.md`, and `config/paths.example.env` (brief Notes / short blocks, same style). Keep this file only if you want the longer setup-time narrative; otherwise it can be dropped from the repo.

Notes from a full machine setup against `README.md`, `SETUP.md`, and `config/paths.example.env`.
Items below are either **inaccurate**, **internally inconsistent**, or **missing but crucial** for following the docs as written.

---

## High priority (would block or mislead a fresh setup)

### 1. NeRSemble path name does not match the evaluation commands

- **Docs say:** `ROOTDIR/data/nersemble_dset/SomeNeRSemble` (`README`, `SETUP`, `paths.example.env`).
- **Commands assume:** `$NERSSEMBLE_DATA/$SUBJECT/sequences/$SEN/images/cam_222200037.mp4` (README §E), i.e. a **subject → sequences** tree.
- That layout is the usual NeRSemble / “multiview” dump (e.g. `024/sequences/SEN-01-.../images/...`), not a folder literally named `SomeNeRSemble`.
- **Fix suggestion:** Document the expected on-disk layout explicitly, and either rename the example path to `nersemble_dset/multiview` or state that `NERSSEMBLE_DATA` must point at the directory that *contains* subject IDs.

### 2. Required helper scripts are not shipped / not in upstream clones

README §E1–E2 call:

| Script | Where README uses it | Reality |
|--------|----------------------|---------|
| `GaussianAvatars/scripts/export_metrical_tracker_to_ga.py` | E1 | **Not** in upstream GaussianAvatars; must be provided separately (or vendored). |
| `GaussianAvatars/scripts/export_smirk_to_ga.py` | E2 | Same. |
| `$REPO_ROOT/scripts/static_rendering/track_folder_for_ga.py` | E2 | Referenced by README but **was not tracked** in this repo at setup time. |
| `$SMIRK_ROOT/smirk-preprocessing/track_folder_for_ga.py` | SETUP SMIRK tree | SETUP lists it under SMIRK; README E2 uses the repo `scripts/` path instead — **two locations, unclear which is canonical**. |

Without these, MICA/SMIRK → GA export cannot follow the README.

### 3. NoW dataset layout is under-specified

Docs only say “put NoW under `ROOTDIR/data/now-dataset/dataset`”.

For **prediction** (`predict_now_*.py`) you need at least:

```text
NOW_DATASET/
  imagepathsvalidation.txt
  NoW_Dataset/final_release_version/iphone_pictures/
  NoW_Dataset/final_release_version/detected_face/
```

For **official scoring** (`now_evaluation` / Docker) you also need:

```text
  scans/
  scans_lmks_onlypp/
```

(and typically `imagepathstest.txt` if using the test list). These are separate downloads on the NoW site; images alone are not enough to run `compute_error.py`.

### 4. NoW Docker evaluation has no concrete steps

README §B: “Then run the official NoW Docker evaluation (`now_evaluation`) on each `predicted_meshes/` folder.”

Missing from README/SETUP:

- Clone URL: `https://github.com/soubhiksanyal/now_evaluation.git`
- Where to put it (e.g. `$ROOTDIR/other/now_evaluation`)
- `docker build -t noweval .` and the mount layout (`dataset` + `preds`)
- That the user must be in the `docker` group (or use `sudo docker`) and often **re-login** after `usermod -aG docker`
- Optional: `NOW_EVAL_ROOT` is used by local helpers but is **absent** from `paths.example.env`

### 5. FLAME 2020 vs upstream GA/VHAP defaults

SETUP says use FLAME 2020 `generic_model.pkl` across models. Upstream Gaussian Avatars / VHAP often default to **FLAME 2023** paths; they must be pointed at FLAME 2020 `generic_model.pkl` for this benchmark’s FLAME assets. Also required but only fully listed under SMIRK:

- `FLAME_texture.npz` (SMIRK, metrical-tracker, VHAP)

SETUP §3 mentions only `generic_model.pkl`.

---

## Medium priority (painful, easy to miss)

### 6. `mediapipe` version pin (tracker / metrical)

Not mentioned. Current `mediapipe` 1.x removes `mp.solutions`, which metrical-tracker-style code still uses. Pin something like **`mediapipe==0.10.14`** (0.10.x) in the `tracker` env.

### 7. `ffmpeg` must be on the conda env PATH

SETUP says “install ffmpeg” at the machine level. SyncNet/`demo_syncnet.py` and many README `ffmpeg` lines invoke the binary from the **active conda env**. Base/system ffmpeg is often invisible after `conda activate tracker` (etc.). Install ffmpeg into at least: `tracker`, `smirk`, `VHAP`, `gaussian-avatars`, and `multirex` (MultiREX’s download script encodes videos with ffmpeg).

### 8. `pytorch3d` build needs more than the one-liner

SETUP SMIRK block: `pip install "git+https://github.com/facebookresearch/pytorch3d.git"`.

On a CUDA 12.8 toolkit / RTX 50 laptop this typically also needs:

- Matching CUDA *toolkit* headers/libs for the build (not only the PyTorch cu128 wheel)
- Low parallelism (e.g. `MAX_JOBS=1`) on ~8 GB VRAM / 32 GB RAM machines to avoid OOM during compile

Same class of issue applies when building GA’s `diff-gaussian-rasterization` / `simple-knn`.

### 9. GA CUDA extension compile on newer nvcc

Fresh builds of `diff-gaussian-rasterization` / `simple-knn` can fail on recent nvcc without small include fixes (e.g. `<cstdint>` in the rasterizer header, `<cfloat>` in `simple_knn.cu`). Not documented; worth a one-line “if build fails on modern CUDA, add these includes” note.

### 10. MultiREX data steps are thinner than needed

SETUP covers conda + FLAME pickle only. Missing:

- Official download: `python -m multirex.scripts.download_videos_and_tracked_meshes ...`
- Then `get_gt_npy_sequences` for `assets/multiface_gt`
- Full dump is large (~15 GB, can take many hours)
- Extra pip deps the Multiface downloader needs (`requests`, `beautifulsoup4`, …) if not pulled by `multirex_conda.sh`
- Default `download_config.json` expression `EXP_ROM07_Facial_Expressions` is **not present for every Multiface identity**; some subjects only expose e.g. `EXP_free_face` under that name filter

### 11. Machine line is WSL-specific

SETUP opens with “Machine: WSL with NVIDIA RTX 5060 GPU and CUDA 12.8”. The rest of the stack works on native Linux the same way. Prefer “example machine” wording so non-WSL readers don’t think WSL is required.

### 12. “CUDA 12.8” vs PyTorch `+cu128`

Easy to misread. What matters for the envs is installing **PyTorch wheels with the `cu128` index** (bundled CUDA runtime). A system toolkit at `/usr/local/cuda-12.8` helps **compiling** extensions; it is not the same thing as “PyTorch uses the system CUDA install.” Worth one clarifying sentence in SETUP.

### 13. Torch pins vs Multirex installer

- Table: `multirex` → `1.11.0+cu113` via their installer. Their installer may resolve differently over time; treat the table as “use MultiREX’s env script,” not a hard pin you must force.
- `VHAP` / `gaussian-avatars` → `2.11.0+cu128`: correct target, but that wheel was not always published on the cu128 index when people first followed SETUP; `2.8.0+cu128` was a workable interim for RTX 50. Optional footnote.

---

## Low priority / polish

### 14. Typo in README

§Evaluation note: `do`cd "$REPO_ROOT"` — missing space after `do`.

### 15. InsightFace weights

MICA / NoW-MICA path needs InsightFace model packs (e.g. `antelopev2`, `buffalo_l`). Not listed in SETUP asset checklist; they often auto-download on first run but fail offline.

### 16. SyncNet S3FD weights path

`download_model.sh` places face detector weights under `detectors/s3fd/weights/` (not an older `face_detection/...` tree). Fine if you run their script; confusing if you hunt for paths from older forks.

### 17. `paths.example.env` gaps vs helpers

Missing optional but useful vars if you document Docker helpers: `NOW_EVAL_ROOT`. Local `paths.env` may also set it; the example file does not.

---

## What *was* accurate enough

- Env **names** and the split of roles (tracker / smirk / VHAP / gaussian-avatars / multirex).
- Overall evaluation structure (A MultiREX, B NoW, C PSNR/LPIPS, E SyncNet pipelines).
- MultiREX ranking protocol notes (front cam, stride 8).
- NeRSemble camera `222200037` and 25 fps SyncNet protocol.
- GA train flags (`-r 2` vs VHAP `-r 256`, 25k iters, white background, bind_to_mesh).
- Pointing SMIRK at FLAME2020 + `SMIRK_em1.pt`.

---

## Suggested doc fixes (checklist)

1. Align NeRSemble path string with the `$SUBJECT/sequences/...` layout used in §E.
2. Vendor or clearly link the three GA/SMIRK helper scripts; pick one canonical path for `track_folder_for_ga.py`.
3. Spell out NoW folder contents (images + lists + scans + landmarks).
4. Add a short NoW Docker subsection (clone, build, run mounts).
5. Add FLAME_texture + FLAME 2020 override notes for GA/VHAP.
6. Pin mediapipe 0.10.x; note ffmpeg-in-env; note pytorch3d / GA extension build caveats.
7. Expand MultiREX download + GT extraction by one short paragraph.
8. Soften WSL-only machine wording; clarify cu128 vs system CUDA.
