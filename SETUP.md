# Setup: trackers, Gaussian Avatars, SyncNet

Example machine: `NVIDIA RTX 5060` (WSL or native Linux). Use PyTorch **`+cu128`** wheels for RTX 50-series (`pip ... --index-url https://download.pytorch.org/whl/cu128`).

Install `ffmpeg` inside the conda envs that call it , i.e. `tracker`, `smirk`, `VHAP`, `gaussian-avatars`, `multirex`, with command such as `conda install -c conda-forge ffmpeg`.

## 1. Configure paths

```bash
cp config/paths.example.env config/paths.env
# Edit ROOTDIR and paths to match your machine (absolute paths OK).
set -a && source config/paths.env && set +a
```

Expected layout:

```text
ROOTDIR/
├── face-tracker-benchmarking/  # this repo you should've cloned
├── GaussianAvatars/            # clone Gaussian Avatars
├── VHAP/                       # clone VHAP
├── syncnet_python/             # clone SyncNet v2 demo
├── other/
│   ├── MICA/                   
│   ├── metrical-tracker/       
│   ├── smirk/
│   ├── RobustVideoMatting/
│   └── now_evaluation/         # optional; for NoW Docker scoring
└── data/                       # datasets
    ├── multirex/ubisoft-laforge-multirex/
    ├── now-dataset/dataset/
    └── nersemble_dset/multiview/   # contains 017/, 024/, ... subject folders
```

Note: If clones or datasets already live elsewhere, override `REPO_ROOT` and each `*_ROOT` / dataset variable with absolute paths in `paths.env`.

Links to all models:

| Component | Source |
|-----------|--------|
| MICA | https://github.com/Zielon/MICA |
| metrical-tracker | https://github.com/Zielon/metrical-tracker |
| SMIRK | https://github.com/georgeretsi/smirk |
| RobustVideoMatting | https://github.com/PeterL1n/RobustVideoMatting |
| Gaussian Avatars | https://github.com/ShenhanQian/GaussianAvatars |
| VHAP | https://github.com/ShenhanQian/VHAP |
| SyncNet | https://github.com/joonson/syncnet_python |
| NoW evaluation | https://github.com/soubhiksanyal/now_evaluation |

## 2. Expected Conda env names

Create these environments:

| Environment | Python | PyTorch | Usage |
|-----|--------|---------|----------|
| `tracker` | 3.9 | 2.8.0+cu128 | MICA, metrical-tracker, SyncNet scoring |
| `smirk` | 3.9 | 2.8.0+cu128 | SMIRK inference |
| `VHAP` | 3.10 | 2.11.0+cu128 | VHAP tracking |
| `gaussian-avatars` | 3.10 | 2.11.0+cu128 | GA train/render |
| `multirex` | 3.8.19 | 1.11.0+cu113 | MultiREX eval (their installer) |

Note: cu128 is for RTX 50-series. On older GPUs, use the matching CUDA wheel from each project's README. 
Below are install steps for each env. Other python packages come from said project's official setup. 

### MICA and metrical-tracker (`tracker` env)

1. Install PyTorch with CUDA (e.g. cu128 on RTX 50-series).
2. MICA: download `data/pretrained/mica.tar` and licensed FLAME `generic_model.pkl` into `MICA/data/`.
3. metrical-tracker: install per its README; ensure `tracker.py` runs.
4. RVM: download `checkpoints/rvm_mobilenetv3.pth` into `RobustVideoMatting/checkpoints/`.
5. Pin `mediapipe==0.10.14` (or another 0.10.x). mediapipe 1.x drops `mp.solutions`, which metrical-tracker still needs.
6. InsightFace packs (`antelopev2` / `buffalo_l`) usually auto-download on first run.

### SMIRK (`smirk` env)

```bash
conda create -y -n smirk python=3.9
conda activate smirk
cd $SMIRK_ROOT
pip install -r requirements.txt
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
MAX_JOBS=1 pip install "git+https://github.com/facebookresearch/pytorch3d.git"   # needs CUDA toolkit headers; keep MAX_JOBS low on 8GB VRAM
pip install trimesh
pip install "numpy<2"
```

Point `SMIRK_ROOT` at one tree that includes:

```text
smirk/pretrained_models/SMIRK_em1.pt
smirk/assets/FLAME2020/generic_model.pkl
smirk/assets/FLAME2020/FLAME_texture.npz
smirk/smirk-preprocessing/track_folder_for_ga.py
```

Note: NeRSemble / README E2 use `scripts/static_rendering/track_folder_for_ga.py` in this repo (same helper; also fine under `smirk-preprocessing/`).

### VHAP (`VHAP` env)

Look at `VHAP/LOCAL_SETUP.md` after cloning VHAP. It needs FLAME, STAR landmarks, and RVM or BackgroundMattingV2 for preprocessing.

Note: Point VHAP at FLAME **2020** `generic_model.pkl` and `FLAME_texture.npz` (upstream may default to FLAME 2023).

### Gaussian Avatars (`gaussian-avatars` env)

Please follow Gaussian Avatars README. You need the FLAME model files for `--bind_to_mesh`.

Note: Place the export helpers used by README E1–E2 in this repo under `scripts/`: `export_metrical_tracker_to_ga.py`, `export_smirk_to_ga.py` (not in upstream GA; README calls them via `"$REPO_ROOT/scripts/..."`).

Note on building it: If `diff-gaussian-rasterization` / `simple-knn` fail to compile on newer nvcc, add `#include <cstdint>` / `#include <cfloat>` where the compiler complains; rebuild with `MAX_JOBS=1`.

NeRSemble training used in previous tests: (MICA / SMIRK)

```bash
python train.py -s data/<id> -m output/<id>_256_25k \
  --bind_to_mesh --white_background -r 2 --sh_degree 0 \
  --lambda_scale 0 --lambda_xyz 0 --iterations 25000
```

Note: The VHAP native canvas uses `-r 256` (i.e. portrait not landscape).

### SyncNet (`tracker` env)

```bash
cd $SYNCNET_ROOT
sh download_model.sh    # data/syncnet_v2.model + detectors/s3fd/weights/sfd_face.pth
conda activate tracker
python demo_syncnet.py --videofile data/example.avi --tmp_dir /tmp/syncnet_test
```

Scoring convention: with `demo_syncnet.py` on `224×224` with 25 fps GA renders (not the full face crop mouth centric pipeline).

### MultiREX evaluation (`multirex` env)

Make MultiREX environment after cloning:

```bash
cd $MULTIREX_ROOT
# then run multirex_conda.sh from the MultiREX repo or create the environment manually
```

Then ensure you copy FLAME model into `ubisoft-laforge-multirex/assets/FLAME/generic_model.pkl`.

Then download videos/meshes (large and will take many hours) and build GT numpy (read their GitHub repo if stuck):

```bash
python -m multirex.scripts.download_videos_and_tracked_meshes \
  --base_installation_folder "./" --download_config "./assets/download_config.json"
python -m multirex.scripts.get_gt_npy_sequences \
  --base_installation_folder "./" --output_multiface_gt_path "./assets/multiface_gt"
```

### NoW evaluation (`now_evaluation` Docker)

See README §B. Clone https://github.com/soubhiksanyal/now_evaluation.git (e.g. under `other/now_evaluation`), set `NOW_EVAL_ROOT`, `docker build -t noweval .`, then mount `$NOW_DATASET` and each tracker's `predicted_meshes/`.

## 3. FLAME 

Get FLAME 2020 assets from https://flame.is.tue.mpg.de/. You need both `generic_model.pkl` and `FLAME_texture.npz`. The same files are reused across MICA, metrical, SMIRK, VHAP, GA, and MultiREX, but the paths differ, so be careful. Force GA/VHAP off any flame2023 default onto these FLAME 2020 files.
