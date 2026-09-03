# Setup: trackers, Gaussian Avatars, SyncNet

Machine: WSL with `NVIDIA RTX 5060 GPU and CUDA 12.8`. Also install `ffmpeg`.

## 1. Configure paths

```bash
cp config/paths.example.env config/paths.env
# Edit ROOTDIR and paths to match your machine.
set -a && source config/paths.env && set +a
```

Expected layout:

```text
ROOTDIR/
├── face-tracker-benchmark/     # this repo you should've cloned
├── GaussianAvatars/            # clone Gaussian Avatars
├── VHAP/                       # clone VHAP
├── syncnet_python/             # clone SyncNet v2 demo
├── other/
│   ├── MICA/                   
│   ├── metrical-tracker/       
│   ├── smirk/
│   └── RobustVideoMatting/
└── data/                       # datasets
    ├── multirex/ubisoft-laforge-multirex/
    ├── now-dataset/dataset/
    └── nersemble_dset/SomeNeRSemble/
```

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

## 2. Expected Conda env names

Create these environments:

| Environment | Python | PyTorch | Usage |
|-----|--------|---------|----------|
| `tracker` | 3.9 | 2.8.0+cu128 | MICA, metrical-tracker, SyncNet scoring |
| `smirk` | 3.9 | 2.8.0+cu128 | SMIRK inference |
| `VHAP` | 3.10 | 2.11.0+cu128 | VHAP tracking |
| `gaussian-avatars` | 3.10 | 2.11.0+cu128 | GA train/render |
| `multirex` | 3.8.19 | 1.11.0+cu113 | MultiREX eval (their installer) |
| `FlashAvatar` | 3.10 | — | RVM matting in the MICA NeRSemble pipeline |

Note: cu128 is for RTX 50-series. On older GPUs, use the matching CUDA wheel from each project's README.

Below are install steps for each env. Other python packages come from said project's official setup. 

### MICA + metrical-tracker (`tracker` env)

1. Install PyTorch with CUDA (e.g. cu128 on RTX 50-series).
2. MICA: download `data/pretrained/mica.tar` and licensed FLAME `generic_model.pkl` into `MICA/data/`.
3. metrical-tracker: install per its README; ensure `tracker.py` runs.
4. RVM: download `checkpoints/rvm_mobilenetv3.pth` into `RobustVideoMatting/checkpoints/`.



### SMIRK (`smirk` env)

```bash
conda create -y -n smirk python=3.9
conda activate smirk
pip install -r smirk/requirements.txt
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install "git+https://github.com/facebookresearch/pytorch3d.git"
```

FLAME assets:

```text
smirk/pretrained_models/SMIRK_em1.pt
smirk/assets/FLAME2020/generic_model.pkl
smirk/assets/FLAME2020/FLAME_texture.npz
```

Note: NeRSemble uses `smirk-preprocessing/track_folder_for_ga.py` on exported GA image folders.

### VHAP (`VHAP` env)

Look at `VHAP/LOCAL_SETUP.md` in after cloning VHAP. It needs FLAME, STAR landmarks, and RVM or BackgroundMattingV2 for preprocessing.

### Gaussian Avatars (`gaussian-avatars` env)

Please follow Gaussian Avatars README. You need the FLAME model files for `--bind_to_mesh`.

NeRSemble training used in previous tests: (MICA / SMIRK)

```bash
python train.py -s data/<id> -m output/<id>_256_25k \
  --bind_to_mesh --white_background -r 2 --sh_degree 0 \
  --lambda_scale 0 --lambda_xyz 0 --iterations 25000
```

Note: The VHAP native canvas uses `-r 256` (i.e. portrait not landscape).

### SyncNet (`tracker` env)

```bash
cd root-dir-of-syncnet
sh download_model.sh    # data/syncnet_v2.model
conda activate tracker
python demo_syncnet.py --videofile data/example.avi --tmp_dir /tmp/syncnet_test
```

Scoring convention: with `demo_syncnet.py` on `224×224` with 25 fps GA renders (not the full face crop mouth centric pipeline).

### MultiREX evaluation (`multirex` env)

Make MultiREX environment after cloning:

```bash
cd root-dir-of-multirex...
# run multirex_conda.sh from the MultiREX repo or create the environment manually
```

Then ensure you copy FLAME model into `ubisoft-laforge-multirex/assets/FLAME/generic_model.pkl`.

## 3. FLAME 

Get FLAME 2020 assets from https://flame.is.tue.mpg.de/. Note that the same `generic_model.pkl` is reused across most models but paths are different so careful.

