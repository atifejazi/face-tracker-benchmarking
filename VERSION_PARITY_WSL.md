# Version parity checklist — WSL reference → Ubuntu

**Purpose:** Make the Ubuntu facetrack machine match the **old WSL machine** that produced the original NeRSemble / MultiREX / NoW results.

**Reference machine (source of truth for these pins):**
- OS: WSL2 Linux on Windows
- GPU: NVIDIA GeForce RTX 5060 Laptop (8151 MiB)
- Live `nvidia-smi` on this machine (2026-10-09): driver **610.43.02** (KMD 610.62), **CUDA UMD Version: 13.3**. That is **normal**. The software stack below targets **CUDA 12.8** (`cu128` wheels + conda/system `cuda-nvcc=12.8`), not CUDA 13 toolkit.
- Compute capability: **sm_120** (RTX 50-series). Extensions must be built with `TORCH_CUDA_ARCH_LIST=12.0`.
- Rebuild recipe twin: `~/conda_rebuild_assets/CONDA_ENVS_REBUILD.md` (same pins).

**Important:** The WSL conda envs were deleted after packaging. Pins below come from recorded `conda list` / `CONDA_ENVS_REBUILD.md` on that machine, not a live freeze. Treat them as the target; if Ubuntu differs, align Ubuntu to these numbers.

On Ubuntu, after `conda activate <env>`, run the check blocks and fill a short PASS/FAIL table.

---

## 0. How to check (copy-paste per env)

```bash
source "$HOME/miniconda3/etc/profile.d/conda.sh"   # or micromamba/miniforge equivalent
conda activate <ENV>

python - <<'PY'
import sys
print("python", sys.version.split()[0])
import torch
print("torch", torch.__version__)
print("torch.version.cuda", torch.version.cuda)
print("cuda_available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device", torch.cuda.get_device_name(0))
    x = torch.ones(1, device="cuda")
    print("sm_120_ok", float(x.item()))  # must print 1.0; availability alone is not enough
try:
    import torchvision
    print("torchvision", torchvision.__version__)
except Exception as e:
    print("torchvision FAIL", e)
try:
    import pytorch3d, pytorch3d._C
    print("pytorch3d", getattr(pytorch3d, "__version__", "?"), "C_ext=OK")
except Exception as e:
    print("pytorch3d FAIL", e)
try:
    import numpy
    print("numpy", numpy.__version__)
except Exception as e:
    print("numpy FAIL", e)
PY

# Toolkit / compiler (needed for pytorch3d / GA CUDA ops builds)
which nvcc; nvcc --version 2>/dev/null | tail -1
echo "CUDA_HOME=${CUDA_HOME:-}"
python -c "import torch; print('torch_lib', torch.__file__)"
```

**PASS rules for GPU envs (`tracker`, `smirk`, `VHAP`, `gaussian-avatars`):**
1. Exact (or documented-equivalent) `torch` / `torchvision` / `pytorch3d` / `numpy` versions below.
2. `torch.version.cuda` starts with **12.8** for cu128 wheels.
3. `sm_120_ok` prints `1.0` (kernel actually runs).
4. `nvcc` reports **release 12.8** when that env is active (conda `cuda-nvcc=12.8`), or system `/usr/local/cuda-12.8` is used consistently for builds.

---

## 1. System / driver (host)

| Item | WSL reference | Ubuntu must match / accept |
|------|---------------|----------------------------|
| GPU | RTX 5060 Laptop | Same class (sm_120) preferred for parity |
| Driver | ~610.x (CUDA UMD may report 13.x) | Driver must run **cu128** apps; newer driver OK |
| System toolkit (optional) | CUDA **12.8** used for compiling extensions | Prefer `/usr/local/cuda-12.8` **or** env-local `cuda-nvcc=12.8` |
| Arch for builds | `TORCH_CUDA_ARCH_LIST=12.0` `FORCE_CUDA=1` | Same |

Wheel CUDA ≠ system toolkit. PyTorch `2.8.0+cu128` / `2.11.0+cu128` ship their own CUDA **runtime**. `nvcc` 12.8 is for **compiling** pytorch3d / nvdiffrast / diff-gaussian-rasterization / simple-knn.

---

## 2. Conda env matrix (benchmark stack)

| Env | Python | torch | torchvision | pytorch3d | numpy | Role |
|-----|--------|-------|-------------|-----------|-------|------|
| `tracker` | **3.9.25** | **2.8.0+cu128** | **0.23.0+cu128** | **0.7.8** | **1.26.4** | MICA, metrical-tracker, SyncNet, RVM |
| `smirk` | **3.9.25** | **2.8.0+cu128** | **0.23.0+cu128** | **0.7.8** | **1.26.4** | SMIRK + NeRSemble E2 helper |
| `VHAP` | **3.10.20** | **2.11.0+cu128** | **0.26.0+cu128** | **0.7.9** | **1.22.3** | VHAP track / preprocess |
| `gaussian-avatars` | **3.10.20** | **2.11.0+cu128** | **0.26.0+cu128** | (not required) | **1.23.5** | GA train / render / LPIPS |
| `multirex` | **3.8.19** | **1.11.0+cu113** | **0.12.0+cu113** | — | **1.23.1** | MultiREX eval (CPU-ish; old CUDA OK) |

Index for 50-series wheels:

```text
https://download.pytorch.org/whl/cu128
```

If `2.11.0+cu128` is missing from the index on Ubuntu, that is a known packaging gap — do **not** silently stay on `2.8.0` for VHAP/GA without recording it. Prefer installing `2.11.0+cu128` when available (that is what WSL used).

---

## 3. Per-env detail (WSL recorded)

### 3.1 `tracker`

| Package | Version |
|---------|---------|
| python | 3.9.25 |
| torch | 2.8.0+cu128 |
| torchvision | 0.23.0+cu128 |
| pytorch3d | 0.7.8 (built for sm_120) |
| numpy | 1.26.4 |
| cuda-nvcc (conda) | 12.8 |
| ffmpeg | 8.0.1 (conda-forge) |
| mediapipe | 0.10.10 (pin **0.10.x**; SETUP also allows 0.10.14 — do **not** use mediapipe 1.x) |
| insightface | 1.0.1 |
| face-alignment | 1.5.0 |
| opencv-python | 4.11.x |
| chumpy | 0.70 |
| yacs | 0.1.8 |
| trimesh | 4.12.2 |
| scikit-image | 0.24.0 |
| scikit-learn | 1.6.1 |
| scipy | 1.13.1 |

Build flags used for pytorch3d:

```bash
export CUDA_HOME="$CONDA_PREFIX"   # after conda install cuda-nvcc=12.8
export TORCH_CUDA_ARCH_LIST="12.0"
export FORCE_CUDA=1
```

Cached wheel on the old machine (optional; may not load on Ubuntu ABI):  
`pytorch3d-0.7.8-cp39-cp39-linux_x86_64.whl` — prefer rebuild from source if import fails.

---

### 3.2 `smirk`

| Package | Version |
|---------|---------|
| python | 3.9.25 |
| torch | 2.8.0+cu128 |
| torchvision | 0.23.0+cu128 |
| pytorch3d | 0.7.8 |
| numpy | **1.26.4** (force after `requirements.txt`; no mixed 1.x/2.x) |
| cuda-nvcc | 12.8 |
| ffmpeg | 7.1.1 (conda-forge) |
| opencv-python | 4.9.0.80 |
| opencv-python-headless | 4.9.0.80 |
| mediapipe | 0.10.10 |
| timm | 0.9.16 |
| omegaconf | 2.3.0 |
| pytorch-lightning | 2.2.1 |
| setuptools | **&lt;81** (Lightning still needs pkg_resources) |
| scikit-image | 0.22.0 |
| scikit-learn | 1.3.2 |
| chumpy | 0.70 |
| trimesh | required for NoW SMIRK scripts |

Install order matters: **torch cu128 first**, then `requirements.txt`, then `numpy==1.26.4 --force-reinstall`, then pytorch3d.

---

### 3.3 `VHAP`

| Package | Version |
|---------|---------|
| python | 3.10.20 |
| torch | **2.11.0+cu128** |
| torchvision | **0.26.0+cu128** |
| pytorch3d | **0.7.9** (sm_120 build) |
| numpy | **1.22.3** (do not bump to 2.x) |
| cuda toolkit / nvcc | **12.8** in env |
| GCC/G++ | **13.3** (CUDA 12.8 wants GCC &lt; 14) |
| ffmpeg | 4.4.0 (conda-forge) |
| nvdiffrast | **0.3.3** from `ShenhanQian/nvdiffrast@backface-culling` |
| face-alignment | 1.5.0 |
| trimesh | 4.12.x |
| chumpy | 0.70 |
| matplotlib | 3.8.0 |
| pandas | 2.0.3 |
| scipy | 1.11.4 |

Env vars used on WSL when building/running:

```bash
export CUDA_HOME="$CONDA_PREFIX"
export CPATH="$CONDA_PREFIX/targets/x86_64-linux/include:${CPATH:-}"
export LIBRARY_PATH="$CONDA_PREFIX/targets/x86_64-linux/lib:${LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
# often also: LD_LIBRARY_PATH="$CONDA_PREFIX/lib/python3.10/site-packages/torch/lib:$LD_LIBRARY_PATH"
export TORCH_CUDA_ARCH_LIST="12.0"
export FORCE_CUDA=1
```

FLAME: **2020** `generic_model.pkl` + `FLAME_texture.npz` (not FLAME 2023 defaults).

---

### 3.4 `gaussian-avatars`

| Package | Version |
|---------|---------|
| python | 3.10.20 |
| torch | **2.11.0+cu128** |
| torchvision | **0.26.0+cu128** |
| numpy | **1.23.5** |
| cuda-nvcc | 12.8 |
| ffmpeg | 8.1.2 |
| nvdiffrast | **0.4.0** (NVlabs) |
| simple-knn | 0.0.0 (compiled from GA submodule) |
| diff-gaussian-rasterization | compiled from GA submodule |
| lpips | 0.1.4 |
| plyfile | 1.1.x |
| roma | 1.5.x |
| tyro | present |
| chumpy | 0.70 |
| opencv-python-headless | present |
| tensorboard | present |

Build with `TORCH_CUDA_ARCH_LIST=12.0`. On modern nvcc, may need `#include <cstdint>` / `#include <cfloat>` patches; `MAX_JOBS=1` on 8 GB VRAM.

pytorch3d is **not** required for the standard GA train/render path used in the benchmark.

---

### 3.5 `multirex`

| Package | Version |
|---------|---------|
| python | 3.8.19 |
| torch | 1.11.0+cu113 |
| torchvision | 0.12.0+cu113 |
| torchaudio | 0.11.0+cu113 |
| numpy | 1.23.1 |
| pandas | 2.0.3 |
| scipy | 1.10.1 |
| trimesh | 4.6.13 |
| chumpy | 0.70 |

Does **not** need sm_120. Prefer MultiREX’s `multirex_conda.sh` once; do not clobber if already correct.

---

## 4. Critical parity (check these first)

If Ubuntu results diverge, verify **in this order**:

1. **`tracker` / `smirk`:** `torch==2.8.0+cu128`, `torchvision==0.23.0+cu128`, `pytorch3d==0.7.8`, `numpy==1.26.4`, `sm_120_ok`.
2. **`VHAP` / `gaussian-avatars`:** `torch==2.11.0+cu128`, `torchvision==0.26.0+cu128` (not 2.8.0 unless documented exception).
3. **`VHAP`:** `pytorch3d==0.7.9`, `nvdiffrast` backface-culling fork, GCC 13.x, FLAME 2020.
4. **`gaussian-avatars`:** local CUDA ops import (`diff_gaussian_rasterization`, `simple_knn`), `nvdiffrast==0.4.0`, `lpips`.
5. **Toolkit:** `nvcc` 12.8 for any rebuild of pytorch3d / GA ops.
6. **Export convention (separate from versions):** MICA→GA must be **legacy moving-camera** (see `AGENT_HANDOFF_NS017_REGRESSION.md`). Wrong export can look like a “version” bug.

---

## 5. Agent deliverable

Fill this table on Ubuntu and paste back:

```text
| Env              | python | torch        | torchvision  | pytorch3d | numpy   | nvcc  | sm_120_ok | notes |
|------------------|--------|--------------|--------------|-----------|---------|-------|-----------|-------|
| tracker          |        |              |              |           |         |       |           |       |
| smirk            |        |              |              |           |         |       |           |       |
| VHAP             |        |              |              |           |         |       |           |       |
| gaussian-avatars |        |              |              | n/a       |         |       |           |       |
| multirex         |        |              |              | n/a       |         | n/a   | n/a       |       |
```

Also report:
- `nvidia-smi` driver line
- Whether system CUDA is 12.8 or only conda nvcc
- Any intentional mismatch (e.g. torch 2.8 on VHAP because 2.11 wheel missing)

---

## 6. One-liner dump (optional full freeze)

```bash
for e in tracker smirk VHAP gaussian-avatars multirex; do
  echo "===== $e ====="
  conda run -n "$e" python -c "import sys,torch; print(sys.version.split()[0], torch.__version__, torch.version.cuda)"
  conda run -n "$e" pip freeze | rg -i '^(torch|torchvision|torchaudio|pytorch3d|numpy|lpips|nvdiffrast|mediapipe|opencv|trimesh|chumpy)=='
done
```

Compare that output to the tables above.
