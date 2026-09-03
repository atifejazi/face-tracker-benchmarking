# Face Tracker Benchmarking with Gaussian Avatars
---
# Overview and Setup
The evaluation is conducted as follows:

| | Pixel/Frames to Geometry | Geometry to Rendered Gaussian Avatars |  
| --- | --- | --- |
| Static (shape) | Geometric error vs GT | Image perceptual quality |
| Dynamic (expression/ <br> speech animation)  | Jitter | Lip-Sync (SyncNet LSE) |

## Ready the Repo Structure 

```bash
git clone face-tracker-benchmarking
cd face-tracker-benchmark
cp config/paths.example.env config/paths.env   # edit paths
set -a && source config/paths.env && set +a

# then set up trackers and GA
# then download datasets rootdir/data/ 
```


## Setup of Trackers and GA
To set up the trackers (MICA, SMIRK, VHAP) and Gaussian Avatars, please refer to `SETUP.md`. This step should be completed before continuing. 

## Setup of Datasets 
| Dataset | Role | Link |
|---------|------|--------|
| **MultiREX** | Static geometry and jitter | [Ubisoft MultiREX](https://github.com/ubisoft/ubisoft-laforge-multirex) put under `MULTIREX_ROOT`. |
| **NoW** | Static geometry validation | [NoW benchmark](https://now.is.tue.mpg.de/) put under `$NOW_DATASET`. |
| **NeRSemble** | Dynamic SyncNet | [NeRSemble](https://github.com/tobias-kirschstein/nersemble)  |

Note: Tests I ran used: Camera `222200037`; Changed FPS from `73` to `25` in trackers and SyncNet. The tests also had MultiREX ranking subset of 8 subjects, with front camera only, and stride 8 frames frames (shared manifest for all trackers).

---
# Evaluation


