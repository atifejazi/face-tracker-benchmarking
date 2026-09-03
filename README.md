# Face Tracker Benchmarking with Gaussian Avatars
--
# Overview and Setup
The evaluation is conducted as follows:

| | Pixel/Frames to Geometry | Geometry to Rendered Gaussian Avatars |  
| --- | --- | --- |
| Static (shape) | Geometric error vs GT | Image perceptual quality |
| Dynamic (expression/ <br> speech animation)  | Jitter | Lip-Sync (SyncNet LSE) |

## Repo Structure 

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
| **MultiREX** | Static geometry and jitter | [Ubisoft MultiREX](https://github.com/ubisoft/ubisoft-laforge-multirex) put under `$MULTIREX_ROOT`. |
| **NoW** | Static scan validation | [NoW benchmark](https://now.is.tue.mpg.de/) — val images under `$NOW_DATASET`. |
| **NeRSemble SomeNeRSemble** | Dynamic SyncNet | Request via [NeRSemble](https://github.com/tobias-kirschstein/nersemble) / lab access. Monocular: `SomeNeRSemble/<subj>/sequences/<SEN>/images/cam_222200037.mp4`. |
| **Dafoe / 1015** | Static PSNR/LPIPS | Lab monocular clips (not redistributable). |

Our NeRSemble subset: subjects **017, 018, 037** × 10 SEN clips (+ dry-run **030** SEN-01 for MICA). Camera **222200037**, native **73 fps** → **25 fps** in trackers and SyncNet.

Our MultiREX ranking subset: **8 subjects**, front camera only, **stride-8** frames on ~30 fps source (shared manifest for all trackers).

