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
git clone <url of this repo> face-tracker-benchmarking
cd face-tracker-benchmarking
cp config/paths.example.env config/paths.env   # edit ROOTDIR (or set absolute paths)
set -a && source config/paths.env && set +a

# then set up trackers and GA
# then download datasets into ROOTDIR/data/ (or point dataset vars at existing downloads)
```

## Setup of Trackers and GA
To set up the trackers (MICA, SMIRK, VHAP) and Gaussian Avatars, please refer to `SETUP.md`. This step should be completed before continuing. 

## Setup of Datasets 
| Dataset | Role | Link |
|---------|------|--------|
| **MultiREX** | Static geometry and jitter | [Ubisoft MultiREX](https://github.com/ubisoft/ubisoft-laforge-multirex) put under `ROOTDIR/data/multirex/ubisoft-laforge-multirex` (`MULTIREX_ROOT` in `paths.env`). |
| **NoW** | Static geometry validation | [NoW benchmark](https://now.is.tue.mpg.de/) put under `ROOTDIR/data/now-dataset/dataset` (`NOW_DATASET` in `paths.env`). |
| **NeRSemble** | Dynamic SyncNet and static rendering | [NeRSemble](https://github.com/tobias-kirschstein/nersemble) — SomeNeRSemble videos under `ROOTDIR/data/nersemble_dset/SomeNeRSemble` (`NERSSEMBLE_DATA` in `paths.env`). |

Note: For ease, bash will fill in folders from the files when `$` environmental variables are used. You may set absolute paths in `paths.env` if clones/datasets are not under one `ROOTDIR`.

Note: Tests I ran used camera `222200037`. I also changed FPS from `73` to `25` in trackers and SyncNet. The tests also had MultiREX ranking subset of 8 subjects, with front camera only, and stride 8 frames (shared manifest for all trackers).

---
# Evaluation
Load paths once per shell session:

```bash
set -a && source config/paths.env && set +a
source "$HOME/miniconda3/etc/profile.d/conda.sh"
mkdir -p "$MULTIREX_RESULTS" "$NERSSEMBLE_RUNS/logs"
```

Note: Python helpers are in `scripts/` and outputs are in `runs/`.

### A. Static geometry: MultiREX
Units: mm

Full ranking pipeline (manifest -> track -> evaluate -> jitter -> put to table):

```bash
set -a && source config/paths.env && set +a
source "$HOME/miniconda3/etc/profile.d/conda.sh"

VIDEOS="$MULTIREX_ROOT/videos_gamma_corrected"
BBOX="$MULTIREX_ROOT/assets/video_bbox_mapping.pickle"
ASSETS="$MULTIREX_ROOT/assets"
GT="$MULTIREX_ROOT/assets/multiface_gt"
ROOT="$MULTIREX_RESULTS"
MANIFEST="$ROOT/manifest.json"
MICA_OUT="$ROOT/params/mica"
VHAP_OUT="$ROOT/params/vhap"
SMIRK_OUT="$ROOT/params/smirk"
VHAP_WORK="$ROOT/vhap_work"
EVAL="$ROOT/eval"
mkdir -p "$MICA_OUT" "$VHAP_OUT" "$SMIRK_OUT" "$VHAP_WORK" "$EVAL"

# 1) front cam only and stride 8
conda activate multirex
python scripts/static_geometry/build_ranking_manifest.py \
  --videos_dir "$VIDEOS" --bbox_pickle "$BBOX" \
  --output "$MANIFEST" --frame_stride 8 --front_only

# 2) SMIRK extract of full videos, then slice to the stride-8
conda activate smirk
python scripts/static_geometry/extract_multirex_smirk.py \
  --videos_dir "$VIDEOS" --bbox_pickle "$BBOX" \
  --output_dir "$ROOT/params/smirk_full" \
  --checkpoint "$SMIRK_ROOT/pretrained_models/SMIRK_em1.pt" --skip_existing
python scripts/static_geometry/slice_smirk_for_ranking.py \
  --manifest "$MANIFEST" --input_dir "$ROOT/params/smirk_full" --output_dir "$SMIRK_OUT" --skip_existing

# 3) MICA extract 
conda activate tracker
python scripts/static_geometry/extract_multirex_mica.py \
  --videos_dir "$VIDEOS" --bbox_pickle "$BBOX" \
  --output_dir "$MICA_OUT" --manifest "$MANIFEST" --skip_existing

# 4) VHAP extract 
conda activate VHAP
python scripts/static_geometry/extract_multirex_vhap.py \
  --videos_dir "$VIDEOS" --bbox_pickle "$BBOX" \
  --output_dir "$VHAP_OUT" --work_dir "$VHAP_WORK" \
  --manifest "$MANIFEST" --n_downsample 2 --batch_size 4 --light_photo --skip_existing

# 5) evaluation and jitter per tracker
conda activate multirex
for tracker in smirk mica vhap; do
  params="$ROOT/params/$tracker"
  out="$EVAL/$tracker"
  mkdir -p "$out"
  python scripts/static_geometry/run_multirex_eval_subsampled.py \
    --input_folder "$params" --output_folder "$out" \
    --assets_path "$ASSETS" --gt_path "$GT" --n_exp_components 100 --overwrite
  python scripts/static_geometry/compute_jitter.py \
    --decoded_dir "$out/decoded_flame_meshes" \
    --output_csv "$out/jitter_by_video.csv" \
    --per_frame_csv "$out/jitter_per_frame.csv" \
    --tracker "$tracker"
done

# 6) ranking table and paired CIs
python scripts/static_geometry/rank_trackers_ci.py \
  --eval_root "$EVAL" --trackers smirk mica vhap \
  --output_csv "$ROOT/ranking_table.csv" \
  --output_md "$ROOT/ranking_table.md" \
  --per_subject_csv "$ROOT/per_subject_means.csv"
```


### B. Static geometry with NoW 
Units: mm

```bash
set -a && source config/paths.env && set +a
source "$HOME/miniconda3/etc/profile.d/conda.sh"

conda activate tracker
python scripts/static_geometry/predict_now_mica.py --device cuda

conda activate smirk
python scripts/static_geometry/predict_now_smirk.py --device cuda

conda activate VHAP
python scripts/static_geometry/predict_now_vhap.py --device cuda
```

Then run the official NoW Docker evaluation (`now_evaluation`) on each `predicted_meshes/` folder. 

### C. Static rendering 
Metrics: PSNR / LPIPS

After tracker the pipeline of GA train to render on a NeRSemble clip (same pipeline as the E1–E3 below), compare GA renders to the exported training images for that clip:

```bash
conda activate gaussian-avatars
python scripts/static_rendering/eval_psnr_lpips.py \
  --pred_dir "$GA_ROOT/output/ns017_SEN01_mica_256_25k/train/ours_25000/renders" \
  --gt_dir "$GA_ROOT/data/ns017_SEN01_mica/images" \
  --output_json runs/nersemble_psnr_lpips.json
```

For SMIRK, swap `mica` for `smirk` in those folder names. For VHAP, use `output/ns017_SEN01_vhap_256w_25k/train/ours_25000/renders` and `data/ns017_SEN01_vhap`.

Note: For GA training, MICA/SMIRK use `-r 2` at 25k iters. Native VHAP uses `-r 256`.
### D. Dynamic geometry: MultiREX jitter

This is similar to part A on decoded meshes. 
Metric: RMS vertex acceleration in mm / (subsampled frame)² on stride-8.

### E. Dynamic rendering: NeRSemble SyncNet

Set clip variables (e.g. subject 017, SEN-01):
```bash
set -a && source config/paths.env && set +a
source "$HOME/miniconda3/etc/profile.d/conda.sh"

SUBJECT=017
SEN=SEN-01-cramp_small_danger
SEN_NUM=$(echo "$SEN" | sed -n 's/^SEN-\([0-9][0-9]*\).*/\1/p')
MP4="$NERSSEMBLE_DATA/$SUBJECT/sequences/$SEN/images/cam_222200037.mp4"
LOGDIR="$NERSSEMBLE_RUNS/logs"
```

#### E1. Pipeline: MICA to GA to SyncNet

```bash
ID="ns${SUBJECT}_SEN${SEN_NUM}_mica"
WORKDIR="$NERSSEMBLE_RUNS/$ID"
CFG="$METRICAL_TRACKER_ROOT/configs/actors/${ID}.yml"
mkdir -p "$LOGDIR" "$WORKDIR/imgs" "$WORKDIR/alpha" "$METRICAL_TRACKER_ROOT/input/$ID"

cp -f "$MP4" "$METRICAL_TRACKER_ROOT/input/$ID/video.mp4"
cat > "$CFG" <<EOF
actor: './input/$ID'
save_folder: './output/'
optimize_shape: true
optimize_jaw: true
begin_frames: 1
keyframes: [ 0, 1 ]
fps: 25
EOF

# MICA identity
conda activate tracker
mkdir -p "$MICA_ROOT/demo/input_$ID" "$MICA_ROOT/demo/arcface_$ID" "$MICA_ROOT/demo/output_$ID"
ffmpeg -y -i "$MP4" -vf "select=eq(n\\,0)" -frames:v 1 "$MICA_ROOT/demo/input_$ID/${ID}.png"
cd "$MICA_ROOT"
python demo.py -i "demo/input_$ID" -a "demo/arcface_$ID" -o "demo/output_$ID" -m data/pretrained/mica.tar
cp "$(find demo/output_$ID -name identity.npy | head -1)" "$METRICAL_TRACKER_ROOT/input/$ID/identity.npy"

# metrical-tracker
cd "$METRICAL_TRACKER_ROOT"
python tracker.py --cfg "./configs/actors/${ID}.yml"

# RVM alphas on tracker crops
conda activate tracker
IMG_DST="$WORKDIR/imgs"
ALPHA_DST="$WORKDIR/alpha"
rm -rf "$IMG_DST" "$ALPHA_DST" && mkdir -p "$IMG_DST" "$ALPHA_DST"
python - <<PY
from pathlib import Path
import shutil
ckpt = Path("$METRICAL_TRACKER_ROOT/output/$ID/checkpoint")
src = Path("$METRICAL_TRACKER_ROOT/output/$ID/input")
dst = Path("$IMG_DST")
ids = sorted(int(p.stem) for p in ckpt.glob("*.frame"))
for i in ids:
    s = src / f"{i:05d}.png"
    if not s.exists():
        raise SystemExit(f"missing crop for frame {i}")
    shutil.copy(s, dst / f"{i:05d}.png")
print("copied", len(ids), "imgs")
PY
cd "$RVM_ROOT"
PYTHONPATH="$RVM_ROOT" python inference.py --variant mobilenetv3 \
  --checkpoint checkpoints/rvm_mobilenetv3.pth --device cuda \
  --input-source "$IMG_DST" --output-type png_sequence \
  --output-alpha "$WORKDIR/rvm_raw" --seq-chunk 4 --num-workers 2
python - <<PY
from pathlib import Path
import shutil
raw = sorted(Path("$WORKDIR/rvm_raw").glob("*.png"))
dst = Path("$ALPHA_DST")
imgs = sorted(Path("$IMG_DST").glob("*.png"))
assert len(raw) == len(imgs), (len(raw), len(imgs))
for a, im in zip(raw, imgs):
    shutil.copy(a, dst / (im.stem + ".jpg"))
print("alphas", len(list(dst.glob("*.jpg"))))
PY

# gaussian avatars setup
conda activate tracker
cd "$GA_ROOT"
TGT="data/$ID"
python scripts/export_metrical_tracker_to_ga.py \
  --track-out "$METRICAL_TRACKER_ROOT/output/$ID" \
  --imgs-dir "$IMG_DST" --alpha-dir "$ALPHA_DST" --tgt-dir "$TGT"

# train and render GA
conda activate gaussian-avatars
cd "$GA_ROOT"
python train.py -s "$TGT" -m "output/${ID}_256_25k" --bind_to_mesh --white_background \
  -r 2 --sh_degree 0 --lambda_scale 0 --lambda_xyz 0 --iterations 25000 --interval 5000
python render.py -m "output/${ID}_256_25k" --skip_val --skip_test -r 2

# syncnet at 224 x 224 and 25 fps
conda activate tracker
AUDIO="$WORKDIR/audio.wav"
ffmpeg -y -i "$MP4" -vn -acodec pcm_s16le -ar 16000 -ac 1 "$AUDIO"
ffmpeg -y -framerate 25 -i "$GA_ROOT/output/${ID}_256_25k/train/ours_25000/renders/%05d.png" \
  -vf "scale=224:224,fps=25" -c:v libx264 -pix_fmt yuv420p "$WORKDIR/renders_noaudio.mp4"
ffmpeg -y -i "$WORKDIR/renders_noaudio.mp4" -i "$AUDIO" \
  -vf "scale=224:224,fps=25" -c:v libx264 -pix_fmt yuv420p \
  -c:a pcm_s16le -ar 16000 -ac 1 -shortest "$WORKDIR/render_with_audio.avi"
cd "$SYNCNET_ROOT"
python demo_syncnet.py --videofile "$WORKDIR/render_with_audio.avi" \
  --tmp_dir "$WORKDIR/syncnet_tmp" --reference "$ID"
```

#### E2. Pipeline: SMIRK to GA to SyncNet 
Note: canvas is the same as mica's 

Requires MICA GA dataset `data/ns${SUBJECT}_SEN${SEN_NUM}_mica` from E1.

```bash
ID="ns${SUBJECT}_SEN${SEN_NUM}_smirk"
MICA_ID="ns${SUBJECT}_SEN${SEN_NUM}_mica"
MICA_DS="$GA_ROOT/data/$MICA_ID"
WORKDIR="$NERSSEMBLE_RUNS/$ID"
TRACK_PT="$WORKDIR/track_params.pt"
mkdir -p "$WORKDIR"


conda activate smirk
cd "$SMIRK_ROOT"
PYTHONPATH=. python smirk-preprocessing/track_folder_for_ga.py \
  --image-dir "$MICA_DS/images" --checkpoint pretrained_models/SMIRK_em1.pt --output "$TRACK_PT"

conda activate gaussian-avatars
cd "$GA_ROOT"
TGT="data/$ID"
python scripts/export_smirk_to_ga.py --track-params "$TRACK_PT" \
  --canonical-dataset "$MICA_DS" --target "$TGT" --focal 1200 --camera-z 1.0
python train.py -s "$TGT" -m "output/${ID}_256_25k" --bind_to_mesh --white_background \
  -r 2 --sh_degree 0 --lambda_scale 0 --lambda_xyz 0 --iterations 25000 --interval 5000
python render.py -m "output/${ID}_256_25k" --skip_val --skip_test -r 2

# syncnet at 224 x 224
conda activate tracker
AUDIO="$WORKDIR/audio.wav"
ffmpeg -y -i "$MP4" -vn -acodec pcm_s16le -ar 16000 -ac 1 "$AUDIO"
ffmpeg -y -framerate 25 -i "$GA_ROOT/output/${ID}_256_25k/train/ours_25000/renders/%05d.png" \
  -vf "scale=224:224,fps=25" -c:v libx264 -pix_fmt yuv420p "$WORKDIR/renders_noaudio.mp4"
ffmpeg -y -i "$WORKDIR/renders_noaudio.mp4" -i "$AUDIO" \
  -vf "scale=224:224,fps=25" -c:v libx264 -pix_fmt yuv420p \
  -c:a pcm_s16le -ar 16000 -ac 1 -shortest "$WORKDIR/render_with_audio.avi"
cd "$SYNCNET_ROOT"
python demo_syncnet.py --videofile "$WORKDIR/render_with_audio.avi" \
  --tmp_dir "$WORKDIR/syncnet_tmp" --reference "$ID"
```

#### E3. Pipeline: VHAP to GA to SyncNet 
note: not identical to mica's canvas so uses a native setup

```bash
ID="ns${SUBJECT}_SEN${SEN_NUM}_vhap"
WORKDIR="$NERSSEMBLE_RUNS/$ID"
mkdir -p "$WORKDIR"
VHAP_IN="$VHAP_ROOT/data/monocular/${ID}.mp4"
TRACK_OUT="$VHAP_ROOT/output/monocular/${ID}_whiteBg_staticOffset"
EXPORT_OUT="$VHAP_ROOT/export/monocular/${ID}_whiteBg_staticOffset_maskBelowLine"
FF_VID_FILTER='scale=224:224:force_original_aspect_ratio=decrease,pad=224:224:(ow-iw)/2:(oh-ih)/2,setsar=1,fps=25'

mkdir -p "$VHAP_ROOT/data/monocular"
cp -f "$MP4" "$VHAP_IN"
conda activate VHAP
cd "$VHAP_ROOT"
python vhap/preprocess_video.py --input "$VHAP_IN" --downsample-scales 2 \
  --matting-method robust_video_matting
python vhap/track.py --data.root-folder "$VHAP_ROOT/data/monocular" \
  --data.sequence "$ID" --data.n-downsample-rgb 2 \
  --exp.output-folder "$TRACK_OUT" --batch-size 4
RUN_DIR=$(ls -d "$TRACK_OUT"/*/ | head -1)
python vhap/export_as_nerf_dataset.py --src-folder "$RUN_DIR" --tgt-folder "$EXPORT_OUT" \
  --background-color white --flame-mode param

conda activate gaussian-avatars
cd "$GA_ROOT"
ln -sfn "$EXPORT_OUT" "data/$ID"
python train.py -s "data/$ID" -m "output/${ID}_256w_25k" --bind_to_mesh --white_background \
  -r 256 --sh_degree 0 --iterations 25000 --interval 5000
python render.py -m "output/${ID}_256w_25k" --skip_val --skip_test -r 256

conda activate tracker
AUDIO="$WORKDIR/audio.wav"
ffmpeg -y -i "$MP4" -vn -acodec pcm_s16le -ar 16000 -ac 1 "$AUDIO"
ffmpeg -y -framerate 25 -i "$GA_ROOT/output/${ID}_256w_25k/train/ours_25000/renders/%05d.png" \
  -vf "$FF_VID_FILTER" -c:v libx264 -pix_fmt yuv420p "$WORKDIR/renders_noaudio.mp4"
ffmpeg -y -i "$WORKDIR/renders_noaudio.mp4" -i "$AUDIO" \
  -vf "$FF_VID_FILTER" -c:v libx264 -pix_fmt yuv420p \
  -c:a pcm_s16le -ar 16000 -ac 1 -shortest "$WORKDIR/render_with_audio.avi"
cd "$SYNCNET_ROOT"
python demo_syncnet.py --videofile "$WORKDIR/render_with_audio.avi" \
  --tmp_dir "$WORKDIR/syncnet_tmp" --reference "$ID"
```

#### E4. Ground Truth (GT) baseline 

```bash
ID="ns${SUBJECT}_SEN${SEN_NUM}_gt"
WORKDIR="$NERSSEMBLE_RUNS/$ID"
mkdir -p "$WORKDIR"
FF_VID_FILTER='scale=224:224:force_original_aspect_ratio=decrease,pad=224:224:(ow-iw)/2:(oh-ih)/2,setsar=1,fps=25'

conda activate tracker
AUDIO="$WORKDIR/audio.wav"
MUX="$WORKDIR/gt_224_25fps.avi"
ffmpeg -y -i "$MP4" -vn -acodec pcm_s16le -ar 16000 -ac 1 "$AUDIO"
ffmpeg -y -i "$MP4" -i "$AUDIO" -vf "$FF_VID_FILTER" -c:v libx264 -pix_fmt yuv420p \
  -c:a pcm_s16le -ar 16000 -ac 1 -shortest "$MUX"
cd "$SYNCNET_ROOT"
python demo_syncnet.py --videofile "$MUX" --tmp_dir "$WORKDIR/syncnet_tmp" --reference "$ID"
```


**SyncNet protocol:** GA render and original audio to ffmpeg w/ 224×224 and 25 fps.
Then the 16 kHz mono to `demo_syncnet.py` and no S3FD mouth crop. 
Metrics: **AV offset** and frames, 0 best.
**Confidence** the higher the better.
**Min dist** the lower the better. Note that VHAP/GT use letterbox pad and MICA/SMIRK use square scale.

---


