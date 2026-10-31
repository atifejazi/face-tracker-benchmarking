#!/usr/bin/env bash
# NeRSemble monocular sentence: MICA / SMIRK / VHAP → GA 25k → render → PSNR/LPIPS → SyncNet
set -eo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
set -a && source "$ROOT/config/paths.env" && set +a
source "$HOME/miniconda3/etc/profile.d/conda.sh"

# Torch CUDA extensions need libc10 on the loader path.
_activate_torch_env() {
  conda activate "$1"
  local _torch_lib
  _torch_lib="$(python -c 'import torch, os; print(os.path.join(os.path.dirname(torch.__file__), "lib"))')"
  export LD_LIBRARY_PATH="${_torch_lib}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
}
activate_ga() { _activate_torch_env gaussian-avatars; }

# GA render.py ffmpeg fails on odd H (yuv420p). Rebuild mp4s with even dims.
fix_ga_mp4s() {
  local out_dir="$1"
  local iter_dir
  iter_dir=$(ls -d "$out_dir"/train/ours_* 2>/dev/null | sort | tail -1)
  [[ -n "$iter_dir" ]] || return 0
  for kind in renders gt; do
    local src="$iter_dir/$kind"
    [[ -d "$src" ]] || continue
    ffmpeg -y -framerate 25 -f image2 -pattern_type glob -i "$src/*.png" \
      -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" -pix_fmt yuv420p "$iter_dir/${kind}.mp4" \
      >/dev/null 2>&1 || true
  done
  ls -la "$iter_dir"/renders.mp4 "$iter_dir"/gt.mp4 2>/dev/null || true
}

activate_vhap() {
  _activate_torch_env VHAP
  # Prefer toolkit with full CUDA headers for any nvdiffrast JIT rebuild.
  export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.8}"
  export PATH="$CUDA_HOME/bin${PATH:+:$PATH}"
  export CPATH="${CUDA_HOME}/targets/x86_64-linux/include${CPATH:+:$CPATH}"
  # nvdiffrast JIT plugin (torch 2.x may not put build dir on sys.path)
  local _nvdr_plugin="$HOME/.cache/torch_extensions/py310_cu128/nvdiffrast_plugin"
  if [[ -d "$_nvdr_plugin" ]]; then
    export PYTHONPATH="${_nvdr_plugin}${PYTHONPATH:+:$PYTHONPATH}"
  fi
}

# Usage: SUBJECT=017 SEN=SEN-01-cramp_small_danger ./scripts/run_nersemble_sentence.sh [all|mica|smirk|vhap|...]
# Optional RUN_TAG (e.g. _20261010) suffixes IDs + SUMMARY so reruns do not clobber prior artifacts.
SUBJECT="${SUBJECT:-017}"
SEN="${SEN:-SEN-01-cramp_small_danger}"
CAM="${CAM:-cam_222200037.mp4}"
RUN_TAG="${RUN_TAG:-}"
SEN_NUM=$(echo "$SEN" | sed -n 's/^SEN-\([0-9][0-9]*\).*/\1/p')
MP4="$NERSSEMBLE_DATA/$SUBJECT/sequences/$SEN/images/$CAM"
LOGDIR="$NERSSEMBLE_RUNS/logs"
SUMMARY="${SUMMARY:-$ROOT/runs/nersemble_${SUBJECT}_SEN${SEN_NUM}${RUN_TAG}}"
mkdir -p "$LOGDIR" "$SUMMARY"
exec > >(tee -a "$SUMMARY/nersemble_pipeline.log") 2>&1

echo "==== START $(date -Is) ===="
echo "MP4=$MP4"
echo "RUN_TAG=${RUN_TAG:-<none>} SUMMARY=$SUMMARY"
test -f "$MP4"

run_mica() {
  local ID="ns${SUBJECT}_SEN${SEN_NUM}_mica${RUN_TAG}"
  local WORKDIR="$NERSSEMBLE_RUNS/$ID"
  local CFG="$METRICAL_TRACKER_ROOT/configs/actors/${ID}.yml"
  mkdir -p "$WORKDIR/imgs" "$WORKDIR/alpha" "$METRICAL_TRACKER_ROOT/input/$ID"

  echo "==== MICA identity $ID ===="
  cp -f "$MP4" "$METRICAL_TRACKER_ROOT/input/$ID/video.mp4"
  # bbox_scale 3.5 (default 2.5): more headroom so chin/forehead survive tight
  # SEN-05 face-detector boxes. get_bbox also shift-keeps size when edge-clamped.
  cat > "$CFG" <<EOF
actor: './input/$ID'
save_folder: './output/'
optimize_shape: true
optimize_jaw: true
begin_frames: 1
keyframes: [ 0, 1 ]
fps: 25
bbox_scale: 3.5
EOF
  conda activate tracker
  mkdir -p "$MICA_ROOT/demo/input_$ID" "$MICA_ROOT/demo/arcface_$ID" "$MICA_ROOT/demo/output_$ID"
  ffmpeg -y -i "$MP4" -vf "select=eq(n\,0)" -frames:v 1 -update 1 "$MICA_ROOT/demo/input_$ID/${ID}.png"
  cd "$MICA_ROOT"
  python demo.py -i "demo/input_$ID" -a "demo/arcface_$ID" -o "demo/output_$ID" -m data/pretrained/mica.tar
  cp "$(find demo/output_$ID -name identity.npy | head -1)" "$METRICAL_TRACKER_ROOT/input/$ID/identity.npy"

  echo "==== metrical-tracker $ID ===="
  cd "$METRICAL_TRACKER_ROOT"
  python tracker.py --cfg "./configs/actors/${ID}.yml"

  echo "==== RVM alphas $ID ===="
  conda activate tracker
  IMG_DST="$WORKDIR/imgs"
  ALPHA_DST="$WORKDIR/alpha"
  rm -rf "$IMG_DST" "$ALPHA_DST" "$WORKDIR/rvm_raw"
  mkdir -p "$IMG_DST" "$ALPHA_DST"
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

  echo "==== export MICA→GA (legacy moving-camera) $ID ===="
  conda activate tracker
  cd "$GA_ROOT"
  TGT="data/$ID"
  rm -rf "$TGT"
  python "$REPO_ROOT/scripts/export_metrical_tracker_to_ga.py" \
    --track-out "$METRICAL_TRACKER_ROOT/output/$ID" \
    --imgs-dir "$IMG_DST" --alpha-dir "$ALPHA_DST" --tgt-dir "$TGT"
  python "$REPO_ROOT/scripts/diagnose_ga_camera_affix.py" \
    --transforms "$TGT/transforms_train.json" \
    --plot "$WORKDIR/mica_cam_path.png" | tee "$WORKDIR/mica_cam_affix.json"

  run_mica_ga "$ID" "$WORKDIR" "$GA_ROOT/$TGT"
}

# Resume point after track+export (dataset already in $GA_ROOT/data/$ID).
run_mica_ga() {
  local ID="${1:-ns${SUBJECT}_SEN${SEN_NUM}_mica${RUN_TAG}}"
  local WORKDIR="${2:-$NERSSEMBLE_RUNS/$ID}"
  local TGT="${3:-$GA_ROOT/data/$ID}"
  test -d "$TGT"

  echo "==== GA train 25k $ID ===="
  activate_ga
  cd "$GA_ROOT"
  rm -rf "output/${ID}_256_25k"
  python train.py -s "$TGT" -m "output/${ID}_256_25k" --bind_to_mesh --white_background \
    -r 2 --sh_degree 0 \
    --lambda_scale 1 --lambda_xyz 0.01 --scaling_lr 0.005 \
    --iterations 25000 --interval 5000 --position_lr_max_steps 25000 \
    --densify_from_iter 500 --densification_interval 100 \
    --densify_until_iter 15000 --opacity_reset_interval 3000
  python render.py -m "output/${ID}_256_25k" --skip_val --skip_test -r 2
  fix_ga_mp4s "output/${ID}_256_25k"

  echo "==== PSNR/LPIPS $ID ===="
  cd "$REPO_ROOT"
  activate_ga
  python scripts/static_rendering/eval_psnr_lpips.py \
    --pred_dir "$GA_ROOT/output/${ID}_256_25k/train/ours_25000/renders" \
    --gt_dir "$GA_ROOT/data/$ID/images" \
    --output_json "$SUMMARY/${ID}_psnr_lpips.json"

  echo "==== SyncNet $ID ===="
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
    --tmp_dir "$WORKDIR/syncnet_tmp" --reference "$ID" | tee "$SUMMARY/${ID}_syncnet.txt"
  echo "MICA_DONE $(date -Is)"
}

run_smirk() {
  local ID="ns${SUBJECT}_SEN${SEN_NUM}_smirk${RUN_TAG}"
  local MICA_ID="ns${SUBJECT}_SEN${SEN_NUM}_mica${RUN_TAG}"
  local MICA_DS="$GA_ROOT/data/$MICA_ID"
  local WORKDIR="$NERSSEMBLE_RUNS/$ID"
  local TRACK_PT="$WORKDIR/track_params.pt"
  mkdir -p "$WORKDIR"
  test -d "$MICA_DS/images"

  echo "==== SMIRK track $ID ===="
  conda activate smirk
  cd "$SMIRK_ROOT"
  PYTHONPATH=. python "$REPO_ROOT/scripts/static_rendering/track_folder_for_ga.py" \
    --image-dir "$MICA_DS/images" --checkpoint pretrained_models/SMIRK_em1.pt --output "$TRACK_PT"

  echo "==== export SMIRK→GA $ID ===="
  activate_ga
  cd "$GA_ROOT"
  TGT="data/$ID"
  rm -rf "$TGT"
  python "$REPO_ROOT/scripts/export_smirk_to_ga.py" --track-params "$TRACK_PT" \
    --canonical-dataset "$MICA_DS" --target "$TGT"
  # Full GA LBS mesh + native SMIRK reprojection gate. Never auto-flip data.
  python "$REPO_ROOT/scripts/diagnose_ga_mesh_overlay.py" \
    --dataset "$TGT" --frames 0,5,10 --ga-root "$GA_ROOT" \
    --track-params "$TRACK_PT" --out_dir "$WORKDIR/mesh_diag"

  echo "==== GA train 25k $ID ===="
  rm -rf "output/${ID}_256_25k"
  python train.py -s "$TGT" -m "output/${ID}_256_25k" --bind_to_mesh --white_background \
    -r 2 --sh_degree 0 --lambda_scale 0 --lambda_xyz 0 --iterations 25000 --interval 5000
  python render.py -m "output/${ID}_256_25k" --skip_val --skip_test -r 2
  fix_ga_mp4s "output/${ID}_256_25k"

  echo "==== PSNR/LPIPS $ID ===="
  cd "$REPO_ROOT"
  activate_ga
  python scripts/static_rendering/eval_psnr_lpips.py \
    --pred_dir "$GA_ROOT/output/${ID}_256_25k/train/ours_25000/renders" \
    --gt_dir "$GA_ROOT/data/$ID/images" \
    --output_json "$SUMMARY/${ID}_psnr_lpips.json"

  echo "==== SyncNet $ID ===="
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
    --tmp_dir "$WORKDIR/syncnet_tmp" --reference "$ID" | tee "$SUMMARY/${ID}_syncnet.txt"
  echo "SMIRK_DONE $(date -Is)"
}

run_vhap() {
  local ID="ns${SUBJECT}_SEN${SEN_NUM}_vhap${RUN_TAG}"
  local WORKDIR="$NERSSEMBLE_RUNS/$ID"
  mkdir -p "$WORKDIR"
  local VHAP_IN="$VHAP_ROOT/data/monocular/${ID}.mp4"
  local TRACK_OUT="$VHAP_ROOT/output/monocular/${ID}_whiteBg_staticOffset"
  local EXPORT_OUT="$VHAP_ROOT/export/monocular/${ID}_whiteBg_staticOffset_maskBelowLine"
  local FF_VID_FILTER='scale=224:224:force_original_aspect_ratio=decrease,pad=224:224:(ow-iw)/2:(oh-ih)/2,setsar=1,fps=25'

  echo "==== VHAP preprocess $ID ===="
  mkdir -p "$VHAP_ROOT/data/monocular"
  cp -f "$MP4" "$VHAP_IN"
  activate_vhap
  cd "$VHAP_ROOT"
  rm -rf "$VHAP_ROOT/data/monocular/$ID"
  python vhap/preprocess_video.py --input "$VHAP_IN" --downsample-scales 2 \
    --matting-method robust_video_matting --rvm-downsample-ratio 0.125

  run_vhap_track "$ID" "$WORKDIR" "$TRACK_OUT" "$EXPORT_OUT" "$FF_VID_FILTER"
}

# Resume after preprocess (images/alpha already under VHAP data/monocular/$ID).
run_vhap_track() {
  local ID="${1:-ns${SUBJECT}_SEN${SEN_NUM}_vhap${RUN_TAG}}"
  local WORKDIR="${2:-$NERSSEMBLE_RUNS/$ID}"
  local TRACK_OUT="${3:-$VHAP_ROOT/output/monocular/${ID}_whiteBg_staticOffset}"
  local EXPORT_OUT="${4:-$VHAP_ROOT/export/monocular/${ID}_whiteBg_staticOffset_maskBelowLine}"
  local FF_VID_FILTER="${5:-scale=224:224:force_original_aspect_ratio=decrease,pad=224:224:(ow-iw)/2:(oh-ih)/2,setsar=1,fps=25}"
  mkdir -p "$WORKDIR"
  test -d "$VHAP_ROOT/data/monocular/$ID"

  echo "==== VHAP track $ID ===="
  activate_vhap
  cd "$VHAP_ROOT"
  rm -rf "$TRACK_OUT"
  python vhap/track.py --data.root-folder "$VHAP_ROOT/data/monocular" \
    --data.sequence "$ID" --data.n-downsample-rgb 2 \
    --exp.output-folder "$TRACK_OUT" --batch-size 1
  RUN_DIR=$(ls -d "$TRACK_OUT"/*/ | head -1)
  rm -rf "$EXPORT_OUT"
  python vhap/export_as_nerf_dataset.py --src-folder "$RUN_DIR" --tgt-folder "$EXPORT_OUT" \
    --background-color white --flame-mode param

  echo "==== GA train 25k VHAP $ID ===="
  activate_ga
  cd "$GA_ROOT"
  ln -sfn "$EXPORT_OUT" "data/$ID"
  rm -rf "output/${ID}_256w_25k"
  python train.py -s "data/$ID" -m "output/${ID}_256w_25k" --bind_to_mesh --white_background \
    -r 256 --sh_degree 0 --iterations 25000 --interval 5000
  python render.py -m "output/${ID}_256w_25k" --skip_val --skip_test -r 256
  fix_ga_mp4s "output/${ID}_256w_25k"

  echo "==== PSNR/LPIPS $ID ===="
  cd "$REPO_ROOT"
  activate_ga
  python scripts/static_rendering/eval_psnr_lpips.py \
    --pred_dir "$GA_ROOT/output/${ID}_256w_25k/train/ours_25000/renders" \
    --gt_dir "$EXPORT_OUT/images" \
    --output_json "$SUMMARY/${ID}_psnr_lpips.json"

  echo "==== SyncNet $ID ===="
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
    --tmp_dir "$WORKDIR/syncnet_tmp" --reference "$ID" | tee "$SUMMARY/${ID}_syncnet.txt"
  echo "VHAP_DONE $(date -Is)"
}

STAGE="${1:-all}"
case "$STAGE" in
  mica) run_mica ;;
  mica_ga) run_mica_ga ;;
  smirk) run_smirk ;;
  vhap) run_vhap ;;
  vhap_track) run_vhap_track ;;
  from_mica_ga)
    run_mica_ga
    run_smirk
    run_vhap
    ;;
  all)
    run_mica
    run_smirk
    run_vhap
    ;;
  *) echo "usage: $0 [all|mica|mica_ga|from_mica_ga|smirk|vhap|vhap_track]"; exit 1 ;;
esac
echo "==== ALL REQUESTED DONE $(date -Is) ===="
