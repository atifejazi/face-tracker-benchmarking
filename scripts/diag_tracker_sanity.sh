#!/usr/bin/env bash
# Track-only sanity check: MICA / SMIRK / VHAP on selected NeRSemble clips,
# then stage GA-format flame_param (and raw SMIRK/VHAP params) for comparison
# against data/past_nersemble_tracked.
#
# Usage:
#   SUBJECTS="017 018 037" SENS="SEN-01 SEN-05 SEN-09" \
#     bash scripts/diag_tracker_sanity.sh [all|mica|smirk|vhap|compare]
set -eo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
set -a && source "$ROOT/config/paths.env" && set +a
source "$HOME/miniconda3/etc/profile.d/conda.sh"

_activate_torch_env() {
  conda activate "$1"
  local _torch_lib
  _torch_lib="$(python -c 'import torch, os; print(os.path.join(os.path.dirname(torch.__file__), "lib"))')"
  export LD_LIBRARY_PATH="${_torch_lib}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
}
activate_ga() { _activate_torch_env gaussian-avatars; }
activate_vhap() {
  _activate_torch_env VHAP
  export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.8}"
  export PATH="$CUDA_HOME/bin${PATH:+:$PATH}"
}

OUT_ROOT="${OUT_ROOT:-$ROOT/runs/tracker_sanity}"
PAST_ROOT="${PAST_ROOT:-$ROOTDIR/data/past_nersemble_tracked}"
CAM="${CAM:-cam_222200037.mp4}"
SUBJECTS="${SUBJECTS:-017 018 037}"
SENS_SHORT="${SENS:-SEN-01 SEN-05 SEN-09}"
LOGDIR="$OUT_ROOT/logs"
mkdir -p "$LOGDIR" "$OUT_ROOT"

resolve_sen() {
  local subject="$1" short="$2"
  local d
  d=$(ls -d "$NERSSEMBLE_DATA/$subject/sequences/${short}-"* 2>/dev/null | head -1)
  [[ -n "$d" ]] || { echo "missing sequence $subject $short" >&2; return 1; }
  basename "$d"
}

stage_mica() {
  local subject="$1" sen_short="$2" sen_full="$3"
  local sen_num="${sen_short#SEN-}"
  local ID="ns${subject}_SEN${sen_num}_mica"
  local WORKDIR="$OUT_ROOT/work/$ID"
  local STAGE="$OUT_ROOT/mica/$subject/$sen_short"
  local CFG="$METRICAL_TRACKER_ROOT/configs/actors/${ID}.yml"
  local MP4="$NERSSEMBLE_DATA/$subject/sequences/$sen_full/images/$CAM"
  mkdir -p "$WORKDIR" "$STAGE" "$METRICAL_TRACKER_ROOT/input/$ID" "$LOGDIR"
  test -f "$MP4"

  echo "==== MICA track $ID ($(date -Is)) ===="
  # Fresh track: wipe prior metrical/MICA demo state for this ID.
  rm -rf "$METRICAL_TRACKER_ROOT/input/$ID" "$METRICAL_TRACKER_ROOT/output/$ID"
  rm -rf "$MICA_ROOT/demo/input_$ID" "$MICA_ROOT/demo/arcface_$ID" "$MICA_ROOT/demo/output_$ID"
  mkdir -p "$METRICAL_TRACKER_ROOT/input/$ID"
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

  conda activate tracker
  mkdir -p "$MICA_ROOT/demo/input_$ID" "$MICA_ROOT/demo/arcface_$ID" "$MICA_ROOT/demo/output_$ID"
  ffmpeg -y -i "$MP4" -vf "select=eq(n\,0)" -frames:v 1 -update 1 "$MICA_ROOT/demo/input_$ID/${ID}.png"
  cd "$MICA_ROOT"
  python demo.py -i "demo/input_$ID" -a "demo/arcface_$ID" -o "demo/output_$ID" -m data/pretrained/mica.tar
  cp "$(find demo/output_$ID -name identity.npy | head -1)" "$METRICAL_TRACKER_ROOT/input/$ID/identity.npy"

  cd "$METRICAL_TRACKER_ROOT"
  python tracker.py --cfg "./configs/actors/${ID}.yml"

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

  conda activate tracker
  cd "$GA_ROOT"
  TGT="data/${ID}_sanity"
  rm -rf "$TGT"
  python "$REPO_ROOT/scripts/export_metrical_tracker_to_ga.py" \
    --track-out "$METRICAL_TRACKER_ROOT/output/$ID" \
    --imgs-dir "$IMG_DST" --alpha-dir "$ALPHA_DST" --tgt-dir "$TGT"

  rm -rf "$STAGE"
  mkdir -p "$STAGE"
  cp -a "$GA_ROOT/$TGT/flame_param" "$STAGE/"
  cp -f "$GA_ROOT/$TGT/canonical_flame_param.npz" "$STAGE/"
  # Keep GA dataset path for SMIRK canvas.
  ln -sfn "$GA_ROOT/$TGT" "$WORKDIR/ga_dataset"
  echo "MICA_STAGED $STAGE"
}

stage_smirk() {
  local subject="$1" sen_short="$2"
  local sen_num="${sen_short#SEN-}"
  local ID="ns${subject}_SEN${sen_num}_smirk"
  local MICA_ID="ns${subject}_SEN${sen_num}_mica"
  local MICA_DS="$GA_ROOT/data/${MICA_ID}_sanity"
  local WORKDIR="$OUT_ROOT/work/$ID"
  local STAGE="$OUT_ROOT/smirk/$subject/$sen_short"
  local TRACK_PT="$WORKDIR/track_params.pt"
  mkdir -p "$WORKDIR" "$STAGE"
  test -d "$MICA_DS/images"

  echo "==== SMIRK track $ID ($(date -Is)) ===="
  conda activate smirk
  cd "$SMIRK_ROOT"
  PYTHONPATH=. python "$REPO_ROOT/scripts/static_rendering/track_folder_for_ga.py" \
    --image-dir "$MICA_DS/images" --checkpoint pretrained_models/SMIRK_em1.pt --output "$TRACK_PT"

  activate_ga
  cd "$GA_ROOT"
  TGT="data/${ID}_sanity"
  rm -rf "$TGT"
  # Default inherit-MICA export (no opencv-y-flip) for tracker-vs-past compare.
  python "$REPO_ROOT/scripts/export_smirk_to_ga.py" --track-params "$TRACK_PT" \
    --canonical-dataset "$MICA_DS" --target "$TGT"

  rm -rf "$STAGE"
  mkdir -p "$STAGE"
  cp -a "$GA_ROOT/$TGT/flame_param" "$STAGE/"
  cp -f "$GA_ROOT/$TGT/canonical_flame_param.npz" "$STAGE/"
  cp -f "$TRACK_PT" "$STAGE/track_params.pt"
  echo "SMIRK_STAGED $STAGE"
}

stage_vhap() {
  local subject="$1" sen_short="$2" sen_full="$3"
  local sen_num="${sen_short#SEN-}"
  local ID="ns${subject}_SEN${sen_num}_vhap"
  local WORKDIR="$OUT_ROOT/work/$ID"
  local STAGE="$OUT_ROOT/vhap/$subject/$sen_short"
  local MP4="$NERSSEMBLE_DATA/$subject/sequences/$sen_full/images/$CAM"
  local VHAP_IN="$VHAP_ROOT/data/monocular/${ID}_sanity.mp4"
  local SEQ="${ID}_sanity"
  local TRACK_OUT="$VHAP_ROOT/output/monocular/${SEQ}_whiteBg_staticOffset"
  local EXPORT_OUT="$VHAP_ROOT/export/monocular/${SEQ}_whiteBg_staticOffset_maskBelowLine"
  mkdir -p "$WORKDIR" "$STAGE" "$VHAP_ROOT/data/monocular"
  test -f "$MP4"

  echo "==== VHAP track $ID ($(date -Is)) ===="
  cp -f "$MP4" "$VHAP_IN"
  activate_vhap
  cd "$VHAP_ROOT"
  rm -rf "$VHAP_ROOT/data/monocular/$SEQ"
  python vhap/preprocess_video.py --input "$VHAP_IN" --downsample-scales 2 \
    --matting-method robust_video_matting --rvm-downsample-ratio 0.125
  rm -rf "$TRACK_OUT"
  python vhap/track.py --data.root-folder "$VHAP_ROOT/data/monocular" \
    --data.sequence "$SEQ" --data.n-downsample-rgb 2 \
    --exp.output-folder "$TRACK_OUT" --batch-size 1
  RUN_DIR=$(ls -d "$TRACK_OUT"/*/ | head -1)
  rm -rf "$EXPORT_OUT"
  python vhap/export_as_nerf_dataset.py --src-folder "$RUN_DIR" --tgt-folder "$EXPORT_OUT" \
    --background-color white --flame-mode param

  rm -rf "$STAGE"
  mkdir -p "$STAGE"
  cp -a "$EXPORT_OUT/flame_param" "$STAGE/"
  cp -f "$EXPORT_OUT/canonical_flame_param.npz" "$STAGE/"
  # Prefer highest-epoch tracked_flame_params_*.npz
  TRACKED=$(ls -1 "$RUN_DIR"/tracked_flame_params_*.npz 2>/dev/null | sort -t_ -k4 -n | tail -1)
  if [[ -n "$TRACKED" ]]; then
    cp -f "$TRACKED" "$STAGE/tracked_flame_params.npz"
  fi
  echo "VHAP_STAGED $STAGE"
}

run_all_mica() {
  for subject in $SUBJECTS; do
    for sen_short in $SENS_SHORT; do
      sen_full=$(resolve_sen "$subject" "$sen_short")
      stage_mica "$subject" "$sen_short" "$sen_full" \
        2>&1 | tee -a "$LOGDIR/mica_${subject}_${sen_short}.log"
    done
  done
}

run_all_smirk() {
  for subject in $SUBJECTS; do
    for sen_short in $SENS_SHORT; do
      stage_smirk "$subject" "$sen_short" \
        2>&1 | tee -a "$LOGDIR/smirk_${subject}_${sen_short}.log"
    done
  done
}

run_all_vhap() {
  for subject in $SUBJECTS; do
    for sen_short in $SENS_SHORT; do
      sen_full=$(resolve_sen "$subject" "$sen_short")
      stage_vhap "$subject" "$sen_short" "$sen_full" \
        2>&1 | tee -a "$LOGDIR/vhap_${subject}_${sen_short}.log"
    done
  done
}

run_compare() {
  conda activate tracker
  cd "$ROOT"
  python scripts/diag_compare_tracks.py \
    --new-root "$OUT_ROOT" \
    --past-root "$PAST_ROOT" \
    --subjects $SUBJECTS \
    --sens $SENS_SHORT \
    --output-json "$OUT_ROOT/compare_report.json" \
    --output-md "$OUT_ROOT/compare_report.md"
  echo "COMPARE_DONE $OUT_ROOT/compare_report.md"
}

STAGE="${1:-all}"
echo "==== tracker sanity START $(date -Is) OUT=$OUT_ROOT STAGE=$STAGE ===="
case "$STAGE" in
  mica) run_all_mica ;;
  smirk) run_all_smirk ;;
  vhap) run_all_vhap ;;
  compare) run_compare ;;
  all)
    run_all_mica
    run_all_smirk
    run_all_vhap
    run_compare
    ;;
  *) echo "usage: $0 [all|mica|smirk|vhap|compare]"; exit 1 ;;
esac
echo "==== tracker sanity DONE $(date -Is) ===="
