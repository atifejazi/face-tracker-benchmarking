#!/usr/bin/env bash
# Official NoW evaluation via Docker (preferred by face-tracker README).
# Falls back to the now_evaluation conda env only if docker is unavailable.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
# shellcheck disable=SC1091
set -a && source "$REPO_ROOT/config/paths.env" && set +a

METHOD="${1:-}"
if [[ -z "$METHOD" ]]; then
  echo "Usage: $0 <mica|smirk|vhap> [extra args...]"
  echo "Expects meshes under \$NOW_RESULTS/<method>/predicted_meshes"
  exit 1
fi
shift || true

PRED="${NOW_RESULTS}/${METHOD}/predicted_meshes"
if [[ ! -d "$PRED" ]]; then
  echo "Missing predictions: $PRED"
  echo "Run predict_now_${METHOD}.py first."
  exit 1
fi
if [[ ! -f "${NOW_DATASET}/imagepathsvalidation.txt" ]]; then
  echo "NoW dataset not found at \$NOW_DATASET=${NOW_DATASET}"
  exit 1
fi

mkdir -p "${NOW_RESULTS}/${METHOD}/results"

# Prefer Docker (README). Use sg docker if this shell hasn't refreshed group membership yet.
docker_cmd() {
  if docker info >/dev/null 2>&1; then
    docker "$@"
  elif command -v sg >/dev/null 2>&1 && getent group docker >/dev/null 2>&1; then
    sg docker -c "docker $(printf '%q ' "$@")"
  else
    return 127
  fi
}

if command -v docker >/dev/null 2>&1 && docker_cmd info >/dev/null 2>&1; then
  if ! docker_cmd image inspect noweval >/dev/null 2>&1; then
    echo "Building noweval Docker image from \$NOW_EVAL_ROOT..."
    docker_cmd build -t noweval "$NOW_EVAL_ROOT"
  fi
  # noweval runs as uid 556 inside the image; host-owned dirs need a writable results mount.
  RESULTS_DIR="${NOW_RESULTS}/${METHOD}/results"
  mkdir -p "$RESULTS_DIR"
  chmod 777 "$RESULTS_DIR"
  # Official now_evaluation Docker flow (README): dataset + predictions; results on host.
  docker_cmd run --ipc host --rm \
    -v "${NOW_DATASET}:/dataset" \
    -v "${PRED}:/preds:ro" \
    -v "${RESULTS_DIR}:/results" \
    noweval \
    --method_identifier "$METHOD" \
    --error_out_path /results \
    --nproc "${NOW_EVAL_NPROC:-1}" \
    "$@"
  echo "Results: ${RESULTS_DIR}"
  exit 0
fi

echo "WARNING: docker not usable; falling back to native now_evaluation conda env."
# shellcheck disable=SC1091
source "${HOME}/miniconda3/etc/profile.d/conda.sh"
conda activate now_evaluation
cd "$NOW_EVAL_ROOT"
python compute_error.py \
  --predicted_mesh_folder "$PRED" \
  --dataset_folder "$NOW_DATASET" \
  --method_identifier "$METHOD" \
  --error_out_path "${NOW_RESULTS}/${METHOD}/results" \
  --nproc "${NOW_EVAL_NPROC:-1}" \
  "$@"
echo "Results: ${NOW_RESULTS}/${METHOD}/results"
