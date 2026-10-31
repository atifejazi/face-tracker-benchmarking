#!/usr/bin/env bash
# One-command dependency preparation; safe to rerun.
set -euo pipefail
REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ ! -f "$REPO_DIR/config/paths.env" ]]; then
  echo 'Copy config/paths.example.env to config/paths.env and configure ROOTDIR first.' >&2
  exit 1
fi
set -a
source "$REPO_DIR/config/paths.env"
set +a
exec "${PYTHON:-python3}" "$REPO_DIR/scripts/setup_dependencies.py" "$@"
