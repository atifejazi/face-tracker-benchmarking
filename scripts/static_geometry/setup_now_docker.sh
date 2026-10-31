#!/usr/bin/env bash
# Run once with: bash scripts/static_geometry/setup_now_docker.sh
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
set -a && source "$REPO_ROOT/config/paths.env" && set +a

sudo systemctl enable --now docker
sudo usermod -aG docker "$USER"
echo "Building noweval image (this can take a while)..."
sudo docker build -t noweval "$NOW_EVAL_ROOT"
echo
echo "Docker image 'noweval' ready."
echo "Log out/in (or: newgrp docker) so 'docker' works without sudo, then:"
echo "  bash scripts/static_geometry/run_now_eval.sh mica"
