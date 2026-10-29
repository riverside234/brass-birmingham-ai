#!/usr/bin/env bash
set -euo pipefail
if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo 'Usage: bash scripts/run-rust-lab-b02.sh artifacts/step5/YOUR_RUN path/to/B01/lab/qualification.json [cuda:0]' >&2
  exit 2
fi
repo_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd -- "$repo_dir"
exec python scripts/qualify-rust-lab-b02.py --run "$1" --b01 "$2" --device "${3:-cuda:0}" \
  --out "artifacts/rust-backend/$(date -u +%Y%m%d-%H%M%S)-b02/lab"
