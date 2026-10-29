#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo 'Usage: bash scripts/run-rust-lab-b01.sh artifacts/step5/YOUR_RUN [cuda:0 cuda:1 ...]' >&2
  exit 2
fi
repo_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd -- "$repo_dir"
run_dir=$1
shift
if [[ $# -eq 0 ]]; then set -- cuda:0; fi
device_args=()
for device in "$@"; do device_args+=(--device "$device"); done

exec python scripts/qualify-rust-lab.py \
  --run "$run_dir" "${device_args[@]}" \
  --profile configs/search/lab-search-v1.json \
  --bundle distribution/rust-lab-b01/b01-search-v1-inputs.zip \
  --bundle-sha256 935443a477d3a52425923e3d7458a15ae8250f4296a6900c5b9088fbef864a76 \
  --out "artifacts/rust-backend/$(date -u +%Y%m%d-%H%M%S)-b01-search-v1/lab"
