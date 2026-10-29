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
  --bundle artifacts/rust-backend/20261028-b01/lab/b01-inputs.zip \
  --bundle-sha256 fc93c4d369ac87a53c07f430a1563070ce74980271086a52993067f5d9edcb72 \
  --out "artifacts/rust-backend/$(date -u +%Y%m%d-%H%M%S)-b01/lab"
