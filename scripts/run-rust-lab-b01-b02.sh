#!/usr/bin/env bash
# Keep training paused throughout. B02 starts only after B01 exits successfully.
set -euo pipefail
if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo 'Usage: bash scripts/run-rust-lab-b01-b02.sh artifacts/step5/YOUR_RUN [cuda:0]' >&2
  exit 2
fi
repo_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd -- "$repo_dir"
run_dir=$1
device=${2:-cuda:0}
evidence="artifacts/rust-backend/$(date -u +%Y%m%d-%H%M%S)-b01-b02-search-v1"
printf 'Qualification evidence: %s\n' "$evidence"
python scripts/qualify-rust-lab.py \
  --run "$run_dir" --device "$device" \
  --profile configs/search/lab-search-v1.json \
  --bundle distribution/rust-lab-b01/b01-search-v1-inputs.zip \
  --bundle-sha256 935443a477d3a52425923e3d7458a15ae8250f4296a6900c5b9088fbef864a76 \
  --out "$evidence/b01/lab"
python scripts/qualify-rust-lab-b02.py \
  --run "$run_dir" --b01 "$evidence/b01/lab/qualification.json" \
  --device "$device" --smoke-timeout 240 --out "$evidence/b02/lab"
printf 'B01: %s\nB02: %s\n' "$evidence/b01/lab/qualification.json" "$evidence/b02/lab/qualification.json"
