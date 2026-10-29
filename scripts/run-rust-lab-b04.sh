#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
if [[ $# -lt 5 || $# -gt 6 ]]; then
  echo 'Usage: bash scripts/run-rust-lab-b04.sh RUN B02_RECEIPT B03_RECEIPT EXPECTED_GENERATION CHECKPOINT_SHA256 [cuda:0]' >&2
  exit 2
fi
python scripts/qualify-rust-lab-b04.py --run "$1" --b02 "$2" --b03 "$3" \
  --expected-generation "$4" --checkpoint-sha256 "$5" --device "${6:-cuda:0}"
