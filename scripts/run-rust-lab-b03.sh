#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
if [[ $# -lt 4 || $# -gt 5 ]]; then
  echo 'Usage: bash scripts/run-rust-lab-b03.sh RUN B02_RECEIPT EXPECTED_GENERATION CHECKPOINT_SHA256 [cuda:0]' >&2
  exit 2
fi
python scripts/qualify-rust-lab-b03.py --run "$1" --b02 "$2" --expected-generation "$3" --checkpoint-sha256 "$4" --device "${5:-cuda:0}"
