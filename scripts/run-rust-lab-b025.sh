#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo 'Usage: bash scripts/run-rust-lab-b025.sh RUN EXPECTED_GENERATION [cpu|cuda:0]' >&2
  exit 2
fi
python scripts/qualify-rust-lab-b025.py --run "$1" --expected-generation "$2" --device "${3:-cpu}"
