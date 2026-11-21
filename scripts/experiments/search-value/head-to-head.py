#!/usr/bin/env python3
"""Compare two search residual weights in the Rust CPU arena."""

from pathlib import Path
from runpy import run_path


if __name__ == '__main__':
    arena = Path(__file__).resolve().parents[1] / 'c-scale-head-to-head' / 'run.py'
    run_path(str(arena))['main'](compare_weights=True)
