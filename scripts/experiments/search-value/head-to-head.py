#!/usr/bin/env python3
"""Run the Rust CPU arena with main Utility versus corrected Search Value."""

from pathlib import Path
from runpy import run_path


if __name__ == '__main__':
    arena = Path(__file__).resolve().parents[1] / 'c-scale-head-to-head' / 'run.py'
    run_path(str(arena))['main'](compare_weights=True)
