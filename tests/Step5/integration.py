"""Compatibility entry point for the active fresh-imitation/PPO integration check."""
from pathlib import Path
import runpy


def run(root, mono=None):
    checks=runpy.run_path(str(Path(__file__).resolve().parents[1]/'Recovery/integration.py'))
    return checks['run'](root,mono=mono)
