"""Offline discovery versions are independent of Step 5 and P0/P1 checkpoints."""
from pathlib import Path
from brass_ai.learning.contracts import digest

VERSION='personality-repertoire/1.0.0'


def identity():
    return dict(version=VERSION,source=digest({p.name:p.read_text(encoding='utf-8')
        for p in sorted(Path(__file__).parent.glob('*.py'))}))
