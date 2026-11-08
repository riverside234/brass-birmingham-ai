"""Resolve historical native runtime paths without rewriting signed receipts."""
from pathlib import Path

from brass_ai.learning.contracts import SOURCE_ROOT
from brass_ai.learning.paths import read_relocation, resolve_project_path


def runtime_path(reference):
    if read_relocation(SOURCE_ROOT) is None:
        return Path(reference)
    return resolve_project_path(SOURCE_ROOT, reference)


def same_pin(left,right):
    return (left.get('bytes')==right.get('bytes') and left['sha256']==right['sha256'] and
            runtime_path(left['path']).resolve()==runtime_path(right['path']).resolve())
