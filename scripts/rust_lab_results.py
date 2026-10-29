"""Portable review copies of lab evidence; original receipts remain unchanged."""
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import uuid

ROOT = Path(__file__).resolve().parents[1]
RESULTS = Path('results/rust-lab-qualification')
SKIP_DIRS = {'target', 'checkout', 'inputs', 'bootstrap', '__pycache__', '.git',
             'payload', '.prepared', '.ppg-eight-cache', '.value-cache'}
TEXT = {'.json', '.log', '.txt', '.md', '.csv'}


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def export(evidence, *, root=ROOT):
    root = Path(root).resolve(); evidence = Path(evidence).resolve()
    base = root/'artifacts/rust-backend'
    if not evidence.is_relative_to(base) or evidence == base or not evidence.is_dir():
        raise ValueError('Export a specific existing artifacts/rust-backend directory')
    name = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'-'+evidence.name+'-'+uuid.uuid4().hex[:8]
    destination = root/RESULTS/name
    destination.mkdir(parents=True, exist_ok=False)
    files = []; omitted = []
    # Prune build trees before traversal. Never follow symlinks out of evidence.
    for current, directories, names in os.walk(evidence, followlinks=False):
        current = Path(current)
        for directory in list(directories):
            path = current/directory
            if directory in SKIP_DIRS or path.is_symlink():
                directories.remove(directory)
                omitted.append(dict(source=path.relative_to(evidence).as_posix(), reason='build/runtime/payload directory or symlink'))
        for filename in sorted(names):
            path = current/filename; relative = path.relative_to(evidence)
            if path.is_symlink() or path.suffix.lower() not in TEXT:
                omitted.append(dict(source=relative.as_posix(), reason='non-report file or symlink'))
                continue
            before = (path.stat().st_size, sha(path))
            target = destination/'evidence'/relative
            compressed = before[0] >= 1024*1024
            if compressed: target = target.with_name(target.name+'.gz')
            target.parent.mkdir(parents=True, exist_ok=True)
            if compressed:
                with path.open('rb') as src, target.open('wb') as raw, gzip.GzipFile(fileobj=raw, mode='wb', mtime=0, filename='') as dst:
                    shutil.copyfileobj(src, dst)
            else:
                shutil.copyfile(path, target)
            if before != (path.stat().st_size, sha(path)):
                raise ValueError('Evidence changed during export: '+str(relative))
            files.append(dict(source=relative.as_posix(), path=target.relative_to(destination).as_posix(),
                source_bytes=before[0], source_sha256=before[1], bytes=target.stat().st_size,
                sha256=sha(target), gzip=compressed))
    # All paths in this index are relative to the export. Original receipt bytes
    # retain lab paths/hashes; this review copy cannot authorize another host.
    manifest = dict(contract='brass-lab-results/1.0.0', created_utc=datetime.now(timezone.utc).isoformat(),
        source=evidence.relative_to(root).as_posix(), source_absolute=str(evidence),
        review_only=True, files=files, omitted=omitted)
    write(destination/'export-manifest.json', manifest)
    (destination/'README.md').write_text(
        '# Lab qualification results\n\nOpen `evidence/runner.json` first (when present), then each stage\'s '
        '`qualification.json` and logs. Large reports are gzip compressed.\n\n'
        '`export-manifest.json` maps original evidence paths to portable files and hashes. '
        'Original receipts are byte-for-byte copies, including lab-local paths. '
        'This is review evidence, not a relocated executable qualification. Build products, '
        'models, replay payloads and runtime dependencies stay on the lab machine.\n', encoding='utf-8')
    return destination


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('evidence', type=Path)
    print(export(parser.parse_args().evidence))
