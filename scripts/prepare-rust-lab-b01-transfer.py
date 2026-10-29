#!/usr/bin/env python3
"""Seal the search_v1 B01 source/input transfer under distribution/; no lab run."""
import json
from pathlib import Path
import sys
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from brass_ai.learning.contracts import read_json
from brass_ai.rust_backend.lab_contract import METHOD, SOURCE, extract_new, native_fingerprint, sha, verify_inputs


def main():
    inputs = ROOT / 'distribution/rust-lab-b01/b01-search-v1-inputs.zip'
    # Packaging checks only: validate the sealed source and local evidence chain.
    with tempfile.TemporaryDirectory(dir=ROOT / 'artifacts', prefix='b01-transfer-') as temp:
        temp = Path(temp)
        extract_new(inputs, temp / 'inputs')
        verify_inputs(temp / 'inputs', read_json(ROOT / 'configs/search/lab-search-v1.json'))
        extract_new(temp / 'inputs/source.zip', temp / 'checkout')
        assert native_fingerprint(temp / 'checkout/native') == SOURCE
    names = [
        'src/brass_ai/rust_backend/lab_contract.py', 'src/brass_ai/rust_backend/lab_inference.py',
        'src/brass_ai/rust_backend/lab_smoke.py', 'src/brass_ai/rust_backend/lab_snapshot.py',
        'configs/search/lab-search-v1.json', 'scripts/prepare-rust-lab-b01.py',
        'scripts/prepare-rust-lab-b01-transfer.py', 'scripts/qualify-rust-lab.py',
        'scripts/run-rust-lab-b01.sh', 'tests/RustBackendLab/test_b01.py',
        'ai-research/RUST_BACKEND_B01_REPORT.md', 'ai-research/SEARCH_V1_DEFAULT_UPDATE.md',
        inputs.relative_to(ROOT).as_posix()]
    files = {name: sha(ROOT / name) for name in names}
    manifest = dict(contract='brass-lab-b01-transfer/2.0.0', method=METHOD, v205_enabled=False,
                    shared_core_source=SOURCE, b01_inputs_sha256=sha(inputs), files=files,
                    b01_qualification='Pending', scope='Source/input transfer; no lab qualification or deployment')
    out = ROOT / 'distribution/rust-lab-b01/b01-search-v1-update.zip'
    with zipfile.ZipFile(out, 'x', zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name in sorted(names):
            archive.writestr(name, (ROOT / name).read_bytes())
        archive.writestr('distribution/rust-lab-b01/search-v1-transfer-manifest.json', json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(dict(path=str(out), bytes=out.stat().st_size, sha256=sha(out)), indent=2))


if __name__ == '__main__':
    main()
