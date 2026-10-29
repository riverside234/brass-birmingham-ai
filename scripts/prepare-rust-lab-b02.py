#!/usr/bin/env python3
"""Create a source-only B02 transfer ZIP; never qualify, train or deploy."""
import hashlib
import json
from pathlib import Path
import sys
import zipfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from brass_ai.rust_backend.lab_contract import METHOD,SOURCE,sha

def main():
    names=['scripts/qualify-rust-lab-b02.py','scripts/run-rust-lab-b02.sh','scripts/run-rust-lab-b01-b02.sh','scripts/prepare-rust-lab-b02.py',
        'src/BrassLabBootstrap/Bootstrap.cs','src/brass_ai/recovery/local_scoring_targets.py',
        'src/brass_ai/rust_backend/lab_collection.py','src/brass_ai/rust_backend/lab_transport.py',
        'src/brass_ai/rust_backend/lab_families.py','src/brass_ai/rust_backend/lab_jobs.py',
        'src/brass_ai/rust_backend/lab_b02_smoke.py','tests/RustBackendLab/test_b02.py',
        'ai-research/RUST_BACKEND_B01_REPORT.md','ai-research/RUST_BACKEND_B02_REPORT.md',
        'ai-research/RUST_AI_BACKEND_MIGRATION_PLAN.md']
    names += [p.relative_to(ROOT).as_posix() for p in sorted((ROOT/'lab-native').rglob('*'))
              if p.is_file() and p.suffix in ('.rs','.toml','.lock') and 'target' not in p.parts]
    files={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in names}
    out=ROOT/'distribution/rust-lab-b02';out.mkdir(parents=True,exist_ok=True)
    manifest=dict(contract='brass-lab-b02-transfer/2.0.0',files=files,b01_qualification='Pending',
        b02_qualification='Pending',method=METHOD,v205_enabled=False,shared_core_source=SOURCE,
        b01_inputs_sha256=sha(ROOT/'distribution/rust-lab-b01/b01-search-v1-inputs.zip'),
        required_b01_transfer_sha256=sha(ROOT/'distribution/rust-lab-b01/b01-search-v1-update.zip'),
        scope='Additive B02 sources only; requires refreshed search_v1 B01 and a passing lab receipt')
    path=out/'b02-search-v1-update.zip'
    with zipfile.ZipFile(path,'x',zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        for name in sorted(names):z.writestr(name,(ROOT/name).read_bytes())
        z.writestr('distribution/rust-lab-b02/search-v1-transfer-manifest.json',json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(dict(path=str(path),bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest()),indent=2))

if __name__=='__main__':main()
