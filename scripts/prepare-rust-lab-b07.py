#!/usr/bin/env python3
"""Seal R6 teacher finalization/deadline repair; preserve passing ordinary work."""
import hashlib
import json
from pathlib import Path
import sys
import zipfile

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
DIRECTORY=ROOT/'distribution/rust-lab-b07-g1250'
R5_SHA='97e5b6535e6e4c9310db4b1bad782ef089df305f3af44937557da532cd340f33'


def main():
    from brass_ai.learning.contracts import require
    from brass_ai.step5.league import implementation_id,implementation_sources,write_json
    from rust_lab_b07_completion import reviewed_source,COMPLETION_CHANGES
    archive=DIRECTORY/'rust-lab-b07-g1250-r6.zip'
    require(not archive.exists(),'Publish a new B07 revision; never reseal')
    source=DIRECTORY/'rust-lab-b07-g1250-r5.zip'
    require(hashlib.sha256(source.read_bytes()).hexdigest()==R5_SHA,'Immutable R5 archive changed')
    with zipfile.ZipFile(source) as z:blobs={n:z.read(n) for n in z.namelist()}
    manifest_name='distribution/rust-lab-b07-g1250/manifest.json'
    original=json.loads(blobs.pop(manifest_name));review=reviewed_source()
    require(review['from_implementation']==original['implementation'],'Unexpected review baseline')
    changes={}
    for name,data in implementation_sources().items():
        before=blobs[name].replace(b'\r\n',b'\n');after=data.encode()
        if before!=after:changes[name]=dict(before=hashlib.sha256(before).hexdigest(),after=hashlib.sha256(after).hexdigest())
    require(changes==review['changes'],'Unreviewed implementation change')
    # The only native changes are compiled with feature teacher. Preserve every
    # ordinary dependency byte and the exact qualified executable receipt.
    require('#[cfg(feature = "teacher")]\npub mod teacher;' in (ROOT/'lab-native/src/lib.rs').read_text(),
            'Teacher module no longer excluded from ordinary build')
    require((ROOT/'lab-native/tests/teacher.rs').read_text().startswith('#![cfg(feature = "teacher")]'),
            'Teacher tests no longer feature gated')
    names={*COMPLETION_CHANGES,
        'configs/learning/selfplay-resume-sources.json',
        'scripts/prepare-rust-lab-b07.py','scripts/qualify-rust-lab-b07.py',
        'scripts/rust_lab_b07_completion.py','scripts/rust_lab_b07_retry.py',
        'scripts/rust_lab_b07_smoke.py','scripts/rust_lab_b07_sealing.py',
        'tests/RustBackendLab/test_b07_completion.py','tests/RustBackendLab/test_b07_sealing.py',
        'tests/RustBackendLab/test_b045_failures.py',
        'ai-research/RUST_AI_BACKEND_MIGRATION_PLAN.md','ai-research/RUST_BACKEND_B07_IMPLEMENTATION.md',
        'ai-research/RUST_BACKEND_B045_SEARCH_VALUE_PLAN.md',
        *('distribution/rust-lab-b07-g1250/'+n for n in ('README.md','reference.json','local-checks.json','source-review.json'))}
    # Keep all unrelated inputs exactly as sealed; no broader qualification rerun.
    for name,expected in original['files'].items():
        path=ROOT/name
        if name in names or not path.is_file():continue
        data=path.read_bytes()
        if path.suffix in original['text']:data=data.replace(b'\r\n',b'\n')
        require(hashlib.sha256(data).hexdigest()==expected,'Unreviewed R6 bundle change: '+name)
    for name in names:
        data=(ROOT/name).read_bytes()
        blobs[name]=data.replace(b'\r\n',b'\n') if Path(name).suffix in original['text'] else data
    for pattern in ('b07-r5-*','r6-*'):
        for path in (DIRECTORY/'evidence').glob(pattern):blobs[path.relative_to(ROOT).as_posix()]=path.read_bytes()
    manifest=dict(implementation=implementation_id(),text=original['text'],
        qualification='R4 ordinary passed and unchanged; R6 teacher deadline/canaries/continuation/adoption Pending',
        files={n:hashlib.sha256(v).hexdigest() for n,v in sorted(blobs.items())})
    write_json(DIRECTORY/'manifest.json',manifest)
    blobs[manifest_name]=(DIRECTORY/'manifest.json').read_bytes()
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for name,data in sorted(blobs.items()):z.writestr(name,data)
    with zipfile.ZipFile(archive) as z:require(z.testzip() is None,'Corrupt R6 archive')
    write_json(DIRECTORY/'bundle.json',dict(archive=archive.name,sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
        bytes=archive.stat().st_size,entries=len(blobs),implementation=implementation_id(),
        qualification=manifest['qualification'],ordinary_qualification_reused=True,teacher_rebuild_required=True,
        production_modified=False))
    print(archive)


if __name__=='__main__':main()
