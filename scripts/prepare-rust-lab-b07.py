#!/usr/bin/env python3
"""Seal immutable B07 code/evidence supplement over the qualified B06 inputs."""
import hashlib
import json
from pathlib import Path
import sys
import zipfile

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
DIRECTORY=ROOT/'distribution/rust-lab-b07-g1250'
B06='results/rust-lab-qualification/20261107T080012822445Z-20261107T075601452661Z-g1250-b06-c1cd94b2-2c29d5e3/evidence'
TEXT={'.py','.rs','.cs','.md','.toml','.lock','.ps1','.sh','.txt'}


def main():
    from brass_ai.learning.contracts import require
    from brass_ai.step5.league import implementation_id,implementation_sources,write_json
    DIRECTORY.mkdir(parents=True,exist_ok=True)
    archive=DIRECTORY/'rust-lab-b07-g1250-r2.zip';require(not archive.exists(),'Publish a new B07 revision; never reseal')
    q=json.loads((ROOT/B06/'qualification.json').read_text());require(q['passed'],'B06 correctness gate missing')
    prior=json.loads((ROOT/'distribution/rust-lab-b06-g1250/bundle.json').read_text())
    source=ROOT/'distribution/rust-lab-b06-g1250'/prior['archive']
    require(hashlib.sha256(source.read_bytes()).hexdigest()==prior['sha256'],'B06 archive changed')
    with zipfile.ZipFile(source) as z:blobs={n:z.read(n) for n in z.namelist()}
    reference=json.loads(blobs['distribution/rust-lab-b06-g1250/reference.json'])
    reference['b06']=dict(path=B06+'/qualification.json',sha256=hashlib.sha256((ROOT/B06/'qualification.json').read_bytes()).hexdigest())
    reference['b06_baseline']=q['performance']
    reference['b07_source_review']=json.loads((DIRECTORY/'source-review.json').read_text())
    reference['source_policy']='B07 process isolation, single writer admission and byte-attested read reuse; explicit backend-only adoption with teacher off'
    write_json(DIRECTORY/'reference.json',reference)
    names=set(implementation_sources())
    for folder,extensions in [('src',{'.py','.cs','.json','.csproj'}),('configs',{'.json'}),('scripts',{'.py','.sh'}),
            ('tests/RustBackendLab',{'.py'}),('tests/Recovery',{'.py'})]:
        names.update(p.relative_to(ROOT).as_posix() for p in (ROOT/folder).rglob('*') if p.is_file()
            and p.suffix in extensions and not any(x in p.parts for x in ('__pycache__','obj','bin')))
    names.update('distribution/rust-lab-b07-g1250/'+n for n in ('README.md','reference.json','local-checks.json','source-review.json'))
    names.update(['ai-research/RUST_AI_BACKEND_MIGRATION_PLAN.md','ai-research/RUST_BACKEND_B07_IMPLEMENTATION.md',
                  'ai-research/RUST_BACKEND_B06_IMPLEMENTATION.md','distribution/rust-lab-g1250/inputs.zip'])
    for name in ('qualification.json','ordinary-checks.json','teacher-checks.json','kernels-checks.json'):
        target='distribution/rust-lab-b07-g1250/evidence/b06-'+name
        blobs[target]=(ROOT/B06/name).read_bytes()
    for name in names:
        path=ROOT/name;data=path.read_bytes();blobs[name]=data.replace(b'\r\n',b'\n') if path.suffix in TEXT else data
    # The old B06 manifest remains archival; the B07 launcher verifies this
    # complete overlay, including newly added files and the exact frozen inputs.
    manifest=dict(implementation=implementation_id(),text=sorted(TEXT),qualification='Pending',
        files={n:hashlib.sha256(v).hexdigest() for n,v in sorted(blobs.items())})
    write_json(DIRECTORY/'manifest.json',manifest)
    blobs['distribution/rust-lab-b07-g1250/manifest.json']=(DIRECTORY/'manifest.json').read_bytes()
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for name,value in sorted(blobs.items()):z.writestr(name,value)
    with zipfile.ZipFile(archive) as z:require(z.testzip() is None,'Corrupt B07 archive')
    write_json(DIRECTORY/'bundle.json',dict(archive=archive.name,sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
        bytes=archive.stat().st_size,entries=len(blobs),implementation=implementation_id(),qualification='Pending'))
    print(archive)


if __name__=='__main__':main()
