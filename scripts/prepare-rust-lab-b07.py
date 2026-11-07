#!/usr/bin/env python3
"""Seal R5 completion tooling over the unchanged, qualified R4 implementation."""
import hashlib
import json
from pathlib import Path
import sys
import zipfile

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
DIRECTORY=ROOT/'distribution/rust-lab-b07-g1250'
RESULT='results/rust-lab-qualification/20261107T110307031706Z-20261107T110018084733Z-g1250-b07-49ed6ce1-fc892c78'
R4_SHA='e979861f03e5508dd5748a54309ad3f5e0ff540b352e435939a871bde23035c2'


def main():
    from brass_ai.learning.contracts import require
    from brass_ai.step5.league import implementation_id,implementation_sources,write_json
    archive=DIRECTORY/'rust-lab-b07-g1250-r5.zip'
    require(not archive.exists(),'Publish a new B07 revision; never reseal')
    source=DIRECTORY/'rust-lab-b07-g1250-r4.zip'
    require(hashlib.sha256(source.read_bytes()).hexdigest()==R4_SHA,'Qualified R4 archive changed')
    with zipfile.ZipFile(source) as z:blobs={n:z.read(n) for n in z.namelist()}
    manifest_name='distribution/rust-lab-b07-g1250/manifest.json'
    original=json.loads(blobs.pop(manifest_name))
    require(original['implementation']==implementation_id(),'R5 must preserve the qualified runtime implementation')
    for name,data in implementation_sources().items():
        require(data==blobs[name].decode('utf-8').replace('\r\n','\n'),
                'R5 cannot reuse ordinary qualification after runtime change: '+name)
    reference=json.loads(blobs['distribution/rust-lab-b07-g1250/reference.json'])
    q=json.loads((ROOT/RESULT/'evidence/qualification.json').read_text())
    require(q['passed'] and q['implementation']==implementation_id() and
            q['ordinary']['matched']['equal'] and q['performance']['useful_accepted_work_gain'],
            'Missing qualified useful R4 ordinary result')
    reuse=dict(result=RESULT,implementation=implementation_id(),runtime_unchanged=True,evidence={})
    for key,name in [('qualification','qualification.json'),('ordinary','ordinary-checks.json'),
                     ('snapshot','snapshot.json.gz'),('v4','v4-parity.json'),('build','ordinary-build.json')]:
        data=(ROOT/RESULT/'evidence'/name).read_bytes()
        target='b07-r4-'+name
        reuse['evidence'][key]=dict(file=target,sha256=hashlib.sha256(data).hexdigest())
        blobs['distribution/rust-lab-b07-g1250/evidence/'+target]=data
        (DIRECTORY/'evidence').mkdir(exist_ok=True)
        (DIRECTORY/'evidence'/target).write_bytes(data)
    reference['completed_ordinary']=reuse
    reference['source_policy']='R5 completion/adoption/one-iteration launcher only; exact R4 runtime, config, recipes, C# V4 and search_v1 unchanged'
    reference['native_runtime_policy']='Reuse exact passing R4 ordinary build; build/test repaired teacher only if a current-source receipt is unavailable'
    reference['b07_source_review']=json.loads((DIRECTORY/'source-review.json').read_text())
    write_json(DIRECTORY/'reference.json',reference)
    names={
        'scripts/prepare-rust-lab-b07.py','scripts/qualify-rust-lab-b07.py','scripts/manage-rust-lab-b07.py',
        'scripts/continue-rust-lab-b07.py','scripts/rust_lab_b07_completion.py',
        'tests/RustBackendLab/test_b07_completion.py',
        'ai-research/RUST_AI_BACKEND_MIGRATION_PLAN.md','ai-research/RUST_BACKEND_B07_IMPLEMENTATION.md',
        *('distribution/rust-lab-b07-g1250/'+n for n in ('README.md','reference.json','local-checks.json','source-review.json'))}
    # Reject accidental changes to any other sealed input, including benchmarks,
    # source-transition registries, tests and historical prerequisites.
    for name,expected in original['files'].items():
        path=ROOT/name
        if name in names or not path.is_file():continue
        data=path.read_bytes()
        if path.suffix in original['text']:data=data.replace(b'\r\n',b'\n')
        require(hashlib.sha256(data).hexdigest()==expected,'Unreviewed R5 bundle change: '+name)
    for name in names:
        data=(ROOT/name).read_bytes()
        blobs[name]=data.replace(b'\r\n',b'\n') if Path(name).suffix in original['text'] else data
    for path in (DIRECTORY/'evidence').glob('r5-*'):
        blobs[path.relative_to(ROOT).as_posix()]=path.read_bytes()
    manifest=dict(implementation=implementation_id(),text=original['text'],
        qualification='R4 ordinary passed; R5 remaining lab/adoption/next commit Pending',
        files={n:hashlib.sha256(v).hexdigest() for n,v in sorted(blobs.items())})
    write_json(DIRECTORY/'manifest.json',manifest)
    blobs[manifest_name]=(DIRECTORY/'manifest.json').read_bytes()
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for name,data in sorted(blobs.items()):z.writestr(name,data)
    with zipfile.ZipFile(archive) as z:require(z.testzip() is None,'Corrupt R5 archive')
    write_json(DIRECTORY/'bundle.json',dict(archive=archive.name,sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
        bytes=archive.stat().st_size,entries=len(blobs),implementation=implementation_id(),
        qualification=manifest['qualification'],ordinary_qualification_reused=True,production_modified=False))
    print(archive)


if __name__=='__main__':main()
