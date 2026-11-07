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
B07='results/rust-lab-qualification/20261107T090047266944Z-20261107T085645018439Z-g1250-b07-d0097dac-faa1535f/evidence'
TEXT={'.py','.rs','.cs','.md','.toml','.lock','.ps1','.sh','.txt'}


def main():
    from brass_ai.learning.contracts import require
    from brass_ai.step5.league import implementation_id,implementation_sources,write_json
    DIRECTORY.mkdir(parents=True,exist_ok=True)
    archive=DIRECTORY/'rust-lab-b07-g1250-r3.zip';require(not archive.exists(),'Publish a new B07 revision; never reseal')
    q=json.loads((ROOT/B06/'qualification.json').read_text());require(q['passed'],'B06 correctness gate missing')
    prior=json.loads((ROOT/'distribution/rust-lab-b06-g1250/bundle.json').read_text())
    source=ROOT/'distribution/rust-lab-b06-g1250'/prior['archive']
    require(hashlib.sha256(source.read_bytes()).hexdigest()==prior['sha256'],'B06 archive changed')
    with zipfile.ZipFile(source) as z:blobs={n:z.read(n) for n in z.namelist()}
    reference=json.loads(blobs['distribution/rust-lab-b06-g1250/reference.json'])
    reference['b06']=dict(path=B06+'/qualification.json',sha256=hashlib.sha256((ROOT/B06/'qualification.json').read_bytes()).hexdigest())
    reference['b06_baseline']=q['performance']
    reference['b07_source_review']=json.loads((DIRECTORY/'source-review.json').read_text())
    reference['source_policy']='B07 compact binary ordinary transport, forced acknowledgements, two publishers and attested reads; explicit backend-only adoption with teacher off'
    # Bound the source review. Ordinary publication changed, so the old phase
    # is retained as historical timing only; the launcher remeasures that scope.
    r2=DIRECTORY/'rust-lab-b07-g1250-r2.zip'
    require(hashlib.sha256(r2.read_bytes()).hexdigest()=='1cda06a47ee5c7b875e55ca011b722549d2d79191dca750b36227c3e4d17df92','Original B07 r2 archive changed')
    with zipfile.ZipFile(r2) as z:
        old_manifest=json.loads(z.read('distribution/rust-lab-b07-g1250/manifest.json'))
        before=z.read('lab-native/Cargo.toml')
        old_sources={n:hashlib.sha256(z.read(n).replace(b'\r\n',b'\n')).hexdigest()
                     for n in implementation_sources() if n in z.namelist()}
    changed={n for n,data in implementation_sources().items()
             if hashlib.sha256(data.encode()).hexdigest()!=old_sources.get(n)}
    require(changed=={'lab-native/Cargo.toml','lab-native/tests/teacher_wire.rs','lab-native/tests/lanes.rs',
        'lab-native/src/collection_wire.rs','lab-native/src/lib.rs','lab-native/src/main.rs',
        'src/brass_ai/rust_backend/lab_transport.py','src/brass_ai/rust_backend/lab_collection.py',
        'src/brass_ai/rust_backend/lab_admission.py','src/brass_ai/rust_backend/lab_families.py',
        'src/brass_ai/rust_backend/lab_pool.py'},'Unreviewed B07 dependency change')
    for name in ('scripts/rust_lab_b07_smoke.py','scripts/rust_lab_b06_ordinary.py','scripts/rust_lab_b06_metrics.py'):
        require(hashlib.sha256((ROOT/name).read_bytes().replace(b'\r\n',b'\n')).hexdigest()==old_manifest['files'][name],
                'Ordinary benchmark implementation changed: '+name)
    blobs['distribution/rust-lab-b07-g1250/evidence/b07-r2-Cargo.toml']=before
    teacher=json.loads(blobs['distribution/rust-lab-b06-g1250/evidence/teacher-build.json'])
    from brass_ai.rust_backend.lab_teacher_overlay import source_fingerprint
    reference['teacher_wire_review']=dict(from_producer=teacher['producer'],to_producer=source_fingerprint(),
        feature='serde_json/float_roundtrip',scope='Exact float decoding; search_v1 equations and both sealed shared-core sources unchanged')
    reference['ordinary_native_source_review']={n:hashlib.sha256(data.encode()).hexdigest()
        for n,data in implementation_sources().items() if n.startswith('lab-native/')}
    reuse=dict(from_implementation=old_manifest['implementation'],to_implementation=implementation_id(),
        result=B07,evidence={},changed_dependencies=sorted(changed))
    for key,name in [('qualification','qualification.json'),('ordinary','ordinary-checks.json')]:
        data=(ROOT/B07/name).read_bytes();target='b07-r2-'+name
        blobs['distribution/rust-lab-b07-g1250/evidence/'+target]=data
        reuse['evidence'][key]=dict(file=target,sha256=hashlib.sha256(data).hexdigest())
    reference['ordinary_baseline']=reuse
    reference['native_runtime_policy']='Rebuild ordinary adapter against unchanged B01 core and teacher against unchanged teacher overlay; reuse exact original C# pins'
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
    for path in (DIRECTORY/'evidence').glob('r3-*'):
        blobs[path.relative_to(ROOT).as_posix()]=path.read_bytes()
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
