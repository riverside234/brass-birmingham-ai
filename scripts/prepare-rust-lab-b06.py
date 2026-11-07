#!/usr/bin/env python3
"""Seal B06 diagnostics and original passing evidence, without weights/binaries."""
import hashlib
import json
from pathlib import Path
import sys
import zipfile

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
DIRECTORY=ROOT/'distribution/rust-lab-b06-g1250'
REPORT='results/joint7-v11-2-g1290-trained-restart-g1250-20261107T041141451177Z'
B02='results/rust-lab-qualification/20261104T012220551953Z-20261104T011422740838Z-g570-all-8bee9d91-e1605e8c/evidence/b02'
B045='results/rust-lab-qualification/20261107T060856941600Z-20261107T060554386081Z-g1250-b045-76165ac4-a1db73ac/evidence'
B05='results/rust-lab-qualification/20261107T070806231803Z-20261107T070539193547Z-g1250-b05-88497ac1-cf4704fd/evidence'
TEXT={'.py','.rs','.cs','.md','.toml','.lock','.ps1','.sh'}


def main():
    from brass_ai.learning.contracts import require,digest
    from brass_ai.rust_backend.lab_contract import sha
    from brass_ai.rust_backend.lab_teacher_overlay import source_fingerprint
    from brass_ai.step5.league import write_json,implementation_id,implementation_sources
    archive=DIRECTORY/'rust-lab-b06-g1250-r2.zip'
    require(not archive.exists(),'B06 bundle already sealed; publish a new revision')
    read=lambda name:json.loads((ROOT/name).read_text())
    state=read(REPORT+'/state.json')
    require(state['iteration']==1250,'Wrong training reference')
    for source in (B02,B045,B05):require(read(source+'/qualification.json')['passed'] is True,'Missing passing prerequisite: '+source)
    reuse=read('distribution/rust-lab-b06-g1250/source-reuse.json')
    require(read(B05+'/qualification.json')['implementation']==reuse['from_implementation'] and
            reuse['to_implementation']==implementation_id(),'B05 source reuse is not reviewed')
    require(set(reuse['changes'])=={'src/brass_ai/rust_backend/lab_collection.py','src/brass_ai/rust_backend/lab_families.py'},
            'B06 reuse exceeds the native branch-window repair')
    old_manifest=read('distribution/rust-lab-b05-g1250/manifest.json')
    for name,change in reuse['changes'].items():
        require(old_manifest['files'][name]==change['before'] and
                hashlib.sha256((ROOT/name).read_bytes().replace(b'\r\n',b'\n')).hexdigest()==change['after'],
                'B06 source reuse diff changed: '+name)
    require(read(B045+'/build.json')['producer']==source_fingerprint(),'Teacher native source changed after B04.5')
    ordinary_sources={p['path']:p['sha256'] for p in read(B02+'/build.json')['source']}
    ordinary_review={}
    for name in ('lab-native/src/lib.rs','lab-native/src/main.rs','lab-native/src/scoring.rs','lab-native/Cargo.toml','lab-native/Cargo.lock'):
        actual=hashlib.sha256((ROOT/name).read_bytes().replace(b'\r\n',b'\n')).hexdigest()
        require(ordinary_sources.get(name)==actual,'Ordinary binary source changed after B02: '+name)
        ordinary_review[name]=actual
    inputs=DIRECTORY/'evidence';inputs.mkdir(parents=True,exist_ok=True)
    copies={
        'ordinary-build.json':B02+'/build.json','teacher-build.json':B045+'/build.json',
        'b02-qualification.json':B02+'/qualification.json','b045-qualification.json':B045+'/qualification.json',
        'b05-qualification.json':B05+'/qualification.json','tables.json.gz':B045+'/tables.json.gz',
        'jobs.json':B045+'/jobs.json','all-seat-inference.json':B045+'/inference.json',
        'all-seat-value-admission.json':B045+'/value-admission.json'}
    for target,source in copies.items():(inputs/target).write_bytes((ROOT/source).read_bytes())
    with zipfile.ZipFile(ROOT/'distribution/rust-lab-g1250/inputs.zip') as z:
        (inputs/'fixtures.json').write_bytes(z.read('fixtures.json'))
    write_json(DIRECTORY/'reference.json',dict(generation=1250,checkpoint_sha256=state['history'][-1]['sha256'],
        model_id=state['history'][-1]['model_id'],config_sha256=digest(state['signature']['config']),
        ordinary_build_sha256=sha(inputs/'ordinary-build.json'),
        ordinary_native_source_review=ordinary_review,
        prerequisites={name:dict(path=path+'/qualification.json',sha256=sha(ROOT/path/'qualification.json'))
                       for name,path in [('B02',B02),('B04.5',B045),('B05',B05)]},
        source_policy='Diagnostic harness plus exact native v9 branch-window repair; B05 unaffected-component evidence retained',
        source_reuse=reuse,
        native_runtime_policy='Reuse exact pinned original lab binaries/dependencies; no previous qualification or build rerun'))
    names=set(implementation_sources())
    for folder,extensions in [('src',{'.py','.cs','.json','.csproj'}),('configs',{'.json'}),('scripts',{'.py','.sh'}),
            ('lab-native',{'.rs','.toml','.lock'}),('tests/RustBackendLab',{'.py'}),('tests/Recovery',{'.py'}),
            ('tests/R2',{'.py','.json'}),('tests/FeatureEngineering',{'.py','.json'}),('tests/GumbelISMCTS/fixtures',{'.json'})]:
        names.update(p.relative_to(ROOT).as_posix() for p in (ROOT/folder).rglob('*') if p.is_file()
            and p.suffix in extensions and not any(x in p.parts for x in ('target','__pycache__','obj','bin')))
    for folder in (REPORT,'results/joint7-v11-2-g1290-trained-restart-g1247-20261107T033857653329Z',
            'results/joint7-v11-2-grouped-lr-critic-g1320-20261027T075134827445Z',
            'results/joint7-v11-2-g1290-trained-restart-g0921-20261105T213224468369Z'):
        names.update((Path(folder)/n).as_posix() for n in ('state.json','recorded-config.json','export-manifest.json') if (ROOT/folder/n).is_file())
    names.update(p.relative_to(ROOT).as_posix() for p in inputs.iterdir())
    names.update('distribution/rust-lab-b06-g1250/'+n for n in ('reference.json','README.md','local-checks.json','source-reuse.json'))
    names.update(['ai-research/RUST_AI_BACKEND_MIGRATION_PLAN.md','ai-research/RUST_BACKEND_B06_IMPLEMENTATION.md',
                  'native/brass-ai-core/tests/data/initial-state.json'])
    blobs={}
    for name in sorted(names):
        p=ROOT/name;value=p.read_bytes();blobs[name]=value.replace(b'\r\n',b'\n') if p.suffix in TEXT else value
    manifest=dict(implementation=implementation_id(),canonical_text_suffixes=sorted(TEXT),
                  files={n:hashlib.sha256(v).hexdigest() for n,v in blobs.items()},qualification='Pending')
    write_json(DIRECTORY/'manifest.json',manifest)
    blobs['distribution/rust-lab-b06-g1250/manifest.json']=(DIRECTORY/'manifest.json').read_bytes()
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for name,value in blobs.items():z.writestr(name,value)
    with zipfile.ZipFile(archive) as z:require(z.testzip() is None,'Corrupt B06 bundle')
    write_json(DIRECTORY/'bundle.json',dict(archive=archive.name,sha256=sha(archive),bytes=archive.stat().st_size,
        entries=len(blobs),implementation=implementation_id(),qualification='Pending'))
    print(archive)


if __name__=='__main__':main()
