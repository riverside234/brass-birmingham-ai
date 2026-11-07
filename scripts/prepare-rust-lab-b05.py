#!/usr/bin/env python3
"""Seal immutable B05 source and successful prerequisite evidence, without weights."""
import hashlib
import json
from pathlib import Path
import sys
import zipfile

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
DIRECTORY=ROOT/'distribution/rust-lab-b05-g1250'
REPORT='results/joint7-v11-2-g1290-trained-restart-g1250-20261107T041141451177Z'
B045='results/rust-lab-qualification/20261107T060856941600Z-20261107T060554386081Z-g1250-b045-76165ac4-a1db73ac/evidence'
B03='results/rust-lab-qualification/20261104T020129387978Z-20261104T015819818200Z-g570-all-4c81aca9-ceb8ad68/evidence/b03'
B04='results/rust-lab-qualification/20261104T022520138504Z-20261104T022431456929Z-g570-b04-845a2a56-36de5c8d/evidence/b04'
TEXT={'.py','.rs','.cs','.md','.toml','.lock','.ps1','.sh'}


def main():
    from brass_ai.learning.contracts import require,digest
    from brass_ai.recovery.utility import reward_spec
    from brass_ai.step5.league import implementation_id,implementation_sources,write_json
    from brass_ai.rust_backend.lab_contract import sha
    archive=DIRECTORY/'rust-lab-b05-g1250-r2.zip'
    require(not archive.exists(),'Bundle already sealed; publish a new version')
    DIRECTORY.mkdir(parents=True,exist_ok=True)
    read=lambda p:json.loads((ROOT/p).read_text())
    state=read(REPORT+'/state.json');previous=read(B045+'/qualification.json')
    require(state['iteration']==1250 and previous['passed'] is True,'Missing g1250/B04.5 prerequisite')
    require(read(B03+'/qualification.json')['passed'] and read(B04+'/qualification.json')['passed'],'Missing B03/B04 prerequisite')
    evidence=DIRECTORY/'evidence';evidence.mkdir(exist_ok=True)
    for name in ('qualification.json','jobs.json','tables.json.gz','head-update.json','value-admission.json'):
        (evidence/name).write_bytes((ROOT/B045/name).read_bytes())
    ordinary=read(B03+'/replay-admission.json')
    # Historical g570 replay retains its original recipe and producer.
    old=read('results/joint7-v11-2-g1290-trained-restart-g0570-20261103T190804654465Z/recorded-config.json')
    require(all(r['receipt']['native_admission']['admission']['reward_id']==digest(reward_spec(old)) for r in ordinary['families']),
            'B03 recorded reward differs')
    ordinary['config']=old
    reference=dict(generation=1250,checkpoint_sha256=state['history'][-1]['sha256'],
        model_id=state['history'][-1]['model_id'],config_sha256=digest(state['signature']['config']),ordinary=ordinary,
        prerequisites={name:dict(path=path+'/qualification.json',sha256=sha(ROOT/path/'qualification.json'))
                       for name,path in [('B03',B03),('B04',B04),('B04.5',B045)]})
    write_json(DIRECTORY/'reference.json',reference)
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
    names.update(p.relative_to(ROOT).as_posix() for p in evidence.iterdir())
    names.update('distribution/rust-lab-b05-g1250/'+n for n in ('reference.json','README.md','local-checks.json'))
    names.update(['ai-research/RUST_AI_BACKEND_MIGRATION_PLAN.md','ai-research/RUST_BACKEND_B05_IMPLEMENTATION.md',
                  'native/brass-ai-core/tests/data/initial-state.json'])
    def data(name):
        p=ROOT/name;value=p.read_bytes()
        return value.replace(b'\r\n',b'\n') if p.suffix in TEXT else value
    blobs={n:data(n) for n in sorted(names)}
    manifest=dict(implementation=implementation_id(),canonical_text_suffixes=sorted(TEXT),
        files={n:hashlib.sha256(v).hexdigest() for n,v in blobs.items()},qualification='Pending')
    write_json(DIRECTORY/'manifest.json',manifest)
    blobs['distribution/rust-lab-b05-g1250/manifest.json']=(DIRECTORY/'manifest.json').read_bytes()
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for name,value in blobs.items():z.writestr(name,value)
    with zipfile.ZipFile(archive) as z:require(z.testzip() is None,'Corrupt B05 archive')
    write_json(DIRECTORY/'bundle.json',dict(archive=archive.name,sha256=sha(archive),bytes=archive.stat().st_size,
        entries=len(blobs),implementation=implementation_id(),qualification='Pending'))
    print(archive)


if __name__=='__main__':main()
