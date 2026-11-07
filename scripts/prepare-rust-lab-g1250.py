#!/usr/bin/env python3
"""Seal a self-contained B04.5 source supplement; historical ZIPs are unnecessary."""
import hashlib
import json
from pathlib import Path
import sys
import zipfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
DIRECTORY=ROOT/'distribution/rust-lab-g1250'
REPORT='results/joint7-v11-2-g1290-trained-restart-g1250-20261107T041141451177Z'
TEXT={'.py','.rs','.cs','.md','.toml','.lock','.ps1','.sh'}


def data(path):
    value=path.read_bytes()
    return value.replace(b'\r\n',b'\n') if path.suffix in TEXT else value


def main():
    from brass_ai.learning.contracts import read_json,require,digest
    from brass_ai.step5.league import implementation_id,implementation_sources,write_json
    from brass_ai.rust_backend.lab_b045_smoke import CHECKPOINT
    from brass_ai.rust_backend.lab_contract import sha
    from brass_ai.rust_backend.lab_teacher_overlay import source_fingerprint
    archive=DIRECTORY/'rust-lab-b045-g1250-r3.zip'
    require(not archive.exists(),'Bundle already sealed; publish a new version')
    state=read_json(ROOT/REPORT/'state.json')
    require(state['iteration']==1250 and state['history'][-1]['sha256']==CHECKPOINT,'Wrong g1250 report')
    require(sha(DIRECTORY/'inputs.zip')=='935443a477d3a52425923e3d7458a15ae8250f4296a6900c5b9088fbef864a76','Sealed search_v1 inputs changed')
    reference=dict(run='artifacts/step5/joint7-v11-2-g1290-trained-restart',generation=1250,checkpoint=state['history'][-1],
        config_sha256=digest(state['signature']['config']),implementation=state['signature']['implementation'],
        report=REPORT,report_state_sha256=sha(ROOT/REPORT/'state.json'),lab_qualification='Pending',
        production_teacher='off',null_receipt='User-authorized for this exact g1250 shadow only')
    write_json(DIRECTORY/'training-reference.json',reference)
    names=set(implementation_sources())
    for folder,extensions in [('src',{'.py','.cs','.json','.csproj'}),('configs',{'.json'}),
        ('scripts',{'.py','.sh'}),('lab-native',{'.rs','.toml','.lock'}),
        ('tests/RustBackendLab',{'.py'}),('tests/Recovery',{'.py'}),('tests/R2',{'.py','.json'}),
        ('tests/FeatureEngineering',{'.py','.json'}),('tests/GumbelISMCTS/fixtures',{'.json'})]:
        names.update(p.relative_to(ROOT).as_posix() for p in (ROOT/folder).rglob('*')
                     if p.is_file() and p.suffix in extensions and not any(x in p.parts for x in ('target','__pycache__','obj','bin')))
    names.update([REPORT+'/recorded-config.json',REPORT+'/state.json',REPORT+'/export-manifest.json',
        'results/joint7-v11-2-grouped-lr-critic-g1320-20261027T075134827445Z/recorded-config.json',
        'results/joint7-v11-2-g1290-trained-restart-g0921-20261105T213224468369Z/recorded-config.json',
        'distribution/rust-lab-g1250/inputs.zip','distribution/rust-lab-g1250/README.md',
        'distribution/rust-lab-g1250/training-reference.json','distribution/rust-lab-g1250/local-checks.json',
        'distribution/rust-lab-toolchain/microsoft.net.compilers.toolset.4.12.0.nupkg',
        'distribution/rust-lab-toolchain/License.txt','distribution/rust-lab-toolchain/README.md',
        'ai-research/RUST_BACKEND_B045_SEARCH_VALUE_PLAN.md','ai-research/RUST_AI_BACKEND_MIGRATION_PLAN.md',
        'ai-research/RUST_BACKEND_B045_IMPLEMENTATION.md','native/brass-ai-core/tests/data/initial-state.json'])
    # No proprietary runtime/checkpoint is shipped. The lab already owns its
    # reference runtime; build_bootstrap verifies every dependency before use.
    blobs={name:data(ROOT/name) for name in sorted(names)}
    manifest=dict(contract='brass-lab-b045-transfer/1.0.0',implementation=implementation_id(),producer=source_fingerprint(),
        canonical_text_suffixes=sorted(TEXT),files={n:hashlib.sha256(v).hexdigest() for n,v in blobs.items()},
        training=reference,previous_bundles_required=False)
    write_json(DIRECTORY/'manifest.json',manifest)
    blobs['distribution/rust-lab-g1250/manifest.json']=data(DIRECTORY/'manifest.json')
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for name,value in blobs.items():
            info=zipfile.ZipInfo(name,date_time=(2026,11,6,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED
            z.writestr(info,value)
    with zipfile.ZipFile(archive) as z:
        require(z.testzip() is None and set(z.namelist())==set(blobs),'Bundle integrity differs')
    write_json(DIRECTORY/'bundle.json',dict(archive=archive.name,sha256=sha(archive),bytes=archive.stat().st_size,
        entries=len(blobs),implementation=implementation_id(),producer=source_fingerprint(),qualification='Pending',
        previous_qualifications_repeated=False))
    print(archive)


if __name__=='__main__':main()
