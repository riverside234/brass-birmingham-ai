#!/usr/bin/env python3
"""Seal B01-B04 for the recorded g570 checkpoint and current reviewed source."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import zipfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
DIRECTORY='distribution/rust-lab-g570'
REPORT='results/joint7-v11-2-g1290-trained-restart-g0570-20261103T190804654465Z'
CONFIG=REPORT+'/recorded-config.json'
ARCHIVES={
    'distribution/rust-lab-b01/b01-search-v1-inputs.zip':'935443a477d3a52425923e3d7458a15ae8250f4296a6900c5b9088fbef864a76',
    'distribution/rust-lab-b01/b01-search-v1-update.zip':'ae634e746dd1864bee281b36f2415112510aeb66a8c5de7faf8d7e4c75dac685',
    'distribution/rust-lab-b02/b02-search-v1-update.zip':'62a9d022bb3daa41f8ad874dfbbf40cb24e3b526c478d559db9db115e4725664',
    'distribution/rust-lab-b025/b025-search-v1-teacher-phase.zip':'038380aab0fcb54c43bff660a098f96cbe7099e17a8abe6d8c02cf3ea0e7332e',
    'distribution/rust-lab-b03/b03-search-v1-admission.zip':'fb6975872aac10a4efb97530b624150dedf78810f91a7800b4950f1d5c4ad0f3',
    'distribution/rust-lab-b04/b04-search-v1-teacher.zip':'fa2ad9b13192786d0417b8be9e4de22a1b28c7797cfe84580891f372f2dd1eca',
}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-only',action='store_true');args=parser.parse_args()
    from brass_ai.learning.contracts import require,digest
    from brass_ai.step5.league import implementation_id,implementation_sources,write_json
    from brass_ai.rust_backend.lab_contract import sha,SOURCE
    from brass_ai.rust_backend.lab_transfer import CONTRACT,RETIRED,implementation_inventory
    from brass_ai.rust_backend.lab_teacher_overlay import source_fingerprint
    out=ROOT/DIRECTORY;archive=out/'rust-lab-b01-b04-g570.zip'
    require(not archive.exists(),'This bundle is already sealed; publish a new version')
    for n,h in ARCHIVES.items():require(sha(ROOT/n)==h,'Changed historical archive: '+n)
    require(all(not (ROOT/n).exists() for n in RETIRED),'Retired AZ_DB source still present')
    state=json.loads((ROOT/REPORT/'state.json').read_text(encoding='utf-8'))
    exported=json.loads((ROOT/REPORT/'export-manifest.json').read_text(encoding='utf-8'))
    config=json.loads((ROOT/CONFIG).read_text(encoding='utf-8'))
    require(state['signature']['config']==config and state['iteration']==570
        and state['history'][-1]['sha256']=='06626c1fb7d0b91f64d939f9fbb3db7c5f3aa072331eb3d773f101e729117ab9'
        and config['diversity']['mode']=='off','Wrong recorded g570 source/config reference')
    reference=dict(run=exported['run'],generation=state['iteration'],games_done=state['games_done'],
        checkpoint=state['history'][-1],implementation=state['signature']['implementation'],report_directory=REPORT,
        state_export_sha256=sha(ROOT/REPORT/'state.json'),report_manifest_sha256=sha(ROOT/REPORT/'export-manifest.json'),
        recorded_config_sha256=digest(state['signature']['config']),requested_config=CONFIG,
        requested_config_sha256=digest(config),recorded_config_is_requested_config=True,
        config_adoption='None: preserve the exact recorded g570 recipe; new g570/entropy options are not adopted',
        qualification_target='Recorded g570 checkpoint via explicit CLI pins, or capture latest with identical recorded config; reject advancement',
        reference_generation_only=False,
        domino_enabled=False,az_db_enabled=False,qualification='Pending')
    write_json(out/'training-reference.json',reference)
    names={'.gitignore',CONFIG,'pyproject.toml','scripts/prepare-rust-lab-g570.py',
        DIRECTORY+'/README.md',DIRECTORY+'/training-reference.json',DIRECTORY+'/local-development.json',
        'results/rust-lab-qualification/README.md',REPORT+'/state.json',REPORT+'/export-manifest.json',REPORT+'/recorded-config.json',
        'native/brass-ai-core/tests/data/initial-state.json','tests/GumbelISMCTS/fixtures/session-v1.json',
        'ai-research/RUST_AI_BACKEND_MIGRATION_PLAN.md','ai-research/TEACHER_SEARCH_PHASE_REVISION_PLAN.md',
        'ai-research/RUST_BACKEND_G570_REFRESH.md',
        'results/joint7-v11-2-g1290-trained-restart-g0370-20261102T221517838215Z/state.json','distribution/RUST_LAB_SEARCH_V1.md'}
    folders=[('src/brass_ai',{'.py'}),('configs',{'.json','.md'}),('scripts',{'.py','.sh'}),
        ('lab-native',{'.rs','.toml','.lock'}),('tests/R2',{'.py','.json'}),('tests/Recovery',{'.py','.json'}),
        ('tests/FeatureEngineering',{'.py','.json'}),('tests/RustBackendLab',{'.py','.json'}),
        ('training/runtime/throughput',{'.json'}),('training/runtime/local-scoring',{'.json'})]
    for folder,suffixes in folders:
        names|={p.relative_to(ROOT).as_posix() for p in (ROOT/folder).rglob('*') if p.is_file()
            and p.suffix in suffixes and not any(x in p.parts for x in ('target','__pycache__','bin','obj'))}
    from brass_ai.search.legacy_runtime import source_paths
    names|={p.relative_to(ROOT).as_posix() for p in source_paths(ROOT)}
    for folder in ('BrassTrainingHost','BrassLocalScoringHost','BrassLabBootstrap'):
        names|={p.relative_to(ROOT).as_posix() for p in (ROOT/'src'/folder).glob('*.cs')}
    for stage in ('b025','b03','b04'):
        base='distribution/rust-lab-'+stage+'/'
        names|={base+n for n in ('bundle.json','transfer-manifest.json','README.md')}
        prior=json.loads((ROOT/base/'transfer-manifest.json').read_text(encoding='utf-8'))
        names|={n for n in prior['files'] if (ROOT/n).is_file() and n not in RETIRED}
    names |= set(implementation_sources())
    names={n for n in names if (ROOT/n).is_file() and n != 'scripts/build-mixed-ai.py'}
    checks=json.loads((out/'local-development.json').read_text(encoding='utf-8'))
    implementation=implementation_id();producer=source_fingerprint()
    require(checks['passed'] and checks['implementation']==implementation and checks['producer']==producer,
        'Bounded development evidence does not match current source')
    require(sha(ROOT/'configs/learning/selfplay-resume-sources.json')==checks['source_review_sha256'],
        'Source approvals changed since continuation checks')
    from brass_ai.recovery.resume import check_signature
    check_signature(state,dict(state['signature'],implementation=implementation),reviewed=True)
    for n,h in checks['logs'].items():require(sha(ROOT/n)==h,'Changed development log: '+n)
    names|=set(checks['logs'])
    # Source comparison tolerates platform line endings. Transfer bytes must
    # remain exact: runtime wire contracts and report receipts hash raw files.
    files={n:(ROOT/n).read_bytes() for n in sorted(names)}
    sources={n:files[n].decode('utf-8').replace('\r\n','\n') for n in implementation_sources()}
    require(digest(sources)==implementation and implementation_id()==implementation
        and source_fingerprint()==producer,'Source changed while preparing the archive')
    manifest=dict(contract=CONTRACT,implementation=implementation,producer=producer,base_core=SOURCE,
        implementation_files=implementation_inventory(sources),
        method='search_v1',v205_enabled=False,domino_enabled=False,az_db_enabled=False,
        teacher_profile='teacher-phase/1.1.0',timing=dict(contract='search-plus-post/1.0.0',search_seconds=4.,post_search_seconds=1.),
        retired_files=list(RETIRED),training_reference=reference,
        files={n:hashlib.sha256(v.replace(b'\r\n',b'\n')).hexdigest() for n,v in files.items()},
        archives=ARCHIVES,local_checks=checks,lab_qualification='Pending',production_teacher_enabled=False,
        results_directory='results/rust-lab-qualification',extraction_order=['Current checkout with deletions, then g570 supplement only'])
    if args.check_only:
        print(json.dumps(dict(passed=True,files=len(files),implementation=implementation,producer=producer)));return
    payload=dict(files)
    payload.update({n:(ROOT/n).read_bytes() for n in (*ARCHIVES,*checks['logs'])})
    encoded=(json.dumps(manifest,indent=2,allow_nan=False)+'\n').encode()
    payload[DIRECTORY+'/transfer-manifest.json']=encoded
    with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        for n,v in sorted(payload.items()):z.writestr(n,v)
    (out/'transfer-manifest.json').write_bytes(encoded)
    receipt=dict(path=archive.relative_to(ROOT).as_posix(),sha256=sha(archive),bytes=archive.stat().st_size,
        entries=len(payload),implementation=implementation,producer=producer,lab_qualification='Pending')
    write_json(out/'bundle.json',receipt);print(json.dumps(receipt,indent=2))


if __name__=='__main__':main()
