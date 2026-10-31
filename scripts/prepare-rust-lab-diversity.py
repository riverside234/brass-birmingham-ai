#!/usr/bin/env python3
"""Seal the g1330 AZ_DB supplement, preserving every original lab archive."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import zipfile

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
DIRECTORY='distribution/rust-lab-g1330'
ARCHIVES={
    'distribution/rust-lab-b01/b01-search-v1-inputs.zip':'935443a477d3a52425923e3d7458a15ae8250f4296a6900c5b9088fbef864a76',
    'distribution/rust-lab-b01/b01-search-v1-update.zip':'ae634e746dd1864bee281b36f2415112510aeb66a8c5de7faf8d7e4c75dac685',
    'distribution/rust-lab-b02/b02-search-v1-update.zip':'62a9d022bb3daa41f8ad874dfbbf40cb24e3b526c478d559db9db115e4725664',
    'distribution/rust-lab-b025/b025-search-v1-teacher-phase.zip':'038380aab0fcb54c43bff660a098f96cbe7099e17a8abe6d8c02cf3ea0e7332e',
    'distribution/rust-lab-b03/b03-search-v1-admission.zip':'fb6975872aac10a4efb97530b624150dedf78810f91a7800b4950f1d5c4ad0f3',
    'distribution/rust-lab-b04/b04-search-v1-teacher.zip':'fa2ad9b13192786d0417b8be9e4de22a1b28c7797cfe84580891f372f2dd1eca',
}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--check-only',action='store_true');args=p.parse_args()
    from brass_ai.learning.contracts import require
    from brass_ai.step5.league import implementation_id,write_json
    from brass_ai.rust_backend.lab_contract import sha,SOURCE
    from brass_ai.rust_backend.lab_teacher_overlay import source_fingerprint
    from brass_ai.rust_backend.lab_diversity_transfer import CONTRACT
    out=ROOT/DIRECTORY;archive=out/'rust-lab-az-db-g1330.zip'
    require(not archive.exists(),'AZ_DB bundle already sealed; publish a new version')
    for name,expected in ARCHIVES.items():require(sha(ROOT/name)==expected,'Changed original archive: '+name)
    log=ROOT/'results/joint7-v11-2-grouped-lr-critic-g1330-20261031T024940047516Z'
    state=json.loads((log/'state.json').read_text());export=json.loads((log/'export-manifest.json').read_text())
    reference=dict(run=export['run'],generation=state['iteration'],games_done=export['games_done'],checkpoint=state['history'][-1],
        implementation=state['signature']['implementation'],report_directory=log.relative_to(ROOT).as_posix(),
        state_export_sha256=sha(log/'state.json'),report_manifest_sha256=sha(log/'export-manifest.json'),
        diversity_mode=state['signature']['config']['diversity']['mode'],teacher_readiness='missing_retained_intrinsic_fit',
        qualification='Pending')
    require(reference['generation']==1330 and reference['diversity_mode']=='apply','Wrong training reference')
    write_json(out/'training-reference.json',reference)
    names={'.gitignore','scripts/prepare-rust-lab-diversity.py',DIRECTORY+'/README.md',DIRECTORY+'/training-reference.json',
        DIRECTORY+'/local-development.json','results/rust-lab-qualification/README.md',
        log.relative_to(ROOT).as_posix()+'/state.json',log.relative_to(ROOT).as_posix()+'/export-manifest.json',
        'native/brass-ai-core/tests/data/initial-state.json','tests/GumbelISMCTS/fixtures/session-v1.json',
        'ai-research/RUST_AI_BACKEND_MIGRATION_PLAN.md','ai-research/AZ_DB_TEACHER_DIVERSITY_REVISION_PLAN.md',
        'ai-research/TEACHER_SEARCH_PHASE_REVISION_PLAN.md','ai-research/RUST_BACKEND_AZ_DB_REPORT.md',
        'ai-research/DOMINO_DIVERSITY_IMPLEMENTATION.md','ai-research/SELFPLAY_G1330_REVIEW.md','docs/TRAINING_RECOVERY_LAB.md'}
    # Carry current sources together, including the user's g1330 DOMiNO fixes.
    # Older ZIPs are inputs to verification/builds; their source snapshots are
    # never extracted over this supplement.
    for directory,suffixes in [('src',{'.py','.cs'}),('configs',{'.json','.md'}),('scripts',{'.py','.sh'}),
            ('lab-native',{'.rs','.toml','.lock'}),('tests/R2',{'.py','.json'}),('tests/Recovery',{'.py','.json'}),
            ('tests/FeatureEngineering',{'.py','.json'}),('tests/RustBackendLab',{'.py','.json'}),
            ('training/runtime/throughput',{'.json'}),('training/runtime/local-scoring',{'.json'})]:
        names|={q.relative_to(ROOT).as_posix() for q in (ROOT/directory).rglob('*') if q.is_file()
                and q.suffix in suffixes and not any(n in q.parts for n in ('target','__pycache__','bin','obj'))}
    for name in ('pyproject.toml','requirements.txt'):
        if (ROOT/name).exists():names.add(name)
    for stage in ('b025','b03','b04'):
        names|={'distribution/rust-lab-'+stage+'/'+n for n in ('bundle.json','transfer-manifest.json','README.md')}
        names|=set(json.loads((ROOT/'distribution'/('rust-lab-'+stage)/'transfer-manifest.json').read_text())['files'])
    checks=json.loads((out/'local-development.json').read_text())
    implementation=implementation_id();producer=source_fingerprint()
    require(checks['passed'] and checks['implementation']==implementation and checks['producer']==producer,
            'Short development evidence must match this source')
    for name,expected in checks['logs'].items():require(sha(ROOT/name)==expected,'Changed development log: '+name)
    names|=set(checks['logs'])
    files={name:(ROOT/name).read_bytes().replace(b'\r\n',b'\n') for name in sorted(names)}
    if args.check_only:
        print(json.dumps(dict(files=len(files),implementation=implementation,producer=producer,passed=True)));return
    manifest=dict(contract=CONTRACT,implementation=implementation,producer=producer,base_core=SOURCE,
        method='search_v1',v205_enabled=False,teacher_profile='teacher-phase/2.0.0',training_reference=reference,
        files={n:hashlib.sha256(v).hexdigest() for n,v in files.items()},archives=ARCHIVES,local_checks=checks,
        lab_qualification='Pending',production_teacher_enabled=False,results_directory='results/rust-lab-qualification',
        extraction_order=['g1330 combined supplement only; historical ZIPs remain sealed build inputs'])
    payload=dict(files)
    # Preserve raw evidence bytes as well as the normalized transfer hashes.
    payload.update({n:(ROOT/n).read_bytes() for n in checks['logs']})
    payload.update({n:(ROOT/n).read_bytes() for n in ARCHIVES})
    encoded=(json.dumps(manifest,indent=2,allow_nan=False)+'\n').encode()
    payload[DIRECTORY+'/transfer-manifest.json']=encoded
    with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        for n,v in sorted(payload.items()):z.writestr(n,v)
    (out/'transfer-manifest.json').write_bytes(encoded)
    receipt=dict(path=archive.relative_to(ROOT).as_posix(),sha256=sha(archive),bytes=archive.stat().st_size,
        entries=len(payload),implementation=implementation,producer=producer,lab_qualification='Pending')
    write_json(out/'bundle.json',receipt);print(json.dumps(receipt,indent=2))


if __name__=='__main__':main()
