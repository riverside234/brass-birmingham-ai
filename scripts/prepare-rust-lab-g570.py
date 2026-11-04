#!/usr/bin/env python3
"""Seal g570-r10: pinned B04 null-readiness assumption and exact prerequisite reuse."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import zipfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
DIRECTORY='distribution/rust-lab-g570'
REPORT='results/joint7-v11-2-g1290-trained-restart-g0570-20261103T190804654465Z'
CONFIG=REPORT+'/recorded-config.json'
ARCHIVES={
    'distribution/rust-lab-toolchain/microsoft.net.compilers.toolset.4.12.0.nupkg':'fe24ef31a6ffcb7c49383d2fd362763dee291ad9b9d98cc0c19ef80203b99ebc',
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
    from brass_ai.rust_backend.lab_contract import sha,SOURCE,extract_new,verify_inputs
    from brass_ai.rust_backend.lab_transfer import CONTRACT,RETIRED,implementation_inventory
    from brass_ai.rust_backend.lab_teacher_overlay import source_fingerprint
    out=ROOT/DIRECTORY;archive=out/'rust-lab-b01-b04-g570-r10.zip'
    require(not archive.exists(),'This bundle is already sealed; publish a new version')
    for n,h in ARCHIVES.items():require(sha(ROOT/n)==h,'Changed historical archive: '+n)
    require(sha(out/'rust-lab-b01-b04-g570.zip')=='7bbceb50b1b2ffa3c3797aa2aae452a90edde2f15be8a2d627b452f7c64a53a4',
            'Previous g570 archive changed')
    require(sha(out/'rust-lab-b01-b04-g570-r2.zip')=='6d7c1a93d1ae9d9d010509e569bfd55801f9069b3f4c6a64233f75c58bfdafb7',
            'Previous g570-r2 archive changed')
    require(sha(out/'rust-lab-b01-b04-g570-r3.zip')=='91542cd7b07330d86129a54aae9aee0c87b6dd758ce2080e1ebc1aae04d721f2',
            'Previous g570-r3 archive changed')
    require(sha(out/'rust-lab-b01-b04-g570-r4.zip')=='02c4780062444fa056dc4adb5757f5d56b9cbb2cbee78eec3c4d28c9b5986993',
            'Previous g570-r4 archive changed')
    require(sha(out/'rust-lab-b01-b04-g570-r5.zip')=='cbd9eb905d4a40cdcedf81d1091d90d4e2189acea2faf84023cf8db65f891968',
            'Previous g570-r5 archive changed')
    require(sha(out/'rust-lab-b01-b04-g570-r6.zip')=='887117aff5c8cfa2976573584c72ecee287695a546e2ae6d72692e0c6935db20',
            'Previous g570-r6 archive changed')
    require(sha(out/'rust-lab-b01-b04-g570-r7.zip')=='dd83356ccaba65d90c8be97b6e6778777b0b16dd3072f87d296a62a91ebc9657',
            'Previous g570-r7 archive changed')
    require(sha(out/'rust-lab-b01-b04-g570-r8.zip')=='8fdb70738e042795b4f9393464f9d33b76ed1435c09b8c93d39d81f11b00b97b',
            'Previous g570-r8 archive changed')
    require(sha(out/'rust-lab-b01-b04-g570-r9.zip')=='c488341f929dce0b3c9ed51511e337d2f71bcc8050d133743db5c84edc852f8a',
            'Previous g570-r9 archive changed')
    require(all(not (ROOT/n).exists() for n in RETIRED),'Retired AZ_DB source still present')
    spec=importlib.util.spec_from_file_location('b01_package_inputs',ROOT/'scripts/qualify-rust-lab.py')
    b01=importlib.util.module_from_spec(spec);spec.loader.exec_module(b01)
    with tempfile.TemporaryDirectory(dir=ROOT/'artifacts',prefix='g570-r10-inputs-') as directory:
        directory=Path(directory)
        extract_new(ROOT/'distribution/rust-lab-b01/b01-search-v1-inputs.zip',directory/'inputs')
        verify_inputs(directory/'inputs',json.loads((ROOT/'configs/search/lab-search-v1.json').read_text(encoding='utf-8')))
        extract_new(directory/'inputs/source.zip',directory/'checkout')
        native_inputs=b01.prepare_test_inputs(directory/'checkout/native')
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
        DIRECTORY+'/prerequisite-reuse.json',
        'results/rust-lab-qualification/README.md',REPORT+'/state.json',REPORT+'/export-manifest.json',REPORT+'/recorded-config.json',
        'native/brass-ai-core/tests/data/initial-state.json','tests/GumbelISMCTS/fixtures/session-v1.json',
        'ai-research/RUST_AI_BACKEND_MIGRATION_PLAN.md','ai-research/TEACHER_SEARCH_PHASE_REVISION_PLAN.md',
        'ai-research/RUST_BACKEND_G570_REFRESH.md',
        'ai-research/RUST_BACKEND_B01_G570_FIX.md',
        'ai-research/RUST_BACKEND_B01_G570_PYTHON_FIX.md',
        'ai-research/RUST_BACKEND_B01_B04_G570_REVIEW.md',
        'ai-research/RUST_BACKEND_B02_COLLECTION_FIX.md',
        'ai-research/RUST_BACKEND_B02_TIMEOUT_FIX.md',
        'ai-research/RUST_BACKEND_B03_CUDA_RESTORE_FIX.md',
        'ai-research/RUST_BACKEND_B03_CUDA_TOPOLOGY_FIX.md',
        'ai-research/RUST_BACKEND_B04_G570_READINESS_FIX.md',
        'distribution/rust-lab-toolchain/README.md','distribution/rust-lab-toolchain/License.txt',
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
    names |= set(b01.TEST_FIXTURES)
    names={n for n in names if (ROOT/n).is_file() and n != 'scripts/build-mixed-ai.py'}
    checks=json.loads((out/'local-development.json').read_text(encoding='utf-8'))
    implementation=implementation_id();producer=source_fingerprint()
    require(checks['passed'] and checks['implementation']==implementation and checks['producer']==producer,
        'Bounded development evidence does not match current source')
    require(checks['b01_inputs_fix']['passed'] and checks['b01_inputs_fix']['native_inputs']==native_inputs,
            'Detached native test-input evidence differs')
    python_checks=checks['b01_python_fix']
    require(python_checks['passed'] and python_checks['test_paths']==list(b01.TEST_PATHS)
            and python_checks['inference']['tests']==28 and python_checks['inference']['skipped']==0,
            'Exact B01 inference suite must pass before packaging')
    review=checks['b04_g570_fix']
    require(review['passed'] and review['implementation']==implementation and review['producer']==producer
            and review['prerequisites_repeated'] is False,'Missing B04 g570 readiness regression evidence')
    with zipfile.ZipFile(out/'rust-lab-b01-b04-g570-r9.zip') as prior:
        old=json.loads(prior.read(DIRECTORY+'/transfer-manifest.json'))
        sources=implementation_sources()
        require(set(sources)==set(old['implementation_files']) and old['producer']==producer,
                'B04 diagnostic repair changed native producer or implementation input set')
        changed={name for name,value in sources.items()
                 if value!=prior.read(name).decode('utf-8').replace('\r\n','\n')}
        require(changed=={'src/brass_ai/rust_backend/lab_b04_smoke.py',
                          'src/brass_ai/rust_backend/lab_prerequisite_reuse.py'},
                'Changed source outside the B04 diagnostic repair')
    for name,expected in review['tested_files'].items():
        require(sha(ROOT/name)==expected,'Changed source after B04 readiness checks: '+name)
    # Preserve historical evidence, checking changed harnesses against the new
    # tests instead of relabelling old qualifications as newly executed.
    retained=[python_checks,checks['b03_cuda_fix'],checks['b03_topology_fix']]
    for prior_checks in retained:
        for name,expected in prior_checks['tested_files'].items():
            require(sha(ROOT/name)==expected or sha(ROOT/name)==review['tested_files'].get(name)
                    or sha(ROOT/name)==checks['b03_topology_fix']['tested_files'].get(name)
                    or (name=='scripts/qualify-rust-lab-b02.py'
                        and sha(ROOT/name)==checks['b03_cuda_fix']['tested_files'].get(name)),
                    'Changed source outside retained and new bounded checks: '+name)
    require(sha(ROOT/'configs/learning/selfplay-resume-sources.json')==checks['source_review_sha256'],
        'Source approvals changed since continuation checks')
    from brass_ai.recovery.resume import check_signature
    check_signature(state,dict(state['signature'],implementation=implementation),reviewed=True)
    for n,h in checks['logs'].items():require(sha(ROOT/n)==h,'Changed development log: '+n)
    names|=set(checks['logs'])
    require(not any(n.startswith('results/rust-lab-qualification/') and n!='results/rust-lab-qualification/README.md'
                    for n in names),'Transfer must not overwrite original lab evidence')
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
        native_test_inputs=native_inputs,
        python_test_paths=list(b01.TEST_PATHS),
        supersedes=dict(path=DIRECTORY+'/rust-lab-b01-b04-g570-r9.zip',
                        sha256='c488341f929dce0b3c9ed51511e337d2f71bcc8050d133743db5c84edc852f8a'),
        qualified_prerequisite_reuse=json.loads((out/'prerequisite-reuse.json').read_text(encoding='utf-8')),
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
