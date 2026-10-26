"""Seal L10 count-mode Gumbel parity and two-owner component profiles locally."""
import argparse
from datetime import datetime,timezone
import os
from pathlib import Path
import runpy
import shutil
import sys
import zipfile

ROOT=Path(__file__).resolve().parents[1]
BASE=runpy.run_path(str(ROOT/'scripts/qualify-rust-scaffold.py'))
L09=runpy.run_path(str(ROOT/'scripts/qualify-rust-inference.py'))
read,write,receipt,sha=(BASE[k] for k in ('read','write','receipt','sha'))
PRIOR=ROOT/'artifacts/rust-backend/20261025-l09-qualified-v2'

def sources():
    added=[ROOT/p for p in ('scripts/verify-rust-search.py','scripts/qualify-rust-search.py','tests/RustBackend/gumbel_brass_reference.py','tests/RustBackend/test_search_trace.py','configs/contracts/rust-backend/gumbel-search-v1.md')]
    return sorted(set(L09['sources']()+added),key=str)

def qualify(out,fixtures):
    assert out.is_relative_to(ROOT/'artifacts/rust-backend') and not out.exists()
    out.mkdir();write(out/'progress.json',dict(status='running',milestone='L10'))
    reference=PRIOR/'reference-source'
    before=BASE['verify_frozen'](reference_source_root=reference);write(out/'preservation-before.json',before)
    prior=[receipt(PRIOR/n) for n in ('qualification.json','final-verification.json','implementation.zip','executables.zip')]
    model=[receipt(p) for p in (PRIOR/'model').iterdir() if p.is_file()]
    provenance=read(fixtures/'provenance.json')
    assert provenance['reference_archive_sha256']==sha(PRIOR/'executables.zip')
    assert provenance['reference_worker_sha256']==sha(fixtures/'reference-worker.exe')
    assert provenance['g_py_sha256']==sha(ROOT/'tests/RustBackend/gumbel_brass_reference.py')
    assert provenance['reference_script_sha256']==sha(ROOT/'scripts/reproduce-gumbel-is-mcts.py')
    assert len(read(fixtures/'index.json'))==44
    fixture_receipts=[receipt(p) for p in sorted(fixtures.glob('*.json'))]
    write(out/'fixtures.json',fixture_receipts)
    source_before=[receipt(p) for p in sources()];write(out/'sources-before.json',source_before)
    identity=BASE['native_fingerprint']()
    run=BASE['Qualification'](out);run.env.pop('PROPTEST_RNG_SEED',None);run.env['CARGO_INCREMENTAL']='0'
    run.env['PYTHONPATH']=os.pathsep.join([str(reference/'src'),*[p for p in run.env['PYTHONPATH'].split(os.pathsep) if Path(p)!=ROOT/'src']])
    run.env['BRASS_ONNX_MANIFEST']=str(PRIOR/'model/manifest.json');run.env['BRASS_ONNX_GOLDEN']=str(PRIOR/'golden.json')
    run.env['BRASS_SEARCH_FIXTURES']=str(fixtures)
    run.cargo_run('fmt','fmt','--all','--','--check')
    run.cargo_run('clippy','clippy','--workspace','--all-targets','--locked','--offline','--','-D','warnings')
    run.run('python-gumbel-reference',[sys.executable,'-m','unittest','discover','-s',ROOT/'tests/GumbelISMCTS','-v'])
    for profile,flags in [('debug',[]),('release',['--release'])]:
        run.cargo_run(profile+'-tests','test','--workspace',*flags,'--locked','--offline')
        run.cargo_run(profile+'-golden-runtime','test','-p','brass-ai-inference','--test','runtime',*flags,'--locked','--offline','--','--ignored','--nocapture')
    builds={}
    sys.path.insert(0,str(reference/'src'))
    from brass_ai.native.protocol import NativeWorker
    for profile,flags in [('debug',[]),('release',['--release']),('profiling',['--profile','profiling'])]:
        run.cargo_run(profile+'-build','build','-p','brass-ai-worker',*flags,'--locked','--offline')
        directory=ROOT/'artifacts/rust-native-target'/BASE['TARGET']/profile
        worker=directory/'brass-ai-worker.exe';binary=directory/'gumbel-trace.exe'
        with NativeWorker(worker) as w:
            hello=w.hello
            assert hello['build']['source_sha256']==identity==BASE['native_fingerprint']()
            assert hello['build']['cargo_lock_sha256']==sha(ROOT/'native/Cargo.lock')
            assert hello['sessions']==2 and hello['threads_per_session']==1
        builds[profile]=dict(worker=receipt(worker),search=receipt(binary),handshake=hello)
        env=dict(run.env,BRASS_NATIVE_WORKER=str(worker),BRASS_SEARCH_BINARY=str(binary))
        run.run(profile+'-boundary',[sys.executable,'-m','unittest','discover','-s',ROOT/'tests/RustBackend','-p','test_search_trace.py','-v'],env=env)
        for suite,folder,pattern in [('inference','RustBackendInference','test*.py'),('features','RustBackendFeatures','test*.py'),('transport','RustBackendNative','test_worker.py')]:
            run.run(profile+'-'+suite,[sys.executable,'-m','unittest','discover','-s',ROOT/'tests'/folder,'-p',pattern,'-v'],env=env)
        args=[sys.executable,ROOT/'scripts/verify-rust-search.py','--fixtures',fixtures,'--binary',binary,'--report',out/(profile+'-parity.json')]
        if profile=='profiling':args+=['--limit','4']
        run.run(profile+'-parity',args,timeout=1800)
        result=read(out/(profile+'-parity.json'));assert result['status']=='passed' and result['toy_cases']==17
        if profile!='profiling':
            assert len(result['root_matrix'])==40 and len(result['brass_cases'])==44
            assert set(result['node_actors'])=={'0','1','2','3'} and len(result['node_phases'])==5
            assert result['leaf_kinds']['terminal']>0 and result['nonroot_completed_edges']>0 and result['cost_adjusted_edges']>0
    run.cargo_run('profiles-build','build','-p','brass-ai-inference','--examples','--release','--locked','--offline')
    run.cargo_run('keys-probe-build','build','-p','brass-ai-core','--example','search_keys_allocations','--release','--locked','--offline')
    index=read(fixtures/'index.json')
    selections={
        'dense-canal':max((r for r in index if r['era']=='CanalEra'),key=lambda r:r['candidates']),
        'wide-rail':max((r for r in index if r['era']=='RailEra' and r['phase']=='action'),key=lambda r:r['candidates']),
        'deep-staged':max(index,key=lambda r:max(len(t['path'])+len(t['chance']) for t in read(fixtures/(r['name']+'-expected.json'))['trajectories'])),
    }
    write(out/'profile-selection.json',selections)
    directory=ROOT/'artifacts/rust-native-target'/BASE['TARGET']/'release/examples'
    for label,item in selections.items():
        case=fixtures/(item['name']+'-request.json')
        for kind,extra in [('profile',[]),('profile-trace',['trace']),('allocations',[])]:
            executable=directory/('search_allocations.exe' if kind=='allocations' else 'search_profile.exe')
            log=run.run(label+'-'+kind,[executable,PRIOR/'model/manifest.json',case,*extra],timeout=300)
            result=read(log)
            assert result['sessions']==2 and result['threads_per_session']==1
            a,b=result['owners'];assert a['thread']!=b['thread']
            assert max(a['started_ns'],b['started_ns'])<min(a['finished_ns'],b['finished_ns'])
            expected=read(fixtures/(item['name']+'-expected.json'))['selected_action']
            assert all(r[1]==expected and r[0]['completed']==item['simulations'] for o in (a,b) for r in o['runs'])
            write(out/(label+'-'+kind+'.json'),result)
        log=run.run(label+'-keys',[directory/'search_keys_allocations.exe',case]);write(out/(label+'-keys.json'),read(log))
    run.cargo_run('dependencies','tree','--workspace','--edges','features','--locked','--offline')
    run.cargo_run('cargo-metadata','metadata','--format-version','1','--locked','--offline')
    notices=out/'notices';notices.mkdir()
    for p in (ROOT/'artifacts/native-toolchain/cargo/registry/src').glob('*/slotmap-1.1.1/LICENSE*'):shutil.copyfile(p,notices/p.name)
    after=BASE['verify_frozen'](reference_source_root=reference);write(out/'preservation-after.json',after)
    assert L09['L05']['preserved_identity'](before)==L09['L05']['preserved_identity'](after)
    assert source_before==[receipt(p) for p in sources()] and identity==BASE['native_fingerprint']()
    assert prior==[receipt(PRIOR/n) for n in ('qualification.json','final-verification.json','implementation.zip','executables.zip')]
    assert model==[receipt(p) for p in (PRIOR/'model').iterdir() if p.is_file()]
    assert fixture_receipts==[receipt(p) for p in sorted(fixtures.glob('*.json'))]
    with zipfile.ZipFile(out/'implementation.zip','w',zipfile.ZIP_DEFLATED) as z:
        for path in sources():z.write(path,path.relative_to(ROOT).as_posix())
    with zipfile.ZipFile(out/'executables.zip','w',zipfile.ZIP_DEFLATED) as z:
        for profile,b in builds.items():
            for kind in ('worker','search'):z.write(ROOT/b[kind]['path'],profile+'/'+Path(b[kind]['path']).name)
        for name in ('search_profile','search_allocations','search_keys_allocations'):z.write(directory/(name+'.exe'),'probes/'+name+'.exe')
        for path in notices.iterdir():z.write(path,'notices/'+path.name)
    reports=[receipt(p) for p in out.glob('*.json') if p.name.endswith(('-parity.json','-profile.json','-profile-trace.json','-allocations.json','-keys.json'))]
    write(out/'qualification.json',dict(status='passed',milestone='L10',completed_utc=datetime.now(timezone.utc).isoformat(),source_sha256=identity,
        search_identity='brass-gumbel-is-mcts/count-v1',belief='L06 current-view approximation',sources=source_before,builds=builds,commands=run.commands,
        prior_evidence=prior,model=model,fixtures=fixture_receipts,reports=reports,archives=[receipt(out/n) for n in ('implementation.zip','executables.zip')],
        local_g890_utility_accepted=True,calibration_receipt=None,installed_backend_changed=False,training_changed=False,
        deadlines_qualified=False,complete_plans_qualified=False,replica_merge_qualified=False,matched_four_second_benchmark=False,next_milestone='L11'))
    write(out/'progress.json',dict(status='passed',milestone='L10'));print('L10 qualification passed',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--fixtures',type=Path,required=True);a=p.parse_args()
    try:qualify(a.out.resolve(),a.fixtures.resolve())
    except BaseException as e:
        if a.out.resolve().is_relative_to(ROOT/'artifacts/rust-backend') and a.out.exists() and not (a.out/'qualification.json').exists():write(a.out/'progress.json',dict(status='failed',milestone='L10',error=repr(e)))
        raise
