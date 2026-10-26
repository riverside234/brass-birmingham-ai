"""Seal L12 two-owner correctness, retirement and four-second engineering sweep."""
import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import runpy
import sys
import zipfile

ROOT=Path(__file__).resolve().parents[1]
L11=runpy.run_path(str(ROOT/'scripts/qualify-rust-sessions.py'))
BASE=L11['BASE'];L09=L11['L09']
read,write,receipt,sha=(BASE[k] for k in ('read','write','receipt','sha'))
PRIOR=ROOT/'artifacts/rust-backend/20261025-l11-qualified-v2'
MODEL=L11['MODEL'];FIXTURES=L11['FIXTURES']


def sources():
    paths=L11['sources']()+[ROOT/p for p in (
        'scripts/reproduce-gumbel-coordinator.py','tests/RustBackend/test_deadlines.py',
        'configs/contracts/rust-backend/gumbel-coordinator-v1.md')]
    return sorted(set(paths),key=str)


def qualify(out):
    assert out.is_relative_to(ROOT/'artifacts/rust-backend') and not out.exists()
    out.mkdir(parents=True);write(out/'progress.json',dict(status='running',milestone='L12'))
    reference=MODEL.parent/'reference-source'
    before=BASE['verify_frozen'](reference_source_root=reference);write(out/'preservation-before.json',before)
    prior=[receipt(PRIOR/n) for n in ('qualification.json','final-verification.json','implementation.zip','executables.zip')]
    assert read(PRIOR/'qualification.json')['status']==read(PRIOR/'final-verification.json')['status']=='passed'
    model=[receipt(p) for p in sorted(MODEL.iterdir()) if p.is_file()]
    fixtures=[receipt(p) for p in sorted(FIXTURES.glob('*.json'))];write(out/'fixtures.json',fixtures)
    source_before=[receipt(p) for p in sources()];write(out/'sources-before.json',source_before)
    identity=BASE['native_fingerprint']()
    run=BASE['Qualification'](out);run.env.pop('PROPTEST_RNG_SEED',None);run.env['CARGO_INCREMENTAL']='0'
    run.env['PYTHONPATH']=os.pathsep.join([str(reference/'src'),*[p for p in run.env['PYTHONPATH'].split(os.pathsep) if Path(p)!=ROOT/'src']])
    run.env['BRASS_ONNX_MANIFEST']=str(MODEL/'manifest.json');run.env['BRASS_ONNX_GOLDEN']=str(MODEL.parent/'golden.json')
    run.cargo_run('fmt','fmt','--all','--','--check')
    run.cargo_run('clippy','clippy','--workspace','--all-targets','--locked','--offline','--','-D','warnings')
    run.run('python-oracles',[sys.executable,'-m','unittest','discover','-s',ROOT/'tests/GumbelISMCTS','-v'])
    sys.path.insert(0,str(reference/'src'))
    from brass_ai.native.protocol import NativeWorker
    builds={};reports={};handshakes={}
    for profile,flags in [('debug',[]),('release',['--release']),('profiling',['--profile','profiling'])]:
        if profile!='profiling':
            run.cargo_run(profile+'-tests','test','--workspace',*flags,'--locked','--offline')
            run.cargo_run(profile+'-ort-goldens','test','-p','brass-ai-inference','--test','runtime',*flags,'--locked','--offline','--','--ignored','--nocapture')
        run.cargo_run(profile+'-build','build','-p','brass-ai-worker',*flags,'--locked','--offline')
        directory=ROOT/'artifacts/rust-native-target'/BASE['TARGET']/profile
        executables={key:directory/(name+'.exe') for key,name in dict(
            worker='brass-ai-worker',trace='gumbel-trace',sessions='gumbel-session',pair='gumbel-pair',allocation_pair='gumbel-pair-alloc').items()}
        builds[profile]={k:receipt(v) for k,v in executables.items()}
        with NativeWorker(executables['worker']) as worker:
            hello=worker.hello
            assert hello['build']['source_sha256']==identity
            assert hello['build']['cargo_lock_sha256']==sha(ROOT/'native/Cargo.lock')
            assert hello['sessions']==2 and hello['threads_per_session']==1
            handshakes[profile]=hello
        env=dict(run.env,BRASS_NATIVE_WORKER=str(executables['worker']),BRASS_SEARCH_BINARY=str(executables['trace']),
                 BRASS_SEARCH_FIXTURES=str(FIXTURES),BRASS_SESSION_BINARY=str(executables['sessions']),BRASS_PAIR_BINARY=str(executables['pair']))
        for label,folder,pattern in [('deadlines','RustBackend','test_deadlines.py'),('sessions','RustBackend','test_sessions.py'),
            ('search-boundary','RustBackend','test_search_trace.py'),('transport','RustBackendNative','test_worker.py'),
            ('inference','RustBackendInference','test*.py'),('features','RustBackendFeatures','test*.py')]:
            run.run(profile+'-'+label,[sys.executable,'-m','unittest','discover','-s',ROOT/'tests'/folder,'-p',pattern,'-v'],env=env)
        if profile!='profiling':
            run.run(profile+'-l10-parity',[sys.executable,ROOT/'scripts/verify-rust-search.py','--fixtures',FIXTURES,
                    '--binary',executables['trace'],'--report',out/(profile+'-l10-parity.json')],timeout=1800)
            run.run(profile+'-coordinator',[sys.executable,ROOT/'scripts/verify-rust-coordinator.py','--pair',executables['pair'],
                    '--sessions',executables['sessions'],'--out',out/(profile+'-coordinator')],timeout=1800)
            result=read(out/(profile+'-coordinator')/'verification.json')
            assert result['status']=='passed' and len(result['cases'])==44 and len(set(map(tuple,result['root_matrix'])))==40
            assert result['coverage']['trajectories']==1536
            reports[profile]=dict(coordinator=receipt(out/(profile+'-coordinator')/'verification.json'),l10=receipt(out/(profile+'-l10-parity.json')))
    run.cargo_run('keys-probe-build','build','-p','brass-ai-core','--example','search_keys_allocations','--release','--locked','--offline')
    probe=ROOT/'artifacts/rust-native-target'/BASE['TARGET']/'release/examples/search_keys_allocations.exe'
    for index in (3,26,41):
        log=run.run(f'keys-{index:03d}',[probe,FIXTURES/f'brass-{index:03d}-request.json'])
        result=read(log);assert result['typed_key_clone']['allocations']==0
        assert result['deep_key_clone_ablation']['allocations']>0
        write(out/f'keys-{index:03d}.json',result)
    run.run('four-second-engineering-sweep',[sys.executable,ROOT/'scripts/benchmark-rust-coordinator.py',
        '--binary',ROOT/builds['release']['pair']['path'],'--alloc',ROOT/builds['release']['allocation_pair']['path'],
        '--out',out/'benchmark','--repetitions','2'],timeout=1800)
    benchmark=read(out/'benchmark/benchmark.json');assert benchmark['status']=='passed'
    reports['benchmark']=receipt(out/'benchmark/benchmark.json')
    run.cargo_run('dependencies','tree','--workspace','--edges','features','--locked','--offline')
    run.cargo_run('metadata','metadata','--format-version','1','--locked','--offline')
    after=BASE['verify_frozen'](reference_source_root=reference);write(out/'preservation-after.json',after)
    assert L09['L05']['preserved_identity'](before)==L09['L05']['preserved_identity'](after)
    assert source_before==[receipt(p) for p in sources()] and identity==BASE['native_fingerprint']()
    assert prior==[receipt(PRIOR/n) for n in ('qualification.json','final-verification.json','implementation.zip','executables.zip')]
    assert model==[receipt(p) for p in sorted(MODEL.iterdir()) if p.is_file()]
    assert fixtures==[receipt(p) for p in sorted(FIXTURES.glob('*.json'))]
    with zipfile.ZipFile(out/'implementation.zip','w',zipfile.ZIP_DEFLATED) as z:
        for path in sources():z.write(path,path.relative_to(ROOT).as_posix())
    with zipfile.ZipFile(out/'executables.zip','w',zipfile.ZIP_DEFLATED) as z:
        for profile,b in builds.items():
            for item in b.values():z.write(ROOT/item['path'],profile+'/'+Path(item['path']).name)
        z.write(probe,'probes/'+probe.name)
    write(out/'qualification.json',dict(status='passed',milestone='L12',completed_utc=datetime.now(timezone.utc).isoformat(),
        source_sha256=identity,identity='brass-gumbel-is-mcts/coordinator-v1',sources=source_before,commands=run.commands,
        builds=builds,handshakes=handshakes,reports=reports,prior_evidence=prior,model=model,fixtures=fixtures,
        archives=[receipt(out/n) for n in ('implementation.zip','executables.zip')],
        singleton_l11_parity=True,complete_plans_qualified=True,replica_merge_qualified=True,
        deep_polling_and_runtime_retirement_qualified=True,cooperative_batching_qualified=True,
        engineering_four_second_sweep=True,production_latency_qualified=False,matched_four_second_benchmark=False,
        strength_qualified=False,live_protocol_integration=False,installed_backend_changed=False,training_changed=False,
        local_g890_utility_accepted=True,calibration_receipt=None,next_milestone='L13'))
    write(out/'progress.json',dict(status='passed',milestone='L12'));print('L12 qualification passed',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    try:qualify(a.out.resolve())
    except BaseException as e:
        if a.out.resolve().is_relative_to(ROOT/'artifacts/rust-backend') and a.out.exists() and not (a.out/'qualification.json').exists():
            write(a.out/'progress.json',dict(status='failed',milestone='L12',error=repr(e)))
        raise
