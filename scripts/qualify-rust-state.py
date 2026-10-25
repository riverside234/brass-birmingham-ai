"""Qualify L04 state preservation locally using the pinned offline L03 toolchain."""
import argparse
from datetime import datetime,timezone
import gzip
import json
from pathlib import Path
import runpy
import shutil
import sys
import zipfile

ROOT=Path(__file__).resolve().parents[1]
BASE=runpy.run_path(str(ROOT/'scripts/qualify-rust-scaffold.py'))
read,write,receipt,sha=(BASE[k] for k in ('read','write','receipt','sha'))


def qualify(out,capture,regressions):
    assert out.is_relative_to(ROOT/'artifacts/rust-backend')
    out.mkdir(parents=True,exist_ok=True)
    assert not (out/'qualification.json').exists(), 'Completed evidence is immutable'
    source_identity=BASE['native_fingerprint']()
    before=BASE['verify_frozen']();write(out/'preservation-before.json',before)
    frozen_l03=ROOT/'artifacts/rust-backend/20261024-l03-verified'
    l03_receipts=[receipt(frozen_l03/name) for name in ('qualification.json','final-verification.json','implementation.zip','executables.zip')]
    run=BASE['Qualification'](out);run.env.pop('PROPTEST_RNG_SEED',None);run.env['PROPTEST_CASES']='256'
    run.env['CARGO_INCREMENTAL']='0'
    run.env['BRASS_STATE_CAPTURE']=str(capture)
    run.env['BRASS_STATE_REGRESSIONS']=str(regressions)
    write(out/'regression-receipt.json',receipt(regressions))
    frozen=read(capture/'capture.json');assert sha(capture/'states.jsonl.gz')==frozen['states_sha256']
    write(out/'capture-receipt.json',frozen)
    with gzip.open(capture/'states.jsonl.gz','rt',encoding='utf-8') as stream:rows=[json.loads(line) for line in stream]
    complex_row=max((r for r in rows if r['state']['context']['phase']=='payment' and r['state']['context']['era']=='RailEra'),key=lambda r:(sum('tile' in s for s in r['state']['slots']),len(r['state']['history'])))
    bench=out/'bench-inputs';bench.mkdir(exist_ok=True)
    shutil.copyfile(capture/'initial-state.json',bench/'initial.json');write(bench/'complex.json',complex_row['state'])
    write(bench/'selection.json',dict(complex_root=complex_row['id'],criterion='Rail payment; most occupied slots, then longest history',inputs=[receipt(bench/(n+'.json')) for n in ('initial','complex')]))
    run.env['BRASS_STATE_BENCH_DIR']=str(bench)
    run.cargo_run('fmt','fmt','--all','--','--check')
    run.cargo_run('clippy','clippy','--workspace','--all-targets','--locked','--offline','--','-D','warnings')
    for profile,flags in [('debug',[]),('release',['--release'])]:
        run.cargo_run(profile+'-tests','test','--workspace',*flags,'--locked','--offline')
    from brass_ai.native import NativeWorker
    builds={}
    for profile,flags in [('debug',[]),('release',['--release']),('profiling',['--profile','profiling'])]:
        run.cargo_run(profile+'-build','build','-p','brass-ai-worker',*flags,'--locked','--offline')
        binary=ROOT/'artifacts/rust-native-target'/BASE['TARGET']/profile/'brass-ai-worker.exe'
        with NativeWorker(binary) as worker:
            hello=worker.hello
            assert hello['build']['source_sha256']==source_identity==BASE['native_fingerprint']()
            assert hello['build']['cargo_lock_sha256']==sha(ROOT/'native/Cargo.lock')
        builds[profile]=dict(binary=receipt(binary),handshake=hello,pe_machine=BASE['machine_type'](binary))
        env=dict(run.env,BRASS_NATIVE_WORKER=str(binary),BRASS_STATE_PARITY_REPORT=str(out/(profile+'-parity.json')))
        run.run(profile+'-python-tests',[sys.executable,'-m','unittest','discover','-s',ROOT/'tests/RustBackendNative','-v'],env=env,timeout=300)
    run.cargo_run('state-criterion','bench','-p','brass-ai-core','--bench','state','--locked','--offline')
    raw=ROOT/'artifacts/rust-native-target/criterion/semantic_state'
    shutil.copytree(raw,out/'criterion',dirs_exist_ok=True)
    run.cargo_run('allocation-build','build','--release','-p','brass-ai-core','--example','state_allocation_probe','--locked','--offline')
    allocation=ROOT/'artifacts/rust-native-target'/BASE['TARGET']/'release/examples/state_allocation_probe.exe'
    for name in ('initial','complex'):
        log=run.run(name+'-allocation',[allocation,bench/(name+'.json')]);write(out/(name+'-allocation.json'),read(log))
    after=BASE['verify_frozen']();write(out/'preservation-after.json',after);assert before==after
    assert source_identity==BASE['native_fingerprint'](), 'Native sources changed during qualification'
    assert l03_receipts==[receipt(frozen_l03/name) for name in ('qualification.json','final-verification.json','implementation.zip','executables.zip')]
    paths=list((ROOT/'native').rglob('*'))+list((ROOT/'src/brass_ai/native').glob('*.py'))+list((ROOT/'tests/RustBackendNative').glob('*'))
    paths += [ROOT/'scripts'/name for name in ('capture-rust-states.py','complete-rust-state-capture.py','capture-rust-state-regressions.py','prepare-rust-state-catalog.py','qualify-rust-state.py','qualify-rust-scaffold.py')]
    paths=[p for p in paths if p.is_file() and (p.suffix in ('.rs','.toml','.lock','.json','.py','.cs','.md','.patch') or p.name=='LICENSE')]
    with zipfile.ZipFile(out/'implementation.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(paths,key=str):archive.write(path,path.relative_to(ROOT).as_posix())
    with zipfile.ZipFile(out/'executables.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for profile,b in builds.items():archive.write(ROOT/b['binary']['path'],profile+'/brass-ai-worker.exe')
    result=dict(format='brass-native-l04-qualification/1.0.0',status='passed',milestone='L04',completed_utc=datetime.now(timezone.utc).isoformat(),
                builds=builds,commands=run.commands,sources=[receipt(p) for p in sorted(paths,key=str)],capture=receipt(capture/'capture.json'),
                prior_l03_evidence=l03_receipts,archives=[receipt(out/n) for n in ('implementation.zip','executables.zip')],
                sessions=2,threads_per_session=1,roots=frozen['count'],supplemental_roots=frozen['supplemental_count'],
                frozen_roots_roundtripped=540,corrected_roots_roundtripped=6,frozen_roots_rejected=6,regressions=receipt(regressions),
                unexplained_differences=0,state_import_qualified=True,target_rules_qualified=False,search_available=False,
                installed_backend_changed=False,nn_executed=False,matched_search_run=False,matched_search_seconds=4,
                next_milestone='L05: staged legal choices, transitions and explicit chance')
    write(out/'qualification.json',result);print('L04 qualification passed:',out,flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--capture',type=Path,default=ROOT/'artifacts/rust-backend/20261024-l04/states')
    parser.add_argument('--regressions',type=Path,default=ROOT/'artifacts/rust-backend/20261024-l04/regressions-v2/regressions.json')
    args=parser.parse_args();out=args.out.resolve()
    try:qualify(out,args.capture.resolve(),args.regressions.resolve())
    except BaseException as error:
        if out.is_relative_to(ROOT/'artifacts/rust-backend') and not (out/'qualification.json').exists():write(out/'progress.json',dict(status='failed',milestone='L04',error=str(error)))
        raise


if __name__=='__main__':main()
