"""Seal L09 sources, export, runtime, three-build parity and two-owner measurements."""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import runpy
import shutil
import sys
import zipfile
ROOT=Path(__file__).resolve().parents[1]
L08=runpy.run_path(str(ROOT/'scripts/qualify-rust-features.py'))
L05,BASE=L08['L05'],L08['BASE']
read,write,receipt,sha=(BASE[k] for k in ('read','write','receipt','sha'))


def sources():
    return sorted(set(L08['source_paths']()+list((ROOT/'tests/RustBackendInference').glob('*.py'))),key=str)


def qualify(out):
    assert out.is_relative_to(ROOT/'artifacts/rust-backend') and not out.exists()
    out.mkdir();write(out/'progress.json',dict(status='running',milestone='L09'))
    reference=L05['prepare_reference'](out)
    before=BASE['verify_frozen'](reference_source_root=reference);write(out/'preservation-before.json',before)
    prior=ROOT/'artifacts/rust-backend/20261025-l08-qualified-v3'
    prior_receipts=[receipt(prior/n) for n in ('qualification.json','final-verification.json','implementation.zip','executables.zip')]
    captures=[ROOT/'artifacts/rust-backend'/n for n in ('20261025-l05-capture-v2','20261025-l05-branches','20261025-l06-capture','20261025-l06-edges')]
    frozen=[L05['verify_capture'](p) for p in captures];write(out/'capture-receipts.json',frozen)
    shutil.copytree(prior/'bench-inputs',out/'bench-inputs')
    source_before=[receipt(p) for p in sources()];write(out/'sources-before.json',source_before)
    identity=BASE['native_fingerprint']()
    run=BASE['Qualification'](out);run.env.pop('PROPTEST_RNG_SEED',None);run.env['CARGO_INCREMENTAL']='0'
    run.env['PYTHONPATH']=os.pathsep.join([str(reference/'src'),*[p for p in run.env['PYTHONPATH'].split(os.pathsep) if Path(p)!=ROOT/'src']])
    model=out/'model'
    run.run('export',[sys.executable,ROOT/'scripts/prepare-rust-onnx.py','--reference',reference,'--out',model])
    run.run('golden-fixtures',[sys.executable,ROOT/'scripts/prepare-rust-inference-fixtures.py','--reference',reference,'--manifest',model/'manifest.json','--out',out/'golden.json'])
    model_receipts=[receipt(p) for p in model.iterdir() if p.is_file()]
    run.env['BRASS_ONNX_MANIFEST']=str(model/'manifest.json');run.env['BRASS_ONNX_GOLDEN']=str(out/'golden.json')
    run.run('generated-layout',[sys.executable,ROOT/'scripts/prepare-rust-feature-layout.py','--reference',reference,'--check'])
    run.cargo_run('fmt','fmt','--all','--','--check')
    run.cargo_run('clippy','clippy','--workspace','--all-targets','--locked','--offline','--','-D','warnings')
    for profile,flags in [('debug',[]),('release',['--release'])]:
        run.cargo_run(profile+'-tests','test','--workspace',*flags,'--locked','--offline')
        run.cargo_run(profile+'-golden-runtime','test','-p','brass-ai-inference','--test','runtime',*flags,'--locked','--offline','--','--ignored','--nocapture')
    builds={}
    sys.path.insert(0,str(reference/'src'))
    from brass_ai.native.protocol import NativeWorker
    for profile,flags in [('debug',[]),('release',['--release']),('profiling',['--profile','profiling'])]:
        run.cargo_run(profile+'-build','build','-p','brass-ai-worker',*flags,'--locked','--offline')
        binary=ROOT/'artifacts/rust-native-target'/BASE['TARGET']/profile/'brass-ai-worker.exe'
        with NativeWorker(binary) as worker:
            hello=worker.hello
            assert hello['build']['source_sha256']==identity==BASE['native_fingerprint']()
            assert hello['build']['cargo_lock_sha256']==sha(ROOT/'native/Cargo.lock')
            assert hello['sessions']==2 and hello['threads_per_session']==1
        builds[profile]=dict(binary=receipt(binary),handshake=hello,pe_machine=BASE['machine_type'](binary))
        env=dict(run.env,BRASS_NATIVE_WORKER=str(binary))
        for suite,folder,pattern in [('inference','RustBackendInference','test*.py'),('features','RustBackendFeatures','test*.py'),('transport','RustBackendNative','test_worker.py')]:
            run.run(profile+'-'+suite+'-tests',[sys.executable,'-m','unittest','discover','-s',ROOT/'tests'/folder,'-p',pattern,'-v'],env=env)
        args=[sys.executable,ROOT/'scripts/verify-rust-inference.py','--reference',reference,'--worker',binary,'--manifest',model/'manifest.json','--report',out/(profile+'-parity.json')]
        if profile=='profiling':args+=['--limit','100']
        run.run(profile+'-parity',args,env=env,timeout=3600)
    run.run('two-owner-batch-sweep',[sys.executable,ROOT/'scripts/benchmark-rust-inference.py','--reference',reference,
        '--worker',ROOT/builds['release']['binary']['path'],'--manifest',model/'manifest.json','--fixtures',out/'bench-inputs','--report',out/'batch-sweep.json'],timeout=1200)
    run.cargo_run('allocations-build','build','-p','brass-ai-inference','--example','allocations','--release','--locked','--offline')
    allocation=ROOT/'artifacts/rust-native-target'/BASE['TARGET']/'release/examples/allocations.exe'
    for name in ('initial','dense','complex'):
        log=run.run(name+'-allocations',[allocation,model/'manifest.json',out/'bench-inputs'/(name+'.json')])
        write(out/(name+'-allocations.json'),read(log))
    run.cargo_run('dependencies','tree','--workspace','--edges','features','--locked','--offline')
    run.cargo_run('cargo-metadata','metadata','--format-version','1','--locked','--offline')
    # Keep license texts with the sealed candidate artifact. No game install.
    notices=out/'notices';notices.mkdir()
    cache=ROOT/'artifacts/native-toolchain/cargo/registry/src'
    for name in ('ort-2.0.0-rc.10','ort-sys-2.0.0-rc.10'):
        for path in cache.glob('*/'+name+'/LICENSE*'):shutil.copyfile(path,notices/(name+'-'+path.name))
    after=BASE['verify_frozen'](reference_source_root=reference);write(out/'preservation-after.json',after)
    assert L05['preserved_identity'](before)==L05['preserved_identity'](after)
    assert prior_receipts==[receipt(prior/n) for n in ('qualification.json','final-verification.json','implementation.zip','executables.zip')]
    assert frozen==[L05['verify_capture'](p) for p in captures]
    assert source_before==[receipt(p) for p in sources()] and identity==BASE['native_fingerprint']()
    assert model_receipts==[receipt(p) for p in model.iterdir() if p.is_file()]
    for overlay in read(out/'python-reference.json')['candidate_overlay']:
        assert receipt(ROOT/overlay['candidate']['path'])==overlay['candidate'] and receipt(ROOT/overlay['executed']['path'])==overlay['executed']
    with zipfile.ZipFile(out/'implementation.zip','w',zipfile.ZIP_DEFLATED) as z:
        for path in sources():z.write(path,path.relative_to(ROOT).as_posix())
    with zipfile.ZipFile(out/'executables.zip','w',zipfile.ZIP_DEFLATED) as z:
        for profile,b in builds.items():z.write(ROOT/b['binary']['path'],profile+'/brass-ai-worker.exe')
        for path in list(model.iterdir())+list(notices.iterdir()):
            if path.is_file():z.write(path,path.relative_to(out).as_posix())
        z.write(allocation,'probes/allocations.exe')
    reports=[receipt(p) for p in out.glob('*parity.json')]+[receipt(out/(n+'.json')) for n in ('batch-sweep','initial-allocations','dense-allocations','complex-allocations')]
    write(out/'qualification.json',dict(status='passed',milestone='L09',completed_utc=datetime.now(timezone.utc).isoformat(),
        source_sha256=identity,builds=builds,commands=run.commands,sources=source_before,prior_evidence=prior_receipts,
        python_reference=receipt(out/'python-reference.json'),model=model_receipts,golden=receipt(out/'golden.json'),reports=reports,
        archives=[receipt(out/n) for n in ('implementation.zip','executables.zip')],sessions=2,threads_per_session=1,
        nn_executed=True,external_evaluator_required=False,calibration_ready=False,installed_backend_changed=False,
        matched_search_run=False,search_implemented=False,next_milestone='L10'))
    write(out/'progress.json',dict(status='passed',milestone='L09'))
    print('L09 qualification passed',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    try:qualify(a.out.resolve())
    except BaseException as error:
        if a.out.resolve().is_relative_to(ROOT/'artifacts/rust-backend') and a.out.exists() and not (a.out/'qualification.json').exists():
            write(a.out/'progress.json',dict(status='failed',milestone='L09',error=repr(error)))
        raise
