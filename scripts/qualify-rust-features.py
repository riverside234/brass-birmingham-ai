"""L08 frozen-source feature/NN parity, failure tests and native profiling."""
import argparse
from datetime import datetime, timezone
import gzip
import json
import os
from pathlib import Path
import runpy
import shutil
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
L05 = runpy.run_path(str(ROOT/'scripts/qualify-rust-rules.py'))
BASE = L05['BASE']
read, write, receipt, sha = (BASE[k] for k in ('read', 'write', 'receipt', 'sha'))


def source_paths():
    paths=list((ROOT/'native').rglob('*'))+list((ROOT/'src/brass_ai/native').glob('*.py'))
    paths+=list((ROOT/'tests/RustBackendFeatures').rglob('*'))+list((ROOT/'tests/RustBackendNative').rglob('*'))
    paths+=list((ROOT/'configs/contracts/rust-backend').glob('*.md'))+list((ROOT/'scripts').glob('*rust*.py'))
    return sorted({p for p in paths if p.is_file() and (p.suffix in ('.rs','.toml','.lock','.json','.py','.cs','.md','.patch') or p.name=='LICENSE')},key=str)


def qualify(out):
    assert out.is_relative_to(ROOT/'artifacts/rust-backend') and not out.exists()
    out.mkdir(parents=True)
    write(out/'progress.json', dict(status='running', milestone='L08'))
    reference = L05['prepare_reference'](out)
    before = BASE['verify_frozen'](reference_source_root=reference)
    write(out/'preservation-before.json', before)
    prior = [ROOT/'artifacts/rust-backend/20261025-l07-qualified-v4'/name for name in
             ('qualification.json', 'final-verification.json', 'implementation.zip', 'executables.zip')]
    prior_receipts = [receipt(p) for p in prior]
    captures = [ROOT/'artifacts/rust-backend'/f for f in
        ('20261025-l05-capture-v2', '20261025-l05-branches', '20261025-l06-capture', '20261025-l06-edges')]
    frozen = [L05['verify_capture'](p) for p in captures]
    for capture in frozen[2:]:
        for field in ('exporter', 'script', 'python_reference', 'rules_capture', 'source'):
            if field in capture: assert receipt(ROOT/capture[field]['path']) == capture[field]
        for field in ('runtime', 'inputs', 'reference_sources'):
            for item in capture.get(field, []): assert receipt(ROOT/item['path']) == item
    write(out/'capture-receipts.json', frozen)
    roots = [json.loads(line) for line in gzip.open(captures[0]/'roots.jsonl.gz', 'rt', encoding='utf-8')]
    development = [r for r in roots if r['partition']=='development' and r['id'].startswith('game-')]
    complex_root = max((r for r in development if r['expected']['state']['context']['phase']=='payment'
                       and r['expected']['state']['context']['era']=='RailEra'),
                      key=lambda r: (sum('tile' in s for s in r['expected']['state']['slots']),len(r['expected']['state']['history'])))
    dense = max((r for r in development if r['expected']['state']['context']['phase']=='action'), key=lambda r:len(r['expected']['choices']))
    selection = dict(initial=next(r for r in roots if r['id']=='game-731001-root-000'), complex=complex_root, dense=dense)
    bench = out/'bench-inputs'; bench.mkdir()
    for name, row in selection.items(): write(bench/(name+'.json'), row['expected']['state'])
    write(bench/'selection.json', dict(roots={k:v['id'] for k,v in selection.items()}, held_out_used=False,
        criterion='Initial; development Rail payment sorted by occupied slots/history; development action with most legal candidates.'))
    source_identity = BASE['native_fingerprint']()
    sources_before = [receipt(p) for p in source_paths()]
    write(out/'sources-before.json', sources_before)
    run = BASE['Qualification'](out)
    run.env.pop('PROPTEST_RNG_SEED', None); run.env['CARGO_INCREMENTAL']='0'
    run.env['PYTHONPATH']=os.pathsep.join([str(reference/'src'), *[p for p in run.env['PYTHONPATH'].split(os.pathsep) if Path(p)!=ROOT/'src']])
    run.env['BRASS_FEATURE_BENCH_DIR']=str(bench)
    run.run('generated-layout', [sys.executable,ROOT/'scripts/prepare-rust-feature-layout.py','--reference',reference,'--check'])
    run.cargo_run('fmt','fmt','--all','--','--check')
    run.cargo_run('clippy','clippy','--workspace','--all-targets','--locked','--offline','--','-D','warnings')
    for profile, flags in [('debug',[]),('release',['--release'])]:
        run.cargo_run(profile+'-tests','test','--workspace',*flags,'--locked','--offline')
    builds = {}
    # Import the separately identified client only after selecting the frozen reference.
    sys.path.insert(0,str(reference/'src'))
    from brass_ai.native.protocol import NativeWorker
    for profile, flags in [('debug',[]),('release',['--release']),('profiling',['--profile','profiling'])]:
        run.cargo_run(profile+'-build','build','-p','brass-ai-worker',*flags,'--locked','--offline')
        binary=ROOT/'artifacts/rust-native-target'/BASE['TARGET']/profile/'brass-ai-worker.exe'
        with NativeWorker(binary) as worker:
            hello=worker.hello
            assert hello['build']['source_sha256']==source_identity==BASE['native_fingerprint']()
            assert hello['build']['cargo_lock_sha256']==sha(ROOT/'native/Cargo.lock')
            assert hello['sessions']==2 and hello['threads_per_session']==1
        builds[profile]=dict(binary=receipt(binary),handshake=hello,pe_machine=BASE['machine_type'](binary))
        env=dict(run.env,BRASS_NATIVE_WORKER=str(binary))
        run.run(profile+'-feature-process-tests',[sys.executable,'-m','unittest','discover','-s',ROOT/'tests/RustBackendFeatures','-v'],env=env)
        run.run(profile+'-worker-regressions',[sys.executable,'-m','unittest','discover','-s',ROOT/'tests/RustBackendNative','-p','test_worker.py','-v'],env=env)
        args=[sys.executable,ROOT/'scripts/verify-rust-features.py','--reference',reference,'--worker',binary,
              '--report',out/(profile+'-feature-parity.json')]
        if profile=='release': args+=['--nn']
        if profile=='profiling': args+=['--limit','100']
        run.run(profile+'-feature-parity',args,env=env,timeout=3600)
    run.cargo_run('criterion','bench','-p','brass-ai-core','--bench','features','--locked','--offline')
    shutil.copytree(ROOT/'artifacts/rust-native-target/criterion/features',out/'criterion')
    for name in selection:
        log=run.cargo_run(name+'-allocations','run','-p','brass-ai-core','--example','features_allocations','--release','--locked','--offline','--',bench/(name+'.json'))
        text=log.read_text(encoding='utf-8'); start=text.index('\n{')+1
        write(out/(name+'-allocations.json'),json.loads(text[start:]))
    run.cargo_run('dependencies','tree','--workspace','--edges','features','--locked','--offline')
    after=BASE['verify_frozen'](reference_source_root=reference)
    write(out/'preservation-after.json',after)
    assert L05['preserved_identity'](before)==L05['preserved_identity'](after)
    assert prior_receipts==[receipt(p) for p in prior]
    assert source_identity==BASE['native_fingerprint']()
    assert frozen==[L05['verify_capture'](p) for p in captures]
    for overlay in read(out/'python-reference.json')['candidate_overlay']:
        assert receipt(ROOT/overlay['candidate']['path'])==overlay['candidate']
        assert receipt(ROOT/overlay['executed']['path'])==overlay['executed']
    paths=source_paths()
    assert sources_before==[receipt(p) for p in paths], 'Candidate/harness sources changed during qualification'
    with zipfile.ZipFile(out/'implementation.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for p in paths: archive.write(p,p.relative_to(ROOT).as_posix())
    with zipfile.ZipFile(out/'executables.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for profile,build in builds.items(): archive.write(ROOT/build['binary']['path'],profile+'/brass-ai-worker.exe')
    reports=[receipt(out/(p+'-feature-parity.json')) for p in builds]
    write(out/'qualification.json',dict(status='passed',milestone='L08',completed_utc=datetime.now(timezone.utc).isoformat(),
        builds=builds,commands=run.commands,sources=[receipt(p) for p in paths],prior_evidence=prior_receipts,
        archives=[receipt(out/n) for n in ('implementation.zip','executables.zip')],captures=[receipt(p/'capture.json') for p in captures],
        python_reference=receipt(out/'python-reference.json'),parity=reports,sessions=2,threads_per_session=1,
        feature_shape=[3128,327],descriptor_shape=[4,34],exact_input_equality=True,output_atol=1e-6,output_rtol=1e-6,
        calibration_ready=False,installed_backend_changed=False,matched_search_run=False,onnx_executed=False,next_milestone='L09'))
    write(out/'progress.json',dict(status='passed',milestone='L08'))
    print('L08 qualification passed',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    try: qualify(args.out.resolve())
    except BaseException as error:
        if args.out.resolve().is_relative_to(ROOT/'artifacts/rust-backend') and args.out.exists() and not (args.out/'qualification.json').exists():
            write(args.out/'progress.json',dict(status='failed',milestone='L08',error=str(error)))
        raise
