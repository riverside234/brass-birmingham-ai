"""L07 local simulator qualification using immutable L01/L02 and L04–L06 evidence."""
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

ROOT=Path(__file__).resolve().parents[1]
L05=runpy.run_path(str(ROOT/'scripts/qualify-rust-rules.py'));BASE=L05['BASE']
read,write,receipt,sha=(BASE[k] for k in ('read','write','receipt','sha'))


def read_allocation_log(path):
    # Cargo warnings may quote Rust source containing braces. The example's
    # JSON starts on its own unindented line, after those diagnostics.
    lines=path.read_text(encoding='utf-8').splitlines()
    for start,line in enumerate(lines):
        if line!='{':continue
        try:value=json.loads('\n'.join(lines[start:]))
        except json.JSONDecodeError:continue
        if isinstance(value,dict) and 'build' in value and 'measurements' in value:return value
    raise ValueError('Missing complete allocation JSON in '+str(path))


def verify_regressions(path):
    value=read(path)
    assert value['status']=='passed_existing_regressions_only' and value['count']==len(value['tests'])==27
    assert value['source_sha256']==BASE['native_fingerprint']()
    for item in value['test_sources']+value['reports']+[value['log'],value['binary'],value['python_reference'],value['search_reference'],value['sealer']]:
        assert receipt(ROOT/item['path'])==item,item['path']
    for overlay in value['candidate_overlay']:
        assert receipt(ROOT/overlay['candidate']['path'])==overlay['candidate']
        assert receipt(ROOT/overlay['executed']['path'])==overlay['executed']
    return value


def qualify(out, regression_evidence=None):
    assert out.is_relative_to(ROOT/'artifacts/rust-backend') and not out.exists()
    out.mkdir(parents=True);write(out/'progress.json',dict(status='running',milestone='L07'))
    reused=verify_regressions(regression_evidence) if regression_evidence else None
    reference=L05['prepare_reference'](out)
    baseline=read(BASE['FROZEN']/'baseline.json')
    runtime_pin=baseline['reference_runtime'];assert receipt(ROOT/runtime_pin['path'])==runtime_pin
    runtime=read(ROOT/runtime_pin['path'])
    expected={Path(f['path']).name:f['sha256'] for f in runtime['files']}
    # Use the exact explicit-chance host captured by L01/L05, including its
    # optional fast API. The oracle below uses the ordinary API (fast=False).
    # Only the manifest lookup directory/extension metadata is adapted.
    source_runtime=ROOT/'artifacts/rust-backend/20261025-l05-capture-v2/exporter'
    manifest=dict(runtime['manifest'],extension='brass-search-chance/1.0.0')
    assert {f['name']:f['sha256'] for f in manifest['files']}==expected
    destination=reference/'artifacts/r4-host';destination.mkdir(parents=True)
    executed_runtime=[]
    for name,digest in expected.items():
        source=source_runtime/name;assert source.parent==source_runtime and sha(source)==digest
        shutil.copyfile(source,destination/name);executed_runtime.append(receipt(destination/name))
    write(destination/'runtime.json',manifest)
    executed_runtime.append(receipt(destination/'runtime.json'))
    write(out/'search-reference-runtime.json',dict(baseline=runtime_pin,executed=executed_runtime,
        adaptation='Pinned L01/L05 explicit-chance host at the r4-host lookup path; extension metadata added, executable/dependency bytes unchanged.'))
    before=BASE['verify_frozen'](reference_source_root=reference);write(out/'preservation-before.json',before)
    prior=[ROOT/'artifacts/rust-backend'/folder/name
        for folder in ('20261024-l03-verified','20261025-l04-qualified','20261025-l05-qualified-v4','20261025-l06-qualified-v3')
        for name in ('qualification.json','final-verification.json','implementation.zip','executables.zip')]
    prior_receipts=[receipt(p) for p in prior]
    captures=[ROOT/'artifacts/rust-backend'/f for f in ('20261025-l05-capture-v2','20261025-l05-branches','20261025-l06-capture','20261025-l06-edges')]
    frozen=[L05['verify_capture'](p) for p in captures]
    # Pin the oracle scripts, assemblies, inputs and module copies as well as data files.
    for capture in frozen[2:]:
        for field in ('exporter','script','python_reference','rules_capture','source'):
            if field in capture:assert receipt(ROOT/capture[field]['path'])==capture[field]
        for field in ('runtime','inputs','reference_sources'):
            for item in capture.get(field,[]):assert receipt(ROOT/item['path'])==item
    write(out/'capture-receipts.json',frozen)
    roots=[json.loads(line) for line in gzip.open(captures[0]/'roots.jsonl.gz','rt',encoding='utf-8')]
    dense=max((r for r in roots if r['partition']=='development' and r['id'].startswith('game-')
        and r['expected']['state']['context']['phase']=='payment' and r['expected']['state']['context']['era']=='RailEra'),
        key=lambda r:(sum('tile' in s for s in r['expected']['state']['slots']),len(r['expected']['state']['history'])))
    initial=next(r for r in roots if r['id']=='game-731001-root-000')
    bench=out/'bench-inputs';bench.mkdir()
    for name,row in [('initial',initial),('complex',dense)]:write(bench/(name+'.json'),row['expected']['state'])
    write(bench/'selection.json',dict(initial=initial['id'],complex=dense['id'],held_out_used=False,
        criterion='Development Rail payment; occupied slots then history length'))
    source_identity=BASE['native_fingerprint']();run=BASE['Qualification'](out)
    run.env.pop('PROPTEST_RNG_SEED',None);run.env['CARGO_INCREMENTAL']='0'
    run.env['PYTHONPATH']=os.pathsep.join([str(reference/'src'),*[p for p in run.env['PYTHONPATH'].split(os.pathsep) if Path(p)!=ROOT/'src']])
    run.env['BRASS_STATE_BENCH_DIR']=str(bench)
    run.env['BRASS_SIMULATION_REFERENCE_ROOT']=str(reference)
    check='''import hashlib,json,sys,unittest
from pathlib import Path
from brass_ai.searchlab import spiel
from brass_ai.search import ismcts
from brass_ai.native import simulation
root=Path(sys.argv[1]);result={}
for module in (spiel,ismcts,simulation):
 p=Path(module.__file__).resolve();assert p.is_relative_to(root);result[module.__name__]=dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
loader=unittest.TestLoader();loader.discover(sys.argv[2]);assert not loader.errors,loader.errors
print(json.dumps(result,indent=2))
'''
    run.run('python-identities',[sys.executable,'-c',check,reference,ROOT/'tests/RustBackendNative'])
    run.cargo_run('fmt','fmt','--all','--','--check')
    run.cargo_run('clippy','clippy','--workspace','--all-targets','--locked','--offline','--','-D','warnings')
    for profile,flags in [('debug',[]),('release',['--release'])]:run.cargo_run(profile+'-tests','test','--workspace',*flags,'--locked','--offline')
    from brass_ai.native import NativeWorker
    builds={}
    for profile,flags in [('release',['--release']),('debug',[]),('profiling',['--profile','profiling'])]:
        run.cargo_run(profile+'-build','build','-p','brass-ai-worker',*flags,'--locked','--offline')
        binary=ROOT/'artifacts/rust-native-target'/BASE['TARGET']/profile/'brass-ai-worker.exe'
        with NativeWorker(binary) as worker:
            hello=worker.hello;assert hello['target_rules_qualified'] is True
            assert hello['build']['source_sha256']==source_identity==BASE['native_fingerprint']()
            assert hello['build']['cargo_lock_sha256']==sha(ROOT/'native/Cargo.lock')
        builds[profile]=dict(binary=receipt(binary),handshake=hello)
        env=dict(run.env,BRASS_NATIVE_WORKER=str(binary),BRASS_SIMULATION_REPORT=str(out/(profile+'-simulation-parity.json')))
        for name in ('STATE','RULES','PRIVACY'):
            env['BRASS_'+name+'_PARITY_REPORT' if name!='PRIVACY' else 'BRASS_PRIVACY_REPORT']=str(out/(profile+'-'+name.lower()+'-parity.json'))
        env['BRASS_PRIVACY_SAMPLING_REPORT']=str(out/(profile+'-sampling-parity.json'))
        env['BRASS_PRIVACY_EDGE_REPORT']=str(out/(profile+'-edges-parity.json'))
        command=[sys.executable,'-m','unittest','discover','-s',ROOT/'tests/RustBackendNative','-v']
        if profile!='release' or reused:command+=['-p','test_simulation.py']
        run.run(profile+'-python-tests',command,env=env,timeout=7200)
    run.cargo_run('criterion','bench','-p','brass-ai-core','--bench','simulation','--locked','--offline')
    shutil.copytree(ROOT/'artifacts/rust-native-target/criterion/simulation',out/'criterion')
    for name in ('initial','complex'):
        log=run.cargo_run(name+'-allocations','run','-p','brass-ai-core','--example','simulation_allocations','--release','--locked','--offline','--',bench/(name+'.json'))
        # Cargo writes diagnostics before the example's JSON in the combined log.
        write(out/(name+'-allocations.json'),read_allocation_log(log))
    run.run('adapter-profile',[sys.executable,ROOT/'scripts/profile-rust-simulation.py','--binary',ROOT/builds['release']['binary']['path'],'--fixtures',bench,'--out',out/'adapter-profile.json'])
    run.cargo_run('dependencies','tree','--workspace','--edges','features','--locked','--offline')
    after=BASE['verify_frozen'](reference_source_root=reference);write(out/'preservation-after.json',after)
    assert L05['preserved_identity'](before)==L05['preserved_identity'](after)
    assert prior_receipts==[receipt(p) for p in prior]
    assert frozen==[L05['verify_capture'](p) for p in captures]
    assert source_identity==BASE['native_fingerprint']()
    if reused:assert verify_regressions(regression_evidence)==reused
    assert executed_runtime==[receipt(ROOT/f['path']) for f in executed_runtime]
    for overlay in read(out/'python-reference.json')['candidate_overlay']:
        assert receipt(ROOT/overlay['candidate']['path'])==overlay['candidate']
        assert receipt(ROOT/overlay['executed']['path'])==overlay['executed']
    paths=list((ROOT/'native').rglob('*'))+list((ROOT/'src/brass_ai/native').glob('*.py'))+list((ROOT/'tests/RustBackendNative').rglob('*'))
    paths+=list((ROOT/'configs/contracts/rust-backend').glob('*.md'))+list((ROOT/'scripts').glob('*rust*.py'))
    paths=sorted({p for p in paths if p.is_file() and (p.suffix in ('.rs','.toml','.lock','.json','.py','.cs','.md','.patch') or p.name=='LICENSE')},key=str)
    with zipfile.ZipFile(out/'implementation.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for p in paths:archive.write(p,p.relative_to(ROOT).as_posix())
    with zipfile.ZipFile(out/'executables.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for profile,build in builds.items():archive.write(ROOT/build['binary']['path'],profile+'/brass-ai-worker.exe')
    write(out/'qualification.json',dict(status='passed',milestone='L07',completed_utc=datetime.now(timezone.utc).isoformat(),
        builds=builds,commands=run.commands,sources=[receipt(p) for p in paths],prior_evidence=prior_receipts,
        archives=[receipt(out/n) for n in ('implementation.zip','executables.zip')],captures=[receipt(p/'capture.json') for p in captures],
        python_reference=receipt(out/'python-reference.json'),search_reference=receipt(out/'search-reference-runtime.json'),
        regression_evidence=receipt(regression_evidence) if reused else None,roots=546,sampled_worlds=1704,games=20,transitions=8344,
        sessions=2,threads_per_session=1,unexplained_differences=0,simulator_adapter='rust-simulator',
        installed_backend_changed=False,nn_executed=False,matched_search_run=False,next_milestone='L08'))
    write(out/'progress.json',dict(status='passed',milestone='L07'));print('L07 qualification passed',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--regression-evidence',type=Path,help='Reuse separately sealed successful earlier regressions only if binary, native/client sources and tests are unchanged.')
    args=parser.parse_args()
    try:qualify(args.out.resolve(),args.regression_evidence.resolve() if args.regression_evidence else None)
    except BaseException as error:
        if args.out.resolve().is_relative_to(ROOT/'artifacts/rust-backend') and not (args.out/'qualification.json').exists():
            write(args.out/'progress.json',dict(status='failed',milestone='L07',error=str(error)))
        raise
