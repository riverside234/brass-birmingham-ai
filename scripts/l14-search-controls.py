"""Independent L14 controls. Never equates Masked visits with Gumbel trajectories.

Uses sealed Python/C# reference modules and the admitted temporary g890 model.
Count-mode comparisons are deliberately not labelled timed whole-turn parity.
"""
import argparse
import copy
import ctypes
from concurrent.futures import ThreadPoolExecutor
import gzip
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import runpy
import shutil
import statistics
import subprocess
import sys
import threading
import time

ROOT=Path(__file__).resolve().parents[1]
sys.dont_write_bytecode=True
RUNNER_SOURCE=Path(__file__).read_bytes()
BASE=runpy.run_path(str(ROOT/'scripts/benchmark-rust-backend.py'))
read,write,sha,receipt=(BASE[k] for k in ('read','write','sha','receipt'))
L10,MODEL=BASE['L10'],BASE['MODEL']
REFERENCE=ROOT/'artifacts/rust-backend/20261025-l09-qualified-v2/reference-source'
HOST_ROOT=ROOT/'artifacts/rust-backend/20261025-l07-qualified-v4/reference-source'
FROZEN=ROOT/'artifacts/rust-backend/20261024-g890-l01-l02'


def pin_runner(folder):
    target=folder/'l14-search-controls.py';target.write_bytes(RUNNER_SOURCE)
    write(folder/'runner-source.json',receipt(target))


def imports():
    env,_=BASE['SUPPORT']['tool_environment']()
    for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
        os.environ[name]='1'
    for path in reversed(env['PYTHONPATH'].split(os.pathsep)):sys.path.insert(0,path)
    sys.path.insert(0,str(REFERENCE/'src'))
    import torch
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    return env


def distribution(values):
    a=sorted(values)
    return dict(n=len(a),median=statistics.median(a),p95=a[math.ceil(.95*len(a))-1],maximum=max(a))


def fixed(out,label=None):
    protocol=BASE['checked'](out);imports()
    from brass_ai.native.protocol import NativeWorker
    gp=runpy.run_path(str(ROOT/'tests/RustBackend/gumbel_brass_reference.py'))
    compare=runpy.run_path(str(ROOT/'scripts/verify-rust-search.py'))['compare']
    native=ROOT/'artifacts/rust-native-target/x86_64-pc-windows-gnu/release/gumbel-trace.exe'
    folder=out/(label or 'fixed-controls');folder.mkdir(exist_ok=False)
    pin_runner(folder)
    selections=[3,26,30,41,0,8]
    write(folder/'protocol.json',dict(indices=selections,repetitions=5,sessions=2,threads_per_session=1,
        scripted='Identical shuffle/policy draws and recorded all-actor NN replies; compare complete traces',
        real='Same pinned ONNX export, one CPU thread per session, fixed taped worlds/draws',
        timed_whole_turn=False,scope='Count-mode tree/rules/adapter control. L11/L12 timed extensions are not implemented by G_py.',
        native_binary=receipt(native),python_source=receipt(ROOT/'tests/RustBackend/gumbel_brass_reference.py')))
    records=[];worker=NativeWorker(L10/'reference-worker.exe',timeout=120)
    pools=[ThreadPoolExecutor(1,thread_name_prefix=f'l14-owner-{i}') for i in range(2)]
    class Owner:
        def __init__(self,index):
            self.index=index
            self.neural=gp['Neural'](MODEL/'model.onnx')
        def simulation(self,unused,kind,**kw):return worker.simulation(self.index,kind,**kw)
        def python(self,case,real):
            evaluator=self.neural if real else gp['Recorded'](case['replies'])
            self.neural.replies=[]
            start=time.perf_counter_ns();cpu=time.thread_time_ns()
            result=gp['search'](self,case['input'],case['config'],evaluator,case['draws'])
            elapsed=time.perf_counter_ns()-start
            if not real:evaluator.finish()
            return result,dict(wall_ns=elapsed,cpu_ns=time.thread_time_ns()-cpu,thread=threading.get_native_id())
    owners=[p.submit(Owner,i).result() for i,p in enumerate(pools)]
    try:
        for index in selections:
            case=read(L10/f'brass-{index:03d}-request.json')['cases'][0]
            expected=read(L10/f'brass-{index:03d}-expected.json')
            for real in (False,True):
                label=f'brass-{index:03d}-'+('ort' if real else 'scripted')
                # Warm each Python owner and its exact model/fixture before timing.
                for f in [p.submit(o.python,case,real) for p,o in zip(pools,owners)]:f.result()
                native_case=copy.deepcopy(case)
                if real:native_case['replies']=None
                request=dict(cases=[native_case]*6)
                if real:request.update(manifest=str(MODEL/'manifest.json'),manifest_sha256=sha(MODEL/'manifest.json'))
                input_path=folder/(label+'-native-input.json');write(input_path,request)
                def run_native(owner):
                    target=folder/(label+f'-native-{owner}.json')
                    subprocess.run([str(native),str(input_path),str(target)],check=True,timeout=240,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
                    return read(target)['results'][1:]
                # Counterbalance implementation order between fixtures/modes.
                result_py=[];result_native=None
                for arm in (['python','rust'] if (index+int(real))%2==0 else ['rust','python']):
                    if arm=='rust':result_native=[f.result() for f in [p.submit(run_native,i) for i,p in enumerate(pools)]]
                    else:
                        for rep in range(5):
                            values=[f.result() for f in [p.submit(o.python,case,real) for p,o in zip(pools,owners)]]
                            for owner,(report,metrics) in enumerate(values):
                                compare(expected,report,tol=2e-4)
                                result_py.append(dict(repeat=rep,owner=owner,completed=len(report['trajectories']),**metrics))
                native_rows=[]
                for owner,reports in enumerate(result_native):
                    for rep,report in enumerate(reports):
                        compare(expected,report,tol=2e-4)
                        native_rows.append(dict(owner=owner,repeat=rep,**report['metrics']))
                # Native total_ns excludes file parse/write and final trace
                # serialization. Python wall includes diagnostic transport/report
                # construction. State this boundary; never call it bare-language speed.
                item=dict(label=label,index=index,real_nn=real,python=result_py,native=native_rows,parity=True,
                    diagnostic_throughput_ratio=statistics.median(r['wall_ns'] for r in result_py)/statistics.median(r['total_ns'] for r in native_rows))
                records.append(item);write(folder/'progress.json',dict(records=records));print('fixed',label,'passed',flush=True)
    finally:
        for p in pools:p.shutdown(wait=True)
        worker.close()
    write(folder/'results.json',dict(passed=True,records=records,
        attribution='Equivalent fixed traces; diagnostic Python includes simulator IPC and report overhead. Native internal timer excludes file transport. Not the L14 timed G_py/G gate.',
        gate_eligible=False,artifacts=[receipt(p) for p in sorted(folder.glob('*.json'))]))


def legacy(out,label=None):
    protocol=BASE['checked'](out);imports()
    historical=read(FROZEN/'local/matched-protocol.json')
    benchmark_path=ROOT/'src/brass_ai/rust_backend/benchmark.py'
    assert sha(benchmark_path)==historical['harness']['src/brass_ai/rust_backend/benchmark.py']
    name='brass_ai.rust_backend.benchmark'
    spec=importlib.util.spec_from_file_location(name,benchmark_path)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    from brass_ai.rust_backend.benchmark import Session,select_root,machine_status,process_samples
    from brass_ai.rust_backend.fixtures import read_gzip,restore,update_recall,advance
    from brass_ai.rust_backend.contracts import canonical
    from brass_ai.learning.checkpoint import load_checkpoint
    from brass_ai.learning.reward import CostLedger
    from brass_ai.search.common import SearchConfig,turn_key
    from brass_ai.search.ismcts import information_key,supported_continuation
    from brass_ai.recovery.parallel import merge_statistics
    from brass_ai.searchlab.spiel import BrassSpielGame
    from brass_ai.native.protocol import NativeWorker
    from brass_ai.native.simulation import NativeSpielGame
    folder=out/(label or 'legacy-controls');folder.mkdir(exist_ok=False)
    pin_runner(folder)
    # The reference loader may materialize its admitted fast host. Never give
    # it a sealed historical evidence directory as a writable runtime root.
    private_host=folder/'reference-host';shutil.copytree(HOST_ROOT,private_host,ignore=shutil.ignore_patterns('__pycache__'))
    # The L07 source archive contains the fast worker but omits its guarded
    # dependency bundle. Admit the existing bundle against that frozen worker's
    # manifest before copying it into this run's private runtime.
    guarded=ROOT/'artifacts/r1-host/guarded'
    fast_manifest=read(private_host/'training/runtime/throughput/search/runtime.json')
    guarded_manifest=read(guarded/'runtime.json')
    assert sha(guarded/'BrassHost.exe')==fast_manifest['baseline_host_sha256']
    assert {f['name']:f['sha256'] for f in guarded_manifest['files'] if f['name']!='BrassHost.exe'}=={
        f['name']:f['sha256'] for f in fast_manifest['files'] if f['name']!='BrassHost.exe'}
    private_guarded=private_host/'artifacts/r1-host/guarded';private_guarded.mkdir(parents=True)
    admitted=[]
    for entry in guarded_manifest['files']:
        name=entry['name'];assert Path(name).name==name
        assert sha(guarded/name)==entry['sha256'],name
        shutil.copyfile(guarded/name,private_guarded/name);admitted.append(receipt(private_guarded/name))
    shutil.copyfile(guarded/'runtime.json',private_guarded/'runtime.json')
    write(folder/'host-admission.json',dict(fast_manifest=receipt(private_host/'training/runtime/throughput/search/runtime.json'),
        guarded_manifest=receipt(private_guarded/'runtime.json'),files=admitted))
    from brass_ai.recovery.host_runtime import training_host
    training_host(private_host,'search')
    regression_pin=read(ROOT/'artifacts/rust-backend/20261025-l04-qualified/qualification.json')['regressions']
    regression_path=ROOT/regression_pin['path'];assert sha(regression_path)==regression_pin['sha256']
    corrected={c['original_id']:c for c in read(regression_path)['cases']}
    selected=[];corrections=[]
    for entry in historical['fixtures']:
        original=FROZEN/'local/fixtures'/entry['path'];assert sha(original)==entry['sha256']
        fixture=read_gzip(original)
        if entry['id'] in corrected:
            correction=corrected[entry['id']]
            assert correction['original_source_sha256']==entry['sha256']
            snapshot=copy.deepcopy(correction['snapshot'])
            for key in ('views','chance','result'):snapshot.setdefault(key,None)
            assert [v['own_hand'] for v in snapshot['views']]==[v['own_hand'] for v in fixture['projection']['views']]
            fixture.update(id=correction['id'],checkpoint=snapshot['checkpoint'],projection=canonical(snapshot))
            target=folder/(correction['id']+'.json');write(target,fixture)
            corrections.append(dict(original=entry,reasons=correction['rejection_reasons'],corrected_fixture=receipt(target)))
            entry=dict(id=correction['id'],seed=entry['seed'],partition=entry['partition'],corrected_fixture=receipt(target),
                       phase=snapshot['context']['phase'],legal_count=len(snapshot['choices']))
        selected.append((entry,fixture))
    write(folder/'fixture-admission.json',dict(regressions=regression_pin,corrections=corrections,
        reason='Use the independently corrected L04 captures for both arms; original invalid authored fixtures remain rejection tests.'))
    identity=read(FROZEN/'baseline.json')
    model,raw=load_checkpoint(ROOT/identity['model']['path']);model.eval()
    config=SearchConfig(seconds=4,contract='masked-ismcts/1.2.0',reward=model.config.utility,simulations=100000,
        pending_paths=1,batch_size=1,workers=1,cpu_role='reference').validate()
    native=NativeWorker(L10/'reference-worker.exe',timeout=60)
    pools=[ThreadPoolExecutor(1,thread_name_prefix=f'l14-legacy-{i}') for i in range(2)]
    def create(i):
        a=Session(model,private_host,config,i)
        try:b=Session(model,private_host,config,i)
        except Exception:
            a.close();raise
        b.policy.game.close()
        for game in b.policy.simulation_games:game.close()
        game=NativeSpielGame(worker=native,session=i,root=private_host)
        b.policy.game=game;b.policy.simulation_games=[game]
        return dict(A=a,B=b)
    owners=[];environment=None;rows=[]
    try:
        owners=[p.submit(create,i).result() for i,p in enumerate(pools)]
        environment=BrassSpielGame(root=private_host,fast=True)
        write(folder/'protocol.json',dict(model_id=raw['model_id'],search=config.to_dict(),
            whole_turn_ms=4000,sessions=2,threads_per_session=1,repetitions=5,fixtures=[e for e,_ in selected],
            order='AB on even repetition; BA on odd; sequential same-machine trials',before=machine_status(),
            reference_source=receipt(benchmark_path)))
        with gzip.open(folder/'turns.jsonl.gz','wt',encoding='utf-8') as stream:
            for rep in range(5):
                for entry,fixture in selected[rep%len(selected):]+selected[:rep%len(selected)]:
                    for arm in (['A','B'] if rep%2==0 else ['B','A']):
                        sessions=[owner[arm] for owner in owners];state=restore(environment.host,fixture['checkpoint'])
                        recall=copy.deepcopy(fixture['recall']);ledger=CostLedger(config.reward,**fixture['ledger']);seed=900000+rep*1000+entry['seed']
                        settings=[f.result() for f in [p.submit(s.prepare,fixture,seed) for p,s in zip(pools,sessions)]]
                        seat=state['context']['actor'];initial=turn_key(state['context']);start=time.monotonic();deadline=start+4
                        continuation={};steps=[];completed=attempted=interrupted=fallbacks=0
                        for step in range(96):
                            if state['chance'] or state['context']['terminal'] or state['context']['actor']!=seat or turn_key(state['context'])!=initial:break
                            update_recall(recall,state);view=state['views'][seat];choices=state['choices'];ledger.observe(view)
                            info=information_key(view,choices,recall[seat]);choice_id=continuation.get(info);work=[];merged=None
                            if len(choices)>1 and state['context']['phase']!='debt' and time.monotonic()<deadline:
                                work=[f.result() for f in [p.submit(s.decide,view,choices,recall[seat],ledger.snapshot(),deadline) for p,s in zip(pools,sessions)]]
                                merged=merge_statistics([w['stats'] for w in work],choices);best=select_root(merged,choices,work[0]['root_priors'])
                                supported=supported_continuation(merged,choices[best]['id'],seat,config.minimum_plan_visits)
                                if supported is not None:continuation=supported;choice_id=choices[best]['id']
                                else:continuation=work[0]['incumbent'];choice_id=choices[work[0]['selected']]['id'];fallbacks+=1
                                completed+=merged['completed'];attempted+=merged['batching']['attempted'];interrupted+=merged['interrupted']
                            if choice_id not in [c['id'] for c in choices]:
                                choice_id=max(choices,key=lambda c:(c['kind']=='finish',c['kind']=='beer' and bool(c.get('merchant')),-choices.index(c)))['id'];fallbacks+=int(len(choices)>1)
                            choice=next(c for c in choices if c['id']==choice_id);ledger.chosen(view,choice);state=advance(environment.host,state,dict(choice=choice_id))
                            steps.append(dict(choice=choice_id,phase=view['context']['phase'],elapsed=time.monotonic()-start,search=merged))
                        else:raise ValueError('legacy completion guard')
                        row=dict(arm=arm,repeat=rep,fixture=entry['id'],completed=completed,attempted=attempted,interrupted=interrupted,
                            fallbacks=fallbacks,seconds=time.monotonic()-start,steps=steps,owners=settings,valid_complete_turn=True)
                        stream.write(json.dumps(row,allow_nan=False)+'\n');stream.flush();rows.append({k:v for k,v in row.items() if k!='steps'})
                        write(folder/'progress.json',dict(turns=len(rows),latest=rows[-1]));print('legacy',arm,entry['id'],completed,flush=True)
    finally:
        for p,owner in zip(pools,owners):
            for s in owner.values():p.submit(s.close).result()
        for p in pools:p.shutdown(wait=True)
        native.close()
        if environment:environment.close()
    write(folder/'results.json',dict(passed=True,rows=rows,after=machine_status(),raw=receipt(folder/'turns.jsonl.gz'),
        scope='Same Python Masked search and synchronous PyTorch CPU evaluator; engine/adapter replacement only. Not a Gumbel speedup or strength result.'))


def native_components(out,label=None):
    protocol=BASE['checked'](out)
    bench=runpy.run_path(str(ROOT/'scripts/benchmark-rust-coordinator.py'))
    folder=out/(label or 'native-components');folder.mkdir(exist_ok=False)
    pin_runner(folder)
    binary=ROOT/'artifacts/rust-native-target/x86_64-pc-windows-gnu/release/gumbel-pair.exe'
    alloc=binary.with_name('gumbel-pair-alloc.exe')
    policy=bench['POLICIES'][0]
    selected=[3,26,30,41,0,8]
    def case(index,rep=0,seconds=60):
        value=bench['case'](index,policy,seconds=seconds,rep=rep,startup=7262519)
        value['config'].update({k:protocol['search_policy'][k] for k in ('k','c1','c2')})
        return value
    write(folder/'protocol.json',dict(sessions=2,threads_per_session=1,ordinary_turn_ms=4000,
        search_policy=protocol['search_policy'],indices=selected,repetitions=5,
        staged_roots='Separate diagnostic roots; excluded from whole-turn implementation ratios',
        startup='Exact retained L13 singleton startup estimate: 7262519 ns',
        binary=receipt(binary),allocation_binary=receipt(alloc),model=receipt(MODEL/'manifest.json')))
    for scripted in (True,False):
        suffix='scripted' if scripted else 'nn'
        bench['execute'](alloc,[case(i) for i in selected],folder,'allocations-'+suffix,scripted)
        bench['execute'](binary,[case(i) for i in selected],folder,'components-'+suffix,scripted)
    rows=bench['execute'](binary,[case(i,rep,4) for rep in range(5) for i in selected],folder,'timed')
    write(folder/'results.json',dict(passed=True,summary=bench['summarize'](rows),
        implementation_speed_gate=False,quality_gate=False,
        source_sha256=BASE['SUPPORT']['native_fingerprint'](),artifacts=[receipt(p) for p in sorted(folder.glob('*.json'))]))


def memory(out,label=None):
    protocol=BASE['checked'](out)
    bench=runpy.run_path(str(ROOT/'scripts/benchmark-rust-coordinator.py'))
    local=runpy.run_path(str(ROOT/'tests/RustBackendLocal/test_local.py'))
    folder=out/(label or 'native-memory');folder.mkdir(exist_ok=False)
    pin_runner(folder)
    worker=local['Worker'](BASE['L13']/'BrassNative/alpha-g890')
    psapi=ctypes.WinDLL('psapi',use_last_error=True)
    psapi.GetProcessMemoryInfo.argtypes=[ctypes.c_void_p,ctypes.POINTER(bench['Memory']),ctypes.c_ulong]
    def sample():
        value=bench['Memory']();value.cb=ctypes.sizeof(value)
        assert psapi.GetProcessMemoryInfo(worker.p._handle,ctypes.byref(value),value.cb),ctypes.get_last_error()
        assert value.working>0 and value.peak>0,'Unavailable counters cannot be interpreted as zero usage'
        return dict(working_bytes=value.working,peak_bytes=value.peak,private_commit_bytes=value.pagefile)
    rows=[]
    try:
        assert (worker.hello['data']['sessions'],worker.hello['data']['threads_per_session'])==(2,1)
        write(folder/'protocol.json',dict(model=receipt(MODEL/'manifest.json'),package=receipt(BASE['L13']/'BrassNative/alpha-g890/build.json'),
            indices=[0,3,8,26],warmup_cycles=1,measured_cycles=5,allowance_ms=4000,
            memory_api='Win32 GetProcessMemoryInfo on the owned production process; Unity Mono process counters return zero and are excluded',
            limits=protocol['memory']))
        for rep in range(-1,5):
            for index in (0,3,8,26):
                before=sample();request=worker.search(index,instance=f'l14-memory-{rep}-{index}')
                worker.send(request);reply=worker.recv();assert reply['status']=='ok' and reply['data']['retired'],reply
                after=sample();worker.send(dict(op='health'));assert not worker.recv()['data']['busy']
                rows.append(dict(warmup=rep<0,repeat=rep,fixture=index,before=before,after=after,response=reply['data']))
                write(folder/'progress.json',dict(rows=rows));print('memory',rep,index,'retired',flush=True)
        measured=[r for r in rows if not r['warmup']]
        peak=max(r['after']['peak_bytes'] for r in measured)
        growth=measured[-1]['after']['working_bytes']-measured[0]['before']['working_bytes']
        write(folder/'results.json',dict(passed=peak<=protocol['memory']['process_peak_bytes'] and growth<=protocol['memory']['post_retirement_growth_bytes'],
            peak_bytes=peak,retained_growth_bytes=growth,turns=len(measured),rows=rows,
            scope='Repeated warmed four-second turns; not a proof against leaks across complete games.'))
    finally:worker.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['fixed','legacy','native','memory']);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--label',help='New attempt directory; existing evidence is never overwritten')
    a=p.parse_args();out=a.out.resolve()
    assert out.is_relative_to(ROOT/'artifacts/rust-backend')
    if a.label:assert a.label not in ('.','..') and all(c.isalnum() or c in '-_' for c in a.label)
    {'fixed':fixed,'legacy':legacy,'native':native_components,'memory':memory}[a.stage](out,a.label)
