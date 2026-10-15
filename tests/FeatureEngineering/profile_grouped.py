"""Bounded profiling of the current grouped target only; no comparison arms.

The fresh worker loads one target checkpoint and its matching named exports.
Search uses the production laptop two-worker 2/5/12-second whole-turn clocks.
"""
import argparse
from contextlib import redirect_stdout
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import traceback


def summary(values):
    ordered=sorted(values)
    return dict(n=len(values),median_ms=statistics.median(values),
        p95_ms=ordered[int(.95*(len(ordered)-1))],minimum_ms=min(values),maximum_ms=max(values),samples_ms=values)


def process_memory():
    try:
        import psutil
        process=psutil.Process();own=process.memory_info();children=process.children(recursive=True)
        return dict(process_rss_bytes=own.rss,process_vms_bytes=own.vms,
            process_peak_rss_bytes=getattr(own,'peak_wset',None),
            descendants_rss_bytes=sum(p.memory_info().rss for p in children if p.is_running()),
            descendants=len(children),scope='RSS snapshots include native runtime allocations; not Python-only allocations.')
    except ImportError as error:
        if sys.platform!='win32':return dict(unavailable=str(error))
        # Standard Windows reader keeps process memory available without an
        # optional Python dependency. Child-process totals need psutil.
        import ctypes
        from ctypes import wintypes
        class Counters(ctypes.Structure):
            _fields_=[('cb',wintypes.DWORD),('PageFaultCount',wintypes.DWORD)]+[
                (name,ctypes.c_size_t) for name in ('PeakWorkingSetSize','WorkingSetSize',
                    'QuotaPeakPagedPoolUsage','QuotaPagedPoolUsage','QuotaPeakNonPagedPoolUsage',
                    'QuotaNonPagedPoolUsage','PagefileUsage','PeakPagefileUsage','PrivateUsage')]
        kernel=ctypes.WinDLL('kernel32',use_last_error=True);reader=ctypes.WinDLL('psapi',use_last_error=True)
        kernel.GetCurrentProcess.restype=wintypes.HANDLE
        reader.GetProcessMemoryInfo.argtypes=[wintypes.HANDLE,ctypes.POINTER(Counters),wintypes.DWORD]
        reader.GetProcessMemoryInfo.restype=wintypes.BOOL
        counts=Counters();counts.cb=ctypes.sizeof(counts)
        if not reader.GetProcessMemoryInfo(kernel.GetCurrentProcess(),ctypes.byref(counts),counts.cb):
            return dict(unavailable=str(ctypes.WinError(ctypes.get_last_error())))
        return dict(process_rss_bytes=counts.WorkingSetSize,process_peak_rss_bytes=counts.PeakWorkingSetSize,
            process_private_commit_bytes=counts.PrivateUsage,descendants_rss_bytes=None,reader='Win32 GetProcessMemoryInfo',
            scope='Current process including native runtime allocations; child-process totals unavailable without psutil.')
    except OSError as error:return dict(unavailable=str(error))


def worker(args):
    started=time.perf_counter();sys.path.insert(0,str(args.root/'src'))
    with redirect_stdout(sys.stderr):
        import platform
        import random
        from dataclasses import replace
        import numpy as np
        import torch
        import onnxruntime as ort
        from brass_ai.bank import sha
        from brass_ai.learning.checkpoint import load_checkpoint
        from brass_ai.learning.batching import collate
        from brass_ai.learning.contracts import compatibility,require
        from brass_ai.learning.model import ModelConfig
        from brass_ai.learning.reward import CostLedger
        from brass_ai.recovery.encoding import PreparedEncoder
        from brass_ai.personality.contracts import context,context_vector
        from brass_ai.personality.checkpoint import load as load_actor
        from brass_ai.personality.search.inference import PersonalityRemote,PersonalityRequest
        from brass_ai.personality.search.contracts import Profile,search_config
        from brass_ai.personality.search.engine import SearchSession
        from brass_ai.personality.difficulty.collection import TrackedChannels
        from brass_ai.personality.laptop.inference import PortableInference
        from brass_ai.personality.laptop.benchmark import processor_name
        from brass_ai.search.common import turn_key
        from brass_ai.searchlab.spiel import BrassSpielGame,action_id
        torch.set_num_threads(2)
        rows=json.loads(args.inputs.read_text(encoding='utf-8'))
        began=time.perf_counter();model,raw=load_checkpoint(args.checkpoint);model.eval()
        checkpoint_load=time.perf_counter()-began
        require(ModelConfig.from_dict({k:v for k,v in model.config.to_dict().items() if k!='utility'})==ModelConfig.preset('medium'),
                'Profile only the current width-304 grouped target')
        receipt=json.loads((args.exports/'receipt.json').read_text(encoding='utf-8'))
        require(receipt['checkpoint']['sha256']==sha(args.checkpoint)
                and receipt['checkpoint']['model_id']==raw['model_id'],'Exports must match this exact checkpoint')
        entries=[{k:e[k] for k in ('path','model_id','actor')} for e in receipt['models'].values()]
        actor_entry=receipt['models']['actor'];utility_entry=receipt['models']['utility']
        actor,actor_raw=load_actor(args.exports/actor_entry['checkpoint']);actor.eval()
        require(actor_raw['model_id']==actor_entry['model_id'],'Actor checkpoint/export differs')
        began=time.perf_counter();service=PortableInference(args.exports,entries,'ort-cpu');service.__enter__()
        service_start=time.perf_counter()-began
        channels=TrackedChannels(service.channels(8));remote=PersonalityRemote(channels.acquire(),actor_entry['model_id'])
        reward=model.config.utility
        config=replace(search_config(reward,seconds=2.,workers=2),cpu_role='laptop').validate()
        session=SearchSession(args.root,service,channels,actor_entry['model_id'],[utility_entry['model_id']],
            config,[Profile(alpha=0.)]*4,dict(bias_ready=False,temperature=1.,scope='Initialized goals; timing only'))
        game=BrassSpielGame(root=args.root,fast=True);encoder=PreparedEncoder()
        encoded=[encoder.encode(r['view'],r['choices']) for r in rows]
        encoded_hash=hashlib.sha256(b''.join(e.state.tobytes()+e.actions.tobytes()+e.stable_to_relative.tobytes() for e in encoded)).hexdigest()
        pools=dict(small=[i for i,r in enumerate(rows) if len(r['choices'])<=6],
            large=[i for i,r in enumerate(rows) if len(r['choices'])>=300],
            rail_ordinary=[i for i,r in enumerate(rows) if r['view']['context']['era']=='RailEra' and r['view']['context']['phase']=='action'])
        require(all(pools.values()),'Corpus must contain both eras and small/large legal sets')
        batches=[]
        for size in (1,2,8):
            for name,indices in pools.items():
                selected=[indices[i%len(indices)] for i in range(size)]
                batch=collate([encoded[i] for i in selected])
                inputs=dict(zip(('states','actions','legal_mask','stable_to_relative'),[t.numpy() for t in batch]))
                inputs['remaining']=np.full((size,4),reward['insolvency']['max_cost'],np.float32)
                actor_inputs=dict(inputs,personality=np.tile(np.asarray(context_vector(context()),np.float32),(size,1)))
                batches.append((name,size,selected,batch,inputs,actor_inputs))
    ready=dict(status='ready',cold_process_seconds=time.perf_counter()-started,checkpoint_load_seconds=checkpoint_load,
        portable_service_start_seconds=service_start,processor=processor_name(),platform=platform.platform(),python=sys.version,
        torch=torch.__version__,onnxruntime=ort.__version__,compatibility=compatibility(),config=model.config.to_dict(),
        parameters=dict(utility=model.parameter_count,personality=actor.parameter_count),encoded_sha256=encoded_hash,
        checkpoint_sha256=sha(args.checkpoint),checkpoint_bytes=args.checkpoint.stat().st_size,
        serialized_exports_bytes={p.name:p.stat().st_size for p in args.exports.glob('*.onnx')},
        parameter_bytes=sum(p.numel()*p.element_size() for p in model.parameters()),memory=process_memory(),
        backend='CPUExecutionProvider',tensor_threads=2,search_workers=2,export_parity=service.parity)
    print(json.dumps(ready),flush=True)

    def timed(call,repeat=15,warm=3):
        for _ in range(warm):call()
        times=[]
        for _ in range(repeat):
            began=time.perf_counter();call();times.append((time.perf_counter()-began)*1000)
        return summary(times)

    def micro():
        measurements={}
        for mode in ('new_encoder','persistent_miss','warm_hit'):
            enc=PreparedEncoder(cache_size=256 if mode=='warm_hit' else 0)
            for row in rows:enc.encode(row['view'],row['choices'])
            values=[]
            for _ in range(3):
                for row in rows:
                    began=time.perf_counter();selected=PreparedEncoder(cache_size=0) if mode=='new_encoder' else enc
                    selected.encode(row['view'],row['choices']);values.append((time.perf_counter()-began)*1000)
            measurements[mode]=summary(values)
        cases=[]
        for name,size,selected,batch,inputs,actor_inputs in batches:
            remaining=torch.from_numpy(inputs['remaining']);personality=torch.from_numpy(actor_inputs['personality'])
            with torch.inference_mode():
                torch_utility=timed(lambda:model(*batch,remaining=remaining))
                torch_actor=timed(lambda:actor(*batch,remaining=remaining,personality=personality))
            ort_utility=timed(lambda:service.graphs[utility_entry['model_id']].run(inputs))
            ort_actor=timed(lambda:service.graphs[actor_entry['model_id']].run(actor_inputs))
            def request():
                requests=[PersonalityRequest(f'profile-{i}',actor_entry['model_id'],rows[index]['view'],rows[index]['choices'],
                    time.monotonic()+60,personality=context(),remaining=tuple(map(float,inputs['remaining'][i]))) for i,index in enumerate(selected)]
                responses=remote.predict(requests)
                require(all(r['status']=='ok' for r in responses),'Complete request failed')
                require(all(r['choice_ids']==[c['id'] for c in rows[index]['choices']] for r,index in zip(responses,selected)),
                        'Candidate order changed')
            remote.encoder.cache.clear();began=time.perf_counter();request();first=(time.perf_counter()-began)*1000
            cases.append(dict(size_class=name,batch=size,candidates=inputs['actions'].shape[1],corpus_rows=selected,
                torch_utility=torch_utility,torch_actor=torch_actor,ort_utility=ort_utility,ort_actor=ort_actor,
                complete_first_request_ms=first,complete_warm_request=timed(request,repeat=7)))
        return dict(encoding=measurements,cases=cases,memory=process_memory(),inference=service.report())

    def whole_turn(command):
        seconds=command['seconds'];index=command['case'];seed=command['seed'];row=rows[index]
        view=row['view'];seat=view['seat'];key=turn_key(view['context'])
        session.finish_turn();session.config=replace(config,seconds=seconds)
        for w in session.workers:w.config=session.config
        session.begin(seed,seed);state=game.from_observation(view,seed)
        require(state.legal==row['choices'],'Root legality differs')
        ledger=CostLedger(reward);ledger.observe(view);rng=random.Random(seed)
        before=dict(session.telemetry);began=time.monotonic();qualified=0;finished=0;deadlines=[];status='turn_complete'
        for _ in range(256):
            if state.is_terminal():break
            if state.is_chance_node():status='chance_boundary';break
            if state.current_player()!=seat or turn_key(state.context)!=key:break
            live=state.views[seat];choices=state.legal;chosen,label,_=session.select(live,choices,rng,ledger)
            if session.turn:deadlines.append(session.turn['deadline'])
            require(0<=chosen<len(choices),'Illegal search choice')
            qualified+=int(label is not None);choice=choices[chosen];ledger.chosen(live,choice)
            state.apply_action(action_id(choice['id']));finished+=int(choice['kind']=='finish')
            if not state.is_terminal() and not state.is_chance_node():ledger.observe(state.views[seat])
        else:raise ValueError('Whole-turn continuation cap exceeded')
        elapsed=time.monotonic()-began;delta={k:session.telemetry[k]-before[k] for k in before}
        require(len(set(deadlines))<=1,'Shared clock reset between actions/subchoices')
        session.finish_turn()
        return dict(case=index,name=row['name'],era=view['context']['era'],root_candidates=len(row['choices']),
            seconds=seconds,elapsed_seconds=elapsed,metrics=delta,qualified_labels=qualified,
            completed_actions=finished,turn_actions_complete=finished>=view['context']['actions_remaining'],
            status=status,shared_deadline=True,seed=seed,memory=process_memory(),
            completed_simulations_per_second=delta['completed']/max(elapsed,1e-9),search_config=session.config.to_dict())

    def python_memory():
        # Allocation instrumentation is deliberately outside timed measurements.
        import tracemalloc
        tracemalloc.start();enc=PreparedEncoder()
        held=[enc.encode(row['view'],row['choices']) for row in rows]
        current,peak=tracemalloc.get_traced_memory();tracemalloc.stop()
        return dict(current_bytes=current,peak_bytes=peak,retained_positions=len(held),
            scope='Python-tracked allocations during a separate feature-encoding pass; excludes most Torch/ORT native allocations.',
            process=process_memory())

    try:
        for line in sys.stdin:
            command=json.loads(line)
            if command['operation']=='close':break
            try:
                with redirect_stdout(sys.stderr):
                    result=micro() if command['operation']=='micro' else python_memory() if command['operation']=='memory' else whole_turn(command)
                print(json.dumps(dict(ok=True,result=result)),flush=True)
            except Exception as error:
                traceback.print_exc(file=sys.stderr);print(json.dumps(dict(ok=False,error=f'{type(error).__name__}: {error}')),flush=True)
    finally:session.close();game.close();channels.close();service.__exit__()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[2])
    parser.add_argument('--checkpoint',type=Path,required=True);parser.add_argument('--exports',type=Path,required=True)
    parser.add_argument('--output',type=Path);parser.add_argument('--inputs',type=Path);parser.add_argument('--worker',action='store_true')
    args=parser.parse_args();args.root=args.root.resolve();args.checkpoint=args.checkpoint.resolve();args.exports=args.exports.resolve()
    if args.worker:worker(args);return
    out=args.output.resolve();assert out.is_relative_to(args.root/'artifacts') and not out.exists(),'Use a new isolated profile directory'
    out.mkdir(parents=True)
    from fe_fixtures import benchmark_positions
    rows=benchmark_positions();inputs=out/'inputs.json';inputs.write_text(json.dumps(rows),encoding='utf-8')
    report=dict(format='grouped-target-profile/1.0.0',micro=[],turns=[],inputs_sha256=hashlib.sha256(inputs.read_bytes()).hexdigest(),
        corpus=dict(positions=len(rows),names=[r['name'] for r in rows],candidates=[len(r['choices']) for r in rows],
                    fixture_reset_seed=1201,fixture_world_seed=99),
        warmup=3,forward_repeats=15,request_repeats=7,micro_blocks=4,
        scope='Current target only. Warm CPU float32 diagnostics; imported utility checkpoint and an initialized zero-bias goal extension. '
              'No strength claim. OS file cache is uncontrolled. Fresh-process startup is one observation; Python allocation tracking runs separately.')
    log=(out/'worker.log').open('w',encoding='utf-8');child=None
    def send(command):
        child.stdin.write(json.dumps(command)+'\n');child.stdin.flush();line=child.stdout.readline()
        if not line:raise RuntimeError('Target worker exited; inspect worker.log')
        reply=json.loads(line)
        if not reply['ok']:raise RuntimeError(reply['error'])
        return reply['result']
    try:
        env=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
        command=[sys.executable,str(Path(__file__).resolve()),'--worker','--root',str(args.root),'--checkpoint',str(args.checkpoint),
            '--exports',str(args.exports),'--inputs',str(inputs)]
        began=time.monotonic();child=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=log,text=True,encoding='utf-8',env=env)
        line=child.stdout.readline()
        if not line:raise RuntimeError('Target worker did not start; inspect worker.log')
        report['cold']=json.loads(line);report['cold']['launch_seconds']=time.monotonic()-began
        assert report['cold']['status']=='ready'
        for block in range(4):
            report['micro'].append(dict(block=block,**send(dict(operation='micro'))))
            print(f'Target micro block {block+1}/4',flush=True)
        cases=[next(i for i,r in enumerate(rows) if r['name']==name) for name in ('settled-next-round','rail-action')]
        for repeat in range(2):
            for case in cases:
                for seconds in (2.,5.,12.):
                    report['turns'].append(dict(repeat=repeat,**send(dict(operation='turn',case=case,seconds=seconds,seed=57731+repeat*101+case))))
                    (out/'partial.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
                    print(f'Target turn: repeat={repeat+1}/2 case={case} budget={seconds}s',flush=True)
        report['python_memory']=send(dict(operation='memory'))
        (out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    finally:
        if child and child.poll() is None:
            try:child.stdin.write('{"operation":"close"}\n');child.stdin.flush();child.wait(timeout=30)
            except Exception:child.terminate();child.wait(timeout=10)
        log.close()


if __name__=='__main__':main()
