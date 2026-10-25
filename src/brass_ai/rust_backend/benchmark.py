"""Frozen variant A: two private one-thread sessions, one 4s turn clock.

This measures the existing Python Masked ISMCTS and admitted C# reference. It
does not implement Rust search or use the upstream PUCT algorithm. The supplied
g890 weights are explicitly mechanical-only until value readiness is qualified.
"""
from __future__ import annotations

from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
import copy
import json
import os
from pathlib import Path
import random
import statistics
import threading
import time

from brass_ai.learning.contracts import digest,read_json,require
from brass_ai.learning.reward import CostLedger
from brass_ai.search.common import SearchConfig,turn_key
from brass_ai.search.ismcts import MaskedSearchPolicy,information_key,supported_continuation
from brass_ai.recovery.parallel import merge_statistics
from .contracts import permitted_request
from .fixtures import (advance,file_sha,read_gzip,restore,update_recall,write_json,harness_receipt)
from brass_ai.learning.batching import Evaluator
from brass_ai.step5.policy import NeuralPolicy


class InlineEvaluator(Evaluator):
    """Same reference encoding/forward/response, on the session owner.

    Normal Evaluator.predict schedules an extra inference thread. This adapter
    excludes it to obey two sessions x one thread; B/C_ref must match this path.
    """
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs);self.owner=threading.get_ident();self.calls=0

    def predict(self,requests):
        require(threading.get_ident()==self.owner,'NN forward left the session thread')
        require(self._executor is None,'Unexpected inference executor')
        self.calls+=1
        return self._predict(requests)


class RootPriorPolicy(NeuralPolicy):
    """Retain the first prediction for the reference forest's prior tie-break.

    Later leaf predictions must not replace it. The forward path, exploration
    distribution and policy identity remain the original NeuralPolicy's.
    """
    capture_root=False
    root_priors=None

    def predict(self,inputs):
        result=super().predict(inputs)
        if self.capture_root:
            self.root_priors=result[0][0].tolist();self.capture_root=False
        return result


def select_root(stats,choices,priors):
    # Same ordering as ParallelMasked._decide: visits, prior, then legal order.
    require(len(priors)==len(choices),'Missing reference root priors')
    return max(range(len(choices)),key=lambda i:(stats['root'].get(choices[i]['id'],{}).get('visits',0),priors[i],-i))


def process_samples(pids):
    """CPU/memory of this harness and only its explicitly owned rules hosts."""
    if os.name!='nt':return dict(status='unavailable on this platform')
    import ctypes
    from ctypes import wintypes
    class Memory(ctypes.Structure):
        _fields_=[('cb',wintypes.DWORD),('PageFaultCount',wintypes.DWORD)]+[
            (name,ctypes.c_size_t) for name in ('PeakWorkingSetSize','WorkingSetSize','QuotaPeakPagedPoolUsage',
                'QuotaPagedPoolUsage','QuotaPeakNonPagedPoolUsage','QuotaNonPagedPoolUsage','PagefileUsage','PeakPagefileUsage')]
    kernel=ctypes.WinDLL('kernel32',use_last_error=True);psapi=ctypes.WinDLL('psapi',use_last_error=True)
    kernel.OpenProcess.argtypes=[wintypes.DWORD,wintypes.BOOL,wintypes.DWORD];kernel.OpenProcess.restype=wintypes.HANDLE
    kernel.CloseHandle.argtypes=[wintypes.HANDLE]
    kernel.GetProcessTimes.argtypes=[wintypes.HANDLE]+[ctypes.POINTER(wintypes.FILETIME)]*4
    psapi.GetProcessMemoryInfo.argtypes=[wintypes.HANDLE,ctypes.POINTER(Memory),wintypes.DWORD]
    result={}
    for pid in pids:
        handle=kernel.OpenProcess(0x1000,False,pid)
        if not handle:result[str(pid)]=dict(error=ctypes.get_last_error());continue
        try:
            values=[wintypes.FILETIME() for _ in range(4)];memory=Memory();memory.cb=ctypes.sizeof(memory)
            row={}
            if kernel.GetProcessTimes(handle,*[ctypes.byref(x) for x in values]):
                seconds=lambda x:((x.dwHighDateTime<<32)|x.dwLowDateTime)/10000000
                row.update(kernel_seconds=seconds(values[2]),user_seconds=seconds(values[3]))
            if psapi.GetProcessMemoryInfo(handle,ctypes.byref(memory),memory.cb):
                row.update(rss_bytes=memory.WorkingSetSize,peak_rss_bytes=memory.PeakWorkingSetSize)
            else:row['memory_error']=ctypes.get_last_error()
            result[str(pid)]=row
        finally:kernel.CloseHandle(handle)
    return result


def machine_status():
    import platform
    result=dict(platform=platform.platform(),logical_cpus=os.cpu_count(),temperature_celsius=None,
                temperature_reason='No qualified thermal sensor in this harness',time_ns=time.time_ns(),
                process_sample=process_samples([os.getpid()]))
    if os.name=='nt':
        import ctypes
        import winreg
        from ctypes import wintypes
        import subprocess
        class Power(ctypes.Structure):
            _fields_=[('ACLineStatus',ctypes.c_ubyte),('BatteryFlag',ctypes.c_ubyte),
                      ('BatteryLifePercent',ctypes.c_ubyte),('SystemStatusFlag',ctypes.c_ubyte),
                      ('BatteryLifeTime',ctypes.c_ulong),('BatteryFullLifeTime',ctypes.c_ulong)]
        power=Power()
        if ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(power)):
            result['power']=dict(ac_line=power.ACLineStatus,battery_percent=power.BatteryLifePercent)
        clocks=[wintypes.FILETIME() for _ in range(3)]
        if ctypes.windll.kernel32.GetSystemTimes(*[ctypes.byref(c) for c in clocks]):
            result['system_cpu_seconds']={name:((c.dwHighDateTime<<32)|c.dwLowDateTime)/10000000
                                         for name,c in zip(('idle','kernel_including_idle','user'),clocks)}
        scheme=subprocess.run(['powercfg','/getactivescheme'],capture_output=True,text=True,check=False)
        result['power_scheme']=dict(exit_code=scheme.returncode,description=scheme.stdout.strip())
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,r'HARDWARE\DESCRIPTION\System\CentralProcessor\0') as key:
            result['cpu']=winreg.QueryValueEx(key,'ProcessorNameString')[0].strip()
    try:
        import psutil
        result['process']=dict(rss=psutil.Process().memory_info().rss,threads=psutil.Process().num_threads())
        result['system_cpu_times']=psutil.cpu_times()._asdict()
    except ImportError:result['resource_sampler']='psutil unavailable'
    return result


class Session:
    def __init__(self,model,root,config,index):
        import torch
        require(torch.get_num_threads()==torch.get_num_interop_threads()==1,
                'Inference thread settings must also hold on each session owner')
        self.index=index;self.owner=threading.get_ident();self.native_owner=threading.get_native_id()
        self.evaluator=InlineEvaluator(copy.deepcopy(model),device='cpu',batch_size=config.batch_size)
        self.policy=MaskedSearchPolicy(None,root,config=config,fast_host=True,evaluator=self.evaluator)
        self.policy.base=RootPriorPolicy(None,evaluator=self.evaluator,focus='neutral')
        self.policy.partition_index=index
        self.policy.teacher_replica=True  # existing completed-statistics path, no label/hash construction
        self.policy.time_budget_primary=True
        self.thread_cpu_start=time.thread_time()

    def prepare(self,fixture,seed):
        require(threading.get_ident()==self.owner,'Reference session moved threads')
        self.tree={};self.cache=OrderedDict();self.rng=random.Random(seed+self.index)
        self.policy.last_statistics=None;self.policy.reward_context=fixture['ledger']
        p=fixture['projection'];seat=p['context']['actor']
        self.policy.base.utility_remaining[self.rng]=tuple(CostLedger(self.policy.config.reward,**fixture['ledger']).remaining)
        # Warm exact encoding/model kernels and the rules process outside timing.
        self.policy.base.predict([(p['views'][seat],p['choices'],self.rng)])
        self.policy.game.from_observation(p['views'][seat],seed,fixture['recall'][seat])
        import torch
        return dict(session=self.index,owner=self.owner,native_owner=self.native_owner,
                    torch_intra_op=torch.get_num_threads(),torch_inter_op=torch.get_num_interop_threads())

    def decide(self,view,choices,recall,ledger,deadline):
        require(threading.get_ident()==self.owner,'Reference session moved threads')
        started=time.monotonic();cpu=time.thread_time();self.policy.last_statistics=None
        self.policy.base.capture_root=True;self.policy.base.root_priors=None
        nn_before=dict(self.evaluator.telemetry)
        self.policy.reward_context=ledger
        self.policy.base.utility_remaining[self.rng]=tuple(CostLedger(self.policy.config.reward,**ledger).remaining)
        selected,_,_=self.policy.decide(view,choices,self.rng,deadline,recall,self.tree,self.cache)
        finished=time.monotonic()
        return dict(session=self.index,owner=self.owner,selected=selected,started=started,retired=finished,
                    root_priors=self.policy.base.root_priors or [1./len(choices)]*len(choices),
                    native_owner=self.native_owner,inference_calls=self.evaluator.calls,
                    inference_executor_created=self.evaluator._executor is not None,
                    nn_work={k:self.evaluator.telemetry[k]-nn_before[k] for k in ('batches','positions','predict_seconds')},
                    search_metrics={k:self.policy.last_statistics[k] for k in ('batching','frontier_leaves','unseen_competitions','opponent_choices')}
                        if self.policy.last_statistics else None,
                    cpu_seconds=time.thread_time()-cpu,stats=self.policy.last_statistics,
                    incumbent=copy.deepcopy(self.policy.last_continuation))

    def close(self):self.policy.close();self.evaluator.close()


def benchmark(root,out,baseline):
    import torch
    from brass_ai.learning.checkpoint import load_checkpoint
    from brass_ai.searchlab.spiel import BrassSpielGame
    out=Path(out);destination=out/'local/baseline-a.json'
    require(not destination.exists(),'Baseline measurements are immutable; select a new evidence run')
    corpus_dir=out/'local/fixtures';corpus=read_json(corpus_dir/'corpus.json')
    require(corpus['complete'],'Capture the full corpus before timing')
    # Six predeclared strata, one largest legal set in each. Never select by timings.
    eligible=[]
    for entry in corpus['roots']:
        if entry['partition']!='development' or 'authored-edge' in entry['tags']:continue
        fixture=read_gzip(corpus_dir/entry['path']);p=fixture['projection'];c=p['context']
        if p['chance'] is None and c['phase']=='action' and not c['terminal']:
            if c['actions_remaining']!=(1 if c['era']=='CanalEra' and c['round']==1 else 2):continue
            eligible.append((entry,fixture))
    selected=[]
    for era in ('CanalEra','RailEra'):
        for low,high in ((1,2),(3,5),(6,8)):
            options=[x for x in eligible if x[1]['projection']['context']['era']==era
                     and low<=x[1]['projection']['context']['round']<=high]
            require(options,'Missing baseline era/round stratum')
            selected.append(max(options,key=lambda x:(x[0]['legal_count'],x[0]['id'])))
    # Ordinary-turn roots for expensive staged sale/resource decisions. Do not
    # give an already-in-progress payment/bonus a fresh whole-turn allowance.
    for identity in ('edge-shared-merchant-beer','edge-rail-action'):
        entry=next(e for e in corpus['roots'] if e['id']==identity)
        fixture=read_gzip(corpus_dir/entry['path'])
        require(fixture['projection']['context']['phase']=='action','Complex benchmark must start at action root')
        c=fixture['projection']['context']
        require(c['actions_remaining']==(1 if c['era']=='CanalEra' and c['round']==1 else 2),
                'Complex benchmark must start at the beginning of an ordinary turn')
        selected.append((entry,fixture))
    load_started=time.perf_counter()
    model,raw=load_checkpoint(Path(root)/baseline['model']['path']);model.eval()
    model_load_seconds=time.perf_counter()-load_started
    require(torch.get_num_threads()==torch.get_num_interop_threads()==1,'Reference inference must remain single threaded')
    config=SearchConfig(seconds=4,contract='masked-ismcts/1.2.0',reward=model.config.utility,
                        simulations=100000,pending_paths=1,batch_size=1,workers=1,cpu_role='reference').validate()
    protocol=dict(format='rust-matched-protocol/1.0.0',variant='A',source=baseline['reference_source']['digest'],
        harness=harness_receipt('src/brass_ai/rust_backend/benchmark.py'),
        model_id=raw['model_id'],model_sha256=baseline['model']['sha256'],model_role=baseline['model']['role'],
        value_readiness=baseline['model']['value_readiness'],search=config.to_dict(),
        sessions=2,threads_per_session=1,torch_intra_op=1,torch_inter_op=1,
        thread_environment={name:os.environ.get(name) for name in
                            ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS')},
        inference='PyTorch CPU synchronous on each session owner; no inference service or cross-session batching',
        reference_overhead='C# rules processes and Python transport threads are included; not Rust in-process execution',
        whole_turn_seconds=4,finalization_reserve_seconds=0,
        fixtures=[e for e,_ in selected],repetitions=5,holdout='withheld from baseline tuning and timing until L14',
        targets=baseline['performance_targets'],
        counters=dict(completed='completed MaskedISMCTS trajectories admitted before deadline',
            attempted='started trajectories including interrupted paths',interrupted='started paths without backup',
            supported_plan='merged complete contingent trace with at least two visits',
            response_ready='final staged turn plan resolved; clock includes reference staged validation/application',
            fallback='no supported merged continuation; reference incumbent or cheap legal completion'),
        no_quality_claim=True)
    write_json(out/'local/matched-protocol.json',protocol) # sealed BEFORE measurements, no candidate results
    before=machine_status();startup=time.perf_counter()
    pools=[ThreadPoolExecutor(1,thread_name_prefix=f'reference-session-{i}') for i in range(2)]
    sessions=[];trials=[];environment=BrassSpielGame(root=root,fast=True)
    try:
        for i,pool in enumerate(pools):sessions.append(pool.submit(Session,model,root,config,i).result())
        owned_pids=[os.getpid(),environment.host.process.pid]+[
            game.host.process.pid for session in sessions for game in [session.policy.game,*session.policy.simulation_games]]
        startup=time.perf_counter()-startup
        # Rotate fixture order deterministically between repetitions.
        with (out/'local/baseline-a-trials.jsonl').open('x',encoding='utf-8') as log:
            for repeat in range(5):
                ordered=selected[repeat%len(selected):]+selected[:repeat%len(selected)]
                for entry,fixture in ordered:
                    preparation=time.perf_counter()
                    state=restore(environment.host,fixture['checkpoint']);recall=copy.deepcopy(fixture['recall'])
                    ledger=CostLedger(config.reward,**fixture['ledger']);seed=900000+repeat*1000+entry['seed']
                    tasks=[p.submit(s.prepare,fixture,seed) for p,s in zip(pools,sessions)]
                    owners=[task.result() for task in tasks]
                    require(len({x['owner'] for x in owners})==2,'Two distinct session owners required')
                    preparation=time.perf_counter()-preparation
                    resources_before=process_samples(owned_pids)
                    seat=state['context']['actor'];initial_turn=turn_key(state['context'])
                    started=time.monotonic();deadline=started+4;completed=attempted=interrupted=0
                    steps=[];continuation={};fallbacks=0;owner_cpu=[0.,0.];last_retired=started
                    for step in range(96):
                        if state['chance'] or state['context']['terminal'] or state['context']['actor']!=seat or turn_key(state['context'])!=initial_turn:break
                        update_recall(recall,state);view=state['views'][seat];choices=state['choices'];ledger.observe(view)
                        info=information_key(view,choices,recall[seat]);identity=continuation.get(info)
                        work=[];merged=None
                        if len(choices)>1 and state['context']['phase']!='debt' and time.monotonic()<deadline:
                            futures=[p.submit(s.decide,view,choices,recall[seat],ledger.snapshot(),deadline) for p,s in zip(pools,sessions)]
                            # Always drain BOTH owners before applying anything, including failure.
                            failure=None
                            for task in futures:
                                try:work.append(task.result())
                                except BaseException as error:failure=failure or error
                            if failure:raise failure
                            merged=merge_statistics([w['stats'] for w in work],choices)
                            best=select_root(merged,choices,work[0]['root_priors'])
                            supported=supported_continuation(merged,choices[best]['id'],seat,config.minimum_plan_visits)
                            if supported is not None:
                                continuation=supported;identity=choices[best]['id']
                            else:
                                continuation=work[0]['incumbent'];identity=choices[work[0]['selected']]['id'];fallbacks+=1
                            completed+=merged['completed'];interrupted+=merged['interrupted'];attempted+=merged['batching']['attempted']
                            for w in work:owner_cpu[w['session']]+=w['cpu_seconds'];last_retired=max(last_retired,w['retired'])
                        if identity not in [c['id'] for c in choices]:
                            identity=max(choices,key=lambda c:(c['kind']=='finish',c['kind']=='beer' and bool(c.get('merchant')),-choices.index(c)))['id']
                            fallbacks+=int(len(choices)>1)
                        choice=next(c for c in choices if c['id']==identity)
                        ledger.chosen(view,choice);state=advance(environment.host,state,dict(choice=identity))
                        steps.append(dict(step=step,phase=view['context']['phase'],actions_remaining=view['context']['actions_remaining'],
                            choice=identity,seconds=time.monotonic()-started,search=merged,
                            owners=[{k:v for k,v in w.items() if k not in ('incumbent','stats')} for w in work]))
                    else:raise ValueError('Reference turn completion guard exceeded')
                    elapsed=time.monotonic()-started
                    trial=dict(fixture=entry['id'],repeat=repeat,seed=seed,preparation_seconds=preparation,session_settings=owners,
                        whole_turn_seconds=4,response_ready_seconds=elapsed,last_search_retirement_seconds=last_retired-started,
                        completed=completed,attempted=attempted,interrupted=interrupted,fallbacks=fallbacks,
                        cpu_seconds_by_session=owner_cpu,steps=steps,valid_complete_turn=True,
                        process_samples_before=resources_before,process_samples_after=process_samples(owned_pids),
                        final_context={k:v for k,v in state['context'].items() if k!='game_id'})
                    trials.append(trial);log.write(json.dumps(trial,allow_nan=False)+'\n');log.flush()
                    print(json.dumps(dict(baseline_trial=len(trials),fixture=entry['id'],completed=completed,
                                          response_ready_seconds=round(elapsed,3))),flush=True)
    finally:
        for pool,session in zip(pools,sessions):pool.submit(session.close).result()
        for pool in pools:pool.shutdown(wait=True)
        environment.close()
    latencies=sorted(t['response_ready_seconds'] for t in trials)
    result=dict(format='rust-baseline-a/1.0.0',protocol_sha256=file_sha(out/'local/matched-protocol.json'),
        trials=len(trials),completed=sum(t['completed'] for t in trials),attempted=sum(t['attempted'] for t in trials),
        interrupted=sum(t['interrupted'] for t in trials),zero_completion_turns=sum(t['completed']==0 for t in trials),
        fallbacks=sum(t['fallbacks'] for t in trials),startup_seconds=startup,checkpoint_load_seconds=model_load_seconds,
        response_ready=dict(median=statistics.median(latencies),observed_p95=latencies[int(.95*(len(latencies)-1))],maximum=max(latencies),
                            scope=f'{len(trials)}-turn reference screen; not a tail-latency or native acceptance claim'),
        by_fixture={entry['id']:dict(completed=[t['completed'] for t in trials if t['fixture']==entry['id']],
                        response_seconds=[t['response_ready_seconds'] for t in trials if t['fixture']==entry['id']]) for entry,_ in selected},
        before=before,after=machine_status(),scope='test NN reference cost only; no Rust speedup or strength claim',
        raw=receipt_trial(out),criteria_frozen=True)
    write_json(destination,result)
    return result


def receipt_trial(out):
    path=Path(out)/'local/baseline-a-trials.jsonl'
    return dict(path='local/baseline-a-trials.jsonl',sha256=file_sha(path),bytes=path.stat().st_size)
