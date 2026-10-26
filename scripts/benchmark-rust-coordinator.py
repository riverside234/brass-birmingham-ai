"""L12 two-owner component/latency study; keeps the cap-one baseline distinct.

All timed comparisons use one four-second turn. This is an engineering sweep,
not L14's implementation-speed or playing-strength acceptance experiment.
"""
import argparse
import copy
import ctypes
from ctypes import wintypes
import hashlib
import json
import math
from pathlib import Path
import random
import runpy
import statistics
import subprocess
import time

ROOT=Path(__file__).resolve().parents[1]
BASE=runpy.run_path(str(ROOT/'scripts/qualify-rust-scaffold.py'))
read,write,sha,receipt=(BASE[k] for k in ('read','write','sha','receipt'))
FIXTURES=ROOT/'artifacts/rust-backend/20261025-l10-capture'
MODEL=ROOT/'artifacts/rust-backend/20261025-l09-qualified-v2/model/manifest.json'
DEVELOPMENT=[3,26,30,27,41]
HELDOUT=[0,8]
POLICIES=[dict(cap=1,pending_limit=1,max_candidate_cells=8192,flush_us=0)]+[
    dict(cap=c,pending_limit=c,max_candidate_cells=c*8192,flush_us=f) for c in (2,4) for f in (0,500,1000,2000)]


def case(index,policy,n=64,seconds=60,rep=0,startup=None):
    item=read(FIXTURES/f'brass-{index:03d}-request.json')['cases'][0]
    inp=copy.deepcopy(item['input']);inp['shuffle_tape']=None;inp['sample_seed']=890+rep
    config=dict(item['config'],simulations=n,seed=7+rep,trace=False)
    mode=dict(kind='count')
    if seconds!=60:
        ctx=inp['context'];manifest=read(MODEL)
        mode=dict(kind='timed',startup=None if startup is None else dict(
            model_id=ctx['model_id'],export_sha256=manifest['export_sha256'],reward_id=ctx['reward_id'],
            phase=inp['view']['context']['phase'],estimate=dict(nanoseconds=startup,
                source=f'L12 scalar-compatible cap-one calibration: brass-{index:03d}; pinned model/runtime; 64 completed trajectories per owner')))
    return dict(name=f'brass-{index:03d}-c{policy["cap"]}-f{policy["flush_us"]}-r{rep}-s{seconds}',
                input=inp,config=config,policy=policy,mode=mode,allowance_ms=seconds*1000)


class Memory(ctypes.Structure):
    _fields_=[('cb',wintypes.DWORD),('faults',wintypes.DWORD),('peak',ctypes.c_size_t),
              ('working',ctypes.c_size_t),('peak_paged',ctypes.c_size_t),('paged',ctypes.c_size_t),
              ('peak_nonpaged',ctypes.c_size_t),('nonpaged',ctypes.c_size_t),('pagefile',ctypes.c_size_t),
              ('peak_pagefile',ctypes.c_size_t)]


def execute(binary,cases,out,label,scripted=False):
    source=out/(label+'-input.json');target=out/(label+'-output.json');log=out/(label+'.log')
    source.write_text(json.dumps(dict(manifest=str(MODEL),manifest_sha256=sha(MODEL),scripted=scripted,cases=cases),
                                 separators=(',',':'),allow_nan=False),encoding='utf-8')
    psapi=ctypes.WinDLL('psapi',use_last_error=True);kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    psapi.GetProcessMemoryInfo.argtypes=[wintypes.HANDLE,ctypes.POINTER(Memory),wintypes.DWORD]
    kernel.GetProcessTimes.argtypes=[wintypes.HANDLE,*[ctypes.POINTER(wintypes.FILETIME)]*4]
    peak=0;start=time.perf_counter()
    with log.open('wb') as stream:
        process=subprocess.Popen([binary,source,target],stdout=stream,stderr=stream,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        while process.poll() is None:
            memory=Memory();memory.cb=ctypes.sizeof(memory)
            if psapi.GetProcessMemoryInfo(process._handle,ctypes.byref(memory),memory.cb):peak=max(peak,memory.peak)
            time.sleep(.025)
        timestamps=[wintypes.FILETIME() for _ in range(4)]
        ok=kernel.GetProcessTimes(process._handle,*[ctypes.byref(t) for t in timestamps])
        cpu=sum(((t.dwHighDateTime<<32)+t.dwLowDateTime)/10_000_000 for t in timestamps[2:]) if ok else None
    assert process.returncode==0,log
    payload=read(target);assert len(payload['results'])==len(cases)
    process_receipt=dict(wall_seconds=time.perf_counter()-start,cpu_seconds=cpu,peak_working_set_bytes=peak,
                         scope='Whole diagnostic process including model startup; per-turn clocks are in each decision',
                         binary=receipt(binary),input=receipt(source),output=receipt(target),log=receipt(log))
    write(out/(label+'-process.json'),process_receipt)
    for row in payload['results']:
        assert row['result']['status']=='complete',(row['name'],row['result'])
        decision=row['result']['decision'];budget=decision['result']['budget'];lanes=decision['lanes']
        assert decision['sessions']==2 and decision['threads_per_session']==1
        assert decision['response_ready_ns']<=budget['deadline_ns']
        for lane in lanes:
            last=lane['outcome']['last_completion_ns']
            assert last is None or last<lane['outcome']['budget']['work_deadline_ns']
            assert lane['outcome']['schedule']['max_pending']<=row['policy']['pending_limit']
        assert not row['busy_after_retirement']
        # Raw proofs remain in the receipted file. Do not retain expanded JSON
        # plans while the next measurement process is running.
        for outcome in [decision['result'],*[lane['outcome'] for lane in lanes]]:
            for key in ('actions','action_ids','eligible_plans','gumbels','snapshot'):
                outcome.pop(key,None)
        decision['selection'].pop('actions',None)
        decision['selection'].pop('pool',None)
    print(label,len(cases),'passed',flush=True)
    return payload['results']


def completed(row):
    return sum(l['outcome']['completed'] for l in row['result']['decision']['lanes'])


def summarize(rows):
    decisions=[r['result']['decision'] for r in rows]
    lanes=[lane for d in decisions for lane in d['lanes']]
    return dict(turns=len(rows),completed=sum(completed(r) for r in rows),
        supported=sum(d['selection']['provenance']=='pooled_search_supported' for d in decisions),
        fallback=sum('fallback' in d['selection']['provenance'] or 'bounded' in d['selection']['provenance'] for d in decisions),
        max_response_ready_ms=max(d['response_ready_ns']/1e6 for d in decisions),
        max_poll_gap_ms=max(l['polls']['max_unchecked_ns']/1e6 for l in lanes),
        max_batch=max(l['neural']['max_batch'] for l in lanes),
        positions=sum(l['neural']['positions'] for l in lanes),batches=sum(l['neural']['batches'] for l in lanes),
        valid_cells=sum(l['neural']['valid_cells'] for l in lanes),padded_cells=sum(l['neural']['padded_cells'] for l in lanes),
        owned_copy_ns=sum(l['polls']['owned_copy_ns'] for l in lanes),
        owner_work_ns=sum(max(0,l['retired_ns']-l['started_ns']) for l in lanes),
        root_wait_ns=sum(l['root_wait_ns'] for l in lanes),
        queue_ns=sum(l['outcome']['schedule']['queue_ns'] for l in lanes),
        component_ns={key:sum(l['outcome']['schedule'][key] for l in lanes) for key in
                      ('sampling_ns','information_ns','lookup_ns','selection_ns','application_ns','evaluation_ns','publication_ns','time_to_leaf_ns')})


def benchmark(binary,alloc,out,repetitions):
    assert repetitions>=2
    out.mkdir(parents=True,exist_ok=False)
    # Frozen before reading any candidate result. Held-out roots do not select policy.
    protocol=dict(development=DEVELOPMENT,heldout=HELDOUT,policies=POLICIES,repetitions=repetitions,
                  seconds=4,selection='At least 5% geometric-mean completed-work improvement; no fewer supported plans. Evaluate chosen candidate on held-out roots; baseline stays available. No strength claim.')
    write(out/'protocol.json',protocol)
    all_roots=DEVELOPMENT+HELDOUT
    calibrated=execute(binary,[case(i,POLICIES[0]) for i in all_roots],out,'calibration')
    estimates={}
    for index,row in zip(all_roots,calibrated):
        lanes=row['result']['decision']['lanes']
        samples=[(l['outcome']['work_stopped_ns']-l['started_ns'])/l['outcome']['completed'] for l in lanes if l['outcome']['completed']]
        estimates[index]=math.ceil(max(samples)) if samples else None
    write(out/'startup-estimates.json',dict(estimates=estimates,source=receipt(out/'calibration-output.json')))
    # Independent allocator process and scripted evaluator attribution.
    for label,scripted in [('native',False),('scripted',True)]:
        rows=execute(alloc,[case(i,POLICIES[0]) for i in DEVELOPMENT],out,'allocations-'+label,scripted)
        assert all(r['allocation_probe'] and r['allocations']['allocations']>0 for r in rows)
        execute(binary,[case(i,POLICIES[0]) for i in DEVELOPMENT],out,'components-'+label,scripted)
    requests=[]
    for rep in range(repetitions):
        order=list(range(len(POLICIES)));random.Random(19890+rep).shuffle(order)
        for index in DEVELOPMENT:
            for p in order: requests.append(case(index,POLICIES[p],seconds=4,rep=rep,startup=estimates[index]))
    swept=execute(binary,requests,out,'development-sweep')
    groups={};comparison=[]
    for policy in POLICIES:
        selected=[r for r in swept if r['policy']==policy];groups[(policy['cap'],policy['flush_us'])]=selected
    baseline=groups[(1,0)];by_pair={(r['name'].split('-c')[0],r['name'].split('-r')[1]):r for r in baseline}
    for policy in POLICIES:
        selected=groups[(policy['cap'],policy['flush_us'])];ratios=[]
        for row in selected:
            ref=by_pair[(row['name'].split('-c')[0],row['name'].split('-r')[1])]
            if completed(ref)>0 and completed(row)>0:ratios.append(completed(row)/completed(ref))
            elif completed(ref)!=completed(row):raise AssertionError('asymmetric zero-work fixture')
        gm=math.exp(statistics.mean(map(math.log,ratios))) if ratios else 1.
        comparison.append(dict(policy=policy,geometric_completed_ratio=gm,**summarize(selected)))
    eligible=[r for r in comparison if r['geometric_completed_ratio']>=1.05 and r['supported']>=comparison[0]['supported']]
    chosen=max(eligible,key=lambda r:r['geometric_completed_ratio'])['policy'] if eligible else POLICIES[0]
    write(out/'development-selection.json',dict(comparison=comparison,candidate=chosen,baseline=POLICIES[0]))
    validation=[]
    for rep in range(repetitions):
        for index in HELDOUT:
            for policy in ([POLICIES[0],chosen] if chosen!=POLICIES[0] else [POLICIES[0]]):
                validation.append(case(index,policy,seconds=4,rep=rep,startup=estimates[index]))
    heldout=execute(binary,validation,out,'heldout-check')
    # All menu allowances exercise the same scheduler and fixed reserve contract.
    settings=execute(binary,[case(i,POLICIES[0],seconds=s,startup=estimates[i]) for i in [3,26] for s in (8,10,12)],out,'other-settings')
    report=dict(status='passed',identity='brass-gumbel-is-mcts/coordinator-v1',baseline=POLICIES[0],engineering_candidate=chosen,
                comparison=comparison,heldout=summarize(heldout),other_settings=summarize(settings),
                source_sha256=BASE['native_fingerprint'](),binary=receipt(binary),allocation_binary=receipt(alloc),model=receipt(MODEL),
                artifacts=[receipt(p) for p in sorted(out.iterdir()) if p.is_file()],
                measured_quality=False,l14_matched_benchmark=False,
                interpretation='Exploratory two-owner engineering sweep. Additional pending paths change selection order. Keep cap-one baseline and qualify quality separately in L14.')
    write(out/'benchmark.json',report)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--binary',type=Path,required=True)
    parser.add_argument('--alloc',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--repetitions',type=int,default=2);args=parser.parse_args()
    benchmark(args.binary.resolve(),args.alloc.resolve(),args.out.resolve(),args.repetitions)
