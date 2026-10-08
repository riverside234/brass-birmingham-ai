"""Parallel, checked self-play ingestion; board histories stay in CPU readers."""
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import nullcontext
import multiprocessing
import time

import torch

from brass_ai.bank import sha
from brass_ai.learning.contracts import require
from .diagnostics import summarize, merge_summaries
from .progress import Progress
from .replay import read_shards
from .updates import prepared


def reader_init():
    torch.set_num_threads(1)


def reader_pool(workers):
    require(type(workers) is int and 1<=workers<=64,'Invalid replay reader CPU allowance')
    return (ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn'),
                                initializer=reader_init) if workers>1 else nullcontext(None))


def _prepare(task):
    began=time.perf_counter();cpu_began=time.process_time()
    records,policy_id,kinds,insolvency=task
    first=records[0];path=first['path']
    require(all(r['sha256']==first['sha256'] and r.get('features')==first.get('features') for r in records),
            'Conflicting self-play shard receipts')
    require(sha(path)==first['sha256'],'Self-play replay changed')
    games=read_shards([path],partition='train',receipts={path:first.get('features')})
    require(all(g['split']=='train' for g in games),'Held-out PPO data')
    expected={r['game_id'] for r in records}
    require(len(expected)==len(records) and {g['game_id'] for g in games}==expected,'Self-play shard membership differs')
    loaded=time.perf_counter()
    diagnostics=summarize(games,{policy_id},kinds)
    diagnosed=time.perf_counter()
    data=prepared(games,lambda r:r['behavior']['policy_id']==policy_id,insolvency=insolvency)
    require(all(r['behavior']['kind']=='model' for r,_,_,_ in data),'External actions in PPO')
    prepared_at=time.perf_counter()
    # Only verified features and learning labels cross IPC. No repeated private
    # hands, public history or board dictionaries are copied into the learner.
    compact=[(dict(seat=r['seat'],chosen=r['chosen'],behavior=r['behavior'],
                   view=dict(context={k:r['view']['context'][k] for k in ('era','phase')}),
                   choices=r['choices'],style=r['style'],intent=r['intent'],teacher=r['teacher'],
                   **{k:r[k] for k in ('_insolvency_cost_to_go','_insolvency_trajectory') if k in r}),
              target,identity,encoded) for r,target,identity,encoded in data]
    headers=[{k:v for k,v in g.items() if k!='records'} for g in games]
    ended=time.perf_counter()
    metrics=dict(worker_cpu_seconds=time.process_time()-cpu_began,worker_elapsed_seconds=ended-began,
        read_validate_seconds=loaded-began,diagnostics_seconds=diagnosed-loaded,
        prepare_seconds=prepared_at-diagnosed,compact_seconds=ended-prepared_at)
    return headers,compact,diagnostics,metrics


def load_collection(records,policy_id,kinds,*,executor=None,workers=1,telemetry=None,insolvency=None):
    """Retain deterministic shard order and the same validation/diagnostics."""
    began=time.perf_counter();by_path={}
    ordered=sorted(records,key=lambda r:r['job'])
    require(ordered and len({r['job'] for r in ordered})==len(ordered),'Duplicate or empty collection jobs')
    require(len({r['game_id'] for r in ordered})==len(ordered),'Duplicate game across shards')
    for record in ordered:by_path.setdefault(record['path'],[]).append(record)
    tasks=[(group,policy_id,kinds,insolvency) for group in by_path.values()]
    parts=[None]*len(tasks);log=Progress(f'self-play prepare (CPU, {min(workers,len(tasks))} readers)',len(tasks),'shards')
    if executor is None:
        for i,task in enumerate(tasks):parts[i]=_prepare(task);log(i+1)
    else:
        remaining=iter(enumerate(tasks));pending={};completed=0
        def submit():
            item=next(remaining,None)
            if item is not None:
                index,task=item;pending[executor.submit(_prepare,task)]=index
        for _ in range(min(len(tasks),2*workers)):submit()
        while pending:
            future=next(as_completed(pending));index=pending.pop(future)
            parts[index]=future.result();completed+=1;log(completed);submit()
    result=([g for games,_,_,_ in parts for g in games],
            [row for _,data,_,_ in parts for row in data],merge_summaries([p[2] for p in parts]))
    if telemetry is not None:
        seconds=time.perf_counter()-began
        totals={k:sum(p[3][k] for p in parts) for k in parts[0][3]}
        telemetry.update(totals,wall_seconds=seconds,shards=len(tasks),learner_rows=len(result[1]),
            readers=min(workers,len(tasks)) if executor is not None else 1,
            worker_average_cores=totals['worker_cpu_seconds']/max(seconds,1e-9),
            max_shard_seconds=max(p[3]['worker_elapsed_seconds'] for p in parts),
            scope='Reader-task CPU excludes process startup and result serialization; worker elapsed phases overlap across readers. Wall includes queue/IPC and parent merge.')
    return result
