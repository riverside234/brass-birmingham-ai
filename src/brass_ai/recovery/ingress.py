"""Parallel, checked self-play ingestion; board histories stay in CPU readers."""
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from contextlib import contextmanager
import multiprocessing
import time
import threading

import torch

from brass_ai.bank import sha
from brass_ai.learning.contracts import require
from .diagnostics import summarize, merge_summaries
from .progress import Progress
from .replay import read_shards
from .updates import prepared


_admission=None

def reader_init(admission=None,runtime=None):
    global _admission
    _admission=admission
    if runtime is not None:
        from .collector_runtime import install
        install(runtime)
    torch.set_num_threads(1)


def reader_pool(workers,admission=None):
    require(type(workers) is int and 1<=workers<=64,'Invalid replay reader CPU allowance')
    return _spawned_readers(workers,admission) if workers>1 else _inline_reader(admission)


def _reader_ready():
    return True


@contextmanager
def _spawned_readers(workers,admission):
    from .collector_runtime import current
    with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn'),
                             initializer=reader_init,initargs=(admission,current())) as pool:
        # ProcessPoolExecutor starts lazily. Start its bounded readers while
        # actors play, rather than making the first durable shards pay for
        # Python/Torch imports. Do not insert a readiness barrier in collection.
        ready=[pool.submit(_reader_ready) for _ in range(workers)]
        yield pool
        for future in ready:require(future.result() is True,'Replay reader initialization failed')


@contextmanager
def _inline_reader(admission):
    # The single-reader path still overlaps live actors. Bind its shared CPU
    # allowance as in spawned readers, without changing the learner's PyTorch
    # thread setting or leaking admission into a later independent invocation.
    global _admission
    previous=_admission;_admission=admission
    try:yield None
    finally:_admission=previous


def _prepare(task):
    if _admission is None:return _prepare_inner(task)
    from .telemetry import Timings
    timings=Timings()
    with _admission.acquire(1,timings):return _prepare_inner(task)


def _prepare_inner(task):
    from . import replay_binding
    before_checks=replay_binding.report()
    began=time.perf_counter();cpu_began=time.process_time()
    records,policy_id,kinds,insolvency,reward=task
    first=records[0];path=first['path']
    require(all(r['sha256']==first['sha256'] and r.get('features')==first.get('features')
                and r.get('native_admission')==first.get('native_admission') for r in records),
            'Conflicting self-play shard receipts')
    native=first.get('native_admission')
    if native is not None:
        from brass_ai.learning.contracts import digest
        from brass_ai.rust_backend.lab_admission import read as read_native
        require(first['sha256']==native['replay_sha256'] and first.get('features')==native.get('features'),
                'Native self-play receipt differs')
        require(native['admission']['reward_id']==digest(reward)
                and native['admission']['job']['provenance']['learner_behavior']['policy_id']==policy_id,
                'Native collection reward/learner differs from preparation')
        # read_native hashes the exact bytes it decodes and checks the feature
        # receipt. Avoid a separate full-file hash and gzip/header parse first.
        games=[read_native(path,native)]
    else:
        games=read_shards([path],partition='train',receipts={path:first.get('features')},
                          _expected_hashes={path:first['sha256']})
    require(all(g['split']=='train' for g in games),'Held-out PPO data')
    expected={r['game_id'] for r in records}
    require(len(expected)==len(records) and {g['game_id'] for g in games}==expected,'Self-play shard membership differs')
    loaded=time.perf_counter()
    diagnostics=summarize(games,{policy_id},kinds)
    from .exploration_metrics import collect as exploration_metrics
    diagnostics['exploration_metrics']=exploration_metrics(games)
    from .branching import expand
    games=expand(games)
    diagnosed=time.perf_counter()
    data=[]
    parents={g['game_id']:g for g in games}
    for game in games:
        vp_parent=parents.get(game.get('branch',{}).get('parent_game_id'))
        if hasattr(game,'critic_table'):
            from .critic_data import check_table
            table=check_table(game.critic_table,game,policy_id)
            require(game.table_kind==kinds[game['setup_group']],'Critic table kind differs from plan')
            eligible={s for s,seat in enumerate(table['seats']) if seat['role'] in ('learner','snapshot')}
            data+=prepared([game],lambda r:r['seat'] in eligible,insolvency=insolvency,reward=reward,vp_parent=vp_parent)
        else:
            data+=prepared([game],lambda r:r['behavior']['policy_id']==policy_id,insolvency=insolvency,reward=reward,vp_parent=vp_parent)
    require(all(r['behavior']['kind']=='model' for r,_,_,_ in data),'External actions in PPO')
    prepared_at=time.perf_counter()
    # Only verified features and learning labels cross IPC. No repeated private
    # hands, public history or board dictionaries are copied into the learner.
    # Keep round for the four-phase anchor probe; era/action phase alone cannot
    # distinguish early/late Canal or Rail after these rows cross IPC. Revision
    # identifies distinct auxiliary decisions and must also survive disk replay.
    compact=[(dict(seat=r['seat'],chosen=r['chosen'],behavior=r['behavior'],
                   view=dict(context={k:r['view']['context'][k] for k in ('era','phase','round','revision')}),
                   choices=r['choices'],style=r['style'],intent=r['intent'],teacher=r['teacher'],
                   **{k:r[k] for k in ('_insolvency_cost_to_go','_insolvency_trajectory','_reward_return','_reward_remaining','_reward_id','_reward_trajectory','_reward_vp_paid',
                       '_critic_role','_table_kind','_seat_descriptors','_descriptor_generation','_branch','_parent_game_id',
                       '_ppo_opening_objective','_local_scoring','_domino','_domino_event') if k in r}),
              target,identity,encoded) for r,target,identity,encoded in data]
    headers=[dict({k:v for k,v in g.items() if k not in ('records','branches')},
        **({'_diversity_assignment':g.diversity_assignment} if hasattr(g,'diversity_assignment') else {}),
        **({'_cotton_seats':g.cotton_seats} if hasattr(g,'cotton_seats') else {})) for g in games]
    # Only small training metadata is detached. Do not let an Owned choice or
    # envelope retain its whole game in the single-reader/no-pickle path.
    memo={};headers=replay_binding.compact_metadata(headers,memo)
    compact=[(replay_binding.compact_metadata(row,memo),replay_binding.compact_metadata(target,memo),identity,encoded)
             for row,target,identity,encoded in compact]
    ended=time.perf_counter()
    metrics=dict(worker_cpu_seconds=time.process_time()-cpu_began,worker_elapsed_seconds=ended-began,
        read_validate_seconds=loaded-began,diagnostics_seconds=diagnosed-loaded,
        prepare_seconds=prepared_at-diagnosed,compact_seconds=ended-prepared_at,
        binding_checks=replay_binding.since(before_checks))
    return headers,compact,diagnostics,metrics


def load_collection(records,policy_id,kinds,*,executor=None,workers=1,telemetry=None,insolvency=None,reward=None,stream=None):
    """Retain deterministic shard order and the same validation/diagnostics."""
    began=time.perf_counter();by_path={}
    ordered=sorted(records,key=lambda r:r['job'])
    require(ordered and len({r['job'] for r in ordered})==len(ordered),'Duplicate or empty collection jobs')
    require(len({r['game_id'] for r in ordered})==len(ordered),'Duplicate game across shards')
    for record in ordered:by_path.setdefault(record['path'],[]).append(record)
    tasks=[(group,policy_id,kinds,insolvency,reward) for group in by_path.values()]
    parts=[None]*len(tasks);log=Progress(f'self-play prepare (CPU, {min(workers,len(tasks))} readers)',len(tasks),'shards')
    if stream is not None:
        for i,task in enumerate(tasks):parts[i]=stream.result(task);log(i+1)
        require(set(stream.futures)==set(by_path),'Streaming replay membership differs')
    elif executor is None:
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
    from .replay_binding import seal_encoded,enabled
    data=[row for _,data,_,_ in parts for row in data]
    if enabled():
        # A reader's capabilities never survive pickle. Establish the learner's
        # own detached numerical boundary once; epochs reuse it thereafter.
        data=[(row,target,identity,seal_encoded(encoded)) for row,target,identity,encoded in data]
    result=([g for games,_,_,_ in parts for g in games],data,merge_summaries([p[2] for p in parts]))
    if telemetry is not None:
        seconds=time.perf_counter()-began
        totals={k:sum(p[3][k] for p in parts) for k,v in parts[0][3].items() if isinstance(v,(int,float))}
        keys={k for p in parts for k in p[3].get('binding_checks',{})}
        totals['binding_checks']={k:sum(p[3].get('binding_checks',{}).get(k,0) for p in parts) for k in keys}
        interval=seconds if stream is None else time.perf_counter()-stream.began
        telemetry.update(totals,wall_seconds=seconds,shards=len(tasks),learner_rows=len(result[1]),
            readers=min(workers,len(tasks)) if executor is not None or (stream is not None and stream.executor is not None) else 1,
            worker_average_cores=totals['worker_cpu_seconds']/max(interval,1e-9),
            max_shard_seconds=max(p[3]['worker_elapsed_seconds'] for p in parts),
            scope='Reader-task CPU excludes process startup and result serialization; worker elapsed phases overlap across readers. Wall is the final join/merge; average cores uses the full streaming interval when enabled.')
        if stream is not None:
            telemetry.update(streaming=True,stream_elapsed_seconds=interval,
                             prepared_before_collection_end=stream.ready_before_finish)
    return result


def _seal_part(future):
    """Establish the learner's numerical boundary for one finished shard."""
    headers,data,diagnostics,metrics=future.result()
    from .replay_binding import seal_encoded,enabled
    if enabled():data=[(row,target,identity,seal_encoded(encoded)) for row,target,identity,encoded in data]
    return headers,data,diagnostics,metrics


class StreamingCollection:
    """Bounded preparation after durable shard publication, before PPO starts.

    Job order, validation, full-game causal returns and diagnostics use the same
    _prepare function as the old barrier path. Only scheduling changes.
    """
    def __init__(self,executor,workers,policy_id,kinds,insolvency=None,reward=None):
        self.executor,self.workers=executor,workers
        self.arguments=(policy_id,kinds,insolvency,reward)
        self.futures={};self.identities={};self.lock=threading.Lock()
        # Seal each shard as it arrives, so the learner does not pay for all
        # 124k rows at once after the last game (about 7 s of single-threaded work).
        self.sealed={};self.sealer=ThreadPoolExecutor(max_workers=4,thread_name_prefix='replay-seal')
        self.slots=threading.Semaphore(max(1,2*workers));self.began=time.perf_counter();self.ready_before_finish=0

    def submit(self,records):
        from concurrent.futures import Future
        from brass_ai.learning.contracts import digest
        groups={}
        for r in records:groups.setdefault(r['path'],[]).append(r)
        for path,group in groups.items():
            group=sorted(group,key=lambda r:r['job']);identity=digest(group)
            self.slots.acquire()
            try:
                with self.lock:
                    require(path not in self.futures,'Repeated streaming shard')
                    task=(group,*self.arguments)
                    if self.executor is None:
                        future=Future()
                        try:future.set_result(_prepare(task))
                        except BaseException as e:future.set_exception(e)
                    else:future=self.executor.submit(_prepare,task)
                    self.futures[path]=future;self.identities[path]=identity
                    future.add_done_callback(lambda _:self.slots.release())
                    self.sealed[path]=self.sealer.submit(_seal_part,future)
            except BaseException:self.slots.release();raise

    def result(self,task):
        from brass_ai.learning.contracts import digest
        group=task[0];path=group[0]['path']
        require(path in self.futures and self.identities[path]==digest(group),'Changed streaming receipt')
        return self.sealed[path].result()

    def finish(self,records,telemetry):
        self.ready_before_finish=sum(f.done() for f in self.futures.values())
        policy,kinds,insolvency,reward=self.arguments
        return load_collection(records,policy,kinds,workers=self.workers,telemetry=telemetry,
                               insolvency=insolvency,reward=reward,stream=self)

    def __enter__(self):return self
    def __exit__(self,kind,*_):
        self.sealer.shutdown(wait=kind is None,cancel_futures=kind is not None)
        if kind is not None:
            for future in self.futures.values():future.cancel()
