"""Ready continuations share the ordinary lane budget; complete families publish once."""
import heapq
import json
import threading
import time
import traceback
import zlib

from brass_ai.learning.contracts import require


def model_cohort(packet):
    """Execution order only: no opponent redraw, rotation or sampling RNG."""
    jobs=packet['jobs']
    return tuple(sorted({p['model_id'] for j in jobs for p in j.get('policies',[]) if 'model_id' in p}))


class ReadyWork:
    """No prefetch backlog: branch-ready work goes to the next free lane.

    Longer continuations precede shorter ones. Branch-producing parents start
    first to reveal dependent work early; remaining parents prefer the prior
    collection's measured duration. Stable job indices break all ties.
    Scheduling never consumes a game RNG or changes its policies.
    """
    def __init__(self,packets,estimates=None,model_cohorts=False):
        self.parents=[];self.branches=[];self.estimates=estimates or {}
        for packet in packets:
            job=packet['jobs'][0]
            heapq.heappush(self.parents,(not bool(job.get('branching')),
                -self.estimates.get(job['provenance']['kind'],1.),
                model_cohort(packet) if model_cohorts else (),job['index'],packet))

    def add_branch(self,packet):
        c=packet['continuation']['row']['view']['context']
        remaining=max(1,16-int(c['round'])-(8 if c['era']=='RailEra' else 0))
        heapq.heappush(self.branches,(-remaining,packet['jobs'][0]['index'],packet))

    def pop(self):
        return heapq.heappop(self.branches if self.branches else self.parents)[-1]

    def __bool__(self):return bool(self.parents or self.branches)


class FamilyMailbox:
    """Join in the origin process, without holding a worker or writer waiting.

    Only a compressed child's JSON crosses IPC. Parent features stay with their
    originating actor, and the ordinary serializer validates the joined family.
    The trusted native checkpoint is never written into replay/training data.
    """
    def __init__(self,inbox,results,stop,publish):
        self.inbox,self.results,self.stop,self.publish=inbox,results,stop,publish
        self.parents={};self.children={};self.lock=threading.Lock()
        self.thread=threading.Thread(target=self.listen,daemon=True)
        self.thread.start()

    @staticmethod
    def key(packet):return packet['schedule_token'],packet['jobs'][0]['index']

    def parent(self,completed,packet):
        key=self.key(packet)
        with self.lock:
            require(key not in self.parents,'Duplicate pending branch parent')
            self.parents[key]=(completed,packet)
            ready=self._ready(key)
        if ready:self.publish(*ready)

    def child(self,message):
        key=tuple(message['key'])
        with self.lock:
            require(key not in self.children,'Duplicate pending branch child')
            self.children[key]=message
            ready=self._ready(key)
        if ready:self.publish(*ready)

    def _ready(self,key):
        if key not in self.parents or key not in self.children:return None
        completed,packet=self.parents.pop(key)
        return completed,packet,self.children.pop(key)

    def listen(self):
        try:
            while True:
                message=self.inbox.get()
                if message is None:return
                self.child(message)
        except BaseException:
            self.stop.set();self.results.put(dict(error=traceback.format_exc()))

    def close(self):
        self.inbox.put(None);self.thread.join(timeout=10)
        require(not self.thread.is_alive(),'Branch family mailbox did not stop')
        if not self.stop.is_set():
            require(not self.parents and not self.children,'Unpublished branch family')


def pack_child(game):
    return zlib.compress(json.dumps(game,separators=(',',':'),allow_nan=False).encode(),1)


def attach_child(game,message):
    from .branching import validate_child
    from .encoding import PreparedEncoder
    game.assert_metadata()
    encoder=PreparedEncoder()
    arrays=[game.encoded(r,encoder) for r in game['records']]
    child=json.loads(zlib.decompress(message['payload']))
    validate_child(game,child)
    game['branches']=[child]
    game.branch_execution=message['execution']
    game.bind(arrays)


def report(events,started,finished,workers,cached=0):
    parents=[e for e in events if e['kind']=='parent']
    branches=[e for e in events if e['kind']=='branch']
    by_parent={e['job']:e for e in parents}
    require(all(e['job'] in by_parent for e in branches),'Missing branch parent timing')
    seconds=sum(e['finished']-e['started'] for e in branches)
    work=sum(e['finished']-e['started'] for e in events)
    parent_end=max((e['finished'] for e in parents),default=started)
    work_end=max((e['finished'] for e in events),default=started)
    waits=[max(0,e['started']-e['queued']) for e in branches]
    return dict(contract='branch-ready-scheduling/1.0.0',workers=workers,cached_parent_games=cached,
        parent_games=len(parents),branch_games=len(branches),collection_wall_seconds=finished-started,
        work_wall_seconds=work_end-started,parent_finish_seconds=parent_end-started,
        branch_tail_seconds=max(0,work_end-parent_end),publication_tail_seconds=max(0,finished-work_end),
        parent_worker_seconds=work-seconds,branch_worker_seconds=seconds,
        branch_worker_seconds_per_worker=seconds/workers if workers else 0.,
        total_work_seconds_per_worker=work/workers if workers else 0.,
        work_utilization=work/(workers*max(work_end-started,1e-9)) if workers else 0.,
        branch_queue_wait_mean_seconds=sum(waits)/len(waits) if waits else 0.,
        branch_queue_wait_max_seconds=max(waits,default=0.),
        branches_started_before_parent_finished=sum(e['started']<by_parent[e['job']]['finished'] for e in branches),
        parent_branch_overlap_seconds=sum(max(0,min(e['finished'],by_parent[e['job']]['finished'])-e['started']) for e in branches),
        scope='Live ordinary lanes only, unchanged worker cap. Worker-seconds include host/inference/CPU waits, not CPU busy time. Work/W is an ideal sharing reference, not measured counterfactual overhead; cached games excluded. Publication includes writer/IPC/backpressure.')
