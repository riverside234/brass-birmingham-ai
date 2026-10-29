"""Bounded ready-work and atomic complete-family publication for B02 diagnostics.

These containers are deliberately NOT four-seat-replay shards. B03 must admit
the explicit native producer before any trainer can consume them.
"""
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import copy
import gzip
import json
from pathlib import Path
import queue
import random
import threading
import time
import uuid

import numpy as np
from brass_ai.learning.contracts import digest, require
from brass_ai.recovery.branching import validate_child
from brass_ai.recovery.branch_scheduler import ReadyWork
from brass_ai.recovery.replay import check_game
from brass_ai.recovery.local_scoring_targets import validate_receipt
from brass_ai.step5.policy import exploration_distribution,context_for
from .lab_collection import frozen_id, route
from .lab_contract import SOURCE

FORMAT='brass-lab-family/1.0.0'


def validate_part(part, job, *, child=False):
    game=check_game(part['game'])
    group=digest(dict(rules='steam-brass-patch-1.1.0',setup_seed=job['seed']))
    parent_id=job.get('stable_game_id') or digest(dict(job=frozen_id(job),seed=job['seed']))
    expected_id=parent_id+'/branch/'+game['branch']['seed'] if child else parent_id
    require(game['setup_group']==group and game['game_id']==expected_id
            and game['opponents']==[job['policies'][(s-job.get('rotation',0))%4]['policy_id'] for s in range(4)],
            'Game identity/assignment differs from frozen job')
    require(bool('branch' in game)==child, 'Family role differs')
    require(len(part['evidence'])==len(game['records']), 'Incomplete behavior evidence')
    if part['producer']['contract']=='brass-lab-collection/1.0.0':
        require(part['producer']['core']['source_sha256']==SOURCE,
                'Native shared core identity differs')
        receipt=part['native_receipt']
        require(receipt['contract']=='brass-lab-collection/1.0.0' and receipt['job']==frozen_id(job)
                and receipt['rows']==len(game['records']) and receipt['result']==game['result']
                and receipt['scoring']==part['scoring'],'Foreign/incomplete native receipt')
        validate_receipt(part['scoring'],game,producer='rust-local-scoring/1.0.0')
    else:
        require(part['producer']['contract']=='brass-reference-collection/1.0.0','Unknown producer')
    counts=[0]*4
    rngs=[random.Random(digest(dict(branch=game['branch']['seed'],seat=s)) if child else
            digest(dict(sampling_seed=job['sampling_seed'],game=job.get('rng_index',0),seat=s))) for s in range(4)]
    if child and game['branch']['contract']=='selfplay-branching/1.0.0':
        # Legacy branches keep their counters; the first recorded post-fork
        # action exposes the inherited value, independently audited by parent.
        counts=list(part['origin_counts'])
    for index,(row,evidence) in enumerate(zip(game['records'],part['evidence'],strict=True)):
        spec=job['policies'][(row['seat']-job.get('rotation',0))%4]
        require(row['behavior']['policy_id']==spec['policy_id'],'Changed frozen policy assignment')
        if child and index==0:
            allowed={'contract','parent_game_id','decision','played','seed','prior_events','raw_probabilities',
                     'chosen','component','probability','component_probability','marginal_probability'}
            require(set(evidence)=={'kind','metadata'} and evidence['kind']=='excluded-fork'
                    and set(evidence['metadata'])==allowed
                    and evidence['metadata']=={k:game['branch'][k] for k in allowed},
                    'Missing/unsafe excluded-fork evidence');continue
        require(set(evidence)<= {'kind','model_id','policy_id','forced','raw_probabilities','feature_version','semantic_id'},
                'Unexpected/private ordinary evidence')
        if spec['kind']!='neural':continue
        b=row['behavior'];phase=row['view']['context']['phase'];seat=row['seat'];opening=spec.get('opening')
        focus=job.get('focuses',{}).get(str(seat),'neutral') if spec['focus']=='mixed' else spec['focus']
        require((row['style'],row['intent'])==context_for(focus),'Frozen personality context differs')
        temperature=opening['temperature'] if opening and phase=='action' and counts[seat]<opening['action_decisions_per_seat'] else spec.get('temperature',1.)
        require(b['temperature']==temperature,'Wrong opening temperature')
        version=spec.get('exploration_version',2)
        if version==3:
            require(b['action_decision']==(counts[seat] if phase=='action' else None),'Opening counter differs')
            require(b['exploration_version']==version and b['exploration']==spec['exploration'],
                    'Frozen exploration settings differ')
        if phase=='action':counts[seat]+=1
        raw=np.asarray(evidence['raw_probabilities'],dtype=np.float64)
        require(raw.shape==(len(row['choices']),) and np.isfinite(raw).all() and (raw>=0).all()
                and abs(raw.sum()-1)<1e-5,'Invalid raw policy evidence')
        require(evidence['model_id']==spec['model_id'] and evidence['policy_id']==spec['policy_id']
                and evidence['semantic_id']==row['choices'][row['chosen']]['id'],'Evidence attribution differs')
        raw/=raw.sum(); p=np.power(raw.clip(1e-30),1/temperature);p/=p.sum()
        q=exploration_distribution(row['choices'],row['intent'],spec.get('exploration_version',2),phase)
        mixed=(1-spec['exploration'])*p+spec['exploration']*q
        require(abs(b['probability']-float(mixed[row['chosen']]))<3e-6,'Behavior denominator differs')
        require(len(row['choices'])!=1 or b['probability']==1.,'Forced choice probability differs')
        # Replay the original per-seat random stream, including forced and
        # intermediate choices. No GPU call or new policy identity is involved.
        if version==3:
            explore=rngs[seat].random()<spec['exploration']
            expected=rngs[seat].choices(range(len(p)),weights=q if explore else p,k=1)[0]
            require(b['component']==('exploration' if explore else 'policy'),'Sampling component differs')
        else:
            mixed/=mixed.sum()
            expected=rngs[seat].choices(range(len(p)),weights=mixed,k=1)[0]
        require(row['chosen']==expected,'Frozen sampling RNG differs')
    return game


def validate_family(value, job, producer):
    require(value['format']==FORMAT and value['job_id']==frozen_id(job) and value['producer']==producer,
            'Stale/foreign family identity')
    require(value['sha256']==digest({k:v for k,v in value.items() if k!='sha256'}),'Family seal differs')
    require(value['training_admitted'] is False,'B02 family cannot self-admit to training')
    require(value['frozen_job']=={k:v for k,v in job.items() if k!='path'},'Frozen job metadata differs')
    parent=validate_part(value['parent'],job)
    child=value['child']
    require(value['parent']['branch_expected']==(child is not None),'Incomplete/unexpected family member')
    if child is not None: validate_child(parent,validate_part(child,job,child=True))
    return value


class FamilyStore:
    def __init__(self, directory, producer):
        self.directory=Path(directory);self.directory.mkdir(parents=True,exist_ok=True)
        self.producer=copy.deepcopy(producer);self.lock=threading.Lock()

    def path(self,job): return self.directory/(frozen_id(job)+'.family.json.gz')

    def load(self,job):
        path=self.path(job)
        if not path.exists():return None
        with gzip.open(path,'rb') as stream:
            data=stream.read(512*1024**2+1)
        require(len(data)<=512*1024**2,'Family exceeds bound')
        return validate_family(json.loads(data),job,self.producer)

    def publish(self,job,parent,child,*,cancel=None):
        require(cancel is None or not cancel.is_set(),'Publication canceled')
        value=dict(format=FORMAT,job_id=frozen_id(job),frozen_job={k:v for k,v in job.items() if k!='path'},
            producer=self.producer,parent=parent,child=child,training_admitted=False)
        value['sha256']=digest(value)
        validate_family(value,job,self.producer)
        with self.lock:
            require(cancel is None or not cancel.is_set(),'Publication canceled')
            old=self.load(job)
            if old is not None:
                require(old['sha256']==value['sha256'],'Duplicate publication differs');return old
            path=self.path(job);temp=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
            data=json.dumps(value,separators=(',',':'),allow_nan=False).encode()
            require(len(data)<=512*1024**2,'Family exceeds publication bound')
            try:
                with temp.open('xb') as raw:
                    with gzip.GzipFile(fileobj=raw,mode='wb',mtime=0,compresslevel=1) as stream:
                        stream.write(data)
                    raw.flush()
                    import os
                    os.fsync(raw.fileno())
                require(cancel is None or not cancel.is_set(),'Publication canceled')
                temp.replace(path)
            finally:temp.unlink(missing_ok=True)
        return value


def collect_families(jobs, store, run_native, run_reference, *, lanes, cancel=None, model_cohorts=False):
    """Callbacks receive (job, continuation, submit_branch). Never wait on sibling.

    At most 2*lanes unfinished families. This bounds the ready checkpoint queue,
    completed sibling buffers, and publication queue independently of job count.
    """
    require(type(lanes) is int and lanes>0,'Invalid ordinary lane count')
    require(len({j['index'] for j in jobs})==len(jobs) and len({frozen_id(j) for j in jobs})==len(jobs),'Duplicate frozen jobs')
    jobs=copy.deepcopy(jobs); by_id={j['index']:j for j in jobs}
    ready=ReadyWork([dict(jobs=[j]) for j in jobs],model_cohorts=model_cohorts)
    messages=queue.Queue(maxsize=2*lanes);active={};pending={};published={};futures={};writes={};forked=set();counts={'rust':0,'reference':0,'cached':0}
    reasons={};began=time.monotonic();peak=0
    for j in jobs:
        saved=store.load(j)
        if saved is not None:published[j['index']]=saved;counts['cached']+=1
    def submit(index,packet):
        require(cancel is None or not cancel.is_set(),'Canceled fork')
        messages.put((index,packet),timeout=10)
    def drain_branches():
        while True:
            try:index,packet=messages.get_nowait()
            except queue.Empty:break
            require(index in active and index not in forked,'Duplicate/stale fork submission')
            forked.add(index);ready.add_branch(dict(jobs=[by_id[index]],continuation=packet))
    with ThreadPoolExecutor(max_workers=lanes) as workers,ThreadPoolExecutor(max_workers=1) as writer:
        while len(published)<len(jobs):
            require(cancel is None or not cancel.is_set(),'Collection canceled')
            drain_branches()
            while ready and len(futures)<lanes:
                # New parents wait when family buffers are full. Ready siblings
                # always run; no lane is held by a finished family member.
                if not ready.branches and len(active)>=2*lanes:break
                packet=ready.pop();job=packet['jobs'][0];index=job['index']
                if index in published:continue
                child=packet.get('continuation');role='child' if child else 'parent'
                if child is None:
                    require(index not in active,'Parent already active');active[index]=True
                    backend,reason=route(job);counts[backend]+=1
                    if reason:reasons[str(index)]=reason
                else:backend='rust'
                call=run_native if backend=='rust' else run_reference
                futures[workers.submit(call,job,child,lambda p,i=index:submit(i,p))]=(index,role)
            peak=max(peak,len(active))
            require(futures or writes,'Incomplete family without runnable work')
            done,_=wait([*futures,*writes],timeout=.05,return_when=FIRST_COMPLETED)
            # A parent's callback happens before its future completes. Drain it
            # before trusting the completed parent's branch_expected flag.
            drain_branches()
            for future in done:
                if future in writes:
                    index=writes.pop(future);published[index]=future.result();active.pop(index);continue
                index,role=futures.pop(future);part=future.result()
                if role=='parent':
                    require(part['branch_expected']==(index in forked),'Parent/fork submission differs')
                family=pending.setdefault(index,{})
                require(role not in family,'Duplicate family member');family[role]=part
                parent=family.get('parent')
                if parent and (not parent['branch_expected'] or 'child' in family):
                    require(parent['branch_expected'] or 'child' not in family,'Unexpected child')
                    require(cancel is None or not cancel.is_set(),'Collection canceled before publication')
                    writes[writer.submit(store.publish,by_id[index],parent,family.get('child'),cancel=cancel)]=index
                    pending.pop(index)
    require(not pending and not futures and not writes and messages.empty(),'Unfinished family state')
    return [published[j['index']] for j in jobs],dict(counts=counts,reference_reasons=reasons,
        wall_seconds=time.monotonic()-began,lanes=lanes,peak_unfinished_families=peak,limit=2*lanes)
