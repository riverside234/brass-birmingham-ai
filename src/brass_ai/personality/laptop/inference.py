"""Portable named heads with private replies and independent CPU sessions."""
from collections import defaultdict
import time
import numpy as np

from brass_ai.learning.contracts import digest,require
from brass_ai.learning.batching import collate
from brass_ai.personality.contracts import context_vector,FACTORS
from brass_ai.recovery.inference import InferenceService
from .export import Graph,check_backend


def execution():
    # One game turn at a time, two private search workers plus two reserved
    # inference/presentation cores. The Unity execution scheduler owns the job.
    return dict(processes=1,hosts_per_process=1,cpu_budget=4,cpu_reserve=2,
        batch_size=64,max_candidate_cells=131072,flush_ms=1.,max_models=5,
        shared_slot_bytes=2*1024*1024)


class _GraphService(InferenceService):
    def __new__(cls,*args,**kwargs):
        return object.__new__(cls)

    def __init__(self,directory,models,backend,threads,config):
        super().__init__(config,'cpu');self.graphs={};self.parity=[]
        try:
            for entry in models:
                graph=Graph(directory/entry['path'],backend,threads)
                require(graph.meta['model_id']==entry['model_id'],'Portable model identity changed')
                self.parity.append(check_backend(graph));self.graphs[entry['model_id']]=graph
        except BaseException:
            InferenceService.__exit__(self);raise

    def _evaluate(self,items):
        replies={};groups=defaultdict(list)
        for key,f in items:
            require(f['model'] in self.graphs and type(f['seat']) is int and 0<=f['seat']<4
                    and np.isfinite(f['deadline']),'Unknown model or observer')
            self.timings.observe('inference.queue_wait',max(0.,time.monotonic()-f['submitted']))
            status='cancelled' if f['cancelled'] else ('expired' if time.monotonic()>=f['deadline'] else None)
            if status:
                self.metrics[status+'_requests']+=1;replies[key]=self._response(f,status)
            else:groups[f['model']].append((key,f))
        for ident,group in sorted(groups.items(),key=lambda x:min(f['deadline'] for _,f in x[1])):
            graph=self.graphs[ident];cfg=graph.meta['config'];actor=graph.meta['actor']
            group.sort(key=lambda x:(int(x[1]['deadline']/.01),len(x[1]['encoded'].choice_ids)))
            while group:
                batch=[];width=0
                while group and len(batch)<self.config['batch_size']:
                    key,f=group[0];e=f['encoded'];n=len(e.choice_ids)
                    require(0<n<=self.config['max_candidate_cells'] and n==len(set(e.choice_ids)), 'Invalid legal candidate count')
                    require(e.state.shape==(cfg['state_dim'],) and e.actions.shape==(n,cfg['action_dim'])
                        and np.isfinite(e.state).all() and np.isfinite(e.actions).all(),'Invalid features')
                    require(np.array_equal(e.stable_to_relative,[(s-f['seat'])%4 for s in range(4)]),'Wrong observer mapping')
                    require(actor==('personality' in f),'Head routing differs')
                    proposed=max(width,n)
                    if batch and proposed*(len(batch)+1)>self.config['max_candidate_cells']:break
                    batch.append(group.pop(0));width=proposed
                live=[]
                for key,f in batch:
                    if time.monotonic()>=f['deadline']:
                        self.metrics['expired_requests']+=1;replies[key]=self._response(f,'expired')
                    else:live.append((key,f))
                if not live:continue
                inputs={k:t.numpy() for k,t in zip(graph.inputs[:4],collate([f['encoded'] for _,f in live]))}
                cap=cfg['utility']['insolvency']['max_cost']
                inputs['remaining']=np.asarray([f.get('remaining') if f.get('remaining') is not None else [cap]*4 for _,f in live],np.float32)
                require(inputs['remaining'].shape==(len(live),4) and np.isfinite(inputs['remaining']).all()
                    and (inputs['remaining']>=0).all() and (inputs['remaining']<=cap+1e-8).all(),'Invalid remaining costs')
                if actor:inputs['personality']=np.stack([context_vector(f['personality']) for _,f in live]).astype('float32')
                with self.timings.phase('inference.forward'):out=graph.run(inputs)
                logits=out['policy_logits'];p=np.exp(logits-logits.max(-1,keepdims=True));p/=p.sum(-1,keepdims=True)
                require(np.isfinite(p).all() and all(np.isfinite(v).all() for k,v in out.items() if k!='policy_logits'), 'Nonfinite export outputs')
                self.metrics['batches']+=1;self.metrics['positions']+=len(live)
                self.metrics['max_batch_positions']=max(self.metrics['max_batch_positions'],len(live))
                self.metrics['padded_candidate_cells']+=len(live)*width
                self.metrics['valid_candidate_cells']+=sum(len(f['encoded'].choice_ids) for _,f in live)
                self.metrics['max_padded_candidate_cells']=max(self.metrics['max_padded_candidate_cells'],len(live)*width)
                for i,(key,f) in enumerate(live):
                    r=self._response(f,'expired' if time.monotonic()>=f['deadline'] else 'ok')
                    if r['status']=='ok':
                        r.update(choice_ids=list(f['encoded'].choice_ids),probabilities=p[i,:len(f['encoded'].choice_ids)].tolist(),
                            win_share=out['win_share_4'][i].tolist(),placement=out['placement_4'][i].tolist(),vp=out['vp_4'][i].tolist(),
                            utility=out['remaining_utility_4'][i].tolist(),reward_id=digest(cfg['utility']))
                        if actor:r.update(goals={k:out['goal_'+k][i].tolist() for k in FACTORS},
                            confidence_logits=out['confidence_logits'][i].tolist(),context_id=digest(f['personality']))
                    else:self.metrics['expired_requests']+=1
                    replies[key]=r
        return replies


class PortableInference(_GraphService):
    """Two single-thread ORT CPU replicas, using the private-envelope queue.

    A free replica consumes the next complete envelope. There is no replica
    affinity tied to reply-lane parity (each worker has several reply lanes).
    Each replica exclusively owns its sessions, metrics and timing samples.
    Shared slots still have exactly one outstanding owner until its reply.
    """
    def __init__(self,directory,models,backend='ort-cpu',*,sessions=None,threads=None):
        import torch
        cpu=backend in ('ort-cpu','openvino-cpu')
        # The measured default is ORT CPU. Other backends keep their existing
        # single-session recipe unless a CPU comparison explicitly requests two.
        sessions=(2 if backend=='ort-cpu' else 1) if sessions is None else sessions
        threads=(1 if sessions==2 else 2) if threads is None else threads
        require(type(sessions) is int and sessions in (1,2) and type(threads) is int
                and threads in (1,2) and sessions*threads<=2 and (cpu or sessions==1),
                'Portable inference has a two-core allowance; GPU uses one session')
        torch.set_num_threads(2)
        config=execution()
        if sessions>1:
            # CPU graphs serve small requests immediately. Do not make both
            # consumers wait for a batching flush, or serialize into a B=2 call.
            config.update(batch_size=1,flush_ms=0.)
        self.replicas=[];self.session_threads=threads
        super().__init__(directory,models,backend,threads,config)
        self.replicas.append(self)
        self.thread.name='brass-portable-inference-0'
        try:
            for _ in range(1,sessions):
                replica=_GraphService(directory,models,backend,threads,config)
                # No channels have been allocated on this replica. The facade
                # alone owns transport lifetime; both consumers use its queue.
                replica.requests.close()
                replica.requests=self.requests;replica.connections=self.connections
                replica.slots=self.slots;replica.cpu=self.cpu;replica.stop=self.stop
                replica.on_failure=self._failed
                replica.thread.name='brass-portable-inference-'+str(len(self.replicas))
                self.replicas.append(replica)
        except BaseException:
            self.__exit__();raise

    def __enter__(self):
        require(not self.started and not self.closed,'Inference service is already started/closed')
        try:
            for replica in self.replicas[1:]:
                replica.thread.start();replica.started=True
            return super().__enter__()
        except BaseException:
            self.__exit__();raise

    def __exit__(self,*args):
        if self.closed:return
        self.stop.set()
        for child in self.children:child.close()
        # Retire EVERY synchronous graph invocation before unmapping a slot or
        # closing its private reply pipe. Cancellation is not retirement.
        until=time.monotonic()+30
        for replica in self.replicas:
            if replica.started:
                replica.thread.join(timeout=max(0.,until-time.monotonic()))
                require(not replica.thread.is_alive(),'Portable inference replica did not shut down')
        super().__exit__(*args)
        for replica in self.replicas:
            replica.graphs.clear();replica.closed=True

    def report(self):
        result=super().report()
        replicas=[dict(index=i,threads=self.session_threads,**r.metrics,timings=r.timings.report(),parity=r.parity,
                       error=str(r.error) if r.error else None) for i,r in enumerate(self.replicas)]
        for key in self.metrics:
            values=[r[key] for r in replicas]
            result[key]=max(values) if key.startswith('max_') else sum(values)
        # Preserve per-replica distributions; a percentile of percentiles is
        # not a pooled percentile. Totals below include both session owners.
        names={key for r in replicas for key in r['timings']}
        result['timings']={}
        for name in sorted(names):
            rows=[r['timings'][name] for r in replicas if name in r['timings']]
            seconds=sum(r['seconds'] for r in rows);calls=sum(r['calls'] for r in rows)
            result['timings'][name]=dict(seconds=seconds,calls=calls,mean_seconds=seconds/calls,
                max_seconds=max(r['max_seconds'] for r in rows),
                percentile_scope='See independent replica distributions')
        result.update(replica_count=len(replicas),session_threads=self.session_threads,replicas=replicas,
            central_batching=len(replicas)==1,routing='next-idle-private-envelope',
            timing_note='Summed CPU replica elapsed times can overlap; wall time belongs to the whole turn')
        return result
