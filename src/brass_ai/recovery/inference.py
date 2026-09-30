"""One frozen-model GPU service shared by spawned CPU game/search workers."""
from collections import OrderedDict, defaultdict
import copy
from dataclasses import dataclass
import multiprocessing
import queue
import threading
import time
import uuid

import numpy as np
import torch

from brass_ai.learning.batching import collate, model_id
from brass_ai.learning.contracts import require
from brass_ai.learning.features import Encoded
from .encoding import PreparedEncoder
from .telemetry import Timings, profile_window, memory


@dataclass
class Endpoint:
    requests: object
    connection: object
    lane: int
    timeout: float = 180.0


class ChannelLease:
    """A spawn-safe pool of private reply channels, one per worker or host lane."""
    def __init__(self, leases, requests):
        self.leases,self.requests=leases,requests
    def acquire(self):
        connection,lane=self.leases.get(timeout=60)
        return Endpoint(self.requests,connection,lane)


class RemoteEvaluator:
    def __init__(self, endpoint, identity, batch_size=64, max_candidate_cells=131072):
        self.endpoint,self.model_id=endpoint,identity
        self.batch_size,self.max_candidate_cells=batch_size,max_candidate_cells
        self.encoder=PreparedEncoder()
        self.telemetry=dict(batches=0,positions=0,max_batch_positions=0,max_padded_candidate_cells=0,predict_seconds=0.0)
        self.telemetry.update(validation_seconds=0.0,encoding_seconds=0.0)

    def predict(self, requests):
        if not requests: return []
        began=time.monotonic();token=uuid.uuid4().hex;frames=[]
        require(len({r.request_id for r in requests})==len(requests),'Duplicate inference request ID')
        for r in requests:
            phase=time.monotonic()
            require(r.expected_model==self.model_id and r.digest()==r.input_digest,'Mutated/mismatched inference request')
            require(np.isfinite(r.deadline),'Invalid inference deadline')
            self.telemetry['validation_seconds']+=time.monotonic()-phase
            if r.stopped():
                # Preserve the R2 stopped-result contract before expensive encoding.
                frames.append(dict(request_id=r.request_id,model=self.model_id,revision=r.view['context']['revision'],
                    seat=r.view['seat'],digest=r.input_digest,deadline=r.deadline,submitted=time.monotonic(),
                    cancelled=bool(r.cancelled and r.cancelled.is_set()),encoded=None))
                continue
            phase=time.monotonic();e=self.encoder.encode(r.view,r.choices,r.style,r.intent)
            self.telemetry['encoding_seconds']+=time.monotonic()-phase
            require(len(e.choice_ids)<=self.max_candidate_cells,'Position exceeds candidate capacity')
            frames.append(dict(request_id=r.request_id,model=self.model_id,revision=r.view['context']['revision'],
                seat=r.view['seat'],digest=r.input_digest,deadline=r.deadline,submitted=time.monotonic(),
                cancelled=bool(r.cancelled and r.cancelled.is_set()),encoded=e))
        self.endpoint.requests.put(dict(lane=self.endpoint.lane,token=token,frames=frames))
        require(self.endpoint.connection.poll(self.endpoint.timeout),'Inference service timeout; incomplete game is aborted')
        response=self.endpoint.connection.recv()
        require(response['token']==token,'Misrouted inference batch')
        require('error' not in response,response.get('error','Inference service failed'))
        self.telemetry['batches']+=1;self.telemetry['positions']+=len(requests)
        self.telemetry['max_batch_positions']=max(self.telemetry['max_batch_positions'],len(requests))
        self.telemetry['predict_seconds']+=time.monotonic()-began
        for r,result in zip(requests,response['responses']):
            stopped=r.stopped()
            if stopped: result['status']=stopped
        return response['responses']


class InferenceService:
    def __init__(self,config,device='cpu',profile=None,profile_steps=4):
        self.config,self.device=config,device
        self.mp=multiprocessing.get_context('spawn')
        self.requests=self.mp.Queue(maxsize=4*max(64,config['processes']*config['hosts_per_process']))
        self.connections={};self.children=[];self.leases=[];self.models={};self.cache=OrderedDict()
        self.lock=threading.RLock();self.stop=threading.Event();self.error=None
        self.timings=Timings();self.profile,self.profile_steps=profile,profile_steps
        self.metrics=dict(batches=0,positions=0,model_cache_hits=0,valid_candidate_cells=0,padded_candidate_cells=0,
            max_batch_positions=0,max_padded_candidate_cells=0,expired_requests=0,cancelled_requests=0)
        self.thread=threading.Thread(target=self._serve,name='brass-inference',daemon=True)

    def register(self,model):
        identity=model_id(model)
        with self.lock:
            if identity not in self.models:
                self.models[identity]=copy.deepcopy(model).cpu().eval().requires_grad_(False)
        return identity

    def register_checkpoint(self,path):
        from brass_ai.learning.checkpoint import load_checkpoint
        with torch.random.fork_rng(devices=[]):
            model,raw=load_checkpoint(path,device='cpu')
        return self.register(model)

    def trim(self, identities):
        """Called only after all round jobs/replies have finished."""
        with self.lock:
            keep=set(identities)
            for key in list(self.models):
                if key not in keep: self.models.pop(key);self.cache.pop(key,None)

    def channels(self,count):
        leases=self.mp.Queue()
        self.leases.append(leases)
        for _ in range(count):
            parent,child=self.mp.Pipe(duplex=True)
            lane=len(self.connections);self.connections[lane]=parent;self.children.append(child)
            leases.put((child,lane))
        return ChannelLease(leases,self.requests)

    def __enter__(self):
        self.thread.start();return self

    def __exit__(self,*_):
        self.stop.set();self.thread.join(timeout=30)
        require(not self.thread.is_alive(),'Inference service did not shut down')
        for c in list(self.connections.values())+self.children: c.close()
        for q in self.leases: q.cancel_join_thread();q.close()
        self.requests.close();self.requests.join_thread()

    def report(self):
        return dict(**self.metrics,timings=self.timings.report(),device=str(self.device),
            central_batching=True,memory=memory(self.device),
            timing_note='Host elapsed time; CUDA forward kernel duration is in optional profiler traces',
            error=str(self.error) if self.error else None)

    def _model(self,identity):
        require(identity in self.models,'Unregistered frozen model')
        if identity not in self.cache:
            self.cache[identity]=copy.deepcopy(self.models[identity]).to(self.device).eval().requires_grad_(False)
        else: self.metrics['model_cache_hits']+=1
        self.cache.move_to_end(identity)
        while len(self.cache)>self.config['max_models']: self.cache.popitem(last=False)
        return self.cache[identity]

    def _evaluate(self,items):
        responses={}
        # Each entry has only tensors derived from one permitted observer.
        by_model=defaultdict(list)
        for key,frame in items:
            require(set(frame)=={'request_id','model','revision','seat','digest','deadline','submitted','cancelled','encoded'},'Invalid inference frame')
            require(type(frame['seat']) is int and 0<=frame['seat']<4 and np.isfinite(frame['deadline']),'Invalid inference attribution')
            self.timings.observe('inference.queue_wait',max(0,time.monotonic()-frame['submitted']))
            status='cancelled' if frame['cancelled'] else ('expired' if time.monotonic()>=frame['deadline'] else None)
            if status:
                self.metrics[status+'_requests']+=1
                responses[key]=self._response(frame,status)
            else: by_model[frame['model']].append((key,frame))
        for identity,group in by_model.items():
            with self.lock: model=self._model(identity)
            group.sort(key=lambda item:len(item[1]['encoded'].choice_ids))
            while group:
                batch=[];maximum=0
                while group and len(batch)<self.config['batch_size']:
                    key,frame=group[0];e=frame['encoded'];n=len(e.choice_ids)
                    require(n>0 and n==len(set(e.choice_ids)) and n<=self.config['max_candidate_cells'],'Invalid encoded choices')
                    require(e.state.shape==(model.config.state_dim,) and e.actions.shape==(n,model.config.action_dim),
                        'Encoded model dimensions differ')
                    require(np.isfinite(e.state).all() and np.isfinite(e.actions).all(),'Nonfinite encoded inputs')
                    require(np.array_equal(e.stable_to_relative,[(s-frame['seat'])%4 for s in range(4)]),'Seat mapping differs')
                    maximum=max(maximum,n)
                    if batch and (len(batch)+1)*maximum>self.config['max_candidate_cells']: break
                    batch.append(group.pop(0))
                live=[]
                for key,frame in batch:
                    if time.monotonic()>=frame['deadline']:
                        self.metrics['expired_requests']+=1
                        responses[key]=self._response(frame,'expired')
                    else: live.append((key,frame))
                if not live: continue
                with torch.inference_mode():
                    with self.timings.phase('inference.pack_transfer'):
                        inputs=collate([f['encoded'] for _,f in live],self.device)
                    with self.timings.phase('inference.forward'):
                        logits,win,rank,vp=model(*inputs)
                    with self.timings.phase('inference.results'):
                        arrays=[a.cpu().numpy() for a in (torch.softmax(logits.float(),-1),win,rank,vp)]
                require(all(np.isfinite(a).all() for a in arrays),'Nonfinite inference output')
                cells=len(live)*max(len(f['encoded'].choice_ids) for _,f in live)
                self.metrics['batches']+=1;self.metrics['positions']+=len(live)
                self.metrics['max_batch_positions']=max(self.metrics['max_batch_positions'],len(live))
                self.metrics['max_padded_candidate_cells']=max(self.metrics['max_padded_candidate_cells'],cells)
                self.metrics['padded_candidate_cells']+=cells
                self.metrics['valid_candidate_cells']+=sum(len(f['encoded'].choice_ids) for _,f in live)
                for j,(key,frame) in enumerate(live):
                    r=self._response(frame,'expired' if time.monotonic()>=frame['deadline'] else 'ok')
                    if r['status']=='expired': self.metrics['expired_requests']+=1
                    if r['status']=='ok':
                        r.update(choice_ids=list(frame['encoded'].choice_ids),
                            probabilities=arrays[0][j,:len(frame['encoded'].choice_ids)].tolist(),
                            win_share=arrays[1][j].tolist(),placement=arrays[2][j].tolist(),vp=arrays[3][j].tolist())
                    responses[key]=r
        return responses

    @staticmethod
    def _response(frame,status):
        return dict(protocol='four-seat-inference/1.0.0',request_id=frame['request_id'],model_id=frame['model'],
            revision=frame['revision'],observer=frame['seat'],input_digest=frame['digest'],status=status)

    def _serve(self):
        try:
            with profile_window(self.profile,self.device,self.profile_steps) as profiler:
                while not self.stop.is_set():
                    try: first=self.requests.get(timeout=.05)
                    except queue.Empty: continue
                    messages=[first];until=time.monotonic()+self.config['flush_ms']/1000
                    until=min(until,min(f['deadline'] for f in first['frames']))
                    while sum(len(m['frames']) for m in messages)<self.config['batch_size']:
                        remaining=until-time.monotonic()
                        if remaining<=0: break
                        try:
                            message=self.requests.get(timeout=remaining);messages.append(message)
                            until=min(until,min(f['deadline'] for f in message['frames']))
                        except queue.Empty: break
                    try:
                        with self.timings.phase('inference.batch'):
                            frames=[((i,j),f) for i,m in enumerate(messages) for j,f in enumerate(m['frames'])]
                            replies=self._evaluate(frames)
                        for i,m in enumerate(messages):
                            self.connections[m['lane']].send(dict(token=m['token'],
                                responses=[replies[i,j] for j in range(len(m['frames']))]))
                        if profiler: profiler.step()
                    except BaseException as error:
                        for m in messages:
                            self.connections[m['lane']].send(dict(token=m['token'],error=str(error)))
                        self.error=error
        except BaseException as error:
            self.error=error
