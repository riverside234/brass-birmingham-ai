"""One frozen-model GPU service shared by spawned CPU game/search workers."""
from collections import OrderedDict, defaultdict
from contextlib import nullcontext
import copy
from dataclasses import dataclass
from dataclasses import field
import multiprocessing
import queue
from pathlib import Path
import threading
import time
import uuid

import numpy as np
import torch

from brass_ai.learning.batching import collate, model_id, predict_encoded, utility_budget
from brass_ai.learning.contracts import require
from brass_ai.learning.features import Encoded
from .encoding import PreparedEncoder
from .telemetry import Timings, profile_window, memory


def validate_inference_devices(devices,device=None):
    """Runtime placement only; never discover/claim GPUs without an explicit list."""
    if devices is None:return None
    require(isinstance(devices,(list,tuple)) and bool(devices),'Specify at least one inference GPU')
    if list(devices)==['auto']:
        require(torch.cuda.is_available(),'CUDA inference GPUs are unavailable')
        devices=['cuda:'+str(i) for i in range(torch.cuda.device_count())]
        require(devices,'CUDA inference GPUs are unavailable')
    parsed=[]
    for name in devices:
        require(isinstance(name,str),'Inference devices must be explicit cuda:N names')
        try:target=torch.device(name)
        except (RuntimeError,ValueError) as error:raise ValueError('Invalid inference GPU '+name) from error
        require(target.type=='cuda' and target.index is not None,'Inference devices must be explicit cuda:N names')
        parsed.append(str(target))
    require(len(set(parsed))==len(parsed),'Duplicate inference GPU')
    require(torch.cuda.is_available(),'CUDA inference GPUs are unavailable')
    count=torch.cuda.device_count()
    require(all(torch.device(name).index<count for name in parsed),'Requested inference GPU is not visible')
    return parsed


def predict_packed(model, encoded, device, candidate_limit, remaining=None):
    """Transfer normal-batch predictions together, without quantizing outputs.

    Five separate CUDA-to-host conversions otherwise synchronize the service
    once per policy/outcome/utility head. The existing oversized-position path
    remains responsible for chunking ALL candidates before the policy softmax.
    """
    if max(len(e.choice_ids) for e in encoded)>candidate_limit:
        return predict_encoded(model,encoded,device,candidate_limit,remaining)
    inputs=collate(encoded,device)
    outputs=model(*inputs,remaining=remaining) if getattr(model.config,'utility',None) is not None else model(*inputs)
    parts=[torch.softmax(outputs[0].float(),-1)]+[out.float() for out in outputs[1:]]
    widths=[part.shape[1] for part in parts]
    packed=torch.cat(parts,dim=1).cpu().numpy()
    offsets=np.cumsum([0]+widths)
    return [packed[:,offsets[i]:offsets[i+1]] for i in range(len(parts))]


@dataclass
class Endpoint:
    requests: object
    connection: object
    lane: int
    timeout: float = 180.0
    pending: object = None
    lock: object = field(default_factory=threading.RLock)
    shared: object = None
    _slot: object = None

    def slot(self):
        if self.shared is not None and self._slot is None:
            from .shared_transport import SharedSlot
            self._slot=SharedSlot(*self.shared)
        return self._slot

    def close(self):
        with self.lock:
            if self._slot is not None:self._slot.close();self._slot=None
            self.connection.close()


class ChannelLease:
    """A spawn-safe pool of private reply channels, one per worker or host lane."""
    def __init__(self, leases, requests):
        self.leases,self.requests=leases,requests
    def acquire(self):
        item=self.leases.get(timeout=60)
        connection,lane=item[:2]
        return Endpoint(self.requests,connection,lane,shared=item[2] if len(item)>2 else None)


class MultiChannelLease:
    """Pickled during spawn; queue objects never travel inside another queue."""
    def __init__(self,leases,requests):self.leases,self.requests=leases,requests
    def acquire(self):
        item=self.leases.get(timeout=60);index,connection,lane=item[:3]
        return Endpoint(self.requests[index],connection,lane,shared=item[3] if len(item)>3 else None)


class RemoteEvaluator:
    def __init__(self, endpoint, identity, batch_size=64, max_candidate_cells=131072):
        self.endpoint,self.model_id=endpoint,identity
        self.batch_size,self.max_candidate_cells=batch_size,max_candidate_cells
        self.encoder=PreparedEncoder()
        self.telemetry=dict(batches=0,positions=0,max_batch_positions=0,max_padded_candidate_cells=0,predict_seconds=0.0)
        self.telemetry.update(validation_seconds=0.0,encoding_seconds=0.0)

    def predict(self, requests):
        if not requests: return []
        return self.poll(self.submit(requests),wait=True)

    def submit(self,requests):
        with self.endpoint.lock:
            return self._submit(requests)

    def _submit(self,requests,owners=None):
        require(requests,'Cannot submit an empty batch')
        require(getattr(self.endpoint,'pending',None) is None,'Private reply channel is still in use')
        began=time.monotonic();token=uuid.uuid4().hex;frames=[]
        require(len({r.request_id for r in requests})==len(requests),'Duplicate inference request ID')
        for owner,r in zip(owners or [self]*len(requests),requests):
            phase=time.monotonic()
            require(owner.endpoint is self.endpoint and r.expected_model==owner.model_id and r.digest()==r.input_digest,'Mutated/mismatched inference request')
            require(np.isfinite(r.deadline),'Invalid inference deadline')
            self.telemetry['validation_seconds']+=time.monotonic()-phase
            if r.stopped():
                # Preserve the R2 stopped-result contract before expensive encoding.
                frames.append(dict(request_id=r.request_id,model=owner.model_id,revision=r.view['context']['revision'],
                    seat=r.view['seat'],digest=r.input_digest,deadline=r.deadline,submitted=time.monotonic(),
                    cancelled=bool(r.cancelled and r.cancelled.is_set()),encoded=None,remaining=r.remaining))
                continue
            phase=time.monotonic();e=owner.encoder.encode(r.view,r.choices,r.style,r.intent)
            self.telemetry['encoding_seconds']+=time.monotonic()-phase
            frames.append(dict(request_id=r.request_id,model=owner.model_id,revision=r.view['context']['revision'],
                seat=r.view['seat'],digest=r.input_digest,deadline=r.deadline,submitted=time.monotonic(),
                cancelled=bool(r.cancelled and r.cancelled.is_set()),encoded=e,remaining=r.remaining))
        slot=self.endpoint.slot();packed=slot.pack_frames(frames,token) if slot is not None else None
        self.endpoint.requests.put(dict(lane=self.endpoint.lane,token=token,frames=packed if packed is not None else frames,shared=packed is not None))
        ticket=dict(token=token,requests=list(requests),began=began)
        self.endpoint.pending=ticket
        return ticket

    def predict_mixed(self,groups):
        """One submission for ready leaves, including distinct frozen models."""
        requests=[r for owner,rows in groups for r in rows]
        owners=[owner for owner,rows in groups for _ in rows]
        with self.endpoint.lock:ticket=self._submit(requests,owners)
        responses=self.poll(ticket,wait=True);offset=0;result=[]
        for _,rows in groups:
            result.append(responses[offset:offset+len(rows)]);offset+=len(rows)
        return result

    def poll(self,ticket,*,wait=False):
        with self.endpoint.lock:
            return self._poll(ticket,wait=wait)

    def _poll(self,ticket,*,wait=False):
        require(getattr(self.endpoint,'pending',None) is ticket,'Unknown/misrouted inference ticket')
        if not wait and not self.endpoint.connection.poll(0):return None
        require(self.endpoint.connection.poll(self.endpoint.timeout),'Inference service timeout; incomplete game is aborted')
        response=self.endpoint.connection.recv()
        self.endpoint.pending=None
        require(response['token']==ticket['token'],'Misrouted inference batch')
        require('error' not in response,response.get('error','Inference service failed'))
        if response.get('shared'):
            require(self.endpoint.slot() is not None,'Missing shared reply slot')
            response['responses']=self.endpoint.slot().unpack_responses(response['responses'],ticket['token'])
        requests=ticket['requests']
        self.telemetry['batches']+=1;self.telemetry['positions']+=len(requests)
        self.telemetry['max_batch_positions']=max(self.telemetry['max_batch_positions'],len(requests))
        self.telemetry['predict_seconds']+=time.monotonic()-ticket['began']
        for r,result in zip(requests,response['responses']):
            stopped=r.stopped()
            if stopped: result['status']=stopped
        return response['responses']


class InferenceService:
    def __new__(cls,config,device='cpu',profile=None,profile_steps=4,*,inference_devices=None):
        if inference_devices is not None:
            require(cls is InferenceService,'Explicit GPU replicas require the Step 5 inference service')
            devices=validate_inference_devices(inference_devices,device)
            if len(devices)>1:
                return MultiInferenceService(config,device,profile,profile_steps,inference_devices=devices)
            instance=super().__new__(cls)
            instance._single_inference_device=devices[0]
            return instance
        return super().__new__(cls)

    def __init__(self,config,device='cpu',profile=None,profile_steps=4,*,inference_devices=None):
        self.learner_device=device
        device=getattr(self,'_single_inference_device',device)
        self.config,self.device=config,device
        # CPU tensors already share host memory; packing them adds a copy.
        self.pack_outputs=torch.device(device).type=='cuda'
        self.mp=multiprocessing.get_context('spawn')
        from .resources import CPUAdmission
        self.cpu=CPUAdmission(self.mp,config.get('cpu_budget',64),config.get('cpu_reserve',4))
        self.requests=self.mp.Queue(maxsize=4*max(64,config['processes']*config['hosts_per_process']))
        self.connections={};self.children=[];self.leases=[];self.models={};self.cache=OrderedDict();self.slots={}
        self.lock=threading.RLock();self.stop=threading.Event();self.error=None
        self.started=self.closed=False
        self.timings=Timings();self.profile,self.profile_steps=profile,profile_steps
        self.metrics=dict(batches=0,positions=0,model_cache_hits=0,valid_candidate_cells=0,padded_candidate_cells=0,
            max_batch_positions=0,max_padded_candidate_cells=0,expired_requests=0,cancelled_requests=0,
            packed_output_batches=0,shared_requests=0,shared_replies=0,inline_requests=0)
        self.on_complete=self.on_failure=None
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
            leases.put(self._channel())
        return ChannelLease(leases,self.requests)

    def _channel(self):
        parent,child=self.mp.Pipe(duplex=True)
        lane=len(self.connections);self.connections[lane]=parent;self.children.append(child)
        from .shared_transport import SharedSlot
        size=self.config.get('shared_slot_bytes',2*1024*1024)
        if size:
            self.slots[lane]=SharedSlot(size=size)
        return child,lane,self.slots[lane].description if size else None

    def __enter__(self):
        require(not self.started and not self.closed,'Inference service is already started/closed')
        self.thread.start();self.started=True;return self

    def __exit__(self,*_):
        if self.closed:return
        self.stop.set()
        # These are the parent's copies of actor reply handles. Keeping them
        # open after actors retire can leave a failed sender waiting forever
        # for a reader that no longer exists.
        for child in self.children:child.close()
        if self.started:
            self.thread.join(timeout=30)
            require(not self.thread.is_alive(),'Inference service did not shut down')
        for c in list(self.connections.values())+self.children: c.close()
        for q in self.leases: q.cancel_join_thread();q.close()
        # An aborted collection can leave queued input with no consumer.
        # Joining its feeder would then wait forever on a full pipe.
        self.requests.cancel_join_thread();self.requests.close()
        for slot in self.slots.values():slot.close()
        self.closed=True

    def report(self):
        return dict(**self.metrics,timings=self.timings.report(),device=str(self.device),
            learner_device=str(self.learner_device),inference_devices=[str(self.device)],replica_count=1,
            central_batching=True,memory=memory(self.device),
            cpu_admission=self.cpu.report(),
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
            require(set(frame) in ({'request_id','model','revision','seat','digest','deadline','submitted','cancelled','encoded'},
                                  {'request_id','model','revision','seat','digest','deadline','submitted','cancelled','encoded','remaining'}),'Invalid inference frame')
            require(type(frame['seat']) is int and 0<=frame['seat']<4 and np.isfinite(frame['deadline']),'Invalid inference attribution')
            self.timings.observe('inference.queue_wait',max(0,time.monotonic()-frame['submitted']))
            status='cancelled' if frame['cancelled'] else ('expired' if time.monotonic()>=frame['deadline'] else None)
            if status:
                self.metrics[status+'_requests']+=1
                responses[key]=self._response(frame,status)
            else: by_model[frame['model']].append((key,frame))
        for identity,group in sorted(by_model.items(),key=lambda pair:min(f['deadline'] for _,f in pair[1])):
            with self.lock: model=self._model(identity)
            # Earliest deadline first within size buckets, so a long choice list
            # cannot starve behind an endless stream of short requests.
            group.sort(key=lambda item:(int(item[1]['deadline']/.01),len(item[1]['encoded'].choice_ids)))
            while group:
                batch=[];maximum=0
                while group and len(batch)<self.config['batch_size']:
                    key,frame=group[0];e=frame['encoded'];n=len(e.choice_ids)
                    require(n>0 and n==len(set(e.choice_ids)),'Invalid encoded choices')
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
                    with self.timings.phase('inference.forward'):
                        budget=utility_budget(model,[f.get('remaining') for _,f in live])
                        remaining=torch.from_numpy(budget).to(self.device) if budget is not None else None
                        predict=predict_packed if self.pack_outputs else predict_encoded
                        arrays=predict(model,[f['encoded'] for _,f in live],self.device,self.config['max_candidate_cells'],remaining)
                        if self.pack_outputs and max(len(f['encoded'].choice_ids) for _,f in live)<=self.config['max_candidate_cells']:
                            self.metrics['packed_output_batches']+=1
                require(all(np.isfinite(a).all() for a in arrays),'Nonfinite inference output')
                cells=min(self.config['max_candidate_cells'],len(live)*max(len(f['encoded'].choice_ids) for _,f in live))
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
                        if len(arrays)>4:
                            from brass_ai.learning.contracts import digest
                            r.update(utility=arrays[4][j].tolist(),reward_id=digest(model.config.utility))
                    responses[key]=r
        return responses

    @staticmethod
    def _response(frame,status):
        return dict(protocol='four-seat-inference/1.0.0',request_id=frame['request_id'],model_id=frame['model'],
            revision=frame['revision'],observer=frame['seat'],input_digest=frame['digest'],status=status)

    def _failed(self,error):
        # Exception contexts may retain numpy views into shared slots.
        # Later broken replies must not overwrite the original inference error.
        with self.lock:
            original=self.error if self.error is not None else RuntimeError(str(error))
            # Publish to the pool before exposing a local failure to its router.
            if self.on_failure is not None:self.on_failure(original)
            self.error=original
        self.cpu.abort()

    def _reply(self,message,response):
        try:self.connections[message['lane']].send(response)
        except BaseException as error:self._failed(error)

    def _serve(self):
        try:
            # Current CUDA device is thread-local. Explicit tensor placement and
            # this context leave the learner's device selection unchanged.
            selected=torch.cuda.device(self.device) if self.pack_outputs else nullcontext()
            with selected,profile_window(self.profile,self.device,self.profile_steps) as profiler:
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
                    frames=[];rows=None;outbound=[]
                    try:
                        with self.timings.phase('inference.batch'):
                            for i,m in enumerate(messages):
                                if m.get('shared'):
                                    require(m['lane'] in self.slots,'Missing shared input slot')
                                    rows=self.slots[m['lane']].unpack_frames(m['frames'],m['token']);self.metrics['shared_requests']+=1
                                else:rows=m['frames'];self.metrics['inline_requests']+=1
                                frames.extend(((i,j),f) for j,f in enumerate(rows))
                            replies=self._evaluate(frames)
                            # Release ndarray views before reusing/unmapping slots.
                            frames.clear();rows=None
                        for i,m in enumerate(messages):
                            output=[replies[i,j] for j in range(len(m['frames']))]
                            slot=self.slots.get(m['lane']);packed=slot.pack_responses(output,m['token']) if slot else None
                            if packed is not None:self.metrics['shared_replies']+=1
                            outbound.append(dict(token=m['token'],shared=packed is not None,
                                responses=packed if packed is not None else output))
                        if profiler: profiler.step()
                    except BaseException as error:
                        outbound=[dict(token=m['token'],error=str(error)) for m in messages]
                        self._failed(error)
                    finally:
                        frames.clear();rows=None
                    # Attempt each reply exactly once. A dead actor must not
                    # duplicate successful replies or starve the other lanes.
                    for message,response in zip(messages,outbound):self._reply(message,response)
                    try:
                        if self.on_complete is not None:self.on_complete(messages)
                    except BaseException as error:self._failed(error)
        except BaseException as error:
            self._failed(error)


def _single_service(config,device,profile,profile_steps):
    return InferenceService(config,device,profile=profile,profile_steps=profile_steps)


class MultiInferenceService:
    """Independent selected-GPU inference replicas with one global CPU gate.

    The learner/optimizer remain on their requested device. Every replica serves
    exactly the same immutable model IDs. Demand routing aggregates compatible
    requests before activating additional replicas; reply lanes remain private.
    """
    def __init__(self,config,device='cpu',profile=None,profile_steps=4,*,inference_devices):
        self.devices=validate_inference_devices(inference_devices,device)
        self.config,self.learner_device,self.device=config,device,self.devices[0]
        self.mp=multiprocessing.get_context('spawn');self.services=[];self.leases=[];self.cursor=0
        self.closed=False
        self.routing=config.get('inference_routing','demand')
        require(self.routing in ('demand','round_robin'),'Unknown inference routing mode')
        self.router_stop=threading.Event();self.router_error=None;self.router_lock=threading.Lock()
        self.failure_lock=threading.Lock();self.first_error=None
        self.loads=[0]*len(self.devices);self.dispatches=[0]*len(self.devices);self.affinity={}
        self.router_started=False
        try:
            for index,target in enumerate(self.devices):
                # PyTorch's process-wide profiler cannot own concurrent
                # sessions. Keep an explicitly requested short trace on the
                # first replica; all devices retain normal phase telemetry.
                trace=Path(profile)/('inference-'+str(index)) if profile is not None and index==0 else None
                self.services.append(_single_service(config,target,trace,profile_steps))
            self.cpu=self.services[0].cpu
            for service in self.services[1:]:service.cpu=self.cpu
            for service in self.services:service.on_failure=self._record_failure
            if self.routing=='demand':
                self.requests=self.mp.Queue(maxsize=4*max(64,config['processes']*config['hosts_per_process']))
                for index,service in enumerate(self.services):
                    service.connections=self.services[0].connections
                    service.slots=self.services[0].slots
                    service.on_complete=lambda messages,i=index:self._complete(i,messages)
                self.router=threading.Thread(target=self._dispatch,name='brass-inference-router',daemon=True)
        except BaseException:
            for service in self.services:service.__exit__(None,None,None)
            raise

    @property
    def models(self):return self.services[0].models

    @property
    def error(self):return self.first_error or self.router_error or next((s.error for s in self.services if s.error is not None),None)

    def _record_failure(self,error):
        # Device order and later router/pipe errors must not choose the cause.
        # Share one first-failure receipt across the router and every replica.
        with self.failure_lock:
            if self.first_error is None:self.first_error=RuntimeError(str(error))

    @property
    def metrics(self):
        reports=[s.metrics for s in self.services]
        return {key:(max(r[key] for r in reports) if key.startswith('max_') else sum(r[key] for r in reports))
                for key in reports[0]}

    def register(self,model):
        identity=model_id(model)
        frozen=copy.deepcopy(model).cpu().eval().requires_grad_(False) if identity not in self.models else self.models[identity]
        for service in self.services:
            with service.lock:service.models.setdefault(identity,frozen)
        return identity

    def register_checkpoint(self,path):
        from brass_ai.learning.checkpoint import load_checkpoint
        with torch.random.fork_rng(devices=[]):model,_=load_checkpoint(path,device='cpu')
        return self.register(model)

    def trim(self,identities):
        for service in self.services:service.trim(identities)

    def channels(self,count):
        require(type(count) is int and count>0,'Missing inference reply lanes')
        leases=self.mp.Queue();self.leases.append(leases)
        if self.routing=='demand':
            for _ in range(count):leases.put(self.services[0]._channel())
            return ChannelLease(leases,self.requests)
        for _ in range(count):
            index=self.cursor%len(self.services);self.cursor+=1
            connection,lane,slot=self.services[index]._channel()
            leases.put((index,connection,lane,slot))
        return MultiChannelLease(leases,tuple(s.requests for s in self.services))

    def __enter__(self):
        require(not self.closed,'Inference pool is closed')
        try:
            for service in self.services:service.__enter__()
            if self.routing=='demand':self.router.start();self.router_started=True
        except BaseException:
            self.__exit__(None,None,None);raise
        return self

    def __exit__(self,*args):
        if self.closed:return
        error=None
        if self.routing=='demand':
            self.router_stop.set()
            if self.router_started:self.router.join(timeout=30)
            require(not self.router_started or not self.router.is_alive(),'Inference router did not shut down')
            # No replica may still write another replica's shared reply slots.
            for service in self.services:service.stop.set()
            for service in self.services:
                for child in service.children:child.close()
            for service in self.services:
                if service.started:service.thread.join(timeout=30)
        for service in self.services:
            try:service.__exit__(*args)
            except BaseException as failure:error=error or failure
        for leases in self.leases:leases.cancel_join_thread();leases.close()
        if self.routing=='demand':self.requests.cancel_join_thread();self.requests.close()
        self.closed=True
        if error is not None:raise error

    def _complete(self,index,messages):
        with self.router_lock:
            self.loads[index]-=sum(len(m['frames']) for m in messages)
            require(self.loads[index]>=0,'Inference routing load underflow')

    def _dispatch(self):
        try:
            while not self.router_stop.is_set():
                try:first=self.requests.get(timeout=.05)
                except queue.Empty:continue
                messages=[first];until=time.monotonic()+self.config['flush_ms']/1000
                until=min(until,min(f['deadline'] for f in first['frames']))
                while sum(len(m['frames']) for m in messages)<self.config['batch_size']:
                    remaining=until-time.monotonic()
                    if remaining<=0:break
                    try:
                        message=self.requests.get(timeout=remaining);messages.append(message)
                        until=min(until,min(f['deadline'] for f in message['frames']))
                    except queue.Empty:break
                # An envelope owns one private reply slot. Never split or mix
                # its identities; the selected service still batches by model.
                for message in sorted(messages,key=lambda m:tuple(f['model'] for f in m['frames'])):
                    models=tuple(sorted({f['model'] for f in message['frames']}))
                    threshold=self.config.get('replica_backlog',max(4,self.config['batch_size']//2))
                    with self.router_lock:
                        preferred=self.affinity.get(models,0)
                        index=preferred if self.loads[preferred]<threshold else min(range(len(self.services)),key=lambda i:(self.loads[i],i))
                        self.loads[index]+=len(message['frames']);self.dispatches[index]+=1
                        self.affinity[models]=index
                        # Bound routing metadata independently of model history.
                        if len(self.affinity)>4*self.config['max_models']:self.affinity.pop(next(iter(self.affinity)))
                    # A failing replica may stop draining its bounded queue.
                    # Keep shutdown interruptible instead of blocking in put().
                    while not self.router_stop.is_set():
                        try:self.services[index].requests.put(message,timeout=.05);break
                        except queue.Full:
                            require(self.services[index].error is None,'Inference replica failed while routing')
                    else:
                        self._complete(index,[message])
                        break
        except BaseException as error:
            self.router_error=RuntimeError(str(error));self._record_failure(error);self.cpu.abort()

    def report(self):
        reports=[s.report() for s in self.services]
        timings={}
        for report in reports:
            for name,phase in report['timings'].items():
                total=timings.setdefault(name,dict(seconds=0.,calls=0,max_seconds=0.))
                total['seconds']+=phase['seconds'];total['calls']+=phase['calls']
                total['max_seconds']=max(total['max_seconds'],phase['max_seconds'])
        for phase in timings.values():phase['mean_seconds']=phase['seconds']/phase['calls']
        mem=dict(host_resident_bytes=reports[0]['memory'].get('host_resident_bytes'))
        for key in ('cuda_allocated_bytes','cuda_reserved_bytes'):
            if any(key in r['memory'] for r in reports):mem[key]=sum(r['memory'].get(key,0) for r in reports)
        return dict(**self.metrics,timings=timings,device=str(self.device),learner_device=str(self.learner_device),inference_devices=list(self.devices),
            central_batching=True,multi_gpu=True,replica_count=len(self.services),memory=mem,
            routing=dict(mode=self.routing,requests_per_replica=list(self.dispatches),pending_positions=list(self.loads)),
            cpu_admission=self.cpu.report(),error=str(self.error) if self.error else None,
            per_device=[dict(selected_device=name,**r) for name,r in zip(self.devices,reports)],
            timing_note='Inference phase totals sum independent devices and may overlap; per-device reports retain latency distributions.',
            profile_scope='Short profiler traces use only the first selected inference replica; phase telemetry covers every replica.')
