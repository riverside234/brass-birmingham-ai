"""Named-head dispatch on the existing central queue, CPU gate and private pipes."""
import copy
from dataclasses import dataclass
import time
import uuid

import numpy as np
import torch

from brass_ai.learning.batching import Request,RequestStopped,accept_response,collate
from brass_ai.learning.encoding_budget import EncodingStopped
from brass_ai.learning.contracts import digest,require
from brass_ai.personality.checkpoint import model_id
from brass_ai.personality.contracts import context,context_vector
from brass_ai.personality.model import PersonalityNet
from brass_ai.recovery.inference import InferenceService,RemoteEvaluator
from brass_ai.search.common import SearchIncomplete
from brass_ai.step5.policy import NeuralPolicy,exploration_distribution


@dataclass(frozen=True)
class PersonalityRequest(Request):
    personality: dict | None=None

    def __post_init__(self):
        object.__setattr__(self,'personality',copy.deepcopy(self.personality))
        context_vector(self.personality)
        super().__post_init__()

    def digest(self):return digest(dict(base=super().digest(),personality=self.personality))


class PersonalityRemote(RemoteEvaluator):
    def _submit(self,requests):
        require(requests and self.endpoint.pending is None,'Private reply lane is busy')
        frames=[];began=time.monotonic();token=uuid.uuid4().hex
        require(len({r.request_id for r in requests})==len(requests),'Duplicate inference request ID')
        for r in requests:
            require(isinstance(r,PersonalityRequest) and r.expected_model==self.model_id and r.digest()==r.input_digest,'Changed personality request')
            require(np.isfinite(r.deadline),'Invalid inference deadline')
            stopped=r.stopped()
            encoded=None
            if not stopped:
                try:
                    encoded=self.encoder.encode(r.view,r.choices,r.style,r.intent,deadline=r.deadline,cancelled=r.cancelled)
                except EncodingStopped as error:
                    stopped=error.reason
            frames.append(dict(request_id=r.request_id,model=self.model_id,revision=r.view['context']['revision'],
                seat=r.view['seat'],digest=r.input_digest,deadline=r.deadline,submitted=began,
                cancelled=bool(stopped=='cancelled' or r.cancelled and r.cancelled.is_set()),encoded=encoded,remaining=r.remaining,
                personality=r.personality))
        self.endpoint.requests.put(dict(lane=self.endpoint.lane,token=token,frames=frames))
        ticket=dict(token=token,requests=list(requests),began=began);self.endpoint.pending=ticket
        return ticket


class SharedInference(InferenceService):
    def register(self,model):
        if not isinstance(model,PersonalityNet):return super().register(model)
        ident=model_id(model)
        with self.lock:
            if ident not in self.models:self.models[ident]=copy.deepcopy(model).cpu().eval().requires_grad_(False)
        return ident

    def _evaluate(self,items):
        ordinary=[(k,f) for k,f in items if 'personality' not in f]
        replies=super()._evaluate(ordinary) if ordinary else {}
        groups={}
        for key,frame in items:
            if 'personality' not in frame:continue
            require(set(frame)=={'request_id','model','revision','seat','digest','deadline','submitted','cancelled','encoded','remaining','personality'},'Invalid named-head frame')
            require(type(frame['seat']) is int and 0<=frame['seat']<4 and np.isfinite(frame['deadline']),'Invalid observer/deadline')
            context_vector(frame['personality'])
            status='cancelled' if frame['cancelled'] else ('expired' if time.monotonic()>=frame['deadline'] else None)
            self.timings.observe('inference.queue_wait',max(0,time.monotonic()-frame['submitted']))
            if status:
                self.metrics[status+'_requests']+=1;replies[key]=self._response(frame,status)
            else:groups.setdefault(frame['model'],[]).append((key,frame))
        for ident,group in groups.items():
            with self.lock:model=self._model(ident)
            require(isinstance(model,PersonalityNet),'Named input routed to an old model')
            group.sort(key=lambda x:(int(x[1]['deadline']/.01),len(x[1]['encoded'].choice_ids)))
            while group:
                batch=[];width=0
                while group and len(batch)<self.config['batch_size']:
                    key,f=group[0];e=f['encoded'];n=len(e.choice_ids)
                    require(0<n<=self.config['max_candidate_cells'] and n==len(set(e.choice_ids)),'Oversized/duplicate legal choices')
                    require(e.state.shape==(model.core.config.state_dim,) and e.actions.shape==(n,model.core.config.action_dim)
                        and np.isfinite(e.state).all() and np.isfinite(e.actions).all(),'Invalid encoded personality input')
                    require(np.array_equal(e.stable_to_relative,[(s-f['seat'])%4 for s in range(4)]),'Observer mapping changed')
                    proposed=max(width,n)
                    if batch and proposed*(len(batch)+1)>self.config['max_candidate_cells']:break
                    batch.append(group.pop(0));width=proposed
                live=[]
                for key,f in batch:
                    if time.monotonic()>=f['deadline']:
                        replies[key]=self._response(f,'expired');self.metrics['expired_requests']+=1
                    else:live.append((key,f))
                if not live:continue
                cap=model.core.config.utility['insolvency']['max_cost']
                remaining=np.asarray([f['remaining'] if f['remaining'] is not None else [cap]*4 for _,f in live],dtype=np.float32)
                require(remaining.shape==(len(live),4) and np.isfinite(remaining).all() and (remaining>=0).all() and (remaining<=cap+1e-8).all(),'Invalid remaining cost budgets')
                with torch.inference_mode(),self.timings.phase('inference.forward'):
                    out=model(*collate([f['encoded'] for _,f in live],self.device),
                        personality=torch.as_tensor(np.stack([context_vector(f['personality']) for _,f in live]),device=self.device),
                        remaining=torch.as_tensor(remaining,device=self.device))
                    p=torch.softmax(out['policy_logits'].float(),-1).cpu().numpy()
                    values={k:out[k].cpu().numpy() for k in ('win_share_4','placement_4','vp_4','remaining_utility_4','confidence_logits')}
                    goals={k:v.cpu().numpy() for k,v in out['goals'].items()}
                require(all(np.isfinite(x).all() for x in [p,*values.values(),*goals.values()]),'Nonfinite named outputs')
                self.metrics['batches']+=1;self.metrics['positions']+=len(live)
                self.metrics['max_batch_positions']=max(self.metrics['max_batch_positions'],len(live))
                cells=len(live)*max(len(f['encoded'].choice_ids) for _,f in live)
                self.metrics['padded_candidate_cells']+=cells
                self.metrics['valid_candidate_cells']+=sum(len(f['encoded'].choice_ids) for _,f in live)
                self.metrics['max_padded_candidate_cells']=max(self.metrics['max_padded_candidate_cells'],cells)
                for j,(key,f) in enumerate(live):
                    r=self._response(f,'expired' if time.monotonic()>=f['deadline'] else 'ok')
                    if r['status']=='ok':
                        r.update(choice_ids=list(f['encoded'].choice_ids),probabilities=p[j,:len(f['encoded'].choice_ids)].tolist(),
                            win_share=values['win_share_4'][j].tolist(),placement=values['placement_4'][j].tolist(),
                            vp=values['vp_4'][j].tolist(),utility=values['remaining_utility_4'][j].tolist(),
                            reward_id=digest(model.core.config.utility),goals={k:v[j].tolist() for k,v in goals.items()},
                            confidence_logits=values['confidence_logits'][j].tolist(),context_id=digest(f['personality']))
                    else:self.metrics['expired_requests']+=1
                    replies[key]=r
        return replies


class PersonalityPolicy(NeuralPolicy):
    def __init__(self,evaluator,*,personality=None,exploration=0.,cancelled=None):
        super().__init__(None,evaluator=evaluator,focus='neutral',exploration=exploration)
        self.cancelled=cancelled;self.set_context(personality or context())

    def set_context(self,value):
        context_vector(value);self.personality=copy.deepcopy(value)
        self.policy_id='personality-policy/'+digest(dict(model=self.evaluator.model_id,context=value,exploration=self.exploration))

    def predict(self,inputs):
        requests=[]
        for view,choices,rng in inputs:
            style,intent=self.context_features(view,rng)
            deadline=min(time.monotonic()+120,self.request_deadline) if self.request_deadline is not None else time.monotonic()+120
            requests.append(PersonalityRequest(uuid.uuid4().hex,self.evaluator.model_id,view,choices,deadline,
                tuple(style),tuple(intent),self.cancelled,self.utility_remaining.get(rng),personality=self.personality))
        responses=self.evaluator.predict(requests);result=[]
        require(len(requests)==len(responses),'Incomplete inference replies')
        for r,response in zip(requests,responses):
            if r.stopped() or response.get('status') in ('expired','cancelled'):raise SearchIncomplete('personality inference deadline/cancel')
            try:
                accept_response(r,response,r.view['context']['revision'],self.evaluator.model_id)
            except RequestStopped as error:
                raise SearchIncomplete('personality inference deadline/cancel') from error
            require(response['context_id']==digest(self.personality),'Private personality context changed')
            p=np.asarray(response['probabilities'],dtype=float)
            p=(1-self.exploration)*p+self.exploration*exploration_distribution(r.choices,r.intent)
            result.append((p/p.sum(),response))
        return result
