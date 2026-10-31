"""Training-only intrinsic companion and transactional DOMiNO iteration state."""
import copy
import time
import weakref
from collections import Counter
import numpy as np
import torch
from torch import nn
from brass_ai.learning.contracts import digest,require
from . import domino_config as C,domino_state as S
from .local_scoring_auxiliary import bundle_id

KEY='domino'


def row_key(ds,i):
    row=ds.rows[i]
    return (ds.data[i][2][0],int(row['seat']),int(row['view']['context']['revision']))


class Diversity:
    def __init__(self,model,config,*,raw=None,generation=0,regime=None):
        self.config=copy.deepcopy(C.validate(config));saved=(raw or {}).get('training',{}).get(KEY)
        adoption=config['adoption'];origin=adoption['generation'] if adoption else 0
        if saved is None:
            require(generation==origin,'Missing DOMiNO training state; weights-only opponents cannot resume training')
            if adoption:
                require(raw['model_id']==adoption['model_id'] and raw.get('step')==generation,
                    'DOMiNO adoption checkpoint/model differs')
            self.state=S.initial(config,generation,regime)
        else:
            require(saved['config']==config and saved['identity']==bundle_id({k:v for k,v in saved.items() if k!='identity'}),
                'DOMiNO checkpoint bundle differs')
            self.state=copy.deepcopy(S.validate(saved['state'],config,generation))
        seed=int(digest(dict(contract=C.CONTRACT,origin=adoption,head='intrinsic-64-zero-output'))[:15],16)
        with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
            torch.manual_seed(seed)
            self.head=nn.Sequential(nn.Linear(model.config.hidden+32,64),nn.GELU(),nn.Linear(64,1))
            nn.init.zeros_(self.head[-1].weight);nn.init.zeros_(self.head[-1].bias)
        self.head.to(next(model.parameters()).device)
        self.optimizers={p:torch.optim.Adam(self.head.parameters(),lr=1e-5) for p in ('ppo','ppg')}
        if saved:
            self.head.load_state_dict(saved['head'],strict=True)
            require(set(saved['optimizers'])==set(self.optimizers),'Missing intrinsic Adam')
            for p,o in self.optimizers.items():o.load_state_dict(saved['optimizers'][p])
            self.validate_tensors()
        self.pending=None;self.targets={};self.train_keys=set();self.heldout_keys=set();self.report={};self.steps=Counter()
        self.target_receipts={};self._indices=weakref.WeakKeyDictionary()
        self.search_normalization_proven=False
        self.search_fit=copy.deepcopy(saved.get('search_fit')) if saved else None
        self.fit_diagnostics=copy.deepcopy(saved.get('fit_diagnostics')) if saved else None
        if self.search_fit is not None:
            from .domino_search_fit import validate
            validate(self.search_fit)
        self.change_regime(regime)

    def change_regime(self,regime):
        require(self.pending is None,'Cannot change DOMiNO evidence regime during an update')
        self.state=S.change_regime(self.state,self.config,regime)

    @property
    def active(self):return self.config['mode']=='apply' and S.readiness(self.state,self.config)=='ready'

    def validate_tensors(self):
        tensors=list(self.head.state_dict().values())+[v for o in self.optimizers.values()
            for s in o.state.values() for v in s.values() if isinstance(v,torch.Tensor)]
        require(all(bool(torch.isfinite(t).all()) for t in tensors),'Nonfinite intrinsic head/Adam state')

    def prepare(self,ds,train_ids,heldout_ids):
        start=time.perf_counter()
        observations,report=S.prepare(ds,train_ids,self.state,self.config)
        self.search_normalization_proven=bool(observations) and all(o['phases']==[7,8,12,4] for o in observations)
        self.pending=S.advance(self.state,self.config,observations,self.state['generation']+1)
        self.targets={row_key(ds,i):r['_domino_target'] for i,r in enumerate(ds.rows) if '_domino_target' in r}
        require(len(self.targets)==sum('_domino_target' in r for r in ds.rows),'Duplicate intrinsic decision identity')
        self.target_receipts={row_key(ds,i):r['_domino']['receipt'] for i,r in enumerate(ds.rows) if '_domino_target' in r}
        self._indices=weakref.WeakKeyDictionary()
        self.train_keys={row_key(ds,i) for i in train_ids if row_key(ds,i) in self.targets}
        self.heldout_keys={row_key(ds,i) for i in heldout_ids if row_key(ds,i) in self.targets}
        self.steps=Counter();self.report=dict(report,contract=C.CONTRACT,mode=self.config['mode'],
            incoming_generation=self.state['generation'],outgoing_generation=self.pending['generation'],
            descriptor_manifest_sha256=self.config['descriptor_manifest_sha256'],
            statistics_regime=self.state['regime'],archived_regimes=len(self.state['archive']),
            task_quality=self.state['quality'],quality_next=self.pending['quality'],
            quality_threshold=self.config['quality_ratio']*self.state['quality'][0] if self.state['quality'][0] is not None else None,
            coefficients_next=S.coefficients(self.pending,self.config).tolist(),
            seat_game_support=self.pending['seat_games'],distinct_game_support=self.pending['distinct_games'],
            prepare_seconds=time.perf_counter()-start,auxiliary_fit_rows=len(self.train_keys) if self.active else 0,
            branches='Inherited contexts, existing task-only objective; excluded from statistics and intrinsic fitting')

    def prediction(self,model,inputs,ds,ix,*,encoded=None,embeddings=None):
        from brass_ai.learning.seat_descriptors import critic_inputs
        ids=ix.detach().cpu().tolist()
        if encoded is None:encoded=model.encode_state(inputs[0])
        if embeddings is None:embeddings=model.descriptor_embeddings(encoded,critic_inputs(ds,ids))
        # Acting observer is always relative seat zero, independently of stable ID.
        return self.head(torch.cat((encoded.context,embeddings[:,0]),-1)).squeeze(-1).float()

    def metadata(self,ds,ids):
        """Iteration-local, read-only row lookup; no mmap state/action hydration.

        Cache small arrays per live dataset, never graphs, policy inputs or stale
        targets. prepare() clears them before the next generation's statistics.
        """
        ids=np.asarray(ids,dtype=np.int64)
        require(ids.ndim==1 and ((ids>=0)&(ids<len(ds.data))).all(),'Invalid intrinsic row indices')
        cache=self._indices.get(ds)
        if cache is None:
            n=len(ds.data)
            cache=dict(known=np.zeros(n,bool),present=np.zeros(n,bool),train=np.zeros(n,bool),
                target=np.zeros(n,np.float64),strategy=np.zeros(n,np.int8))
            self._indices[ds]=cache
        missing=np.unique(ids[~cache['known'][ids]])
        if len(missing):
            if hasattr(ds,'diversity_metadata'):rows=ds.diversity_metadata(missing)
            else:
                rows=[(row_key(ds,int(i)),ds.rows[int(i)].get('_domino'),
                       ds.roles[int(i)],bool(ds.rows[int(i)].get('_branch'))) for i in missing]
            for i,(key,a,role,branch) in zip(missing,rows):
                if key in self.targets:
                    require(not branch and role=='learner' and a is not None and a['training_role']=='learner' and
                        a['receipt']==self.target_receipts[key],'Intrinsic target context/role differs from current collection')
                    cache['present'][i]=True;cache['train'][i]=key in self.train_keys
                    cache['target'][i]=self.targets[key];cache['strategy'][i]=a['strategy_id']
                cache['known'][i]=True
            self.report['metadata_rows_read']=self.report.get('metadata_rows_read',0)+len(missing)
        return tuple(cache[k][ids] for k in ('present','train','target','strategy'))

    def eligible(self,ds,ids,training=True):
        return self.metadata(ds,ids)[1 if training else 0].tolist()

    def combine(self,task,ds,actor_ids,frozen):
        result=task.clone();ids=[];raw=[];contribution=[]
        c=S.coefficients(self.state,self.config)
        if self.active:
            require(frozen is not None,'Active DOMiNO requires pre-PPO intrinsic predictions')
            present,_,targets,z=self.metadata(ds,actor_ids)
            ids=np.asarray(actor_ids,dtype=np.int64)[present];z=z[present]
            # Scalar CPU bookkeeping in float64, followed by one device gather
            # and scatter. Avoid one GPU kernel/synchronization per PPO row.
            raw=targets[present]-frozen.detach().cpu().numpy()[ids].astype(np.float64)
            contribution=(1-c[z])*raw
            ix=torch.as_tensor(ids,device=task.device)
            result[ix]=task[ix]*task.new_tensor(c[z])+task.new_tensor(contribution)
        self.report['advantages']=dict(rows=len(ids),task_rms=float(task[actor_ids].square().mean().sqrt()) if actor_ids else 0.,
            intrinsic_rms=float(np.sqrt(np.mean(np.square(raw)))) if len(raw) else 0.,
            intrinsic_contribution_rms=float(np.sqrt(np.mean(np.square(contribution)))) if len(contribution) else 0.,
            normalization='One shared normalization after combining unnormalized streams; branches task-only')
        return result

    def begin(self,phase):
        self.optimizers[phase].zero_grad(set_to_none=True)

    def step(self,phase,rate):
        o=self.optimizers[phase]
        if self.active and any(p.grad is not None for p in self.head.parameters()):
            require(np.isfinite(rate) and rate>0,'Invalid intrinsic companion rate')
            for g in o.param_groups:g['lr']=rate
            torch.nn.utils.clip_grad_norm_(self.head.parameters(),1.,error_if_nonfinite=True);o.step();self.steps[phase]+=1
        o.zero_grad(set_to_none=True)

    def snapshot(self):
        return copy.deepcopy(dict(head=self.head.state_dict(),optimizers={p:o.state_dict() for p,o in self.optimizers.items()},
            steps=self.steps,search_fit=self.search_fit,fit_diagnostics=self.fit_diagnostics))

    def restore(self,s):
        self.head.load_state_dict(s['head'])
        for p,o in self.optimizers.items():o.load_state_dict(s['optimizers'][p])
        self.steps=copy.deepcopy(s['steps'])
        self.search_fit=copy.deepcopy(s.get('search_fit'));self.fit_diagnostics=copy.deepcopy(s.get('fit_diagnostics'))

    def payload(self):
        require(self.pending is not None,'DOMiNO checkpoint requires pending committed evidence')
        self.validate_tensors()
        s=dict(config=self.config,state=self.pending,head=self.head.state_dict(),optimizers={p:o.state_dict() for p,o in self.optimizers.items()})
        if self.search_fit is not None:s.update(search_fit=self.search_fit,fit_diagnostics=self.fit_diagnostics)
        return dict(s,identity=bundle_id(s))

    def committed(self,generation):
        require(self.pending and self.pending['generation']==generation,'DOMiNO commit generation differs')
        self.state=self.pending;self.pending=None

    def diagnostics(self,model,ds,ids,update):
        from .value_phase import value_parts
        chosen=[i for i in ids if row_key(ds,i) in self.targets];error=0.;by_strategy={}
        prediction_squared=target_squared=error_sum=0.
        model.eval()
        with torch.no_grad():
            for part,ix,inputs in ds.value_batches(value_parts(ds,chosen,update,model),model):
                out=self.prediction(model,inputs,ds,ix)
                target=torch.tensor([self.targets[row_key(ds,i)] for i in part],device=out.device)
                prediction_squared+=float(out.double().square().sum())
                target_squared+=float(target.double().square().sum())
                error_sum+=float((out.double()-target.double()).sum())
                errors=(out-target).square().cpu().tolist();error+=sum(errors)
                for i,err in zip(part,errors):
                    z=str(ds.rows[i]['_domino']['strategy_id']);s=by_strategy.setdefault(z,dict(rows=0,sum=0.))
                    s['rows']+=1;s['sum']+=err
        self.report['intrinsic_heldout']=dict(rows=len(chosen),mse=error/len(chosen) if chosen else None,
            zero_predictor_mse=target_squared/len(chosen) if chosen else None,bias=error_sum/len(chosen) if chosen else None,
            prediction_rms=(prediction_squared/len(chosen))**.5 if chosen else None,
            target_rms=(target_squared/len(chosen))**.5 if chosen else None,
            by_strategy={z:dict(rows=s['rows'],mse=s['sum']/s['rows']) for z,s in by_strategy.items()})
        self.report['retained_optimizer_steps']=dict(self.steps)
        if self.search_fit is not None and self.search_fit['collection_generation']==self.state['generation']:
            self.fit_diagnostics=copy.deepcopy(self.report['intrinsic_heldout'])


class PPGCompanions:
    """One rollback unit for existing local heads plus the new intrinsic Adam."""
    def __init__(self,local,diversity,optimizer,value_weight):
        self.local,self.diversity,self.optimizer,self.value_weight=local,diversity,optimizer,value_weight
        self.active=True;self.valid=0;self.batch_size=0
        self.attempted_rows=set();self.retained_rows=set()
        self.attempted_steps=self.retained_steps=self.attempted_presentations=self.retained_presentations=0
        self.fit_batch=[];self._local_loss=None
    def inventory(self,*args):
        if self.local is not None:self.local.inventory(*args)
    def begin(self,phase):
        if self.local is not None:self.local.begin(phase)
        self.diversity.begin(phase)
    def batch(self,ds,ids):
        valid=self.diversity.eligible(ds,ids)
        self.fit_batch=[i for i,keep in zip(ids,valid) if keep] if self.diversity.active else []
        self.valid=sum(valid);self.batch_size=len(ids)
    def forward(self,model,inputs,ds,ix,**kwargs):
        from .updates import value_output
        captured=[]
        capture=captured.append if self.diversity.active and self.valid else None
        out,loss=(self.local.forward(model,inputs,ds,ix,capture_encoded=capture,**kwargs) if self.local is not None else
                  (value_output(model,inputs,ds,ix,capture_encoded=capture,**kwargs),None))
        # Report each companion's own loss; the caller backpropagates their sum.
        self._local_loss=loss.detach() if loss is not None else None
        if self.diversity.active and self.valid:
            ids=ix.detach().cpu().tolist();_,valid,targets,_=self.diversity.metadata(ds,ids)
            if any(valid):
                pred=self.diversity.prediction(model,inputs,ds,ix,encoded=captured[0] if captured else None)
                truth=pred.new_tensor(targets)
                auxiliary=(pred-truth).square()*pred.new_tensor(valid)*(self.batch_size/self.valid)*self.value_weight*self.diversity.config['intrinsic_value_weight']
                loss=auxiliary if loss is None else loss+auxiliary
        return out,loss
    def account_loss(self,loss,weights,denominator,phase):
        if self.local is not None and self._local_loss is not None:
            self.local.account_loss(self._local_loss,weights,denominator,phase)
    def step(self,phase):
        if self.local is not None:self.local.step(phase)
        before=self.diversity.steps[phase]
        self.diversity.step(phase,self.optimizer.param_groups[0]['lr'])
        if self.diversity.steps[phase]>before and self.fit_batch:
            self.attempted_steps+=1;self.retained_steps+=1
            self.attempted_presentations+=len(self.fit_batch);self.retained_presentations+=len(self.fit_batch)
            self.attempted_rows.update(self.fit_batch);self.retained_rows.update(self.fit_batch)
    def snapshot(self):
        return dict(local=self.local.snapshot() if self.local is not None else None,diversity=self.diversity.snapshot(),
            retained_rows=self.retained_rows.copy(),retained_steps=self.retained_steps,retained_presentations=self.retained_presentations)
    def restore(self,s):
        if self.local is not None:self.local.restore(s['local'])
        self.diversity.restore(s['diversity'])
        self.retained_rows=s['retained_rows'].copy();self.retained_steps=s['retained_steps'];self.retained_presentations=s['retained_presentations']
    def report(self):
        return dict(eligible_rows=len(self.diversity.train_keys) if self.diversity.active else 0,
            attempted_steps=self.attempted_steps,retained_steps=self.retained_steps,
            attempted_presentations=self.attempted_presentations,retained_presentations=self.retained_presentations,
            attempted_distinct_rows=len(self.attempted_rows),retained_distinct_rows=len(self.retained_rows),
            status=self.diversity.config['mode'] if self.diversity.config['mode']!='apply' else
                'task_only_bootstrap' if not self.diversity.active else 'retained_fit' if self.retained_steps else 'no_retained_fit',
            scope='Current ordinary learner critic-train targets only. Excludes prior ages, branches, snapshots and heldouts; PPO preservation steps do not fit targets. Retained counts follow joint rollback.')
