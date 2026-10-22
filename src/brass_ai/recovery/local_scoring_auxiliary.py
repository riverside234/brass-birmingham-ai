"""Companion ownership/scoring optimization, sharing the existing critic encoding."""
import copy
import hashlib
import math
import time
from pathlib import Path
from collections import defaultdict,OrderedDict

import numpy as np
import torch
from torch.nn import functional as F
from brass_ai.learning.contracts import digest,require,keys
from brass_ai.learning.local_scoring_heads import LocalScoringHeads,CONTRACT as HEAD
from . import local_scoring_targets as targets

KEY='local_scoring_auxiliary'
CONTRACT='local-scoring-training/1.0.0'
DEFAULT=dict(contract=CONTRACT,head_contract=HEAD,target=targets.identity(),mode='apply',
    head_width=64,vp_scale=20.,loss_weights=dict(slots_owner=.05,links_owner=.05,slots_vp=.01,links_vp=.01),
    optimizer=dict(kind='Adam',learning_rate=1e-4,weight_decay=0.,grad_clip=1.),
    head_only_iterations=2,ramp_iterations=5,adoption=None)


def validate_config(config):
    keys(config,tuple(DEFAULT))
    require(config['mode'] in ('off','shadow','apply'),'Unknown local scoring mode')
    for name in ('contract','head_contract','target','head_width','vp_scale','head_only_iterations','ramp_iterations'):
        require(config[name]==DEFAULT[name], 'Local scoring contract differs: '+name)
    require(set(config['loss_weights'])==set(DEFAULT['loss_weights']) and all(type(v) in (int,float) and math.isfinite(v) and v>=0 for v in config['loss_weights'].values()),'Invalid local scoring loss weights')
    o=config['optimizer'];keys(o,tuple(DEFAULT['optimizer']))
    require(o['kind']=='Adam' and o['weight_decay']==0 and o['grad_clip']==1. and
        type(o['learning_rate']) in (int,float) and math.isfinite(o['learning_rate']) and 0<o['learning_rate']<=.001,'Invalid companion optimizer')
    a=config['adoption']
    if a is not None:
        keys(a,('generation','checkpoint_sha256','model_id','fresh_ppg_buffer'))
        require(type(a['generation']) is int and a['generation']>=0 and a['fresh_ppg_buffer'] is True and
            all(isinstance(a[k],str) and len(a[k])==64 and all(x in '0123456789abcdef' for x in a[k]) for k in ('checkpoint_sha256','model_id')),'Invalid local scoring adoption binding')
    return config


def bundle_id(value):
    h=hashlib.sha256()
    def visit(x):
        if isinstance(x,torch.Tensor):
            t=x.detach().cpu().contiguous();h.update(str((str(t.dtype),tuple(t.shape))).encode());h.update(t.numpy().tobytes())
        elif isinstance(x,dict):
            for k in sorted(x,key=str):h.update(str(k).encode());visit(x[k])
        elif isinstance(x,(list,tuple)):
            for v in x:visit(v)
        else:h.update(repr(x).encode())
    visit(value);return h.hexdigest()


def head_identity():
    from brass_ai.learning import local_scoring_heads
    return hashlib.sha256(Path(local_scoring_heads.__file__).read_bytes().replace(b'\r\n',b'\n')).hexdigest()


class Companion:
    def __init__(self,model,config,*,raw=None,generation=0,adopting=False):
        self.config=copy.deepcopy(validate_config(config));self.generation=generation
        saved=(raw or {}).get('training',{}).get(KEY)
        origin=config['adoption'] or (saved['origin'] if saved is not None else dict(generation=0,model_id=(raw or {}).get('model_id','fresh')))
        self.origin=copy.deepcopy(origin);self.age=generation-origin['generation']
        require(self.age>=0,'Auxiliary activation origin is in the future')
        self.seed=int(digest(dict(origin=origin,head=HEAD))[:15],16)
        # fork_rng restores CPU and all initialized CUDA streams; no Python or
        # NumPy RNG is used by initialization. Placement consumes no randomness.
        with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
            torch.manual_seed(self.seed);self.heads=LocalScoringHeads(model.config.hidden)
        self.heads.to(next(model.parameters()).device)
        self.optimizers={p:torch.optim.Adam(self.heads.parameters(),lr=config['optimizer']['learning_rate'],weight_decay=0.) for p in ('ppo','ppg')}
        self.retained_steps=dict(ppo=0,ppg=0);self.retained_samples=dict(rows=0,games=0,era_maps=0)
        self.retained=set();self.reset_report()
        if saved is None:
            require(generation==0 or adopting,'Missing adopted local scoring companion checkpoint')
        else:
            keys(saved,('contract','config','config_sha256','head_sha256','origin','seed','generation','age','heads','optimizers','retained_steps','retained_samples','identity'))
            require(saved['identity']==bundle_id({k:v for k,v in saved.items() if k!='identity'}),'Corrupt companion bundle identity')
            require(saved['config_sha256']==digest(saved['config']) and saved['head_sha256']==head_identity(),'Companion architecture/config hash differs')
            expected=copy.deepcopy(saved['config']);expected['mode']=config['mode']
            require(expected==config and saved['contract']==CONTRACT and saved['origin']==origin and saved['seed']==self.seed and
                    saved['generation']==generation and saved['age']==self.age,'Companion checkpoint/config/progress differs')
            self.heads.load_state_dict(saved['heads'],strict=True)
            require(set(saved['optimizers'])=={'ppo','ppg'},'Missing companion Adam')
            for p,o in self.optimizers.items():o.load_state_dict(saved['optimizers'][p])
            self.retained_steps=copy.deepcopy(saved['retained_steps'])
            self.retained_samples=copy.deepcopy(saved['retained_samples'])
            for values,names in ((self.retained_steps,('ppo','ppg')),(self.retained_samples,('rows','games','era_maps'))):
                require(set(values)==set(names) and all(type(x) is int and x>=0 for x in values.values()),'Invalid companion retained counters')
            for phase,o in self.optimizers.items():
                require(len(o.param_groups)==1 and len(o.param_groups[0]['params'])==len(list(self.heads.parameters()))
                    and o.param_groups[0]['weight_decay']==0 and math.isfinite(o.param_groups[0]['lr'])
                    and 0<o.param_groups[0]['lr']<=config['optimizer']['learning_rate']*(1+1e-12),'Invalid companion Adam settings')
        self.active=config['mode']!='off';self.pending=False
        self.target_cache=OrderedDict()

    @property
    def scale(self):
        if self.config['mode']!='apply':return 0.
        return max(0.,min(1.,(self.age-self.config['head_only_iterations']+1)/self.config['ramp_iterations']))

    def reset_report(self):
        self.attempted=set();self.retained=set();self.stats=defaultdict(float)
        self.available=set();self.unavailable=set();self.map_support={};self.map_masked={}
        self.rollbacks=0;self.restorations=0;self.steps=dict(ppo=0,ppg=0);self.gradient_norms=[]

    def begin(self,phase,*,reset=False,rate_factor=1.):
        optimizer=self.optimizers[phase]
        if not self.active:
            optimizer.zero_grad(set_to_none=True);self.pending=False;self.batch_used=set();return
        if reset:optimizer.state.clear()
        optimizer.zero_grad(set_to_none=True);self.pending=False;self.batch_used=set()
        require(math.isfinite(rate_factor) and 0<rate_factor<=1+1e-12,'Invalid companion phase LR factor')
        for group in optimizer.param_groups:group['lr']=self.config['optimizer']['learning_rate']*min(1.,rate_factor)

    def forward(self,model,inputs,ds,ix,*,dropout=None):
        from .updates import value_output
        ids=ix.detach().cpu().tolist() if isinstance(ix,torch.Tensor) else list(ix)
        selected=[targets.map_for(ds.rows[i]) for i in ids]
        if not self.active or not any(x is not None for x in selected):return value_output(model,inputs,ds,ix,dropout=dropout),None
        from brass_ai.learning.seat_descriptors import critic_inputs
        start=time.perf_counter()
        encoded=model.encode_state(inputs[0]);embeddings=model.descriptor_embeddings(encoded,critic_inputs(ds,ids,dropout=dropout))
        heads=model.official_heads(encoded,inputs[3],embeddings)
        out=(inputs[1].new_zeros(inputs[1].shape[:2]),*heads)
        if hasattr(model,'predict_utility'):
            out+= (model.predict_utility(encoded.context,inputs[3],heads,None if ds.remaining is None else ds.remaining[ix],embeddings),)
        predictions=self.heads(encoded,embeddings,self.scale)
        loss=torch.zeros(len(ids),device=inputs[0].device,dtype=torch.float32)
        for kind,(owner,vp) in predictions.items():
            count=owner.shape[1];classes=np.zeros((len(ids),count),np.int64)
            values=np.zeros((len(ids),count,4),np.float32);mask=np.zeros((len(ids),count),np.bool_)
            for b,(i,target) in enumerate(zip(ids,selected)):
                if target is None:continue
                token=(ds.rows[i]['_local_scoring']['envelope']['sha256'],target['era'],int(ds.rows[i]['seat']),kind)
                if token not in self.target_cache:
                    mapping=np.asarray(ds.encoded[i].stable_to_relative)
                    stable=np.array([r['owner']-1 for r in target[kind]])
                    relative=mapping[stable.clip(0)];present=stable>=0
                    owner_class=np.where(present,relative+1,0)
                    val=np.zeros((count,4),np.float32)
                    val[np.arange(count)[present],relative[present]]=np.array([r['vp'] for r in target[kind]])[present]/self.config['vp_scale']
                    valid=np.array([r['valid'] for r in target[kind]],np.bool_)
                    self.target_cache[token]=(owner_class,val,valid)
                    if len(self.target_cache)>4096:self.target_cache.popitem(last=False)
                self.target_cache.move_to_end(token)
                classes[b],values[b],mask[b]=self.target_cache[token]
            m=torch.as_tensor(mask,device=owner.device);den=m.sum(-1).clamp_min(1)
            cls=torch.as_tensor(classes,device=owner.device);truth=torch.as_tensor(values,device=vp.device)
            ce=F.cross_entropy(owner.float().transpose(1,2),cls,reduction='none')
            huber=F.smooth_l1_loss(vp.float(),truth,reduction='none').mean(-1)
            loss+=self.config['loss_weights'][kind+'_owner']*(ce*m).sum(-1)/den
            loss+=self.config['loss_weights'][kind+'_vp']*(huber*m).sum(-1)/den
            self.stats[kind+'_presentations']+=int(m.sum())
            self.stats[kind+'_ce_sum']+=float((ce.detach()*m).sum())
            self.stats[kind+'_none']+=int(((cls==0)&m).sum())
            self.stats[kind+'_vp_abs_sum']+=float(((vp.detach().float()-truth).abs().sum(-1)*m).sum())*self.config['vp_scale']
        for i,target in zip(ids,selected):
            if target is not None:
                row=ds.rows[i];game=ds.data[i][2][0]
                token=(game,target['era'],int(row['view']['context']['revision']),int(row['seat']))
                self.batch_used.add(token);self.attempted.add(token)
                if (game,target['era']) not in self.map_support:
                    self.map_support[(game,target['era'])]=sum(int(r['valid']) for k in ('slots','links') for r in target[k])
                    self.map_masked[(game,target['era'])]=sum(int(not r['valid']) for k in ('slots','links') for r in target[k])
        self.pending=True;self.stats['forward_seconds']+=time.perf_counter()-start
        return out,loss

    def step(self,phase):
        if self.active and self.pending:
            norm=torch.nn.utils.clip_grad_norm_(self.heads.parameters(),1.,error_if_nonfinite=True)
            start=time.perf_counter();self.optimizers[phase].step();self.stats['optimizer_seconds']+=time.perf_counter()-start
            self.gradient_norms.append(float(norm));self.steps[phase]+=1;self.retained_steps[phase]+=1
            self.retained.update(self.batch_used)
        self.optimizers[phase].zero_grad(set_to_none=True);self.pending=False

    def snapshot(self):
        return copy.deepcopy(dict(heads=self.heads.state_dict(),adam=self.optimizers['ppg'].state_dict(),
            retained=self.retained,retained_steps=self.retained_steps,steps=self.steps))

    def restore(self,saved):
        # A final selected-state reload is not necessarily a discarded update.
        changed=self.retained_steps!=saved['retained_steps'] or self.retained!=saved['retained']
        self.heads.load_state_dict(saved['heads']);self.optimizers['ppg'].load_state_dict(saved['adam'])
        self.retained=copy.deepcopy(saved['retained']);self.retained_steps=copy.deepcopy(saved['retained_steps']);self.steps=copy.deepcopy(saved['steps'])
        self.optimizers['ppg'].zero_grad(set_to_none=True);self.pending=False;self.restorations+=1;self.rollbacks+=int(changed)

    def account_loss(self,loss,weights,denominator,phase):
        self.stats[phase+'_weighted_auxiliary_loss_sum']+=float((loss.detach()*weights).sum()/denominator)

    def payload(self,generation):
        require(generation in (self.generation,self.generation+1),'Noncommitted companion progress jump')
        result=dict(contract=CONTRACT,config=self.config,config_sha256=digest(self.config),head_sha256=head_identity(),origin=self.origin,seed=self.seed,generation=generation,
            age=generation-self.origin['generation'],heads=self.heads.state_dict(),
            optimizers={p:o.state_dict() for p,o in self.optimizers.items()},retained_steps=self.retained_steps,
            retained_samples={k:v+(self.usage()[k] if generation==self.generation+1 else 0) for k,v in self.retained_samples.items()})
        return dict(result,identity=bundle_id(result))

    def committed(self,generation):
        require(generation==self.generation+1,'Auxiliary commit discontinuity')
        self.retained_samples={k:v+self.usage()[k] for k,v in self.retained_samples.items()}
        self.generation=generation;self.age=generation-self.origin['generation'];self.reset_report()

    def usage(self):
        return dict(rows=len(self.retained),games=len({r[0] for r in self.retained}),era_maps=len({r[:2] for r in self.retained}))

    def report(self):
        def support(rows):
            maps={r[:2] for r in rows}
            return dict(rows=len(rows),games=len({r[0] for r in rows}),era_maps=len(maps),
                distinct_entity_targets=sum(self.map_support.get(k,0) for k in maps),
                masked_entity_targets=sum(self.map_masked.get(k,0) for k in maps))
        return dict(contract=CONTRACT,mode=self.config['mode'],origin=self.origin,activation_age=self.age,
            shared_gradient_scale=self.scale,parameter_count=sum(p.numel() for p in self.heads.parameters()),
            available=support(self.available),unavailable=support(self.unavailable),attempted=support(self.attempted),retained=support(self.retained),companion_steps=self.steps,
            companion_lr={p:o.param_groups[0]['lr'] for p,o in self.optimizers.items()},
            gradient_norm_max=max(self.gradient_norms,default=0.),rollbacks=self.rollbacks,state_restorations=self.restorations,
            training=dict(self.stats),training_scope='Attempted presentations/loss totals, including rejected intervals. Retained counts follow joint checkpoint selection.',
            target=targets.identity(),audit_discrepancies=0,scope='Critic-train rows only; official held-out loss and exact policy KL remain the acceptance criteria. Invalid scoring audits fail admission.')

    def inventory(self,dataset,ids):
        for i in ids:
            row=dataset.rows[i];target=targets.map_for(row)
            game=dataset.data[i][2][0];era=row['view']['context']['era']
            if target is None:
                self.unavailable.add((game,era,row['view']['context'].get('revision',i),int(row['seat'])));continue
            self.available.add((game,era,int(row['view']['context']['revision']),int(row['seat'])))
            if (game,era) not in self.map_support:
                self.map_support[(game,era)]=sum(int(r['valid']) for k in ('slots','links') for r in target[k])
                self.map_masked[(game,era)]=sum(int(not r['valid']) for k in ('slots','links') for r in target[k])

    def diagnostics(self,model,dataset,heldout,*,maximum=2048):
        """Bounded ordinary held-out families; no fitting or gate decisions."""
        from .updates import autocast
        from brass_ai.learning.seat_descriptors import critic_inputs
        began=time.perf_counter();strata=defaultdict(list)
        for i in heldout:
            row=dataset.rows[i]
            if row.get('_branch') or dataset.roles[i]!='learner' or targets.map_for(row) is None:continue
            strata[(dataset.tables[i],row['view']['context']['era'])].append(i)
        ids=[]
        for n in range(maximum):
            present=False
            for part in strata.values():
                if n<len(part) and len(ids)<maximum:ids.append(part[n]);present=True
            if not present or len(ids)>=maximum:break
        groups={};model.eval();self.heads.eval()
        from brass_ai.learning.state_layout import state_layout
        segments={s.name:s for s in state_layout().segments}
        with torch.no_grad():
            for start in range(0,len(ids),64):
                chosen=ids[start:start+64];ix,inputs,_=dataset.value_batch(chosen,model)
                encoded=model.encode_state(inputs[0]);embeddings=model.descriptor_embeddings(encoded,critic_inputs(dataset,chosen))
                predictions=self.heads(encoded,embeddings,0.)
                for b,i in enumerate(chosen):
                    row=dataset.rows[i];target=targets.map_for(row);mapping=dataset.encoded[i].stable_to_relative
                    for kind,(logits,values) in predictions.items():
                        key=dataset.tables[i]+'/'+target['era']+'/'+kind
                        if key not in groups:groups[key]=dict(rows=0,entities=0,positive=0,none=0,ce_sum=0.,brier_sum=0.,
                            vp_abs_sum=0.,vp_squared_sum=0.,vp_total_abs_sum=0.,positive_errors=0,confusion=[[0]*5 for _ in range(5)],
                            current_empty=0,current_occupied=0)
                        g=groups[key];g['rows']+=1
                        probabilities=logits[b].float().softmax(-1).cpu().numpy();pred=values[b].float().cpu().numpy()*self.config['vp_scale']
                        truth=np.zeros_like(pred);valid=np.array([r['valid'] for r in target[kind]])
                        # Existing inputs alone describe current occupancy.
                        for j,label in enumerate(target[kind]):
                            if not valid[j]:continue
                            owner=int(mapping[label['owner']-1])+1 if label['owner'] else 0
                            if owner:truth[j,owner-1]=label['vp']
                            predicted=int(probabilities[j].argmax());g['entities']+=1;g['positive']+=int(owner>0);g['none']+=int(owner==0)
                            g['positive_errors']+=int(owner>0 and predicted!=owner);g['confusion'][owner][predicted]+=1
                            g['ce_sum']-=math.log(max(1e-30,float(probabilities[j,owner])))
                            target_one=np.eye(5)[owner];g['brier_sum']+=float(((probabilities[j]-target_one)**2).sum())
                            # Feature manifests name the owner block; unknown
                            # layouts are reported as unavailable, never guessed.
                            prefix=('slot.' if kind=='slots' else 'link.')+label['id']
                            segment=segments.get(prefix+'.owner',segments.get(prefix))
                            if segment is not None:
                                state=dataset.encoded[i].state
                                occupied=bool(state[segment.start:segment.start+4].sum())
                                g['current_occupied']+=int(occupied);g['current_empty']+=int(not occupied)
                        diff=(pred-truth)[valid];g['vp_abs_sum']+=float(np.abs(diff).sum());g['vp_squared_sum']+=float((diff**2).sum())
                        g['vp_total_abs_sum']+=float(np.abs(diff.sum(0)).sum())
        for g in groups.values():
            count=max(1,g['entities']);g.update(owner_ce=g['ce_sum']/count,owner_brier=g['brier_sum']/count,
                vp_mae=g['vp_abs_sum']/(4*count),vp_rmse=math.sqrt(g['vp_squared_sum']/(4*count)),
                vp_sum_mae_per_seat=g['vp_total_abs_sum']/(4*max(1,g['rows'])))
        return dict(rows=len(ids),games=len({dataset.data[i][2][0] for i in ids}),groups=groups,seconds=time.perf_counter()-began,
            scope='At most 2048 ordinary learner critic-held-out rows balanced by era/table; branches and snapshot seats excluded. These families retain their existing PPO actor use.')
