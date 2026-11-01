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
ACCEPTED_CONTRACT='local-scoring-training/2.0.0'
DIAGNOSTICS_EVERY_ITERATIONS=5
DIAGNOSTICS_MAX_ROWS=512
DEFAULT=dict(contract=CONTRACT,head_contract=HEAD,target=targets.identity(),mode='apply',
    head_width=64,vp_scale=20.,loss_weights=dict(slots_owner=.05,links_owner=.05,slots_vp=.01,links_vp=.01),
    optimizer=dict(kind='Adam',learning_rate=1e-4,weight_decay=0.,grad_clip=1.),
    head_only_iterations=2,ramp_iterations=5,adoption=None)


def accepted_warmup(config):return config.get('contract')==ACCEPTED_CONTRACT


def fresh_config(config):
    """Opt-in fresh heads; no dependency on panel, branching or GAE startup."""
    validate_config(config)
    result=copy.deepcopy(config)
    result.update(contract=ACCEPTED_CONTRACT,head_only_iterations=2,minimum_fitting_steps=32,
        ramp_iterations=5,adoption=None)
    result['optimizer']['learning_rate']=1e-4
    return validate_config(result)


def diagnostics_schedule(generation):
    require(type(generation) is int and generation>0,'Invalid auxiliary diagnostics generation')
    return dict(contract='local-scoring-diagnostics/2.0.0',generation=generation,
        measured=generation%DIAGNOSTICS_EVERY_ITERATIONS==0,every_iterations=DIAGNOSTICS_EVERY_ITERATIONS,
        maximum_rows=DIAGNOSTICS_MAX_ROWS,
        next_generation=(generation//DIAGNOSTICS_EVERY_ITERATIONS+1)*DIAGNOSTICS_EVERY_ITERATIONS)


def validate_config(config):
    work=accepted_warmup(config)
    keys(config,tuple(DEFAULT)+(('minimum_fitting_steps',) if work else ()))
    require(config['mode'] in ('off','shadow','apply'),'Unknown local scoring mode')
    for name in ('contract','head_contract','target','head_width','vp_scale','head_only_iterations','ramp_iterations'):
        if work and name in ('contract','head_only_iterations','ramp_iterations'):continue
        require(config[name]==DEFAULT[name], 'Local scoring contract differs: '+name)
    if work:
        require(all(type(config[k]) is int and config[k]>0 for k in
            ('head_only_iterations','minimum_fitting_steps','ramp_iterations'))
            and config['adoption'] is None,'Accepted-work auxiliary warmup requires a fresh initialization')
    require(set(config['loss_weights'])==set(DEFAULT['loss_weights']) and all(type(v) in (int,float) and math.isfinite(v) and v>=0 for v in config['loss_weights'].values()),'Invalid local scoring loss weights')
    o=config['optimizer'];keys(o,tuple(DEFAULT['optimizer']))
    require(o['kind']=='Adam' and o['weight_decay']==0 and o['grad_clip']==1. and
        type(o['learning_rate']) in (int,float) and math.isfinite(o['learning_rate']) and 0<o['learning_rate']<=.001,'Invalid companion optimizer')
    if work:require(o['learning_rate']==1e-4,'Fresh companion Adam base LR must be 1e-4')
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


def initialization_receipt(companion,payload):
    require(companion.work_warmup and companion.generation==0,'Fresh auxiliary receipt requires generation zero')
    return dict(contract='fresh-local-scoring-heads/1.0.0',kind='fresh',generation=0,
        head_contract=companion.config['head_contract'],seed=companion.seed,model_id=companion.origin['model_id'],
        config_sha256=digest(companion.config),initial_bundle_identity=payload['identity'],
        head_only_iterations=companion.config['head_only_iterations'],minimum_fitting_steps=companion.config['minimum_fitting_steps'],
        ramp_iterations=companion.config['ramp_iterations'],initialization='normal_hidden_zero_final_outputs',
        optimizer='fresh',rng='forked_preserved',progress='retained_PPG_fitting_at_commit')


def verify_initialization(state,raw,config):
    receipt=state.get('auxiliary_initialization');saved=raw.get('training',{}).get(KEY)
    require(isinstance(receipt,dict) and isinstance(saved,dict)
        and raw['training'].get('auxiliary_initialization')==receipt,
        'Missing fresh auxiliary initialization/checkpoint receipt')
    keys(receipt,('contract','kind','generation','head_contract','seed','model_id','config_sha256',
        'initial_bundle_identity','head_only_iterations','minimum_fitting_steps','ramp_iterations',
        'initialization','optimizer','rng','progress'))
    require(receipt['contract']=='fresh-local-scoring-heads/1.0.0' and receipt['kind']=='fresh'
        and receipt['generation']==0 and receipt['config_sha256']==digest(config)
        and receipt['head_contract']==config['head_contract']
        and all(receipt[k]==config[k] for k in ('head_only_iterations','minimum_fitting_steps','ramp_iterations'))
        and receipt['initialization']=='normal_hidden_zero_final_outputs' and receipt['optimizer']=='fresh'
        and receipt['rng']=='forked_preserved' and receipt['progress']=='retained_PPG_fitting_at_commit'
        and saved['origin']==dict(generation=0,model_id=receipt['model_id'])
        and saved['seed']==receipt['seed'] and saved['generation']==state['iteration']
        and saved['config']==config,'Fresh auxiliary checkpoint provenance differs')
    if state['iteration']==0:
        require(saved['identity']==receipt['initial_bundle_identity'] and raw['model_id']==receipt['model_id'],
            'Initial auxiliary weights/identity differ')


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
        self.work_warmup=accepted_warmup(config)
        from . import local_scoring_warmup as warmup
        self.warmup=warmup.initial() if self.work_warmup else None
        self.retained=set();self.reset_report()
        if saved is None:
            require(generation==0 or adopting,'Missing adopted local scoring companion checkpoint')
        else:
            keys(saved,('contract','config','config_sha256','head_sha256','origin','seed','generation','age','heads','optimizers','retained_steps','retained_samples','identity')+
                (('warmup',) if self.work_warmup else ()))
            require(saved['identity']==bundle_id({k:v for k,v in saved.items() if k!='identity'}),'Corrupt companion bundle identity')
            require(saved['config_sha256']==digest(saved['config']) and saved['head_sha256']==head_identity(),'Companion architecture/config hash differs')
            expected=copy.deepcopy(saved['config']);expected['mode']=config['mode']
            require(expected==config and saved['contract']==config['contract'] and saved['origin']==origin and saved['seed']==self.seed and
                    saved['generation']==generation and saved['age']==self.age,'Companion checkpoint/config/progress differs')
            self.heads.load_state_dict(saved['heads'],strict=True)
            require(set(saved['optimizers'])=={'ppo','ppg'},'Missing companion Adam')
            for p,o in self.optimizers.items():o.load_state_dict(saved['optimizers'][p])
            self.retained_steps=copy.deepcopy(saved['retained_steps'])
            self.retained_samples=copy.deepcopy(saved['retained_samples'])
            if self.work_warmup:
                self.warmup=copy.deepcopy(warmup.validate(saved['warmup'],config,generation))
                require(self.warmup['fitting_steps']==self.retained_steps['ppg'],
                    'Auxiliary fitting counters differ from retained PPG steps')
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
        if self.work_warmup:
            from .local_scoring_warmup import scale
            return scale(self.warmup,self.config)
        return max(0.,min(1.,(self.age-self.config['head_only_iterations']+1)/self.config['ramp_iterations']))

    def reset_report(self):
        self.attempted=set();self.retained=set();self.stats=defaultdict(float)
        self.device_stats={}
        self.available=set();self.unavailable=set();self.map_support={};self.map_masked={}
        self.rollbacks=0;self.restorations=0;self.steps=dict(ppo=0,ppg=0);self.gradient_norms=[]
        self.fitting_steps=0;self.fitting_used=set()

    def begin(self,phase,*,reset=False,rate_factor=1.):
        optimizer=self.optimizers[phase]
        self.phase=phase;self.batch_fitting=set()
        if not self.active:
            optimizer.zero_grad(set_to_none=True);self.pending=False;self.batch_used=set();return
        if reset:optimizer.state.clear()
        optimizer.zero_grad(set_to_none=True);self.pending=False;self.batch_used=set()
        require(math.isfinite(rate_factor) and 0<rate_factor<=1+1e-12,'Invalid companion phase LR factor')
        for group in optimizer.param_groups:group['lr']=self.config['optimizer']['learning_rate']*min(1.,rate_factor)

    def metadata(self,ds,ids):
        if hasattr(ds,'scoring_metadata'):return ds.scoring_metadata(ids)
        result=[]
        for i in ids:
            row=ds.rows[i];target=targets.map_for(row);context=row['view']['context']
            mapping=np.asarray(ds.encoded[i].stable_to_relative)
            key=(row['_local_scoring']['envelope']['sha256'],target['era'],int(row['seat']),tuple(mapping)) if target else None
            revision=context['revision'] if target is not None else context.get('revision',i)
            result.append((target,key,(ds.data[i][2][0],context['era'],int(revision),int(row['seat'])),mapping))
        return result

    def target_arrays(self,metadata,kind,count):
        # One conversion per sealed game/era/seat mapping, then vectorized
        # expansion of the unique records in this batch. Cache only small CPU
        # labels, never encoded states, model outputs or autograd graphs.
        unique={};values=[];inverse=[]
        for target,key,_,mapping in metadata:
            token=(key,kind)
            if token not in unique:
                unique[token]=len(values)
                if target is None:
                    value=(np.zeros(count,np.int64),np.zeros((count,4),np.float32),np.zeros(count,np.bool_))
                else:
                    value=self.target_cache.get(token)
                    if value is None:
                        stable=np.array([r['owner']-1 for r in target[kind]])
                        relative=mapping[stable.clip(0)];present=stable>=0
                        cls=np.where(present,relative+1,0);vp=np.zeros((count,4),np.float32)
                        vp[np.arange(count)[present],relative[present]]=np.array([r['vp'] for r in target[kind]])[present]/self.config['vp_scale']
                        value=(cls,vp,np.array([r['valid'] for r in target[kind]],np.bool_))
                        self.target_cache[token]=value;self.stats['target_cache_misses']+=1
                        if len(self.target_cache)>65536:self.target_cache.popitem(last=False)
                    else:self.stats['target_cache_hits']+=1
                    self.target_cache.move_to_end(token)
                values.append(value)
            inverse.append(unique[token])
        return tuple(np.stack([v[j] for v in values])[inverse] for j in range(3))

    def forward(self,model,inputs,ds,ix,*,dropout=None,detach=False,capture_encoded=None):
        from .updates import value_output
        ids=ix.detach().cpu().tolist() if isinstance(ix,torch.Tensor) else list(ix)
        if not self.active:return value_output(model,inputs,ds,ix,dropout=dropout,detach=detach,capture_encoded=capture_encoded),None
        metadata=self.metadata(ds,ids)
        if not any(x[0] is not None for x in metadata):return value_output(model,inputs,ds,ix,dropout=dropout,detach=detach,capture_encoded=capture_encoded),None
        if self.work_warmup:
            require(self.phase=='ppg','Fresh auxiliary target fitting is restricted to PPG')
        from brass_ai.learning.seat_descriptors import critic_inputs
        start=time.perf_counter()
        with torch.set_grad_enabled(torch.is_grad_enabled() and not detach):encoded=model.encode_state(inputs[0])
        if detach:
            from .ppg_phasic import stop_value_gradient
            encoded=stop_value_gradient(encoded)
        embeddings=model.descriptor_embeddings(encoded,critic_inputs(ds,ids,dropout=dropout))
        if capture_encoded is not None:capture_encoded(encoded)
        heads=model.official_heads(encoded,inputs[3],embeddings)
        out=(inputs[1].new_zeros(inputs[1].shape[:2]),*heads)
        if hasattr(model,'predict_utility'):
            out+= (model.predict_utility(encoded.context,inputs[3],heads,None if ds.remaining is None else ds.remaining[ix],embeddings),)
        predictions=self.heads(encoded,embeddings,self.scale)
        loss=torch.zeros(len(ids),device=inputs[0].device,dtype=torch.float32)
        for kind,(owner,vp) in predictions.items():
            classes,values,mask=self.target_arrays(metadata,kind,owner.shape[1])
            m=torch.as_tensor(mask,device=owner.device);den=m.sum(-1).clamp_min(1)
            cls=torch.as_tensor(classes,device=owner.device);truth=torch.as_tensor(values,device=vp.device)
            ce=F.cross_entropy(owner.float().transpose(1,2),cls,reduction='none')
            huber=F.smooth_l1_loss(vp.float(),truth,reduction='none').mean(-1)
            loss+=self.config['loss_weights'][kind+'_owner']*(ce*m).sum(-1)/den
            loss+=self.config['loss_weights'][kind+'_vp']*(huber*m).sum(-1)/den
            self.accumulate(kind+'_presentations',m.sum())
            self.accumulate(kind+'_ce_sum',(ce.detach()*m).sum())
            self.accumulate(kind+'_none',((cls==0)&m).sum())
            self.accumulate(kind+'_vp_abs_sum',((vp.detach().float()-truth).abs().sum(-1)*m).sum()*self.config['vp_scale'])
        critic_weights=ds.critic_weights[ix].detach().cpu().tolist() if self.work_warmup else [1.]*len(ids)
        for (target,_,token,_),weight in zip(metadata,critic_weights):
            if target is not None:
                game=token[0]
                valid=any(any(r['valid'] for r in target[k]) and
                    (self.config['loss_weights'][k+'_owner']>0 or self.config['loss_weights'][k+'_vp']>0)
                    for k in ('slots','links')) and weight>0
                if valid:
                    self.batch_fitting.add(token)
                if valid or not self.work_warmup:
                    self.batch_used.add(token);self.attempted.add(token)
                if (game,target['era']) not in self.map_support:
                    self.map_support[(game,target['era'])]=sum(int(r['valid']) for k in ('slots','links') for r in target[k])
                    self.map_masked[(game,target['era'])]=sum(int(not r['valid']) for k in ('slots','links') for r in target[k])
        self.pending=bool(self.batch_used);self.stats['forward_seconds']+=time.perf_counter()-start
        return out,loss

    def step(self,phase):
        if self.active and self.pending:
            norm=torch.nn.utils.clip_grad_norm_(self.heads.parameters(),1.,error_if_nonfinite=True)
            start=time.perf_counter();self.optimizers[phase].step();self.stats['optimizer_seconds']+=time.perf_counter()-start
            detached=norm.detach().double()
            self.device_stats['gradient_norm_max']=torch.maximum(self.device_stats.get('gradient_norm_max',torch.zeros_like(detached)),detached)
            self.steps[phase]+=1;self.retained_steps[phase]+=1
            self.retained.update(self.batch_used)
            if phase=='ppg' and self.batch_fitting:
                self.fitting_steps+=1;self.fitting_used.update(self.batch_fitting)
        self.optimizers[phase].zero_grad(set_to_none=True);self.pending=False

    def preservation_used(self,dataset,indices):
        """Track output-distillation work without reading/fitting scoring targets."""
        require(self.active,'Inactive companion cannot preserve outputs')
        for i in indices:
            row=dataset.rows[i]
            token=(dataset.data[i][2][0],row['view']['context']['era'],int(row['view']['context']['revision']),int(row['seat']))
            self.batch_used.add(token);self.attempted.add(token)
        self.pending=True

    def snapshot(self):
        return copy.deepcopy(dict(heads=self.heads.state_dict(),adam=self.optimizers['ppg'].state_dict(),
            retained=self.retained,retained_steps=self.retained_steps,steps=self.steps,
            fitting_steps=self.fitting_steps,fitting_used=self.fitting_used))

    def restore(self,saved):
        # A final selected-state reload is not necessarily a discarded update.
        changed=self.retained_steps!=saved['retained_steps'] or self.retained!=saved['retained']
        self.heads.load_state_dict(saved['heads']);self.optimizers['ppg'].load_state_dict(saved['adam'])
        self.retained=copy.deepcopy(saved['retained']);self.retained_steps=copy.deepcopy(saved['retained_steps']);self.steps=copy.deepcopy(saved['steps'])
        self.fitting_steps=saved['fitting_steps'];self.fitting_used=copy.deepcopy(saved['fitting_used'])
        self.optimizers['ppg'].zero_grad(set_to_none=True);self.pending=False;self.restorations+=1;self.rollbacks+=int(changed)

    def account_loss(self,loss,weights,denominator,phase):
        self.accumulate(phase+'_weighted_auxiliary_loss_sum',(loss.detach()*weights).sum()/denominator)

    def accumulate(self,name,value):
        # Detached scalar totals only: no graphs and no per-microbatch device
        # synchronization. Safety checks/gradient clipping still run normally.
        value=value.detach().double()
        if name not in self.device_stats:self.device_stats[name]=torch.zeros_like(value)
        self.device_stats[name].add_(value)

    def payload(self,generation):
        require(generation in (self.generation,self.generation+1),'Noncommitted companion progress jump')
        result=dict(contract=self.config['contract'],config=self.config,config_sha256=digest(self.config),head_sha256=head_identity(),origin=self.origin,seed=self.seed,generation=generation,
            age=generation-self.origin['generation'],heads=self.heads.state_dict(),
            optimizers={p:o.state_dict() for p,o in self.optimizers.items()},retained_steps=self.retained_steps,
            retained_samples={k:v+(self.usage()[k] if generation==self.generation+1 else 0) for k,v in self.retained_samples.items()})
        if self.work_warmup:
            result['warmup']=self.next_warmup() if generation==self.generation+1 else copy.deepcopy(self.warmup)
            require(result['warmup']['fitting_steps']==self.retained_steps['ppg'],
                'Auxiliary warmup must account for exactly the retained fitting steps')
        return dict(result,identity=bundle_id(result))

    def committed(self,generation):
        require(generation==self.generation+1,'Auxiliary commit discontinuity')
        if self.work_warmup:self.warmup=self.next_warmup()
        self.retained_samples={k:v+self.usage()[k] for k,v in self.retained_samples.items()}
        self.generation=generation;self.age=generation-self.origin['generation'];self.reset_report()

    def usage(self):
        return dict(rows=len(self.retained),games=len({r[0] for r in self.retained}),era_maps=len({r[:2] for r in self.retained}))

    def next_warmup(self):
        from .local_scoring_warmup import advance
        return advance(self.warmup,self.config,self.generation+1,self.fitting_steps,
            len({r[0] for r in self.fitting_used}))

    def report(self):
        names=list(self.device_stats)
        totals=dict(zip(names,torch.stack([self.device_stats[k] for k in names]).cpu().tolist())) if names else {}
        gradient_norm=totals.pop('gradient_norm_max',0.)
        def support(rows):
            maps={r[:2] for r in rows}
            return dict(rows=len(rows),games=len({r[0] for r in rows}),era_maps=len(maps),
                distinct_entity_targets=sum(self.map_support.get(k,0) for k in maps),
                masked_entity_targets=sum(self.map_masked.get(k,0) for k in maps))
        progress={}
        if self.work_warmup:
            from .local_scoring_warmup import report
            progress=dict(warmup=dict(before=report(self.warmup,self.config),after_commit=report(self.next_warmup(),self.config),
                retained_fitting_steps=self.fitting_steps,distinct_fitting_games=len({r[0] for r in self.fitting_used}),
                distinct_fitting_rows=len(self.fitting_used),
                scope='This iteration: retained PPG target fitting only. PPO preservation, rejected intervals and invalid/zero-weight targets excluded. Ramp changes after checkpoint commit.'))
        return dict(contract=self.config['contract'],mode=self.config['mode'],origin=self.origin,activation_age=self.age,
            **progress,
            shared_gradient_scale=self.scale,parameter_count=sum(p.numel() for p in self.heads.parameters()),
            available=support(self.available),unavailable=support(self.unavailable),attempted=support(self.attempted),retained=support(self.retained),companion_steps=self.steps,
            companion_lr={p:o.param_groups[0]['lr'] for p,o in self.optimizers.items()},
            gradient_norm_max=gradient_norm,rollbacks=self.rollbacks,state_restorations=self.restorations,
            training=dict(self.stats,**totals),training_scope='Detached device totals transferred together at reporting; attempted presentations/loss totals include rejected intervals. Retained counts follow joint checkpoint selection.',
            target=targets.identity(),audit_discrepancies=0,scope='Critic-train rows only; official held-out loss and exact policy KL remain the acceptance criteria. Invalid scoring audits fail admission.')

    def inventory(self,dataset,ids):
        ids=list(ids)
        for start in range(0,len(ids),4096):
            for target,_,token,_ in self.metadata(dataset,ids[start:start+4096]):
                game,era=token[:2]
                if target is None:self.unavailable.add(token);continue
                self.available.add(token)
                if (game,era) not in self.map_support:
                    self.map_support[(game,era)]=sum(int(r['valid']) for k in ('slots','links') for r in target[k])
                    self.map_masked[(game,era)]=sum(int(not r['valid']) for k in ('slots','links') for r in target[k])

    def diagnostics(self,model,dataset,heldout,*,maximum=DIAGNOSTICS_MAX_ROWS):
        """Bounded ordinary held-out families; no fitting or gate decisions."""
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
        groups={};group_games=defaultdict(set);was_training=model.training;heads_training=self.heads.training
        model.eval();self.heads.eval();label_cache={}
        try:
          with torch.no_grad():
            for start in range(0,len(ids),64):
                chosen=ids[start:start+64];ix,inputs,_=dataset.value_batch(chosen,model)
                encoded=model.encode_state(inputs[0]);embeddings=model.descriptor_embeddings(encoded,critic_inputs(dataset,chosen))
                predictions=self.heads(encoded,embeddings,0.)
                # One transfer per output/batch, instead of per row and role.
                # All entity reductions below are vectorized on CPU.
                predictions={kind:(logits.float().softmax(-1).cpu().numpy(),
                    values.float().cpu().numpy()*self.config['vp_scale'])
                    for kind,(logits,values) in predictions.items()}
                for b,i in enumerate(chosen):
                    row=dataset.rows[i];target=targets.map_for(row);mapping=dataset.encoded[i].stable_to_relative
                    for kind,(probabilities_batch,values_batch) in predictions.items():
                        key=dataset.tables[i]+'/'+target['era']+'/'+kind
                        group_games[key].add(dataset.data[i][2][0])
                        if key not in groups:groups[key]=dict(rows=0,entities=0,positive=0,none=0,ce_sum=0.,
                            vp_abs_sum=0.,positive_errors=0,occupied_ce_sum=0.,occupied_vp_abs_sum=0.)
                        g=groups[key];g['rows']+=1
                        probabilities=probabilities_batch[b];pred=values_batch[b]
                        token=(row['_local_scoring']['envelope']['sha256'],target['era'],kind,tuple(mapping))
                        if token not in label_cache:
                            labels=target[kind];valid=np.array([r['valid'] for r in labels],dtype=bool)
                            owners=np.array([int(mapping[r['owner']-1])+1 if r['owner'] else 0 for r in labels])
                            truth=np.zeros_like(pred);positive=np.flatnonzero(owners)
                            truth[positive,owners[positive]-1]=[labels[j]['vp'] for j in positive]
                            label_cache[token]=(valid,owners,truth)
                        valid,owners,truth=label_cache[token];owner=owners[valid]
                        probs=probabilities[valid];predicted=probs.argmax(-1)
                        g['entities']+=len(owner);g['positive']+=int((owner>0).sum());g['none']+=int((owner==0).sum())
                        g['positive_errors']+=int(((owner>0)&(predicted!=owner)).sum())
                        ce=-np.log(np.maximum(probs[np.arange(len(owner)),owner].astype(np.float64),1e-30))
                        error=np.abs((pred-truth)[valid])
                        g['ce_sum']+=float(ce.sum());g['vp_abs_sum']+=float(error.sum())
                        g['occupied_ce_sum']+=float(ce[owner>0].sum())
                        g['occupied_vp_abs_sum']+=float(error[owner>0].sum())
        finally:
            model.train(was_training);self.heads.train(heads_training)
        for key,g in groups.items():
            g['games']=len(group_games[key])
            count=max(1,g['entities']);g.update(owner_ce=g['ce_sum']/count,vp_mae=g['vp_abs_sum']/(4*count))
            occupied=g['positive']
            g['occupied']=dict(entities=occupied,owner_ce=g['occupied_ce_sum']/occupied if occupied else None,
                owner_error_rate=g['positive_errors']/occupied if occupied else None,
                vp_mae=g['occupied_vp_abs_sum']/(4*occupied) if occupied else None)
        return dict(rows=len(ids),games=len({dataset.data[i][2][0] for i in ids}),groups=groups,seconds=time.perf_counter()-began,
            scope=f'At most {maximum} ordinary learner critic-held-out rows balanced by era/table; branches and snapshot seats excluded. Slots/links reported separately; occupied means ground-truth era-end owner is nonzero. VP MAE is in VP units across four seats. Report-only; no gate or fitting changes.')
