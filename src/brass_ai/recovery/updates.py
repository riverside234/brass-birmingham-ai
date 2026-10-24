"""Effective minibatch PPO, full-legal-choice BC and vectorized four-seat losses."""
from contextlib import nullcontext
from collections import deque,Counter
from concurrent.futures import ThreadPoolExecutor
import math
import numpy as np
import torch
from .learning_rate import step as scheduled_step,report as learning_rate_report
from torch.nn import functional as F

from brass_ai.learning.batching import collate, model_id
from brass_ai.learning.contracts import require
from brass_ai.learning.replay import terminal_targets, validate_teacher
from brass_ai.step5.policy import exploration_distribution
from brass_ai.step5.updates import bounded_batches
from .encoding import PreparedEncoder
from .telemetry import Timings, profile_window, memory
from .progress import Progress
from .insolvency import returns as insolvency_returns,advantage_costs


def prepared(games, predicate=None, neutral=False, insolvency=None, reward=None, outcome_targets=None):
    """Targets once per game; filter BEFORE encoding; cached arrays survive epochs."""
    require(reward is None or insolvency==reward['insolvency'],'Return/cost reward mismatch')
    encoder, data = PreparedEncoder(), []
    for game in games:
        if hasattr(game,'assert_metadata'): game.assert_metadata()
        target = terminal_targets(game['result']) if outcome_targets is None else outcome_targets[game['game_id']]
        prior=([min(insolvency['max_cost'],n*insolvency['event_cost']) for n in game['branch']['prior_events']]
               if 'branch' in game and insolvency is not None else None)
        costs,summaries=insolvency_returns(game['records'],insolvency,prior) if insolvency is not None else (None,None)
        if reward is not None:
            from brass_ai.learning.reward import replay_returns
            labels,rewards=replay_returns(game['records'],game['result'],reward,prior)
        annotated=set()
        for step,row in enumerate(game['records']):
            if 'branch' in game and step==0:
                if hasattr(game,'encoded'):game.encoded(row,encoder)
                continue
            if predicate is None or predicate(row):
                r = dict(row, style=[0.0]*4, intent=[0.0]*8) if neutral else row
                if hasattr(game,'ppo_opening_objective'):
                    r=dict(r,_ppo_opening_objective=game.ppo_opening_objective)
                if reward is not None:
                    r=dict(r,_reward_return=labels[step]['utility'],_reward_remaining=labels[step]['remaining'],_reward_id=rewards['reward_id'])
                    if row['seat'] not in annotated:
                        seat=row['seat']
                        r['_reward_trajectory']=dict(**{k:v[seat] for k,v in rewards['components'].items()},
                            cost=rewards['cost'][seat],utility=rewards['utility'][seat])
                if costs is not None:
                    r=dict(r,_insolvency_cost_to_go=costs[step])
                    if row['seat'] not in annotated:
                        r['_insolvency_trajectory']=summaries[row['seat']];annotated.add(row['seat'])
                encoded = game.encoded(r,encoder,allow_context=neutral) if hasattr(game,'encoded') else encoder.encode(r['view'], r['choices'], r['style'], r['intent'])
                if hasattr(game,'critic_table'):
                    from .critic_data import annotate
                    r=annotate(r,encoded,game.critic_table,game.table_kind)
                if 'branch' in game:
                    r=dict(r,_branch=True,_parent_game_id=game['branch']['parent_game_id'])
                if 'local_scoring_targets' in game:
                    from .local_scoring_targets import row_target
                    r=dict(r,_local_scoring=row_target(game,r))
                data.append((r, target, (game['game_id'],row['seat']), encoded))
            elif costs is not None and hasattr(game,'encoded'):
                # Opponent observations also determine the learner's causal
                # costs. Filtering their training rows cannot waive the seal.
                game.encoded(row,encoder)
    require(data, 'No eligible learning decisions')
    return data


def value_losses(outputs, targets):
    _, win, rank, vp = outputs[:4]
    win, rank, vp = win.float(), rank.float(), vp.float()
    w, r, v = targets
    ce = -(w * win.clamp_min(1e-9).log()).sum(-1)
    rank_loss = F.smooth_l1_loss((rank-1)/3, (r-1)/3, reduction='none').mean(-1)
    vp_loss = F.smooth_l1_loss(vp/100, v/100, reduction='none').mean(-1)
    return ce + .25*rank_loss + .1*vp_loss, ce, rank_loss, vp_loss


def bc_loss(logits, chosen):
    """Denominator is every legal action, not singleton search coverage."""
    return F.cross_entropy(logits.float(), chosen, reduction='none')


class Dataset:
    def __init__(self, data, device, policy=True, progress=None):
        self.data, self.device = data, device
        self.rows=[d[0] for d in data]
        self.encoded=[d[3] for d in data]
        from .batch_cache import InputAudit, BatchCache
        self.input_audit=InputAudit()
        self.scan_cache=BatchCache()
        # Selectors are admitted when their candidates are actually collated.
        # Eager full-action hashing duplicated policy admission and made a
        # state-only critic Dataset pay for every unused action as well.
        from brass_ai.learning.features import check_encoded
        for row in self.encoded:check_encoded(row)
        log=Progress(progress,5,'tensor groups') if progress else None
        self.chosen=torch.tensor([r['chosen'] for r in self.rows],device=device)
        self.seats=torch.tensor([r['seat'] for r in self.rows],device=device)
        from brass_ai.learning.seat_descriptors import unknown_numpy,validate
        self.descriptors=torch.tensor(np.stack([validate(r.get('_seat_descriptors',unknown_numpy())) for r in self.rows]),device=device)
        self.roles=[r.get('_critic_role','learner') for r in self.rows]
        self.tables=[r.get('_table_kind','selfplay') for r in self.rows]
        self.critic_weights=torch.ones(len(data),device=device)
        self.old=torch.tensor([r['behavior']['probability'] for r in self.rows],device=device)
        if log:log(1)
        self.targets=tuple(torch.tensor([d[1][k] for d in data],device=device,dtype=torch.float32)
            for k in ('win_share','placement','vp'))
        self.utility=self.remaining=None
        if any('_reward_return' in r for r in self.rows):
            require(all('_reward_return' in r and '_reward_remaining' in r for r in self.rows),'Mixed reward targets')
            self.utility=torch.tensor([r['_reward_return'] for r in self.rows],device=device,dtype=torch.float32)
            self.remaining=torch.tensor([r['_reward_remaining'] for r in self.rows],device=device,dtype=torch.float32)
        if log:log(2)
        self.q=[exploration_distribution(r['choices'],r['intent'],r['behavior'].get('exploration_version',2),
                    r.get('view',{}).get('context',{}).get('phase')).astype(np.float32) for r in self.rows] if policy else None
        self.temperatures=torch.tensor([r['behavior'].get('temperature',1.) for r in self.rows],device=device)[:,None]
        if log:log(3)
        self.forced_cpu=[len(r['choices'])==1 for r in self.rows]
        self.forced=torch.tensor(self.forced_cpu,device=device)
        if log:log(4)
        self.bc_weights=torch.tensor([r.get('_bc_weight',1.0) for r in self.rows],device=device,dtype=torch.float32)
        if log:log(5)

    def microbatches(self, indices, size, cells, bucket=True):
        indices=list(indices)
        if bucket: indices.sort(key=lambda i: len(self.encoded[i].choice_ids))
        selected=[self.data[i] for i in indices]
        for start, part in bounded_batches(selected,size,cells):
            yield indices[start:start+len(part)]

    def _batch_rows(self, indices, value_only=False):
        rows=[self.encoded[i] for i in indices]
        if value_only:
            from brass_ai.learning.features import Encoded,check_encoded
            # Check the original shapes before narrowing: slicing must not
            # hide a changed action/choice count after Dataset construction.
            # Unused actions do not enter a stock state-only value head. The
            # consumed candidate still receives full selector admission below.
            for row in rows:check_encoded(row)
            rows=[Encoded(row.state,row.actions[:1],row.stable_to_relative,
                          row.choice_ids[:1],row.feature_version) for row in rows]
        return rows

    def batch(self, indices, exploration=0, temperature=1, *, value_only=False):
        ix=torch.tensor(indices,device=self.device)
        inputs=collate(self._batch_rows(indices,value_only),self.device,
                       audit=None if value_only else self.input_audit)
        width=inputs[1].shape[1]
        q=np.zeros((len(indices),width),dtype=np.float32)
        if exploration:
            for j,i in enumerate(indices): q[j,:len(self.q[i])]=self.q[i]
        return ix, inputs, torch.from_numpy(q).to(self.device)

    def input_seal(self, indices, exploration=0):
        return self.input_audit.seal([self.encoded[i] for i in indices],
                                    [self.q[i] for i in indices] if exploration else ())

    def _pinned_batch(self, indices, exploration=0, *, value_only=False, cache=False):
        # A small background queue prepares/pins CPU tensors while CUDA trains.
        # These private buffers are never mutated after asynchronous transfer.
        return self._host_batch(indices,exploration,value_only=value_only,cache=cache,pinned=True)

    def _host_batch(self, indices, exploration=0, *, value_only=False, cache=False, pinned=False):
        key=(tuple(indices),bool(exploration),value_only,pinned)
        seal=self.input_seal(indices,exploration) if cache else None
        if cache:
            hit=self.scan_cache.get(key,seal)
            if hit is not None:return hit
        ix,inputs,q=self.batch_cpu(indices,exploration,value_only=value_only)
        tensors=(ix,*inputs,q)
        if pinned:tensors=tuple(t.pin_memory() for t in tensors)
        return self.scan_cache.put(key,seal,tensors) if cache else tensors

    def batch_cpu(self, indices, exploration=0, *, value_only=False):
        inputs=collate(self._batch_rows(indices,value_only),audit=None if value_only else self.input_audit)
        q=np.zeros((len(indices),inputs[1].shape[1]),dtype=np.float32)
        if exploration:
            for j,i in enumerate(indices):q[j,:len(self.q[i])]=self.q[i]
        return torch.tensor(indices),inputs,torch.from_numpy(q)

    def supervised_batches(self, parts, executor=None):
        for part,ix,inputs,_ in self.policy_batches(parts,executor=executor):
            yield part,ix,inputs

    def value_batch(self, indices, model):
        return self.batch(indices,value_only=state_value_model(model))

    def value_batches(self, parts, model, executor=None):
        for part,ix,inputs,_ in self.policy_batches(parts,executor=executor,value_only=state_value_model(model)):
            yield part,ix,inputs

    def policy_batches(self, parts, exploration=0, executor=None, *, value_only=False, cache=False):
        if hasattr(executor,'batches'):
            yield from executor.batches(self,parts,exploration,value_only=value_only,cache=cache)
            return
        if executor is None:
            for part in parts:
                if cache:
                    host=self._host_batch(part,exploration,value_only=value_only,cache=True)
                    tensors=tuple(t.to(self.device) for t in host)
                    yield part,tensors[0],tensors[1:-1],tensors[-1]
                else:
                    ix,inputs,q=self.batch(part,exploration,value_only=value_only)
                    yield part,ix,inputs,q
            return
        parts=iter(parts);pending=deque()
        def submit():
            part=next(parts,None)
            if part is not None:pending.append((part,executor.submit(self._pinned_batch,part,exploration,value_only=value_only,cache=cache)))
        for _ in range(2):submit()
        while pending:
            part,future=pending.popleft()
            host=future.result()
            tensors=tuple(t.to(self.device,non_blocking=True) for t in host)
            submit()
            yield part,tensors[0],tensors[1:-1],tensors[-1]


def state_value_model(model):
    """Only stock state-only value heads can omit unused candidate scoring.

    Keep one real legal candidate for the unchanged forward signature. Value
    phases discard its logits; policy, BC and teacher phases retain all choices.
    Subclasses may define action-dependent values and keep the full batch.
    """
    from brass_ai.learning.model import FourSeatNet
    from brass_ai.learning.utility_model import UtilityNet
    return type(model) in (FourSeatNet,UtilityNet)


def value_output(model,inputs,dataset,indices,*,dropout=None,unknown=False,detach=False):
    from brass_ai.learning.model import ARCHITECTURE
    from brass_ai.learning.seat_descriptors import critic_inputs
    extra={}
    if getattr(model.config,'architecture',None)==ARCHITECTURE:
        ids=indices.detach().cpu().tolist() if isinstance(indices,torch.Tensor) else list(indices)
        extra['descriptors']=critic_inputs(dataset,ids,dropout=dropout,unknown=unknown)
    if dataset.remaining is not None:extra['remaining']=dataset.remaining[indices]
    if detach:
        from .ppg_phasic import stop_value_gradient
        require(state_value_model(model),'Detached PPO critic requires the supported state-only model')
        with torch.no_grad():encoded=stop_value_gradient(model.encode_state(inputs[0]))
        heads=model.heads_from_encoded(encoded,inputs[3],extra.get('remaining'),extra.get('descriptors'))
        return (inputs[1].new_zeros(inputs[1].shape[:2]),)+heads
    return model(*inputs,**extra)


def rehearsal_dataset(data,device):
    """Reuse immutable audited BC tensors across PPO rounds on one device."""
    if isinstance(data,Dataset):
        require(data.chosen.device==torch.empty(0,device=device).device,'Rehearsal device changed')
        require(data.q is None,'Rehearsal must use supervised tensors')
        return data
    return Dataset(data,device,policy=False) if data else None


def probabilities(logits, q, exploration, temperature):
    return (1-exploration)*F.softmax(logits.float()/temperature,-1)+exploration*q


def row_probabilities(dataset, indices, logits, q, behavior):
    # Legacy callers retain their single temperature; v3 is an explicit
    # per-row sampling contract, including the opening decision counter.
    version=getattr(behavior,'exploration_version',2)
    temperature=dataset.temperatures[indices] if version==3 else behavior.temperature
    return probabilities(logits,q,behavior.exploration,temperature)


def verify_behavior_metadata(dataset, behavior):
    version=getattr(behavior,'exploration_version',2)
    for row in dataset.rows:
        b=row['behavior']
        require(b.get('exploration_version',2)==version,'Wrong/stale exploration version')
        if version==3:
            from .exploration import check_behavior
            check_behavior(b,row['view']['context']['phase'])
            opening=behavior.opening
            expected=(opening['temperature'] if opening and b['action_decision'] is not None and
                      b['action_decision']<opening['action_decisions_per_seat'] else behavior.temperature)
            require(b['exploration']==behavior.exploration and b['temperature']==expected,
                    'Wrong/stale exploration or opening temperature')


def entropy_term(logits, mixed, dataset, indices, behavior, config):
    if not getattr(behavior,'raw_entropy',False):
        entropy=-(mixed*mixed.clamp_min(1e-30).log()).sum(-1)
        return entropy,config['entropy_weight']*entropy
    from .exploration import group
    raw=F.softmax(logits.float(),-1)
    entropy=-(raw*raw.clamp_min(1e-30).log()).sum(-1)
    ids=indices.detach().cpu().tolist() if isinstance(indices,torch.Tensor) else indices
    weights=torch.tensor([behavior.entropy_alphas[group(dataset.rows[i]['view']['context'])] for i in ids],
                         device=logits.device)
    return entropy,weights*entropy


def autocast(config, device):
    if config['precision']=='bf16':
        require(str(device).startswith('cuda') and torch.cuda.is_bf16_supported(), 'BF16 requires a supported CUDA device')
        return torch.autocast(device_type='cuda',dtype=torch.bfloat16)
    return nullcontext()


def policy_measure(model, dataset, indices, behavior, config, timings, verify=False, advantages=None, executor=None):
    indices=list(indices)
    if not indices: return 0.0
    if verify:verify_behavior_metadata(dataset,behavior)
    total=torch.zeros((),device=dataset.device);valid=torch.ones((),dtype=torch.bool,device=dataset.device)
    with torch.no_grad(), timings.phase('ppo.policy_measure'):
        parts=dataset.microbatches(indices,config['microbatch_size'],config['max_candidate_cells'],config['bucket'])
        # The initial all-row likelihood verification runs once and includes
        # forced rows. Reserve the bounded cache for repeated actor-only scans;
        # otherwise that one large sweep would displace every reusable batch.
        for part,ix,inputs,q in dataset.policy_batches(parts,behavior.exploration,executor,cache=not verify):
            out=value_output(model,inputs,dataset,ix)
            p=row_probabilities(dataset,ix,out[0],q,behavior)
            selected=p.gather(1,dataset.chosen[ix,None]).squeeze(1).clamp_min(1e-30)
            old=dataset.old[ix]
            if verify:
                valid &= torch.isclose(selected,old,rtol=3e-3,atol=3e-6).all()
                if getattr(dataset,'opening_reference',None) is not None:
                    dataset.opening_reference.capture(ix,out[0])
            ratio=selected/old
            total+=((ratio-1)-ratio.log()).sum()
            if advantages is not None:
                target=dataset.utility if dataset.utility is not None else dataset.targets[0]
                baseline=out[4] if dataset.utility is not None else out[1]
                advantages[ix]=target[ix,dataset.seats[ix]]-baseline[torch.arange(len(ix),device=ix.device),dataset.seats[ix]]
        if verify:
            require(bool(valid),'Wrong/stale behavior probabilities')
            if getattr(dataset,'opening_reference',None) is not None:dataset.opening_reference.freeze()
        result=float(total)/len(indices)
    return result


def ppo_update(model,optimizer,games,behavior,config,*,profile=None,profile_steps=4,rehearsal=None,rehearsal_weight=0,learner_data=None,insolvency=None,reward=None,sample_groups=None,anchor_model=None,phases=None,preparation_pool=None):
    if config.get('actor_nonforced'):
        return ppo_nonforced(model,optimizer,games,behavior,config,profile=profile,profile_steps=profile_steps,
                             rehearsal=rehearsal,rehearsal_weight=rehearsal_weight,learner_data=learner_data,insolvency=insolvency,reward=reward,sample_groups=sample_groups,anchor_model=anchor_model,phases=phases,preparation_pool=preparation_pool)
    require(phases is None,'GAE/value phase requires the nonforced actor recipe')
    require(insolvency is None and reward is None,'Combined reward requires the nonforced actor recipe')
    require(all(g['split']=='train' for g in games),'Held-out PPO data')
    require(model_id(model)==behavior.evaluator.model_id,'Collection model differs from learner')
    timings=Timings()
    with timings.phase('ppo.prepare'):
        data=prepared(games,lambda r:r['behavior']['policy_id']==behavior.policy_id)
        require(all(d[0]['behavior']['kind']=='model' for d in data),'External actions in PPO')
        device=next(model.parameters()).device
        dataset=Dataset(data,device)
        demo=rehearsal_dataset(rehearsal,device) if rehearsal_weight else None
    count=len(data)
    advantages=torch.zeros(count,device=device)
    model.eval()
    policy_measure(model,dataset,range(count),behavior,config,timings,True,advantages)
    mean,std=advantages.mean(),advantages.std(unbiased=False)
    stats=dict(mean=float(mean),std=float(std))
    if config['normalize_advantages']: advantages=(advantages-mean)/std.clamp_min(1e-8)
    probe=torch.randperm(count).tolist()[:min(count,1024)]
    steps=0; stopped=False; epochs=[]; clip_events=0
    with profile_window(profile,device,profile_steps) as profiler:
        for epoch in range(config['epochs']):
            order=torch.randperm(count).tolist()
            totals=torch.zeros(7,device=device); seen=0
            model.train()
            for start in range(0,count,config['effective_batch_size']):
                batch=order[start:start+config['effective_batch_size']]
                optimizer.zero_grad(set_to_none=True)
                with timings.phase('ppo.forward_backward'):
                    for part in dataset.microbatches(batch,config['microbatch_size'],config['max_candidate_cells'],config['bucket']):
                        ix,inputs,q=dataset.batch(part,behavior.exploration,behavior.temperature)
                        with autocast(config,device): outputs=model(*inputs)
                        p=row_probabilities(dataset,ix,outputs[0],q,behavior)
                        new=p.gather(1,dataset.chosen[ix,None]).squeeze(1).clamp_min(1e-30)
                        ratio=new/dataset.old[ix]
                        a=advantages[ix]
                        actor=-torch.minimum(ratio*a,ratio.clamp(1-config['clip'],1+config['clip'])*a)
                        value,win,rank,vp=value_losses(outputs,tuple(t[ix] for t in dataset.targets))
                        entropy,bonus=entropy_term(outputs[0],p,dataset,part,behavior,config)
                        ((actor+config['value_weight']*value-bonus).sum()/len(batch)).backward()
                        clip=((ratio-1).abs()>config['clip']).float()
                        totals+=torch.stack([x.detach().sum() for x in (actor,value,entropy,clip,win,rank,vp)])
                        seen+=len(part)
                    if demo:
                        sample=torch.randint(len(demo.data),(min(len(batch),len(demo.data)),)).tolist()
                        for part in demo.microbatches(sample,config['microbatch_size'],config['max_candidate_cells'],config['bucket']):
                            ix,inputs,_=demo.batch(part)
                            with autocast(config,device): out=model(*inputs)
                            (rehearsal_weight*bc_loss(out[0],demo.chosen[ix]).sum()/len(sample)).backward()
                with timings.phase('ppo.optimizer'):
                    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),config['grad_clip'],error_if_nonfinite=True)
                    clip_events+=int(float(norm)>config['grad_clip'])
                    scheduled_step(optimizer,ppo=True);steps+=1
                if profiler: profiler.step()
                model.eval()
                kl=policy_measure(model,dataset,probe,behavior,config,timings)
                if kl>config['target_kl']:
                    stopped=True;break
                model.train()
            full_kl=policy_measure(model,dataset,range(count),behavior,config,timings)
            values=(totals/max(1,seen)).cpu().tolist()
            epochs.append(dict(epoch=epoch,**dict(zip(('actor','value','entropy','clip_fraction','win_loss','rank_loss','vp_loss'),values)),
                post_update_kl=full_kl,probe_kl=kl,gradient_norm=float(norm),positions=seen))
            if stopped or full_kl>config['target_kl']: stopped=True;break
    return dict(format='effective-minibatch-ppo/2.0.0',epochs=epochs,optimizer_steps=steps,
        learning_rates=[group['lr'] for group in optimizer.param_groups],learning_rate_schedule=learning_rate_report(optimizer),
        learner_decisions=count,learner_trajectories=len({d[2] for d in data}),forced_decisions=int(dataset.forced.sum()),
        effective_batch_size=config['effective_batch_size'],microbatch_size=config['microbatch_size'],
        advantage=stats,kl_probe_rows=len(probe),kl_early_stop=stopped,gradient_clip_steps=clip_events,
        timings=timings.report(),memory=memory(device),rehearsal_weight=rehearsal_weight,
        rehearsal_active=demo is not None,rehearsal_rows=len(demo.data) if demo else 0)


def phase_probe(dataset,indices,limit):
    """Proportional era/phase sample of the nonforced PPO decision population.

    Equal quotas per phase over-represented rare bonus/payment choices, hiding
    drift in the dominant action phase. Largest-remainder quotas preserve the
    population mixture; full-dataset sampled KL is still recorded separately.
    """
    from collections import defaultdict
    groups=defaultdict(list)
    for i in indices:
        context=dataset.rows[i]['view']['context']
        groups[(context['era'],context['phase'])].append(i)
    count=sum(map(len,groups.values()));size=min(limit,count)
    if not count:return []
    quota={key:size*len(rows)//count for key,rows in groups.items()}
    order=sorted(groups,key=lambda key:(-(size*len(groups[key])%count),key))
    for key in order[:size-sum(quota.values())]:quota[key]+=1
    output=[]
    for key in sorted(groups):
        rows=groups[key]
        output.extend(rows[j] for j in torch.randperm(len(rows)).tolist()[:quota[key]])
    return output


def ppo_nonforced(model,optimizer,games,behavior,config,*,profile=None,profile_steps=4,rehearsal=None,rehearsal_weight=0,learner_data=None,insolvency=None,reward=None,sample_groups=None,anchor_model=None,phases=None,preparation_pool=None):
    cuda=next(model.parameters()).device.type=='cuda'
    from contextlib import ExitStack
    from .ppo_preparation import PreparationPool,Preparation
    workers=config.get('preparation_workers',0)
    # Avoid spawning/importing Torch workers for tiny diagnostics. Production
    # collections are much larger; explicit Preparation tests cover small rows.
    useful=workers and (learner_data is None or len(learner_data)>=2048)
    with ExitStack() as stack:
        if useful:
            pool=preparation_pool or stack.enter_context(PreparationPool(workers))
            require(pool.workers==workers,'PPO preparation pool/config differ')
            executor=stack.enter_context(Preparation(pool))
        else:executor=stack.enter_context(ThreadPoolExecutor(max_workers=2)) if cuda else None
        result=_ppo_nonforced(model,optimizer,games,behavior,config,profile=profile,profile_steps=profile_steps,
                              rehearsal=rehearsal,rehearsal_weight=rehearsal_weight,learner_data=learner_data,executor=executor,insolvency=insolvency,reward=reward,sample_groups=sample_groups,anchor_model=anchor_model,phases=phases)
        if workers:result['cpu_preparation']=(executor.report() if useful else
            dict(workers=0,requested_workers=workers,reason='Fewer than 2048 rows; inline/thread preparation avoids process overhead'))
        return result


def _ppo_nonforced(model,optimizer,games,behavior,config,*,profile,profile_steps,rehearsal,rehearsal_weight,learner_data,executor,insolvency,reward,sample_groups=None,anchor_model=None,phases=None):
    from .ppg_phasic import detached
    from . import dcpg_config
    from . import ppo_kl_control
    exact_only=ppo_kl_control.active(config)
    detach_critic=detached(config)
    preserve=dcpg_config.enabled(config)
    from .ppg_g210 import active as restored_ppg
    if preserve:require(phases is not None and (dcpg_config.active(phases.options['value_phase']) or restored_ppg(phases.options['value_phase'])),
        'Value preservation requires the delayed fresh-target phase')
    require(all(g['split']=='train' for g in games),'Held-out PPO data')
    timings=Timings();device=next(model.parameters()).device
    with timings.phase('ppo.behavior_identity'):
        require(model_id(model)==behavior.evaluator.model_id,'Collection model differs from learner')
    from .learning_rate import KEY as WARMUP_KEY
    warmup=optimizer.param_groups[0].get(WARMUP_KEY)
    warmup_done=warmup is None or warmup['completed_steps']>=warmup['steps']
    with timings.phase('ppo.prepare'):
        data=learner_data if learner_data is not None else prepared(games,lambda r:r['behavior']['policy_id']==behavior.policy_id,insolvency=insolvency,reward=reward)
        require(data and all(d[0]['behavior']['kind']=='model' and d[0]['behavior']['policy_id']==behavior.policy_id
                            and d[0].get('_critic_role','learner')=='learner' for d in data),
                'External or stale actions in PPO')
        ds=Dataset(data,device);verify_behavior_metadata(ds,behavior)
        if 'opening_objective' in config:
            from .opening_objective import Reference
            ds.opening_reference=Reference(ds,behavior,config['opening_objective'])
        demo=rehearsal_dataset(rehearsal,device) if rehearsal_weight else None
    count=len(data);actor_ids=(~ds.forced).nonzero().flatten().cpu().tolist()
    with timings.phase('ppo.critic_prepare'):
        critic_ids=phases.prepare(ds) if phases is not None else list(range(count))
    critic_ds=phases.critic_dataset if phases is not None else ds
    critic_count=len(critic_ids)
    planned_steps=config['epochs']*math.ceil(len(actor_ids)/config['effective_batch_size'] if actor_ids else critic_count/config['critic_batch_size'])
    calibration=None;reward_report=None
    if reward is not None:
        from brass_ai.learning.contracts import digest
        from .utility import calibrate
        require(insolvency==reward['insolvency'] and model.config.utility==reward and all(r['_reward_id']==digest(reward) for r in ds.rows),
                'PPO model/return reward version differs')
        require(all(r['teacher'] is None for r in ds.rows),'Search distributions cannot enter fresh PPO rollouts')
        with timings.phase('ppo.utility_calibration'):
            calibration=(dict(required=False,steps=0,reason='DCPG fresh targets are fitted only in the value phase')
                if preserve else calibrate(model,optimizer,critic_ds,config,indices=critic_ids,executor=executor))
        trajectories=[r['_reward_trajectory'] for r in ds.rows if '_reward_trajectory' in r]
        require(len(trajectories)==len({d[2] for d in data}),'Missing reward components')
        means={k:sum(t[k] for t in trajectories)/len(trajectories) for k in trajectories[0]}
        reward_report=dict(spec=reward,undiscounted=True,components=means,
            weighted={k:reward[k]*means[k] for k in ('win','placement','vp')},trajectories=len(trajectories))
        from .review_metrics import reward_distribution
        reward_report['distribution']=reward_distribution(trajectories,reward)
    advantages=torch.zeros(count,device=device);model.eval()
    frozen=None
    if preserve:
        from .value_preservation import FrozenValues,TrunkGradients
        with timings.phase('ppo.freeze_values'):
            frozen=FrozenValues(model,critic_ds,config,getattr(phases,'local_scoring',None),executor)
    # Verify likelihoods for ALL rows, including deterministic continuation rows.
    policy_measure(model,ds,range(count),behavior,config,timings,True,None if frozen else advantages,executor)
    if frozen is not None:
        baseline=frozen.values['utility'][:count].to(device)
        advantages=(ds.utility-baseline)[torch.arange(count,device=device),ds.seats]
    costs,insolvency_report=advantage_costs(data,insolvency)
    if reward is not None:
        insolvency_report['objective']='Combined terminal utility minus future capped insolvency costs'
    # Archived win-only recipes subtract costs here. The combined head already
    # predicts the SAME cost-to-go as its target; never subtract costs twice.
    if reward is None:advantages-=torch.tensor(costs,device=device,dtype=advantages.dtype)
    mc_useful=advantages[actor_ids]
    gae_report=None
    if phases is not None:
        with timings.phase('ppo.gae'):
            advantages,gae_report=phases.advantages(advantages,actor_ids)
    useful=advantages[actor_ids];mean=useful.mean() if actor_ids else torch.zeros((),device=device)
    std=useful.std(unbiased=False) if actor_ids else torch.zeros((),device=device)
    from .review_metrics import distribution,CategoricalProbe,PPO_KL_GUARD
    advantage_distribution=distribution(useful)
    target=ds.utility if ds.utility is not None else ds.targets[0]
    actor_returns=target[actor_ids,ds.seats[actor_ids]]
    return_distribution=distribution(actor_returns)
    variance=float(actor_returns.var(unbiased=False)) if actor_ids else 0.
    explained_variance=1-float(mc_useful.var(unbiased=False))/variance if variance>1e-12 else None
    if config['normalize_advantages'] and actor_ids: advantages[actor_ids]=(useful-mean)/std.clamp_min(1e-8)
    with timings.phase('ppo.probe_prepare'):
        probe=phase_probe(ds,actor_ids,config['kl_probe_rows'])
        categorical_probe=CategoricalProbe(model,ds,probe,behavior,config)
    gradient_norms=[]
    epochs=[];steps=0;critic_steps=0;clips=0;stopped=False;rehearsal_steps=0
    stop_reason=None;max_categorical_kl=0.;max_behavior_kl=0.
    actor_used=set();critic_used=set();actor_presentations=Counter();critic_presentations=Counter();critic_roles=Counter()
    groups=[(sample_groups or {}).get(d[2][0],'unknown') for d in data]
    with profile_window(profile,device,profile_steps) as profiler:
        for epoch in range(config['epochs']):
            order=[actor_ids[j] for j in torch.randperm(len(actor_ids)).tolist()]
            critic_order=[critic_ids[j] for j in torch.randperm(critic_count).tolist()];critic_cursor=0
            actor_batches=[order[i:i+config['effective_batch_size']] for i in range(0,len(order),config['effective_batch_size'])]
            if not actor_batches: actor_batches=[[] for _ in range(math.ceil(critic_count/config['critic_batch_size']))]
            actor_totals=torch.zeros(3,device=device);value_totals=torch.zeros(4,device=device)
            actor_seen=critic_seen=0;norm=torch.zeros((),device=device);kl=exact_kl=0.0
            utility_squared=torch.zeros((),device=device);utility_absolute=torch.zeros((),device=device)
            for batch in actor_batches:
                if critic_cursor>=critic_count: critic_order=[critic_ids[j] for j in torch.randperm(critic_count).tolist()];critic_cursor=0
                crit=critic_order[critic_cursor:critic_cursor+config['critic_batch_size']];critic_cursor+=len(crit)
                optimizer.zero_grad(set_to_none=True);model.train()
                companion=getattr(phases,'local_scoring',None)
                if companion is not None:companion.begin('ppo')
                gradient_terms=TrunkGradients(model) if frozen else None
                with timings.phase('ppo.forward_backward'):
                    parts=ds.microbatches(batch,config['microbatch_size'],config['max_candidate_cells'],config['bucket'])
                    for part,ix,inputs,q in ds.policy_batches(parts,behavior.exploration,executor):
                        with autocast(config,device): out=model(*inputs)
                        p=row_probabilities(ds,ix,out[0],q,behavior)
                        new=p.gather(1,ds.chosen[ix,None]).squeeze(1).clamp_min(1e-30)
                        a=advantages[ix]
                        if getattr(ds,'opening_reference',None) is not None:
                            actor,ratio=ds.opening_reference.loss(ix,out[0],new,a,config['clip'])
                        else:
                            ratio=new/ds.old[ix]
                            actor=-torch.minimum(ratio*a,ratio.clamp(1-config['clip'],1+config['clip'])*a)
                        entropy,bonus=entropy_term(out[0],p,ds,part,behavior,config)
                        if gradient_terms is not None:gradient_terms.add_actor(actor.sum()/len(batch))
                        ((actor-bonus).sum()/len(batch)).backward()
                        actor_totals+=torch.stack([x.detach().sum() for x in (actor,entropy,((ratio-1).abs()>config['clip']).float())])
                        actor_seen+=len(part)
                    if gradient_terms is not None:gradient_terms.before_value()
                    parts=critic_ds.microbatches(crit,config['microbatch_size'],config['max_candidate_cells'],config['bucket'])
                    for part,ix,inputs in critic_ds.value_batches(parts,model,executor):
                        if frozen is not None:
                            model.eval()
                            with autocast(config,device):preserved=frozen.losses(model,inputs,ix)
                            values=tuple(preserved[k] for k in ('core','win','rank','vp'))
                            loss=values[0];local_loss=preserved['local'] if companion is not None and companion.active else None
                            if local_loss is not None:companion.preservation_used(critic_ds,part)
                        else:
                            with autocast(config,device):
                                drop=phases.dropout(epoch,'ppo') if phases is not None else None
                                if companion is not None:out,local_loss=companion.forward(model,inputs,critic_ds,ix,dropout=drop,detach=detach_critic)
                                else:out,local_loss=value_output(model,inputs,critic_ds,ix,dropout=drop,detach=detach_critic),None
                            values=value_losses(out,tuple(t[ix] for t in critic_ds.targets))
                            loss=values[0]
                            if critic_ds.utility is not None:
                                diff=out[4].float()-critic_ds.utility[ix]
                                utility_squared+=diff.detach().square().sum();utility_absolute+=diff.detach().abs().sum()
                                loss=loss+diff.square().mean(-1)
                        combined=config['value_weight']*loss
                        if local_loss is not None:
                            combined=combined+local_loss
                            companion.account_loss(local_loss,critic_ds.critic_weights[ix],len(crit),'ppo')
                        ((combined*critic_ds.critic_weights[ix]).sum()/len(crit)).backward()
                        value_totals+=torch.stack([v.detach().sum() for v in values]);critic_seen+=len(part)
                    if gradient_terms is not None:
                        frozen.gradients.append(gradient_terms.finish(steps+1));del gradient_terms
                    if demo:
                        sample=torch.randint(len(demo.data),(min(config['effective_batch_size'],len(demo.data)),)).tolist()
                        meaningful=[i for i in sample if not demo.forced_cpu[i]]
                        parts=demo.microbatches(meaningful,config['microbatch_size'],config['max_candidate_cells'],config['bucket'])
                        for part,ix,inputs in demo.supervised_batches(parts,executor):
                            with autocast(config,device): out=model(*inputs)
                            (rehearsal_weight*bc_loss(out[0],demo.chosen[ix]).sum()/len(meaningful)).backward()
                        rehearsal_steps+=int(bool(meaningful))
                with timings.phase('ppo.optimizer'):
                    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),config['grad_clip'],error_if_nonfinite=True)
                    clips+=int(float(norm)>config['grad_clip']);scheduled_step(optimizer,ppo=True);steps+=1
                    if companion is not None:companion.step('ppo')
                    gradient_norms.append(float(norm))
                    critic_steps+=int(bool(crit) and config['value_weight']>0)
                    actor_used.update(batch);critic_used.update(crit)
                    actor_presentations.update(groups[i] for i in batch)
                    critic_presentations.update(groups[i] for i in crit if i<count and critic_ds.roles[i]=='learner')
                    critic_roles.update(critic_ds.roles[i] for i in crit)
                if profiler: profiler.step()
                model.eval()
                with timings.phase('ppo.kl_probe'):
                    measured=categorical_probe.measure_step(model,executor=executor)
                exact_kl=measured['categorical_kl'];kl=measured['sampled_kl']
                max_categorical_kl=max(max_categorical_kl,exact_kl)
                max_behavior_kl=max(max_behavior_kl,measured.get('behavior_categorical_kl',exact_kl))
                if exact_kl>config['target_kl']:
                    stopped=True;stop_reason=('temperature1_categorical_probe' if 'opening_objective' in config else 'categorical_probe');break
                if measured.get('behavior_categorical_kl',0.)>config['target_kl']:
                    stopped=True;stop_reason='behavior_categorical_probe';break
            full=policy_measure(model,ds,actor_ids,behavior,config,timings,executor=executor)
            require(math.isfinite(full),'Nonfinite full-dataset PPO sampled KL; refusing further updates')
            if not exact_only and not stopped and full>config['target_kl']:
                stopped=True;stop_reason='sampled_full_dataset'
            aa=(actor_totals/max(1,actor_seen)).cpu().tolist();vv=(value_totals/max(1,critic_seen)).cpu().tolist()
            epochs.append(dict(epoch=epoch,actor=aa[0],entropy=aa[1],clip_fraction=aa[2],value=vv[0],
                win_loss=vv[1],rank_loss=vv[2],vp_loss=vv[3],post_update_kl=full,probe_kl=kl,probe_categorical_kl=exact_kl,
                gradient_norm=float(norm),actor_positions=actor_seen,critic_positions=critic_seen,positions=actor_seen))
            if 'opening_objective' in config:
                epochs[-1].update(probe_behavior_categorical_kl=measured['behavior_categorical_kl'] if steps else 0.,
                    probe_temperature1_categorical_kl=exact_kl)
            if reward is not None and not preserve:epochs[-1]['utility_prediction']=dict(mse=float(utility_squared)/max(1,4*critic_seen),mae=float(utility_absolute)/max(1,4*critic_seen))
            if preserve:epochs[-1]['value_scope']='Frozen-output preservation, not fresh-target fitting'
            if stopped:break
    from .ordinary import sample_usage
    if phases is not None:
        phases.after_ppo(model,actor_ids,probe,categorical_probe,
            critic_steps=critic_steps,critic_rows=len(critic_used) if critic_steps else 0,
            ppo_rate=optimizer.param_groups[0]['lr'] if steps else None,ppo_steps=steps)
    with timings.phase('ppo.probe_final'):
        final_probe=categorical_probe.measure(model,anchor=anchor_model)
    preservation_report=None
    if frozen is not None:
        with timings.phase('ppo.preservation_final'):
            preservation_report=frozen.report(model,critic_ids,probe,executor)
        preservation_report['actor_kl']=final_probe['categorical_kl']
        preservation_report['actor_kl_scope']='Unchanged raw-T1 PPO probe and separate behavior-policy guard; compare prior detach runs on the same recipe'
    return dict(format='nonforced-actor-ppo/2.2.0' if 'opening_objective' in config else 'nonforced-actor-ppo/2.1.0',epochs=epochs,optimizer_steps=steps,
        value_gradient=dict(mode='preserve-outputs' if preserve else 'detach-encoder' if detach_critic else 'shared',
            scope=('Frozen outputs only, with shared encoder/descriptor/head gradients; all fresh targets in PPG' if preserve else
                'PPO outcome, utility and local scoring losses; heads and critic-only descriptors train; actor/rehearsal unchanged; PPG encoder gradients enabled')),
        **(dict(value_preservation=preservation_report) if preserve else {}),
        **(dict(opening_objective=ds.opening_reference.report()) if 'opening_objective' in config else {}),
        branch_sample_usage=dict(actor_distinct_used=sum(bool(ds.rows[i].get('_branch')) for i in actor_used),
            critic_distinct_used=sum('/branch/' in critic_ds.data[i][2][0] for i in critic_used),
            scope='Distinct branch continuation rows in retained PPO steps; branch move and prefix excluded'),
        **(dict(gae=gae_report,critic_split=phases.split) if phases is not None else {}),
        optimizer_steps_planned=planned_steps,optimizer_steps_fraction=steps/planned_steps if planned_steps else None,
        controller_warmup_done=warmup_done,
        sample_usage=sample_usage(groups,actor_ids,actor_used,{i for i in critic_used if i<count},actor_presentations,
            Counter({k:v for k,v in critic_presentations.items()})),
        critic_rows=dict(available=len(critic_ds.data),train=len(critic_ids),
            optimizer_steps=critic_steps,
            available_by_role=dict(Counter(critic_ds.roles)),
            train_by_role=dict(Counter(critic_ds.roles[i] for i in critic_ids)),
            presentations_by_role=dict(critic_roles),
            learner_distinct_used=sum(critic_ds.roles[i]=='learner' for i in critic_used),
            snapshot_distinct_used=sum(critic_ds.roles[i]=='snapshot' for i in critic_used),
            snapshot_weight=phases.critic_state['snapshot_weight'] if phases is not None and phases.critic_state else None),
        insolvency=insolvency_report,
        reward=reward_report,utility_calibration=calibration,
        learner_decisions=count,meaningful_decisions=len(actor_ids),forced_decisions=int(ds.forced.sum()),
        learner_trajectories=len({d[2] for d in data}),effective_batch_size=config['effective_batch_size'],
        microbatch_size=config['microbatch_size'],critic_batch_size=config['critic_batch_size'],
        advantage=dict(mean=float(mean),std=float(std),scope='nonforced actor decisions'),
        advantage_distribution=advantage_distribution,return_distribution=return_distribution,
        baseline_explained_variance=explained_variance,
        baseline_scope='Before PPO, after any head calibration; nonforced learner decisions, in-sample',
        legal_actions=distribution([len(ds.rows[i]['choices']) for i in actor_ids]),
        categorical_probe=final_probe,gradient_norms=distribution(gradient_norms),
        input_cache=dict(probe=categorical_probe.cache_report(),scans=ds.scan_cache.report(),
                         admission=ds.input_audit.report()),
        kl_estimator=('probe_categorical_kl/categorical_kl: exact KL(old raw T1 || current raw T1), per-step guard and LR signal; behavior_categorical_kl: independent exact behavior-policy guard at the same limit; '
            if 'opening_objective' in config else 'probe_categorical_kl: exact KL(old behavior || current behavior), per-step guard; ')+
            'post_update_kl/probe_kl: sampled behavior chosen-action ((ratio-1)-log(ratio)); '+
            ('diagnostics only, never a stop or LR signal. ' if exact_only else 'full means all nonforced rows, retained as secondary epoch guard. ')+
            'Behavior reconstruction is verified separately before updates.',
        kl_guard=dict(contract=ppo_kl_control.CONTRACT if exact_only else 'ppo-categorical-kl-guard/2.0.0' if 'opening_objective' in config else PPO_KL_GUARD,
            target_kl=config['target_kl'],probe_positions=len(probe),
            requested_probe_positions=config['kl_probe_rows'],sampled_kl_stop_enabled=not exact_only,
            probe_checks=steps,full_dataset_checks=len(epochs),max_categorical_kl=max_categorical_kl,
            **(dict(max_behavior_categorical_kl=max_behavior_kl,
                operative_policy='raw-temperature-1',behavior_guard_preserved=True) if 'opening_objective' in config else {}),
            stop_reason=stop_reason,stop_step=steps if stopped else None),
        kl_probe_rows=len(probe),kl_probe_sampling='proportional-era-phase/1.0.0',
        learning_rates=[group['lr'] for group in optimizer.param_groups],learning_rate_schedule=learning_rate_report(optimizer),
        kl_early_stop=stopped,gradient_clip_steps=clips,
        rehearsal_active=demo is not None,rehearsal_rows=len(demo.data) if demo else 0,
        rehearsal_steps=rehearsal_steps,rehearsal_weight=rehearsal_weight if demo else 0,
        phases={phase:sum(ds.rows[i]['view']['context']['phase']==phase for i in actor_ids)
                for phase in sorted({r['view']['context']['phase'] for r in ds.rows})},timings=timings.report(),memory=memory(device))


def supervised_epoch(model,optimizer,data,config,*,training=True,kind='bc',weight=1.0,profile=None,progress=None,validation_weights=None,auxiliary_step=None,fit_observer=None):
    """Optionally report a second BC weighting from the same validation outputs."""
    require(kind in ('bc','teacher'),'Unknown supervised label kind')
    require(auxiliary_step is None or (training and kind=='teacher'),'Auxiliary gradients require a teacher training step')
    require(fit_observer is None or (not training and kind=='teacher'),'Fit observation is diagnostic only')
    # Search visits teach a policy distribution. A sparse teacher's eventual
    # game outcome is not a search value target. PPO/BC already fit the critic.
    value_weight=config['value_weight'] if kind=='bc' else 0.
    auxiliary_used=False
    timings=Timings();device=next(model.parameters()).device
    with timings.phase('supervised.prepare'):
        dataset=data if isinstance(data,Dataset) else Dataset(data,device,policy=False)
    data=dataset.data
    count=len(dataset.data);order=torch.randperm(count).tolist() if training else list(range(count))
    reweighted=None
    if validation_weights is not None:
        require(not training and kind=='bc','Additional BC weights are validation-only')
        validation_weights=torch.as_tensor(validation_weights,device=device,dtype=torch.float32)
        require(validation_weights.shape==(count,) and bool(torch.isfinite(validation_weights).all())
                and bool((validation_weights>0).all()),'Invalid validation weights')
        reweighted=torch.zeros((),device=device)
    totals=torch.zeros(10,device=device);steps=0;clips=0
    diagnostics={}
    log=Progress(progress,count) if progress else None
    predictions=torch.empty(count,device=device,dtype=torch.long)
    row_losses=torch.empty(count,device=device)
    model.train(training)
    pool=ThreadPoolExecutor(max_workers=2) if device.type=='cuda' else nullcontext(None)
    with pool as executor,torch.set_grad_enabled(training),profile_window(profile,device) as profiler:
        for start in range(0,count,config['effective_batch_size']):
            batch=order[start:start+config['effective_batch_size']]
            # Eligibility comes from the legal-choice list already on the CPU.
            # Reading a CUDA scalar for every row (again for every microbatch)
            # introduces thousands of synchronizations per optimizer step.
            actor_count=sum(not dataset.forced_cpu[i] for i in batch)
            if training: optimizer.zero_grad(set_to_none=True)
            with timings.phase('supervised.forward_backward'):
                parts=dataset.microbatches(batch,config['microbatch_size'],config['max_candidate_cells'],config['bucket'])
                for part,ix,inputs in dataset.supervised_batches(parts,executor):
                    with autocast(config,device): out=model(*inputs)
                    if fit_observer is not None:fit_observer(part,data,out)
                    if kind=='bc':
                        natural=bc_loss(out[0],dataset.chosen[ix])
                        policy=natural*dataset.bc_weights[ix]
                        if reweighted is not None:
                            # Balanced validation changes only the BC weights.
                            # Keep the original per-microbatch reduction order,
                            # sharing predictions/value losses from this pass.
                            reweighted+=(natural*validation_weights[ix]).sum()
                        teacher_entropy=teacher_kl=torch.zeros_like(policy)
                    else:
                        width=out[0].shape[1]
                        coverage=np.zeros((len(part),width),dtype=bool);target=np.zeros((len(part),width),dtype=np.float32)
                        confidence=[]
                        for j,i in enumerate(part):
                            teacher=data[i][0]['teacher'];validate_teacher(teacher,len(data[i][0]['choices']))
                            n=len(teacher['coverage']);coverage[j,:n]=teacher['coverage'];target[j,:n]=teacher['probabilities']
                            confidence.append(teacher['confidence'])
                        cov=torch.from_numpy(coverage).to(device);tar=torch.from_numpy(target).to(device)
                        logp=F.log_softmax(out[0].float().masked_fill(~cov,-1e9),-1)
                        confidence=torch.tensor(confidence,device=device)
                        policy=-(tar*logp).sum(-1)*confidence
                        teacher_entropy=-(tar*tar.clamp_min(1e-30).log()).sum(-1)*confidence
                        teacher_kl=policy-teacher_entropy
                        natural=policy
                    value,win,rank,vp=value_losses(out,tuple(t[ix] for t in dataset.targets))
                    meaningful=~dataset.forced[ix]
                    if kind=='bc':
                        if config.get('actor_nonforced'):
                            loss=weight*(policy*meaningful).sum()/max(1,actor_count)+config['value_weight']*value.sum()/len(batch)
                        else:loss=(weight*policy+config['value_weight']*value).sum()/len(batch)
                    else:
                        # Do not attach a zero-weight value graph: Adam's old
                        # moments/weight decay would still move value heads.
                        loss=(weight*(policy*meaningful).sum()/max(1,actor_count) if config.get('actor_nonforced')
                              else weight*policy.sum()/len(batch))
                    if training: loss.backward()
                    correct=((out[0].argmax(-1)==dataset.chosen[ix]) & meaningful).float()
                    totals+=torch.stack([x.detach().sum() for x in (policy,value,win,rank,vp,correct,meaningful.float(),natural,teacher_entropy,teacher_kl)])
                    predictions[ix]=out[0].detach().argmax(-1);row_losses[ix]=natural.detach()
            if training:
                if auxiliary_step is not None:auxiliary_used |= bool(auxiliary_step())
                norm=torch.nn.utils.clip_grad_norm_(model.parameters(),config['grad_clip'],error_if_nonfinite=True)
                clips+=int(float(norm)>config['grad_clip']);scheduled_step(optimizer);steps+=1
            if profiler: profiler.step()
            if log:
                completed=min(start+len(batch),count)
                if device.type=='cuda' and completed*100>=log.next*count:
                    torch.cuda.synchronize(device)  # Milestones report completed GPU work, not queued kernels.
                log(completed)
    # One device read per diagnostic vector/epoch, including searched teachers.
    predicted=predictions.cpu().tolist();losses=row_losses.cpu().tolist()
    with timings.phase('supervised.diagnostics'):
        for i in order:
            row=data[i][0]
            if dataset.forced_cpu[i]:continue
            target=row['choices'][row['chosen']];guess=row['choices'][predicted[i]]
            key='|'.join([row['behavior']['policy_id'],row['view']['context']['era'],row['view']['context']['phase'],target['kind']])
            bucket=diagnostics.setdefault(key,dict(decisions=0,loss=0.,correct=0,family_correct=0))
            bucket['decisions']+=1;bucket['loss']+=losses[i];bucket['correct']+=predicted[i]==row['chosen']
            bucket['family_correct']+=(guess['kind'],guess.get('industry'))==(target['kind'],target.get('industry'))
    p,v,w,r,vp,correct,meaningful,natural,teacher_entropy,teacher_kl=totals.cpu().tolist()
    actor_denominator=max(1,meaningful) if config.get('actor_nonforced') else count
    report=dict(policy_loss=p/actor_denominator,natural_policy_loss=natural/actor_denominator,value_loss=v/count,
        loss=weight*p/actor_denominator+value_weight*v/count,value_loss_training=bool(value_weight),
        win_loss=w/count,rank_loss=r/count,vp_loss=vp/count,meaningful_top1=correct/max(1,meaningful),
        meaningful_decisions=int(meaningful),decisions=count,optimizer_steps=steps,gradient_clip_steps=clips,
        learning_rate_schedule=learning_rate_report(optimizer),
        **(dict(teacher_entropy=teacher_entropy/actor_denominator,teacher_kl=teacher_kl/actor_denominator) if kind=='teacher' else {}),
        root_action_top1=sum(v['correct'] for k,v in diagnostics.items() if '|action|' in k)/max(1,sum(v['decisions'] for k,v in diagnostics.items() if '|action|' in k)),
        root_family_top1=sum(v['family_correct'] for k,v in diagnostics.items() if '|action|' in k)/max(1,sum(v['decisions'] for k,v in diagnostics.items() if '|action|' in k)),
        by_teacher_era_phase_action={k:dict(decisions=v['decisions'],natural_policy_loss=v['loss']/v['decisions'],
            top1=v['correct']/v['decisions'],family_top1=v['family_correct']/v['decisions']) for k,v in diagnostics.items()},
        timings=timings.report(),memory=memory(device))
    if auxiliary_used:
        report.update(value_loss_training=True,
            loss_scope='Policy loss only here; added auxiliary value loss and usage are reported in collection.teacher_value')
    if reweighted is None:return report
    from copy import deepcopy
    balanced_total=float(reweighted)
    return report,dict(deepcopy(report),policy_loss=balanced_total/actor_denominator,
                       loss=weight*balanced_total/actor_denominator+value_weight*v/count)


def distill_update(model,optimizer,games,config,*,steps=8,weight=.25,quality=None,auxiliary=None):
    from collections import Counter
    from .config import DISTILLATION,bounded_fields
    from .teacher_schedule import PHASES,phase_for
    quality=dict(DISTILLATION if quality is None else quality)
    bounded_fields(quality,DISTILLATION,'distillation')
    require(type(steps) is int and steps>0,'Invalid distillation step cap')
    require(all(g['split']=='train' for g in games),'Held-out search teachers')
    require(len({g['game_id'] for g in games})==len(games),'Duplicate search teacher games')
    source='ismcts-root-visits/1.2.0' if getattr(model.config,'utility',None) is not None else 'ismcts-root-visits/1.1.0'
    require(all(r['teacher'] is None or (r['teacher']['source']==source and
                (getattr(model.config,'utility',None) is None or r['teacher']['search']['reward']==model.config.utility)) for g in games for r in g['records']),
            'Historical neural/search replay cannot enter fresh distillation')
    eligible=lambda r:len(r['choices'])>1 and r['teacher'] is not None and r['teacher']['confidence']>0 and sum(r['teacher']['coverage'])>=2
    phases=Counter();setups=set();positions=set();count=0
    for game in games:
        for row in game['records']:
            if not eligible(row):continue
            validate_teacher(row['teacher'],len(row['choices']))
            phase=phase_for(row['view']['context']);require(phase is not None,'Unknown teacher phase')
            position=(game['setup_group'],row['seat'],row['view']['context']['revision'])
            require(position not in positions,'Duplicate qualified teacher position')
            positions.add(position)
            count+=1;phases[phase]+=1;setups.add(game['setup_group'])
    width=min(config['effective_batch_size'],quality['batch_size'])
    support=dict(teacher_rows=count,teacher_setups=len(setups),samples_by_phase={p:phases[p] for p,_,_,_ in PHASES},
                 quality=quality,requested_steps=steps,value_loss_training=False,
                 distill_batch_size=width,sample_presentation_cap=width*steps)
    def usage(seen,data):
        used=Counter(phase_for(data[i][0]['view']['context']) for i in seen)
        presented=Counter()
        for i,n in seen.items():presented[phase_for(data[i][0]['view']['context'])]+=n
        return dict(distinct_labels_used=len(seen),unused_qualified_labels=count-len(seen),
            sample_presentations=sum(seen.values()),max_observed_reuse=max(seen.values(),default=0),
            used_samples_by_phase={p:used[p] for p,_,_,_ in PHASES},
            unused_samples_by_phase={p:phases[p]-used[p] for p,_,_,_ in PHASES},
            sample_presentations_by_phase={p:presented[p] for p,_,_,_ in PHASES})
    if count<quality['min_samples'] or len(setups)<quality['min_games'] or any(phases[p]<quality['min_samples_per_phase'] for p,_,_,_ in PHASES):
        return dict(updated=False,optimizer_steps=0,reason='Insufficient diverse search teachers; no optimizer update',
                    teacher_policy_drift=None,**usage({},[]),**support)
    if getattr(model.config,'utility',None) is not None:
        from .utility import calibration_ready
        from brass_ai.step5.league import implementation_id
        identity=model_id(model);source_id=implementation_id()
        require(all(r['teacher']['model_id']==identity and r['teacher']['source_fingerprint']==source_id
                    for g in games for r in g['records'] if eligible(r)),
                'Search teaching requires fresh current-model/current-source targets')
        require(calibration_ready(model),'Calibrate the utility head before search teaching')
    data=prepared(games,eligible,outcome_targets=auxiliary.outcome_targets if auxiliary is not None else None)
    from .review_metrics import teacher_drift_probe
    from types import SimpleNamespace
    probe=teacher_drift_probe(model,data,SimpleNamespace(exploration=0.,temperature=1.),config)
    def fit():
        probe=auxiliary.probe if auxiliary is not None else None
        if probe:probe.start_fit()
        report=supervised_epoch(model,None,data,config,training=False,kind='teacher',weight=weight,
                                fit_observer=probe.observe_fit if probe else None)
        return {k:report[k] for k in ('policy_loss','teacher_entropy','teacher_kl','meaningful_top1','root_action_top1','root_family_top1')}
    before=fit()
    if auxiliary is not None:auxiliary.begin(model)
    reports=[];seen=Counter()
    teacher_config=dict(config,effective_batch_size=width)
    for _ in range(quality['max_reuse']):
        order=torch.randperm(len(data)).tolist()
        for start in range(0,len(order),width):
            if len(reports)>=steps:break
            indices=order[start:start+width];seen.update(indices)
            auxiliary_step=(lambda:auxiliary.backward(model,len(reports),width)) if auxiliary is not None else None
            reports.append(supervised_epoch(model,optimizer,[data[i] for i in indices],teacher_config,kind='teacher',weight=weight,
                                            auxiliary_step=auxiliary_step))
        if len(reports)>=steps:break
    after=fit()
    if auxiliary is not None:auxiliary.post_update(model)
    drift=probe.measure(model) if probe is not None else None
    if drift is not None:drift['scope']='Exact KL(pre-distillation policy || post-distillation policy) on a bounded qualified-label probe; before utility recalibration, training data, not playing strength'
    invalidated=getattr(model.config,'utility',None) is not None and bool(reports)
    if invalidated:
        # Policy-only updates still move the shared state encoder, which feeds
        # both official predictions and the remaining-utility correction. The
        # old fit receipt cannot qualify a subsequent teacher from these weights.
        # Fresh ordinary PPO data will run the existing bounded head calibration.
        model.utility_calibration=dict(model.utility_calibration,ready=False,
            invalidated_by='teacher-mixed-value' if auxiliary is not None else 'policy-distillation',
            reason='Shared trunk changed; fresh matching utility calibration required')
    support['value_loss_training']=auxiliary is not None and bool(auxiliary.seen)
    return dict(updated=True,optimizer_steps=sum(r['optimizer_steps'] for r in reports),updates=reports,
        **usage(seen,data),teacher_policy_drift=drift,
        fit_before=before,fit_after=after,fit_scope='same qualified collection; training fit, not held-out strength',
        utility_calibration_invalidated=invalidated,
        learning_rates=[group['lr'] for group in optimizer.param_groups],**support)
