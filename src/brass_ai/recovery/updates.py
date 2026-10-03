"""Effective minibatch PPO, full-legal-choice BC and vectorized four-seat losses."""
from contextlib import nullcontext
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import math
import numpy as np
import torch
from torch.nn import functional as F

from brass_ai.learning.batching import collate, model_id
from brass_ai.learning.contracts import require
from brass_ai.learning.replay import terminal_targets, validate_teacher
from brass_ai.step5.policy import exploration_distribution
from brass_ai.step5.updates import bounded_batches
from .encoding import PreparedEncoder
from .telemetry import Timings, profile_window, memory
from .progress import Progress


def prepared(games, predicate=None, neutral=False):
    """Targets once per game; filter BEFORE encoding; cached arrays survive epochs."""
    encoder, data = PreparedEncoder(), []
    for game in games:
        if hasattr(game,'assert_metadata'): game.assert_metadata()
        target = terminal_targets(game['result'])
        for row in game['records']:
            if predicate is None or predicate(row):
                r = dict(row, style=[0.0]*4, intent=[0.0]*8) if neutral else row
                encoded = game.encoded(r,encoder,allow_context=neutral) if hasattr(game,'encoded') else encoder.encode(r['view'], r['choices'], r['style'], r['intent'])
                data.append((r, target, (game['game_id'],row['seat']), encoded))
    require(data, 'No eligible learning decisions')
    return data


def value_losses(outputs, targets):
    _, win, rank, vp = outputs
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
        log=Progress(progress,5,'tensor groups') if progress else None
        self.chosen=torch.tensor([r['chosen'] for r in self.rows],device=device)
        self.seats=torch.tensor([r['seat'] for r in self.rows],device=device)
        self.old=torch.tensor([r['behavior']['probability'] for r in self.rows],device=device)
        if log:log(1)
        self.targets=tuple(torch.tensor([d[1][k] for d in data],device=device,dtype=torch.float32)
            for k in ('win_share','placement','vp'))
        if log:log(2)
        self.q=[exploration_distribution(r['choices'],r['intent']).astype(np.float32) for r in self.rows] if policy else None
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

    def batch(self, indices, exploration=0, temperature=1):
        ix=torch.tensor(indices,device=self.device)
        inputs=collate([self.encoded[i] for i in indices],self.device)
        width=inputs[1].shape[1]
        q=np.zeros((len(indices),width),dtype=np.float32)
        if exploration:
            for j,i in enumerate(indices): q[j,:len(self.q[i])]=self.q[i]
        return ix, inputs, torch.from_numpy(q).to(self.device)

    def _pinned_batch(self, indices):
        # A small background queue prepares/pins CPU tensors while CUDA trains.
        # These private buffers are never mutated after asynchronous transfer.
        return tuple(t.pin_memory() for t in (torch.tensor(indices),*collate([self.encoded[i] for i in indices])))

    def supervised_batches(self, parts, executor=None):
        if executor is None:
            for part in parts:
                ix,inputs,_=self.batch(part)
                yield part,ix,inputs
            return
        parts=iter(parts);pending=deque()
        def submit():
            part=next(parts,None)
            if part is not None:pending.append((part,executor.submit(self._pinned_batch,part)))
        for _ in range(2):submit()
        while pending:
            part,future=pending.popleft()
            host=future.result()
            tensors=tuple(t.to(self.device,non_blocking=True) for t in host)
            submit()
            yield part,tensors[0],tensors[1:]


def probabilities(logits, q, exploration, temperature):
    return (1-exploration)*F.softmax(logits.float()/temperature,-1)+exploration*q


def autocast(config, device):
    if config['precision']=='bf16':
        require(str(device).startswith('cuda') and torch.cuda.is_bf16_supported(), 'BF16 requires a supported CUDA device')
        return torch.autocast(device_type='cuda',dtype=torch.bfloat16)
    return nullcontext()


def policy_measure(model, dataset, indices, behavior, config, timings, verify=False, advantages=None):
    indices=list(indices)
    if not indices: return 0.0
    total=0.0
    with torch.no_grad(), timings.phase('ppo.policy_measure'):
        for part in dataset.microbatches(indices,config['microbatch_size'],config['max_candidate_cells'],config['bucket']):
            ix,inputs,q=dataset.batch(part,behavior.exploration,behavior.temperature)
            out=model(*inputs)
            p=probabilities(out[0],q,behavior.exploration,behavior.temperature)
            selected=p.gather(1,dataset.chosen[ix,None]).squeeze(1).clamp_min(1e-30)
            old=dataset.old[ix]
            if verify:
                require(torch.allclose(selected,old,rtol=3e-3,atol=3e-6),'Wrong/stale behavior probabilities')
            ratio=selected/old
            total+=float(((ratio-1)-ratio.log()).sum())
            if advantages is not None:
                advantages[ix]=dataset.targets[0][ix,dataset.seats[ix]]-out[1][torch.arange(len(ix),device=ix.device),dataset.seats[ix]]
    return total/len(indices)


def ppo_update(model,optimizer,games,behavior,config,*,profile=None,profile_steps=4,rehearsal=None,rehearsal_weight=0):
    if config.get('actor_nonforced'):
        return ppo_nonforced(model,optimizer,games,behavior,config,profile=profile,profile_steps=profile_steps,
                             rehearsal=rehearsal,rehearsal_weight=rehearsal_weight)
    require(all(g['split']=='train' for g in games),'Held-out PPO data')
    require(model_id(model)==behavior.evaluator.model_id,'Collection model differs from learner')
    timings=Timings()
    with timings.phase('ppo.prepare'):
        data=prepared(games,lambda r:r['behavior']['policy_id']==behavior.policy_id)
        require(all(d[0]['behavior']['kind']=='model' for d in data),'External actions in PPO')
        device=next(model.parameters()).device
        dataset=Dataset(data,device)
        demo=Dataset(rehearsal,device,policy=False) if rehearsal and rehearsal_weight else None
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
                        p=probabilities(outputs[0],q,behavior.exploration,behavior.temperature)
                        new=p.gather(1,dataset.chosen[ix,None]).squeeze(1).clamp_min(1e-30)
                        ratio=new/dataset.old[ix]
                        a=advantages[ix]
                        actor=-torch.minimum(ratio*a,ratio.clamp(1-config['clip'],1+config['clip'])*a)
                        value,win,rank,vp=value_losses(outputs,tuple(t[ix] for t in dataset.targets))
                        entropy=-(p*p.clamp_min(1e-30).log()).sum(-1)
                        ((actor+config['value_weight']*value-config['entropy_weight']*entropy).sum()/len(batch)).backward()
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
                    optimizer.step();steps+=1
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
        learner_decisions=count,learner_trajectories=len({d[2] for d in data}),forced_decisions=int(dataset.forced.sum()),
        effective_batch_size=config['effective_batch_size'],microbatch_size=config['microbatch_size'],
        advantage=stats,kl_probe_rows=len(probe),kl_early_stop=stopped,gradient_clip_steps=clip_events,
        timings=timings.report(),memory=memory(device),rehearsal_weight=rehearsal_weight,
        rehearsal_active=demo is not None,rehearsal_rows=len(demo.data) if demo else 0)


def phase_probe(dataset,indices,limit):
    """Bounded nonforced probe covering phases; full final KL remains required."""
    from collections import defaultdict
    groups=defaultdict(list)
    for i in indices: groups[dataset.rows[i]['view']['context']['phase']].append(i)
    queues=[]
    for phase in sorted(groups):
        group=groups[phase];queues.append([group[j] for j in torch.randperm(len(group)).tolist()])
    output=[]
    while queues and len(output)<limit:
        for q in list(queues):
            if len(output)>=limit: break
            output.append(q.pop())
            if not q: queues.remove(q)
    return output


def ppo_nonforced(model,optimizer,games,behavior,config,*,profile=None,profile_steps=4,rehearsal=None,rehearsal_weight=0):
    require(all(g['split']=='train' for g in games),'Held-out PPO data')
    require(model_id(model)==behavior.evaluator.model_id,'Collection model differs from learner')
    timings=Timings();device=next(model.parameters()).device
    with timings.phase('ppo.prepare'):
        data=prepared(games,lambda r:r['behavior']['policy_id']==behavior.policy_id)
        require(all(d[0]['behavior']['kind']=='model' for d in data),'External actions in PPO')
        ds=Dataset(data,device);demo=Dataset(rehearsal,device,policy=False) if rehearsal and rehearsal_weight else None
    count=len(data);actor_ids=(~ds.forced).nonzero().flatten().cpu().tolist()
    advantages=torch.zeros(count,device=device);model.eval()
    # Verify likelihoods for ALL rows, including deterministic continuation rows.
    policy_measure(model,ds,range(count),behavior,config,timings,True,advantages)
    useful=advantages[actor_ids];mean=useful.mean() if actor_ids else torch.zeros((),device=device)
    std=useful.std(unbiased=False) if actor_ids else torch.zeros((),device=device)
    if config['normalize_advantages'] and actor_ids: advantages[actor_ids]=(useful-mean)/std.clamp_min(1e-8)
    probe=phase_probe(ds,actor_ids,config['kl_probe_rows'])
    epochs=[];steps=0;clips=0;stopped=False;rehearsal_steps=0
    with profile_window(profile,device,profile_steps) as profiler:
        for epoch in range(config['epochs']):
            order=[actor_ids[j] for j in torch.randperm(len(actor_ids)).tolist()]
            critic_order=torch.randperm(count).tolist();critic_cursor=0
            actor_batches=[order[i:i+config['effective_batch_size']] for i in range(0,len(order),config['effective_batch_size'])]
            if not actor_batches: actor_batches=[[] for _ in range(math.ceil(count/config['critic_batch_size']))]
            actor_totals=torch.zeros(3,device=device);value_totals=torch.zeros(4,device=device)
            actor_seen=critic_seen=0;norm=torch.zeros((),device=device);kl=0.0
            for batch in actor_batches:
                if critic_cursor>=count: critic_order=torch.randperm(count).tolist();critic_cursor=0
                crit=critic_order[critic_cursor:critic_cursor+config['critic_batch_size']];critic_cursor+=len(crit)
                optimizer.zero_grad(set_to_none=True);model.train()
                with timings.phase('ppo.forward_backward'):
                    for part in ds.microbatches(batch,config['microbatch_size'],config['max_candidate_cells'],config['bucket']):
                        ix,inputs,q=ds.batch(part,behavior.exploration,behavior.temperature)
                        with autocast(config,device): out=model(*inputs)
                        p=probabilities(out[0],q,behavior.exploration,behavior.temperature)
                        new=p.gather(1,ds.chosen[ix,None]).squeeze(1).clamp_min(1e-30)
                        ratio=new/ds.old[ix];a=advantages[ix]
                        actor=-torch.minimum(ratio*a,ratio.clamp(1-config['clip'],1+config['clip'])*a)
                        entropy=-(p*p.clamp_min(1e-30).log()).sum(-1)
                        ((actor-config['entropy_weight']*entropy).sum()/len(batch)).backward()
                        actor_totals+=torch.stack([x.detach().sum() for x in (actor,entropy,((ratio-1).abs()>config['clip']).float())])
                        actor_seen+=len(part)
                    for part in ds.microbatches(crit,config['microbatch_size'],config['max_candidate_cells'],config['bucket']):
                        ix,inputs,_=ds.batch(part)
                        with autocast(config,device): out=model(*inputs)
                        values=value_losses(out,tuple(t[ix] for t in ds.targets))
                        (config['value_weight']*values[0].sum()/len(crit)).backward()
                        value_totals+=torch.stack([v.detach().sum() for v in values]);critic_seen+=len(part)
                    if demo:
                        sample=torch.randint(len(demo.data),(min(config['effective_batch_size'],len(demo.data)),)).tolist()
                        meaningful=[i for i in sample if not demo.forced_cpu[i]]
                        for part in demo.microbatches(meaningful,config['microbatch_size'],config['max_candidate_cells'],config['bucket']):
                            ix,inputs,_=demo.batch(part)
                            with autocast(config,device): out=model(*inputs)
                            (rehearsal_weight*bc_loss(out[0],demo.chosen[ix]).sum()/len(meaningful)).backward()
                        rehearsal_steps+=int(bool(meaningful))
                with timings.phase('ppo.optimizer'):
                    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),config['grad_clip'],error_if_nonfinite=True)
                    clips+=int(float(norm)>config['grad_clip']);optimizer.step();steps+=1
                if profiler: profiler.step()
                model.eval();kl=policy_measure(model,ds,probe,behavior,config,timings)
                if kl>config['target_kl']: stopped=True;break
            full=policy_measure(model,ds,actor_ids,behavior,config,timings)
            aa=(actor_totals/max(1,actor_seen)).cpu().tolist();vv=(value_totals/max(1,critic_seen)).cpu().tolist()
            epochs.append(dict(epoch=epoch,actor=aa[0],entropy=aa[1],clip_fraction=aa[2],value=vv[0],
                win_loss=vv[1],rank_loss=vv[2],vp_loss=vv[3],post_update_kl=full,probe_kl=kl,
                gradient_norm=float(norm),actor_positions=actor_seen,critic_positions=critic_seen,positions=actor_seen))
            if stopped or full>config['target_kl']: stopped=True;break
    return dict(format='nonforced-actor-ppo/2.1.0',epochs=epochs,optimizer_steps=steps,
        learner_decisions=count,meaningful_decisions=len(actor_ids),forced_decisions=int(ds.forced.sum()),
        learner_trajectories=len({d[2] for d in data}),effective_batch_size=config['effective_batch_size'],
        microbatch_size=config['microbatch_size'],critic_batch_size=config['critic_batch_size'],
        advantage=dict(mean=float(mean),std=float(std),scope='nonforced actor decisions'),
        kl_probe_rows=len(probe),kl_early_stop=stopped,gradient_clip_steps=clips,
        rehearsal_active=demo is not None,rehearsal_rows=len(demo.data) if demo else 0,
        rehearsal_steps=rehearsal_steps,rehearsal_weight=rehearsal_weight if demo else 0,
        phases={phase:sum(ds.rows[i]['view']['context']['phase']==phase for i in actor_ids)
                for phase in sorted({r['view']['context']['phase'] for r in ds.rows})},timings=timings.report(),memory=memory(device))


def supervised_epoch(model,optimizer,data,config,*,training=True,kind='bc',weight=1.0,profile=None,progress=None):
    require(kind in ('bc','teacher'),'Unknown supervised label kind')
    timings=Timings();device=next(model.parameters()).device
    with timings.phase('supervised.prepare'):
        dataset=data if isinstance(data,Dataset) else Dataset(data,device,policy=False)
    data=dataset.data
    count=len(dataset.data);order=torch.randperm(count).tolist() if training else list(range(count))
    totals=torch.zeros(8,device=device);steps=0;clips=0
    diagnostics={}
    log=Progress(progress,count) if progress else None
    if kind=='bc':
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
                    if kind=='bc':
                        natural=bc_loss(out[0],dataset.chosen[ix])
                        policy=natural*dataset.bc_weights[ix]
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
                        policy=-(tar*logp).sum(-1)*torch.tensor(confidence,device=device)
                        natural=policy
                    value,win,rank,vp=value_losses(out,tuple(t[ix] for t in dataset.targets))
                    meaningful=~dataset.forced[ix]
                    if config.get('actor_nonforced'):
                        loss=weight*(policy*meaningful).sum()/max(1,actor_count)+config['value_weight']*value.sum()/len(batch)
                    else: loss=(weight*policy+config['value_weight']*value).sum()/len(batch)
                    if training: loss.backward()
                    correct=((out[0].argmax(-1)==dataset.chosen[ix]) & meaningful).float()
                    totals+=torch.stack([x.detach().sum() for x in (policy,value,win,rank,vp,correct,meaningful.float(),natural)])
                    if kind=='bc':
                        predictions[ix]=out[0].detach().argmax(-1);row_losses[ix]=natural.detach()
            if training:
                norm=torch.nn.utils.clip_grad_norm_(model.parameters(),config['grad_clip'],error_if_nonfinite=True)
                clips+=int(float(norm)>config['grad_clip']);optimizer.step();steps+=1
            if profiler: profiler.step()
            if log:
                completed=min(start+len(batch),count)
                if device.type=='cuda' and completed*100>=log.next*count:
                    torch.cuda.synchronize(device)  # Milestones report completed GPU work, not queued kernels.
                log(completed)
    if kind=='bc':
        # One device read per diagnostic vector/epoch, not two per microbatch.
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
    p,v,w,r,vp,correct,meaningful,natural=totals.cpu().tolist()
    actor_denominator=max(1,meaningful) if config.get('actor_nonforced') else count
    return dict(policy_loss=p/actor_denominator,natural_policy_loss=natural/actor_denominator,value_loss=v/count,
        loss=weight*p/actor_denominator+config['value_weight']*v/count,
        win_loss=w/count,rank_loss=r/count,vp_loss=vp/count,meaningful_top1=correct/max(1,meaningful),
        meaningful_decisions=int(meaningful),decisions=count,optimizer_steps=steps,gradient_clip_steps=clips,
        root_action_top1=sum(v['correct'] for k,v in diagnostics.items() if '|action|' in k)/max(1,sum(v['decisions'] for k,v in diagnostics.items() if '|action|' in k)),
        root_family_top1=sum(v['family_correct'] for k,v in diagnostics.items() if '|action|' in k)/max(1,sum(v['decisions'] for k,v in diagnostics.items() if '|action|' in k)),
        by_teacher_era_phase_action={k:dict(decisions=v['decisions'],natural_policy_loss=v['loss']/v['decisions'],
            top1=v['correct']/v['decisions'],family_top1=v['family_correct']/v['decisions']) for k,v in diagnostics.items()},
        timings=timings.report(),memory=memory(device))


def distill_update(model,optimizer,games,config,*,steps=8,weight=.25):
    require(all(g['split']=='train' for g in games),'Held-out search teachers')
    require(all(r['teacher'] is None or r['teacher']['source']=='ismcts-root-visits/1.1.0' for g in games for r in g['records']),
            'Historical neural/search replay cannot enter fresh distillation')
    eligible=lambda r:r['teacher'] is not None and r['teacher']['confidence']>0 and sum(r['teacher']['coverage'])>=2
    if not any(eligible(r) for g in games for r in g['records']):
        return dict(updated=False,reason='No completed Masked ISMCTS visits on two alternatives')
    data=prepared(games,eligible)
    reports=[]
    # Sample effective batches; steps means optimizer steps, not whole epochs.
    for _ in range(steps):
        sample=[data[i] for i in torch.randint(len(data),(min(config['effective_batch_size'],len(data)),)).tolist()]
        reports.append(supervised_epoch(model,optimizer,sample,config,kind='teacher',weight=weight))
    return dict(updated=True,teacher_rows=len(data),optimizer_steps=sum(r['optimizer_steps'] for r in reports),updates=reports)
