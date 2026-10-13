"""Bounded P1 head warm-up and fresh on-policy joint-loss pilot, not P2 search."""
from pathlib import Path
import math
import shutil
import numpy as np
import torch

from brass_ai.bank import sha
from brass_ai.learning.batching import collate
from brass_ai.learning.contracts import digest,project_path,read_json,require
from brass_ai.learning.replay import terminal_targets
from brass_ai.learning.reward import replay_returns
from brass_ai.recovery.progress import Progress
from brass_ai.recovery.retention import owned,run_lock
from brass_ai.recovery.updates import value_losses
from brass_ai.recovery.replay import read_shards
from brass_ai.step5.league import write_json
from . import checkpoint,handoff
from .contracts import contract,local_file,accepted_contract
from .collection import collect,distribution
from .labels import loss_rows,confidence_report
from .model import import_parent,base_outputs
from .traces import examples,read_collection

DEFAULTS=dict(learning_rate=1e-5,goal_learning_rate=1e-4,epochs=2,batch_size=2048,critic_batch_size=2048,microbatch_size=512,
              max_candidate_cells=131072,clip=.2,target_kl=.01,entropy=.005,value_weight=.5,goal_weight=.05,grad_clip=5.)


def resolved_config(manifest):
    source=manifest['signature']['config'];update=source['update']
    cfg=dict(DEFAULTS,learning_rate=source['learning_rate'])
    for key in ('microbatch_size','max_candidate_cells','epochs','clip','target_kl','value_weight','grad_clip','critic_batch_size'):
        cfg[key]=update[key]
    cfg.update(batch_size=update['effective_batch_size'],entropy=update['entropy_weight'])
    return cfg


def parts(data,indexes,cfg):
    group=[];width=0
    for i in indexes:
        n=len(data[i]['encoded'].choice_ids)
        require(n<=cfg['max_candidate_cells'],'Oversized P1 candidate set; increase the bounded cell limit')
        proposed=max(width,n)
        if group and (len(group)>=cfg['microbatch_size'] or (len(group)+1)*proposed>cfg['max_candidate_cells']):
            yield group;group=[];width=0
        group.append(i);width=max(width,n)
    if group:yield group


def forward(model,data,indexes,device):
    batch=[data[i] for i in indexes]
    inputs=collate([d['encoded'] for d in batch],device)
    z=torch.as_tensor(np.stack([d['personality'] for d in batch]),device=device)
    remaining=torch.tensor([d['remaining'] for d in batch],device=device,dtype=torch.float32)
    return model(*inputs,personality=z,remaining=remaining)


def metrics(model,data,device,cfg=DEFAULTS):
    losses=[];confidence=[];utility_square=utility_abs=0.;official=np.zeros(3)
    model.eval()
    with torch.inference_mode():
        for ix in parts(data,range(len(data)),cfg):
            out=forward(model,data,ix,device);batch=[data[i] for i in ix]
            loss,mask=loss_rows(out,[d['goals'] for d in batch]);losses.extend(loss[mask].cpu().tolist())
            confidence.extend(torch.sigmoid(out['confidence_logits']).cpu().tolist())
            target=torch.tensor([d['utility'] for d in batch],device=device)
            error=out['remaining_utility_4']-target
            utility_square+=float(error.square().sum());utility_abs+=float(error.abs().sum())
            targets=tuple(torch.tensor([d['targets'][k] for d in batch],device=device) for k in ('win_share','placement','vp'))
            official+=np.asarray([float(x.sum()) for x in value_losses(base_outputs(out),targets)[1:]])
    return dict(goal_loss=float(np.mean(losses)) if losses else None,goal_rows=len(losses),
        confidence=confidence_report(confidence,[d['goals'] for d in data]),
        utility_mse=utility_square/(4*len(data)),utility_mae=utility_abs/(4*len(data)),
        official_head_losses=dict(zip(('win','rank','vp'),(official/len(data)).tolist())))


def fit(model,data,device,*,epochs=3,joint=False,cfg=None,resume=None):
    cfg=dict(DEFAULTS,**(cfg or {}));require(type(epochs) is int and 1<=epochs<=10,'P1 uses 1-10 bounded passes')
    require(data,'No P1 training examples')
    require(all(d['split']=='train' for d in data),'Evaluation/calibration data cannot train the personality model')
    require(not joint or all(d['on_policy'] for d in data),'PPO rejects legacy/opponent/auxiliary-only rows')
    require(joint or any(any(d['goals']['mask'].values()) or np.asarray(d['goals']['confidence_mask']).any() for d in data),
            'No observed goal labels; collect fresh complete discovery games')
    for d in data:require(all(math.isfinite(x) for x in d['utility']),'Nonfinite remaining utility')
    if joint:model.requires_grad_(True)
    else:model.freeze_core()
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
        lr=cfg['learning_rate'] if joint else cfg['goal_learning_rate'],weight_decay=.0001)
    if resume is not None:
        require(resume['optimizer'] is not None,'Missing P1 optimizer resume state')
        optimizer.load_state_dict(resume['optimizer']);checkpoint.restore_rng(resume)
    before=metrics(model,data,device,cfg);frozen={k:t.detach().cpu().clone() for k,t in model.core.state_dict().items()} if not joint else None
    old=[];advantages=torch.zeros(len(data),device=device)
    actor_ids=[i for i,d in enumerate(data) if len(d['row']['choices'])>1]
    meaningful=torch.tensor([len(d['row']['choices'])>1 for d in data],device=device)
    if joint:
        with torch.no_grad():
            for ix in parts(data,range(len(data)),cfg):
                out=forward(model,data,ix,device);batch=[data[i] for i in ix]
                p=distribution(out['policy_logits'],[d['row']['choices'] for d in batch])
                for j,(i,d) in enumerate(zip(ix,batch)):
                    actual=float(p[j,d['row']['chosen']]);stored=d['row']['behavior']['probability']
                    require(abs(actual-stored)<=max(2e-6,2e-5*stored),'PPO behavior likelihood/context mismatch')
                    old.append(p[j,:len(d['row']['choices'])].detach())
                    advantages[i]=d['utility'][d['row']['seat']]-out['remaining_utility_4'][j,d['row']['seat']]
        if meaningful.any():
            values=advantages[meaningful];advantages[meaningful]=(values-values.mean())/values.std(unbiased=False).clamp_min(1e-8)
    reports=[];steps=0;progress=Progress('P1 joint pilot' if joint else 'P1 goal warm-up',epochs,'epochs')
    for epoch in range(epochs):
        model.train();order=torch.randperm(len(data)).tolist();stopped=False;last_kl=0.;actor_seen=critic_seen=0
        shuffled=[actor_ids[j] for j in torch.randperm(len(actor_ids)).tolist()] if joint else []
        batches=[shuffled[i:i+cfg['batch_size']] for i in range(0,len(shuffled),cfg['batch_size'])]
        width=cfg['critic_batch_size'] if joint else cfg['batch_size']
        if not batches:batches=[[] for _ in range(math.ceil(len(data)/width))]
        cursor=0
        for actor_batch in batches:
            if cursor>=len(order):order=torch.randperm(len(data)).tolist();cursor=0
            batch=order[cursor:cursor+width];cursor+=len(batch)
            optimizer.zero_grad(set_to_none=True)
            # As in Step 5, actor batches contain only genuine choices and
            # critic/goal batches independently sample all learner decisions.
            for ix in parts(data,actor_batch,cfg):
                out=forward(model,data,ix,device);rows=[data[i] for i in ix]
                p=distribution(out['policy_logits'],[d['row']['choices'] for d in rows])
                chosen=torch.tensor([d['row']['chosen'] for d in rows],device=device)
                previous=torch.stack([old[i][d['row']['chosen']] for i,d in zip(ix,rows)])
                ratio=p.gather(1,chosen[:,None]).squeeze(1)/previous
                a=advantages[ix];surrogate=torch.minimum(ratio*a,ratio.clamp(1-cfg['clip'],1+cfg['clip'])*a)
                entropy=-(p*p.clamp_min(1e-30).log()).sum(-1)
                loss=(-surrogate-cfg['entropy']*entropy).sum()/len(actor_batch)
                require(bool(torch.isfinite(loss)),'Nonfinite P1 actor loss');loss.backward();actor_seen+=len(ix)
            n_goal=sum(any(data[i]['goals']['mask'].values()) or np.asarray(data[i]['goals']['confidence_mask']).any() for i in batch)
            for ix in parts(data,batch,cfg):
                out=forward(model,data,ix,device);rows=[data[i] for i in ix]
                goal,_=loss_rows(out,[d['goals'] for d in rows])
                loss=(cfg['goal_weight'] if joint else 1.)*goal.sum()/max(1,n_goal)
                if joint:
                    targets=tuple(torch.tensor([d['targets'][k] for d in rows],device=device) for k in ('win_share','placement','vp'))
                    values=value_losses(base_outputs(out),targets)[0]
                    returns=torch.tensor([d['utility'] for d in rows],device=device)
                    value=values+(out['remaining_utility_4']-returns).square().mean(-1)
                    loss+=cfg['value_weight']*value.sum()/len(batch)
                    critic_seen+=len(ix)
                require(bool(torch.isfinite(loss)),'Nonfinite P1 loss');loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],cfg['grad_clip'],error_if_nonfinite=True)
            optimizer.step();steps+=1
            if joint:
                total=0.;count=0
                with torch.inference_mode():
                    for ix in parts(data,range(len(data)),cfg):
                        out=forward(model,data,ix,device);p=distribution(out['policy_logits'],[data[i]['row']['choices'] for i in ix])
                        for j,i in enumerate(ix):
                            if meaningful[i]:
                                q=old[i];total+=float((q*(q.clamp_min(1e-30).log()-p[j,:len(q)].clamp_min(1e-30).log())).sum());count+=1
                last_kl=total/max(1,count)
                if last_kl>cfg['target_kl']:stopped=True;break
        reports.append(dict(epoch=epoch+1,kl=last_kl,kl_stopped=stopped,actor_positions=actor_seen,critic_positions=critic_seen));progress(epoch+1)
        if stopped:break
    if frozen is not None:
        require(all(torch.equal(v,model.core.state_dict()[k].cpu()) for k,v in frozen.items()),'Warm-up changed the parent core')
        require(not torch.count_nonzero(model.context_projection.weight),'Warm-up changed neutral projection')
    after=metrics(model,data,device,cfg)
    report=dict(phase='joint-fresh-ppo' if joint else 'goal-head-only',before=before,after=after,
        optimizer_steps=steps,epochs=reports,config=cfg,rows=len(data),actor_rows=len(actor_ids) if joint else 0,
        reward=model.core.config.utility,scope='Bounded P1 operation/fit check, not playing-strength or style qualification')
    return optimizer,report


def initialize(root,bundle,run,device):
    root,run=owned(root,run);bundle=project_path(root,bundle)
    manifest,parent=handoff.load(bundle)
    if (run/'state.json').exists():
        state=read_json(run/'state.json')
        accepted_contract(state['contract'])
        require(state['handoff_sha256']==sha(bundle/'manifest.json'),'P1 run/handoff changed')
        model,raw=checkpoint.load(checkpoint.committed_path(run,state),device)
        require(raw['model_id']==state['model_id'],'P1 checkpoint identity changed')
        return root,run,manifest,parent.to(device),model,state,raw
    require(not run.exists() or not any(p.name!='.active-run.json' for p in run.iterdir()),'Use a new personality run')
    torch.manual_seed(2910000);model,transfer=import_parent(parent);model=model.to(device)
    run.mkdir(parents=True,exist_ok=True);checkpoint.save(run/'initial.pt',model,training=dict(handoff=manifest['parent'],transfer=transfer))
    identity=checkpoint.save(run/'model.pt',model,training=dict(phase='imported',handoff=manifest['parent']))
    state=dict(format=checkpoint.PILOT,model_path='model.pt',contract=contract(),handoff=bundle.relative_to(root).as_posix(),
        handoff_sha256=sha(bundle/'manifest.json'),model_id=identity,model_sha256=sha(run/'model.pt'),
        warmup_runs=0,joint_runs=0,validation_runs=0,reports=[],transfer=transfer)
    write_json(run/'state.json',state)
    write_json(run/'pins.json',dict(paths=[(run/'initial.pt').relative_to(root).as_posix(),
        (bundle/'manifest.json').relative_to(root).as_posix()]))
    return root,run,manifest,parent.to(device),model,state,None


def execute(root,bundle,run,*,mode='warmup',epochs=3,games=0,workers=4,device='cpu',mono=None):
    require(mode in ('warmup','joint','validate'),'Unknown P1 phase')
    require(mode=='warmup' or 1<=games<=32,'Joint/calibration requires fresh games')
    _,run=owned(root,run)
    with run_lock(run):
        root,run,manifest,parent,model,state,raw=initialize(root,bundle,run,device)
        require(mode=='warmup' or state['warmup_runs']>0,'Run goal-head warm-up before the joint/calibration pilot')
        require(mode!='warmup' or not state['joint_runs'],'Warm-up is frozen-core preparation; use a separate run after joint training')
        key=dict(warmup='warmup_runs',joint='joint_runs',validate='validation_runs')[mode]
        directory=run/f'{mode}-{state[key]:04d}'
        require(not directory.exists(),'Uncommitted P1 phase exists; preserve it under the run\'s interrupted/ directory, then retry from the committed checkpoint')
        reward=manifest['reward'];partition='validation' if mode=='validate' else 'train';cfg=resolved_config(manifest)
        if games:
            collect(project_path(root,manifest['signature']['config']['host']),model,parent,directory/'collection',
                games=games,workers=workers,seed=2910000+state[key]*10000+dict(warmup=0,joint=1000000,validate=2000000)[mode],
                partition=partition,device=device,mono=mono)
            data,metadata,episodes=read_collection(directory/'collection',checkpoint.model_id(model),partition=partition,reward=reward)
        else:
            require(mode=='warmup' and manifest['traces'],'No handoff traces; use warmup --games 4 to collect fresh discovery data')
            episodes=read_shards([local_file(project_path(root,bundle),t['path']) for t in manifest['traces']],partition='train')
            data=examples(episodes,reward,roots_only=True);metadata=None
        require(all(g['split']==partition for g in episodes),'Training/calibration split contamination')
        if mode=='validate':
            report=dict(phase='held-out-goal-calibration',metrics=metrics(model,data,device,cfg),model_id=checkpoint.model_id(model),
                scope='No gradients or model promotion; confidence remains diagnostic')
        else:
            continuing=raw if raw and raw['training']['phase']==mode else None
            # Enlarging the trainable parameter set starts fresh Adam groups,
            # while the saved pilot RNG still governs the next minibatch order.
            if raw is not None and continuing is None:checkpoint.restore_rng(raw)
            optimizer,report=fit(model,data,device,epochs=epochs,joint=mode=='joint',cfg=cfg,resume=continuing)
            report['parent_outputs_preserved']=mode=='warmup'
            if mode=='joint':
                # The parent's receipt is preserved in the handoff. It does
                # not certify the changed style-conditioned leaf evaluator.
                model.core.utility_calibration=dict(ready=False,reward_id=digest(reward),
                    version='personality-utility-review/1.0.0',reason='Joint P1 update; recheck before P2 teaching',
                    training_error=report['after']['utility_mse'],parent=manifest['utility_calibration'])
            # Publish weights before the state pointer, keeping every committed
            # checkpoint recoverable if the report/state write is interrupted.
            model_path=directory/'model.pt'
            checkpoint.save(model_path,model,optimizer,training=dict(phase=mode,handoff=manifest['parent'],report=report))
            state.update(format=checkpoint.PILOT,contract=contract(),model_id=checkpoint.model_id(model),model_sha256=sha(model_path),
                model_path=model_path.relative_to(run).as_posix())
        # Report official outcomes separately from utility or auxiliary fit.
        seats=[(g,next(i for i,p in enumerate(g['opponents']) if p.startswith('personality-pilot/'))) for g in episodes] if games else []
        report['official']=({k:float(np.mean([terminal_targets(g['result'])[k][s] for g,s in seats])) for k in ('win_share','placement','vp')} if seats else None)
        if metadata:
            profile_seats={}
            for g,s in seats:
                profile=next(x['context']['profile'] for x in metadata['contexts'][g['game_id']] if x['role']=='learner')
                profile_seats.setdefault(profile,[]).append((g,s))
            report['by_profile']={p:dict(games=len(selected),**{k:float(np.mean([
                terminal_targets(g['result'])[k][s] for g,s in selected])) for k in ('win_share','placement','vp')})
                for p,selected in profile_seats.items()}
        if seats:
            components=[(replay_returns(g['records'],g['result'],reward)[1],s) for g,s in seats]
            report['reward_components']=dict(
                raw={k:float(np.mean([r['components'][k][s] for r,s in components])) for k in ('win','placement','vp')},
                weighted={k:float(np.mean([reward[k]*r['components'][k][s] for r,s in components])) for k in ('win','placement','vp')},
                cost=float(np.mean([r['cost'][s] for r,s in components])),utility=float(np.mean([r['utility'][s] for r,s in components])))
        report.update(games=len(episodes),split=partition,parameters=model.validate_parameter_count())
        write_json(directory/'report.json',report)
        state[key]+=1;state['reports']=(state['reports']+[dict(path=(directory/'report.json').relative_to(root).as_posix(),phase=mode)])[-8:]
        write_json(run/'state.json',state)
        if mode!='validate':
            # Compatibility convenience copy only. Readers above and in P2 use
            # the committed pointer even if this final copy is interrupted.
            temporary=run/'model.pt.tmp'
            try:
                shutil.copyfile(model_path,temporary);temporary.replace(run/'model.pt')
            finally:temporary.unlink(missing_ok=True)
    return report
