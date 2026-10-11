"""Reserved teacher collection/accounting and separate policy-only distillation."""
from pathlib import Path
import time
import numpy as np
import torch

from brass_ai.bank import sha
from brass_ai.learning.contracts import digest,project_path,read_json,require
from brass_ai.personality import checkpoint
from brass_ai.personality.contracts import PROFILES,context_vector
from brass_ai.personality.training import forward,parts,DEFAULTS
from brass_ai.recovery.encoding import PreparedEncoder
from brass_ai.recovery.retention import owned,run_lock
from brass_ai.recovery.teacher_schedule import PHASES,phase_for,search_allowance
from brass_ai.step5.league import seeds_for,write_json
from .contracts import Profile,identity,search_config,controller
from .readiness import load
from .collection import collect,read

CADENCE=160
WARMUP=640
SEARCH_GAMES=4
WALL_FRACTION=.9
DISTILL_STEPS=8


def ordinary_receipts(root,pilot,state):
    """Only committed fresh P1 PPO collections count; never eval/search games."""
    from brass_ai.personality.traces import read_collection
    root,pilot=owned(root,pilot);pstate=read_json(pilot/'state.json')
    for report in sorted(pilot.glob('joint-*/report.json')):
        key=report.relative_to(root).as_posix()
        if key in state['ordinary']:
            require(state['ordinary'][key]==sha(report),'Previously counted ordinary report changed');continue
        payload=read_json(report)
        require(payload['phase']=='joint-fresh-ppo' and payload['split']=='train','Not ordinary learner training')
        # A report left behind before the state commit is not eligible.
        number=int(report.parent.name.rsplit('-',1)[1]);require(number<pstate['joint_runs'],'Uncommitted PPO iteration')
        _,meta,games=read_collection(report.parent/'collection',reward=payload['reward'])
        require(len(games)==payload['games'],'Ordinary count differs')
        state['ordinary'][key]=sha(report);state['ordinary_games']+=len(games);state['ordinary_seconds']+=meta['seconds']
        state['latest_iteration']=key


def due(state,seconds):
    if state['ordinary_games']<state['next_games']:return 'not_due'
    if state['iteration_attempts'].get(state['latest_iteration'],0)>=2:return 'iteration_collection_limit'
    estimate=SEARCH_GAMES*(4*4*seconds+state['ordinary_seconds']/max(1,state['ordinary_games']))
    if estimate>search_allowance(state['ordinary_seconds'],state['search_seconds'],WALL_FRACTION):return 'wall_time_deferred'
    return None


def collect_due(root,bundle,pilot,run,*,tables=4,device='cpu',mono=None):
    root,run=owned(root,run);model,parents,ready,_,pool=load(root,bundle)
    from brass_ai.learning.batching import model_id as base_id
    with run_lock(run):
        pilot=project_path(root,pilot)
        require(ready['origin']['pilot']==pilot.relative_to(root).as_posix()
            and ready['origin']['pilot_model_sha256']==read_json(pilot/'state.json')['model_sha256'],
            'Learner changed since preparation; prepare the current learner before teaching')
        path=run/'state.json'
        state=read_json(path) if path.exists() else dict(format='personality-teacher-schedule/1.0.0',contract=identity(),ordinary={},
            ordinary_games=0,ordinary_seconds=0.,search_seconds=0.,next_games=WARMUP,iteration_attempts={},latest_iteration='',attempts=[])
        require(state['contract']==identity(),'P2 teacher schedule source changed')
        ordinary_receipts(root,pilot,state)
        config=search_config(ready['reward']);reason=due(state,config.seconds)
        if reason:
            write_json(path,state);return dict(collected=False,reason=reason,ordinary_games=state['ordinary_games'],next_games=state['next_games'])
        number=len(state['attempts']);directory=run/f'collection-{number:04d}'
        seeds,_=seeds_for(73000000+state['next_games'],SEARCH_GAMES,'train')
        jobs=[]
        for i,seed in enumerate(seeds):
            profiles=[Profile(style=PROFILES[(i+s)%len(PROFILES)]) for s in range(4)]
            controllers=[controller(ready['model']['model_id'],base_id(parents[0]),p,config,pool)['candidate_id'] for p in profiles]
            jobs.append(dict(index=i,seed=seed,sampling_seed=seed*7,partition='train',neural_seats=list(range(4)),
                profiles=[p.to_dict() for p in profiles],controllers=controllers,parent_indices=[0]*4))
        reserve=SEARCH_GAMES*(16*config.seconds+state['ordinary_seconds']/max(1,state['ordinary_games']))
        attempt=dict(path=directory.relative_to(root).as_posix(),status='started',charged_seconds=reserve,model_id=ready['model']['model_id'])
        state['next_games']+=CADENCE;state['search_seconds']+=reserve
        key=state['latest_iteration'];state['iteration_attempts'][key]=state['iteration_attempts'].get(key,0)+1
        state['attempts'].append(attempt);write_json(path,state);start=time.monotonic()
        try:
            report=collect(root,bundle,directory,jobs,config,tables=tables,device=device,mono=mono,teacher_game=True)
            attempt.update(status='complete',manifest_sha256=sha(directory/'manifest.json'))
        except BaseException:
            attempt['status']='failed';raise
        finally:
            elapsed=time.monotonic()-start
            state['search_seconds']+=elapsed-reserve;attempt['charged_seconds']=elapsed;write_json(path,state)
        return dict(collected=True,path=str(directory),seconds=report['seconds'],schedule=state)


def distill(root,bundle,collection,output,*,device='cpu'):
    root,out=owned(root,output);collection=project_path(root,collection)
    require(not out.exists(),'Use a new distilled model directory')
    model,_,ready,_,_=load(root,bundle,device)
    manifest,pairs=read(collection,teacher_game=True)
    require(manifest['spec']['ready_id']==ready['id'] and manifest['spec']['model_id']==ready['model']['model_id'],'Teacher is stale for this learner/readiness')
    phases={p:0 for p,_,_,_ in PHASES};setups=set();data=[];encoder=PreparedEncoder()
    for game,extra in pairs:
        require(game['split']=='train','Held-out teacher cannot train')
        for row,x in zip(game['records'],extra['records']):
            label=x['teacher']
            if label is None:continue
            phase=phase_for(row['view']['context']);require(phase in phases,'Teacher phase missing')
            require(row['view']['context']['round']==extra['reservations'][row['seat']][phase],'Teacher outside its reserved round')
            phases[phase]+=1;setups.add(game['setup_group'])
            data.append(dict(row=row,encoded=encoder.encode(row['view'],row['choices'],row['style'],row['intent']),
                personality=context_vector(label['personality']),remaining=label['remaining'],target=label['target']['probabilities']))
    quality=dict(samples=len(data),setups=len(setups),by_phase=phases)
    if len(data)<16 or len(setups)<2 or any(n<2 for n in phases.values()):return dict(updated=False,reason='insufficient_phase_coverage',quality=quality)
    with run_lock(collection):
        usage=collection/'distillation-use.json';require(not usage.exists(),'Teacher collection already presented; no unlimited reuse')
        # Claim before gradients. A crash cannot grant another reuse allowance.
        write_json(usage,dict(model=ready['model']['model_id'],output=out.relative_to(root).as_posix(),max_presentations=2,status='started'))
        model.requires_grad_(False)
        for module in (model.core.state_encoder,model.core.action_encoder,model.core.policy_scorer,model.context_projection):module.requires_grad_(True)
        frozen={k:v.detach().cpu().clone() for k,v in model.state_dict().items() if k.startswith(('core.utility_head','core.outcomes','goal_','confidence_'))}
        optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-5,weight_decay=.0001)
        steps=0;seen=[0]*len(data);losses=[]
        for _ in range(2):
            order=torch.randperm(len(data)).tolist()
            for start in range(0,len(order),16):
                if steps>=DISTILL_STEPS:break
                ix=order[start:start+16];optimizer.zero_grad(set_to_none=True)
                for part in parts(data,ix,DEFAULTS):
                    result=forward(model,data,part,device);logp=torch.log_softmax(result['policy_logits'].float(),-1)
                    target=torch.zeros_like(logp)
                    for j,i in enumerate(part):target[j,:len(data[i]['target'])]=torch.tensor(data[i]['target'],device=device);seen[i]+=1
                    loss=-(target*logp.masked_fill(~torch.isfinite(logp),0.)).sum()/len(ix)
                    require(torch.isfinite(loss).item(),'Nonfinite distillation');loss.backward();losses.append(float(loss.detach()))
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],5.,error_if_nonfinite=True);optimizer.step();steps+=1
        require(max(seen)<=2 and all(torch.equal(v,model.state_dict()[k].cpu()) for k,v in frozen.items()),'Distillation changed value/goal heads or over-reused labels')
        model.core.utility_calibration=dict(ready=False,reward_id=digest(ready['reward']),reason='Shared trunk distilled; recalibrate before new teachers')
        out.mkdir(parents=True);ident=checkpoint.save(out/'model.pt',model,optimizer,training=dict(phase='p2-policy-distillation',parent=ready['model'],teacher=manifest['id']))
        report=dict(updated=True,contract=identity(),handoff_sha256=ready['handoff_sha256'],parent_ready_id=ready['id'],
            parent_ready=project_path(root,bundle).relative_to(root).as_posix(),
            model_id=ident,model_sha256=sha(out/'model.pt'),teacher_id=manifest['id'],quality=quality,
            steps=steps,max_presentations=max(seen),mean_batch_loss=float(np.mean(losses)),search_ready=False,value_loss_training=False)
        write_json(out/'report.json',report);write_json(usage,dict(status='complete',**report))
        write_json(out/'pins.json',dict(paths=[(project_path(root,bundle)/'ready.json').relative_to(root).as_posix(),
            (collection/'manifest.json').relative_to(root).as_posix(),(out/'model.pt').relative_to(root).as_posix()]))
    return report
