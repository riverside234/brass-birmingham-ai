"""Bounded fresh-data utility check and separate goal-confidence calibration."""
import copy
from pathlib import Path
import numpy as np
import torch

from brass_ai.bank import sha
from brass_ai.learning.contracts import digest,project_path,read_json,require
from brass_ai.learning.checkpoint import load_checkpoint
from brass_ai.personality import checkpoint,handoff
from brass_ai.personality.contracts import local_file
from brass_ai.personality.training import forward,parts,metrics,DEFAULTS
from brass_ai.recovery.retention import owned,run_lock
from brass_ai.step5.league import write_json
from .contracts import READY,identity
from .calibration import collect_examples


def confidence_calibration(logits,labels,groups):
    x=np.asarray(logits,dtype=float);y=np.asarray([d['confidence'] for d in labels]);mask=np.asarray([d['confidence_mask'] for d in labels],dtype=bool)
    require(x.shape==y.shape==mask.shape and np.isfinite(x).all(),'Goal calibration shape differs')
    temperatures=(.5,.75,1.,1.5,2.)
    def loss(t):
        p=1/(1+np.exp(-np.clip(x/t,-50,50)))
        return float(np.square(p-y)[mask].mean()) if mask.any() else 1.
    t=min(temperatures,key=lambda t:(loss(t),abs(t-1)))
    probability=1/(1+np.exp(-np.clip(x/t,-50,50)))
    from brass_ai.personality.labels import confidence_report
    calibration=confidence_report(probability,labels)
    return dict(temperature=t,brier=loss(t),ece=calibration['ece'],labels=int(mask.sum()),setups=len(set(groups)),
        bias_ready=len(set(groups))>=2 and mask.sum()>=64 and loss(t)<=.25 and calibration['ece']<=.10,
        scope='Post-hoc temperature on separate calibration setups; not release evidence')


def distilled_source(root,pilot,state,distilled,handoff_sha256):
    """Resolve only a teacher update of this exact, still-current P1 learner.

    Sharing a handoff is insufficient: several pilots (and later generations
    of one pilot) can have that parent while carrying different PPO weights and
    optimizer moments. Do not attribute an old teacher branch to a newer pilot.
    """
    distilled=project_path(root,distilled);report=read_json(distilled/'report.json')
    require(report.get('updated') and report['contract']==identity()
        and report['handoff_sha256']==handoff_sha256,'Unversioned/distinct-parent distillation continuation')
    _,_,parent_ready,_,_=load(root,report['parent_ready'])
    require(report['parent_ready_id']==parent_ready['id']
        and parent_ready['handoff_sha256']==handoff_sha256
        and parent_ready['origin']['pilot']==pilot.relative_to(root).as_posix()
        and parent_ready['origin']['pilot_model_sha256']==state['model_sha256'],
        'PPO source changed since this distillation; prepare the matching learner')
    path=distilled/'model.pt';require(sha(path)==report['model_sha256'],'Distilled model changed')
    return path


def prepare(root,bundle,pilot,output,*,games=20,calibration_games=8,workers=4,device='cpu',mono=None,distilled=None):
    root,out=owned(root,output);_,pilot=owned(root,pilot);bundle=project_path(root,bundle)
    require(not out.exists() and not out.is_relative_to(pilot),'Use a new separate P2 readiness directory')
    require(1<=games<=32 and 1<=calibration_games<=32,'Bounded readiness games required')
    with run_lock(pilot):
        manifest,parent=handoff.load(bundle);state=read_json(pilot/'state.json')
        require(state['warmup_runs']>0 and state['handoff_sha256']==sha(bundle/'manifest.json'),'Finish P1 warm-up with this handoff')
        model_path=checkpoint.committed_path(pilot,state)
        if distilled is not None:
            model_path=distilled_source(root,pilot,state,distilled,sha(bundle/'manifest.json'))
        model,raw=checkpoint.load(model_path,device);parent=parent.to(device)
        origin=dict(model_id=checkpoint.model_id(model),sha256=sha(model_path),path=model_path.relative_to(root).as_posix(),
            pilot=pilot.relative_to(root).as_posix(),pilot_model_sha256=state['model_sha256'])
        copied={k:v.detach().cpu().clone() for k,v in model.state_dict().items() if not k.startswith('core.utility_head.')}
        original_utility=copy.deepcopy(model.core.utility_head.state_dict())
        out.mkdir(parents=True)
        seed=61000000+int(origin['model_id'][:7],16)%1000000
        data=collect_examples(root,bundle,model,parent,manifest,out/'utility-data',games=games,workers=workers,device=device,mono=mono,seed=seed,partition='train')
        before=metrics(model,data,device);model.requires_grad_(False);model.core.utility_head.requires_grad_(True)
        optimizer=torch.optim.AdamW(model.core.utility_head.parameters(),lr=1e-3,weight_decay=0.)
        for _ in range(3):
            order=torch.randperm(len(data)).tolist()
            for start in range(0,len(order),2048):
                ix=order[start:start+2048];optimizer.zero_grad(set_to_none=True)
                for part in parts(data,ix,DEFAULTS):
                    outp=forward(model,data,part,device);target=torch.tensor([data[i]['utility'] for i in part],device=device)
                    ((outp['remaining_utility_4']-target).square().mean(-1).sum()/len(ix)).backward()
                torch.nn.utils.clip_grad_norm_(model.core.utility_head.parameters(),5.,error_if_nonfinite=True);optimizer.step()
        after=metrics(model,data,device)
        require(all(torch.equal(v,model.state_dict()[k].cpu()) for k,v in copied.items()),'Utility calibration moved policy/goal/official heads')
        model.core.utility_calibration=dict(version=READY,reward_id=digest(manifest['reward']),ready=after['utility_mse']<=before['utility_mse']+1e-8,
            before=before['utility_mse'],after=after['utility_mse'],origin=origin)
        held=collect_examples(root,bundle,model,parent,manifest,out/'goal-calibration',games=calibration_games,workers=workers,device=device,mono=mono,
            seed=seed+2000000,partition='validation')
        held_after=metrics(model,held,device);fitted_utility=copy.deepcopy(model.core.utility_head.state_dict())
        model.core.utility_head.load_state_dict(original_utility);held_before=metrics(model,held,device)
        model.core.utility_head.load_state_dict(fitted_utility)
        model.core.utility_calibration['ready'] &= held_after['utility_mse']<=1.10*held_before['utility_mse']+.005
        model.core.utility_calibration.update(heldout_before=held_before['utility_mse'],heldout_after=held_after['utility_mse'])
        logits=[]
        with torch.inference_mode():
            for part in parts(held,range(len(held)),DEFAULTS):logits.extend(forward(model,held,part,device)['confidence_logits'].cpu().tolist())
        goal=confidence_calibration(logits,[d['goals'] for d in held],[d['setup_group'] for d in held])
        ident=checkpoint.save(out/'model.pt',model,training=dict(phase='p2-readiness',origin=origin,utility=after,goal_calibration=goal))
        result=dict(format=READY,contract=identity(),model=dict(path='model.pt',sha256=sha(out/'model.pt'),model_id=ident),
            handoff=bundle.relative_to(root).as_posix(),handoff_sha256=sha(bundle/'manifest.json'),origin=origin,reward=manifest['reward'],
            utility_ready=model.core.utility_calibration['ready'],goal_calibration=goal,utility_before=before,utility_after=after,
            context_coverage=dict(training_saved_intent_rows=sum(d['saved_intent'] for d in data),validation_saved_intent_rows=sum(d['saved_intent'] for d in held)),
            heldout=held_after,heldout_before=held_before,data=[dict(path=p.relative_to(out).as_posix(),sha256=sha(p))
                for directory in ('utility-data','goal-calibration') for p in sorted((out/directory).glob('*')) if p.is_file()],
            scope='Numerical/fit readiness, not playing-strength or style qualification')
        result['id']=digest(result);write_json(out/'ready.json',result)
        write_json(out/'pins.json',dict(paths=[(out/'model.pt').relative_to(root).as_posix(),
            (bundle/'manifest.json').relative_to(root).as_posix()]+[(out/d['path']).relative_to(root).as_posix() for d in result['data']]))
    return result


def load(root,directory,device='cpu'):
    directory=project_path(root,directory);receipt=read_json(directory/'ready.json')
    require(receipt['format']==READY and receipt['contract']==identity() and receipt['id']==digest({k:v for k,v in receipt.items() if k!='id'}),'Readiness source/receipt changed')
    require(receipt['utility_ready'],'P2 utility fit failed; collect new calibration data')
    for data in receipt['data']:require(sha(local_file(directory,data['path']))==data['sha256'],'Readiness calibration data changed')
    path=local_file(directory,receipt['model']['path']);require(sha(path)==receipt['model']['sha256'],'Ready model changed')
    model,_=checkpoint.load(path,device);require(checkpoint.model_id(model)==receipt['model']['model_id'],'Ready model identity differs')
    bundle=project_path(root,receipt['handoff']);require(sha(bundle/'manifest.json')==receipt['handoff_sha256'],'Handoff changed')
    manifest,parent=handoff.load(bundle)
    require(model.core.config.utility==receipt['reward']==manifest['reward'] and model.core.utility_calibration['ready']
        and model.core.utility_calibration['reward_id']==digest(receipt['reward']),'Reward/calibration differs')
    pool=[parent]
    for ref in manifest['references'][:3]:
        m,_=load_checkpoint(local_file(bundle,ref['path']),device='cpu');pool.append(m)
    from brass_ai.learning.batching import model_id as base_id
    pool_id=digest([base_id(m) for m in pool])
    return model,pool,receipt,manifest,pool_id
