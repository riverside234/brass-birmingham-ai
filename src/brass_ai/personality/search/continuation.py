"""Copy a distilled learner into a new compatible P1 pilot; source stays frozen."""
import copy
import shutil
import torch
from brass_ai.bank import sha
from brass_ai.learning.contracts import project_path,read_json,require
from brass_ai.personality import checkpoint
from brass_ai.personality.contracts import contract
from brass_ai.recovery.retention import owned,run_lock
from brass_ai.step5.league import write_json
from .contracts import identity
from .readiness import load


def continue_pilot(root,pilot,distilled,output):
    root,out=owned(root,output);pilot=project_path(root,pilot);distilled=project_path(root,distilled)
    require(not out.exists() and not out.is_relative_to(pilot),'Use a new successor pilot directory')
    with run_lock(pilot):
        state=read_json(pilot/'state.json');report=read_json(distilled/'report.json')
        pilot_model=checkpoint.committed_path(pilot,state)
        require(report['updated'] and report['contract']==identity(),'Distillation source differs')
        _,_,ready,_,_=load(root,report['parent_ready'])
        require(report['parent_ready_id']==ready['id'] and ready['origin']['pilot']==pilot.relative_to(root).as_posix()
            and ready['origin']['pilot_model_sha256']==state['model_sha256'],
            'PPO source changed since this distillation; do not overwrite a newer learner')
        require(report['handoff_sha256']==state['handoff_sha256'] and sha(distilled/'model.pt')==report['model_sha256'],'Distilled lineage/weights differ')
        model,new_raw=checkpoint.load(distilled/'model.pt');_,old_raw=checkpoint.load(pilot_model)
        require(not model.core.utility_calibration['ready'],'Distilled model must await fresh utility checking')
        optimizer=None;phase='p2-distilled'
        if old_raw['training']['phase']=='joint' and old_raw['optimizer'] is not None:
            # Same named tensor layout. Preserve PPO Adam moments; the separate
            # policy-only teacher optimizer is not substituted for PPO Adam.
            model.requires_grad_(True);optimizer=torch.optim.AdamW(model.parameters())
            optimizer.load_state_dict(old_raw['optimizer']);phase='joint'
        out.mkdir(parents=True);shutil.copyfile(pilot/'initial.pt',out/'initial.pt')
        torch.set_rng_state(old_raw['rng'])
        if old_raw['cuda_rng'] and torch.cuda.is_available():
            require(len(old_raw['cuda_rng'])==torch.cuda.device_count(),'CUDA topology changed on P2 continuation')
            torch.cuda.set_rng_state_all(old_raw['cuda_rng'])
        model_id=checkpoint.save(out/'model.pt',model,optimizer,training=dict(phase=phase,p2_distillation=report,
            scope='Fresh-rollout successor; old PPO moments retained when compatible'))
        new_state=copy.deepcopy(state);new_state.update(format=checkpoint.PILOT,contract=contract(),model_id=model_id,model_sha256=sha(out/'model.pt'),model_path='model.pt',
            p2_parent=dict(pilot=pilot.relative_to(root).as_posix(),sha256=state['model_sha256'],teacher=report['teacher_id']))
        write_json(out/'state.json',new_state)
        write_json(out/'pins.json',dict(paths=[(out/'model.pt').relative_to(root).as_posix(),(out/'initial.pt').relative_to(root).as_posix(),
            (pilot/'state.json').relative_to(root).as_posix(),(distilled/'report.json').relative_to(root).as_posix(),report['parent_ready']+'/ready.json']))
        return dict(output=str(out),model_id=model_id,ppo_optimizer_preserved=optimizer is not None,
            joint_runs_preserved=state['joint_runs'],search_ready=False,next='Collect fresh joint-pilot rollouts, then prepare again before teaching')
