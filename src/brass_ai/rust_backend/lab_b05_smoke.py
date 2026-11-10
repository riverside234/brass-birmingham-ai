"""B05 bounded fixed-data updates in an isolated, fully restored lab fork."""
import copy
import gzip
import json
from pathlib import Path
from unittest.mock import patch

import torch

from brass_ai.learning.contracts import SOURCE_ROOT, require, digest
from brass_ai.learning.checkpoint import save_checkpoint, load_checkpoint, resume_optimizer
from brass_ai.learning.search_value import ordinary_parameters
from brass_ai.recovery.search_value_config import adopt
from brass_ai.recovery.search_value_iteration import SearchValueIteration, stamp
from brass_ai.step5.league import write_json
from . import lab_continuation as continuation
from .lab_snapshot import tree_digest, verify_unchanged


def ordinary_step(restored, config, output):
    """Actual PPO/PPG kernels, current recipe/Adams; small authored input only.

    Keep cached ages and all source dependencies. Bound only diagnostic row
    selection; do not widen KL limits or replace current losses/optimizers.
    """
    from brass_ai.recovery import updates, value_phase
    from test_ppg_eight import collect
    from test_local_scoring import annotate
    model,optimizer,phases=restored.model,restored.optimizer,restored.phases
    generation=restored.state['iteration']+1
    # The authored tensor factory runs on CPU. Bind its fixed actions to the
    # current behavior and compute their exact likelihoods on the learner device.
    behavior,data=collect(copy.deepcopy(model).cpu(),phases,generation,branches=True)
    from brass_ai.recovery.league import Behavior
    from brass_ai.learning.batching import model_id
    from brass_ai.recovery import entropy_schedule
    schedule=config['update']['entropy_schedule']
    progress=restored.state.get('exploration',{}).get('entropy_schedule',{}).get('progress',0)
    alphas=entropy_schedule.weights(schedule,progress,generation)
    receipt=dict(contract=schedule['contract'],config_id=digest(schedule),generation=generation,progress=progress,alphas=alphas)
    behavior=Behavior(model_id(model),config['exploration'],dict(version=3,opening=config['opening'],
        raw_entropy=True,alphas=alphas,entropy_schedule=receipt))
    for row,_,_,_ in data+phases.snapshot_data:
        row['behavior']=dict(kind='model',policy_id=('diagnostic-snapshot' if row.get('_critic_role')=='snapshot' else behavior.policy_id),probability=1.,temperature=1.,
            exploration=behavior.exploration,exploration_version=3,component='policy',action_decision=4)
        row['_ppo_opening_objective']=copy.deepcopy(config['update']['opening_objective'])
    phases.collection_source.update(policy_id=behavior.policy_id,model_id=behavior.evaluator.model_id)
    ds=updates.Dataset(data,next(model.parameters()).device)
    model.eval()
    with torch.no_grad():
        for part in ds.microbatches(range(len(data)),32,131072):
            ix,inputs,q=ds.batch(part,behavior.exploration)
            p=updates.row_probabilities(ds,ix,model(*inputs)[0],q,behavior)
            for j,i in enumerate(part):data[i][0]['behavior']['probability']=float(p[j,data[i][0]['chosen']])
    annotate(data);annotate(phases.snapshot_data)
    # Authored tensors exercise the optimizer/caches. They never claim to be
    # new Rust rollouts or qualify production strength.
    report=updates.ppo_update(model,optimizer,[dict(split='train')],behavior,config['update'],
        learner_data=data,insolvency=config['insolvency'],reward=model.config.utility,phases=phases)
    fit=value_phase.fit
    def bounded(*args,**kwargs):
        args=list(args);args[4]=list(args[4])[:16]
        kwargs['value_train_ids']=list(kwargs['value_train_ids'])[:32]+list(kwargs['value_train_ids'])[-32:]
        # Deduplicate if the current and previous slices overlap.
        kwargs['value_train_ids']=list(dict.fromkeys(kwargs['value_train_ids']))
        if kwargs.get('gate_ids') is not None:kwargs['gate_ids']=list(kwargs['gate_ids'])[:16]
        return fit(*args,**kwargs)
    with patch.object(value_phase,'fit',bounded):value=phases.fit(model)
    phases.finish(generation=generation,update=report,anchor_gate=None,rating=None)
    payload=copy.deepcopy(restored.raw['training'])
    payload['actor_value_phases']=phases.payload()
    if restored.companion is not None:payload['local_scoring_auxiliary']=restored.companion.payload(generation)
    path=output/'ordinary-next.pt'
    require(not path.exists() and not path.with_suffix('.pt.tmp').exists(),'Diagnostic checkpoint already exists')
    save_checkpoint(path,model,optimizer,step=generation,training=payload)
    loaded,raw=load_checkpoint(path,device=str(next(model.parameters()).device))
    adam=torch.optim.AdamW(ordinary_parameters(loaded));resume_optimizer(raw,adam)
    require(tree_digest(model.state_dict())==tree_digest(loaded.state_dict()) and
            tree_digest(optimizer.state_dict())==tree_digest(adam.state_dict()),'Ordinary next checkpoint differs')
    from brass_ai.recovery.actor_value import TrainingPhases
    resumed=TrainingPhases(loaded,config,raw=raw,generation=generation,root=phases.root,run=phases.run)
    saved=phases.payload()
    for key,actual in [('options',resumed.options),('gae',resumed.state),('beta_clone',resumed.beta),
                       ('kl_trips',resumed.trips),('optimizer',resumed.optimizer.state_dict()),
                       ('rolling_buffer',resumed.buffer.receipts),('critic_data',resumed.critic_state)]:
        require(tree_digest(saved[key])==tree_digest(actual),'Next PPG state/ages/Adam differs: '+key)
    if restored.companion is not None:
        from brass_ai.recovery.local_scoring_auxiliary import Companion
        companion=Companion(loaded,config['local_scoring_auxiliary'],raw=raw,generation=generation)
        require(tree_digest(companion.payload(generation))==tree_digest(payload['local_scoring_auxiliary']),
                'Next local-scoring weights/Adams/counters differ')
    return dict(ppo_steps=report['optimizer_steps'],ppg_steps=value['optimizer_steps'],
        attempted_ppg_steps=value['attempted_steps'],generation=generation,checkpoint_restored=True,
        current_recipe=digest(config),ages=[r['generation'] for r in resumed.buffer.receipts],
        row_bounds=dict(actor=len(data),critic_train=64,critic_validation=16),
        scope='Isolated current-checkpoint PPO/PPG and G-to-G+1 component commit; authored inputs, no new games')


def search_commit(model, config, tables, jobs, output, generation):
    # Start from the pinned unmodified model, not the ordinary diagnostic update.
    c=adopt(config);c['teacher_phase']['mode']='on';c['teacher_phase']['search_value']['stage']='A'
    lifecycle=SearchValueIteration(output,c,dict(iteration=generation),model)
    lifecycle.admit(tables,jobs)
    lifecycle.prepare_ppg(None,generation+1)
    ordinary={n:tree_digest(p) for n,p in model.state_dict().items() if not n.startswith('search_value_head.')}
    lifecycle.finish_ppg()
    require(lifecycle.phase.companion.steps > 0,'No admitted stage-A head update')
    require(ordinary=={n:tree_digest(p) for n,p in model.state_dict().items() if not n.startswith('search_value_head.')},
            'Stage A changed ordinary parameters')
    optimizer=torch.optim.AdamW(ordinary_parameters(model))
    payload=lifecycle.payload(generation+1)
    save_checkpoint(output/'search-next.pt',model,optimizer,step=generation+1,training=dict(search_value=payload))
    loaded,raw=load_checkpoint(output/'search-next.pt',device=str(next(model.parameters()).device))
    state=dict(iteration=generation+1,search_value=stamp(payload));lifecycle.committed(state)
    restored=SearchValueIteration(output,c,state,loaded,raw)
    require(tree_digest(restored.phase.payload())==tree_digest(lifecycle.phase.payload()),'Search commit/restore differs')
    return dict(head_steps=restored.phase.companion.steps,checkpoint_restored=True,
        value_rows=sum(r['rows'] for r in restored.phase.buffer.receipts),ordinary_parameters_unchanged=True,
        scope='Frozen B04.5 labels and isolated stage-A fixture; no production readiness authorization')


def replay_equivalence(reference):
    from . import lab_admission as admission
    from .lab_b03_smoke import compare_preparation
    from brass_ai.bank import sha
    reports=[]
    for entry in reference['families'][:3]:
        record=entry['receipt'];path=Path(record['path'])
        require(path.is_file() and sha(path)==record['sha256'],'Restore the original B03 sealed replay: '+str(path))
        game=admission.read(path,record['native_admission'],allow_diagnostic=True)
        with gzip.open(path,'rt',encoding='utf-8') as f:_,family=map(json.loads,f.readlines())
        # Preserve each historical producer/model/reward binding. Only compare
        # the two readers; never rebrand these g570 games as g1250 collection.
        old=record['native_admission']['admission']
        rows=compare_preparation(game,family,reference['config'],old['job']['provenance']['learner_behavior']['policy_id'])
        reports.append(dict(path=str(path),rows=len(rows),equal=True,original_generation=old['generation']))
    require(reports,'Missing fixed ordinary family evidence')
    return dict(families=reports,new_games=0,scope='Original B03 reference/native fixed-family preparation; original lineage retained')


def task_teacher_step(device):
    """Use the real ordinary actor backward pass as the teacher-share baseline."""
    from test_b025 import PPO
    from brass_ai.recovery.updates import ppo_update
    fixture=PPO();model,behavior,game,config,term=fixture.fixture('on')
    try:
        model.to(device)
        optimizer=torch.optim.AdamW(ordinary_parameters(model),lr=1e-5)
        report=ppo_update(model,optimizer,[game],behavior,config,teacher_term=term)
        # Once beta is fixed, actor norms are sampled on one step in four.
        measured=[s for s in term.steps if s['actor_norm'] is not None]
        require(measured and all(s['actor_norm']>0 and s['raw_share'] is not None for s in measured),
                'Teacher did not measure the actual ordinary actor gradient')
        require(len(term.steps)==report['optimizer_steps'],'Teacher added an optimizer step')
        return dict(passed=True,device=str(device),optimizer_steps=report['optimizer_steps'],
            teacher=term.report(),scope='Authored task-only PPO/teacher kernel, not production enablement')
    finally:fixture.doCleanups()


def run(args, reference):
    root=SOURCE_ROOT;out=args.out
    incoming=continuation.preflight(root,args.run,generation=reference['generation'],
        checkpoint_sha256=reference['checkpoint_sha256'],devices=[args.device])
    require(incoming['snapshot']['model_id']==reference['model_id'] and
            incoming['snapshot']['config_sha256']==reference['config_sha256'],
            'B05 model/config differs from the qualified g1250 reference')
    write_json(out/'continuation-preflight.json',incoming)
    try:
        fork=continuation.create_fork(root,out/'continuation',incoming)
        restored=continuation.restore(fork,device=args.device,allow_cuda_topology_change=args.allow_cuda_topology_change)
        write_json(out/'restoration.json',restored.report)
        write_json(out/'fixed-families.json',replay_equivalence(reference['ordinary']))
        ordinary=ordinary_step(restored,restored.state['signature']['config'],out)
        write_json(out/'ordinary-update.json',ordinary)
        model,_=load_checkpoint(incoming['snapshot']['checkpoint']['path'],device=args.device)
        inputs=root/'distribution/rust-lab-b05-g1250/evidence'
        tables=json.loads(gzip.decompress((inputs/'tables.json.gz').read_bytes()))
        jobs=json.loads((inputs/'jobs.json').read_text())['jobs']
        search=search_commit(model,restored.state['signature']['config'],tables,jobs,out/'search',reference['generation'])
        write_json(out/'search-update.json',search)
        teacher=task_teacher_step(args.device)
        write_json(out/'teacher-update.json',teacher)
        return dict(passed=True,lab_qualification='Passed for pinned g1250 fixed-data diagnostic scope',
            ordinary_update=ordinary,search_update=search,teacher_update=teacher,production_modified=False,new_games=0)
    finally:
        continuation.verify_dependencies(root,incoming);verify_unchanged(incoming['snapshot'])
