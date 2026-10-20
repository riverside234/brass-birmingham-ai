"""Explicit checkpoint extension and bounded, head-only return calibration."""
import copy
from dataclasses import replace
import torch

from brass_ai.learning.contracts import digest,require
from brass_ai.learning.reward import specification
from brass_ai.learning.utility_model import UtilityNet,utility_config

MIGRATION='four-seat-utility-extension/1.0.0'


def reward_spec(config):return specification(config['reward'],config['insolvency']) if 'reward' in config else None


def cap_upgrade(previous,current):
    if not isinstance(previous,dict) or not isinstance(current,dict):return False
    expected=copy.deepcopy(previous)
    if expected.get('insolvency')!=dict(event_cost=.005,max_cost=.025):return False
    expected['insolvency']=dict(event_cost=.005,max_cost=.075)
    return expected==current


def event_cost_upgrade(previous,current):
    if not isinstance(previous,dict) or not isinstance(current,dict):return False
    expected=copy.deepcopy(previous)
    if expected.get('insolvency')!=dict(event_cost=.005,max_cost=.075):return False
    expected['insolvency']=dict(event_cost=.0075,max_cost=.075)
    return expected==current


def migrate(model,optimizer,spec):
    if spec is None:return None
    if getattr(model.config,'utility',None) is not None:
        if model.config.utility==spec:return None
        event_upgrade=event_cost_upgrade(model.config.utility,spec)
        require(getattr(model,'_reviewed_cap_upgrade',False) and (cap_upgrade(model.config.utility,spec) or event_upgrade),
                'Checkpoint reward differs; explicit reward migration required')
        before=model.config.to_dict()
        model.config=replace(model.config,utility=copy.deepcopy(spec))
        model.utility_calibration=None;model._reviewed_cap_upgrade=False
        return dict(version='insolvency-event-cost/1.0.0' if event_upgrade else 'insolvency-cap-extension/1.0.0',from_config=before,to_config=model.config.to_dict(),
                    preserved='All weights, Adam moments/steps and RNG; fresh ordinary utility calibration required')
    before=model.config.to_dict();device=next(model.parameters()).device
    # Do not perturb the restored rollout/PPO RNG or compatible tensors/moments.
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(17041)
        model.config=utility_config(model.config,copy.deepcopy(spec));model.__class__=UtilityNet;model.add_utility_head()
    model.utility_head.to(device);model.utility_calibration=None
    require(len(optimizer.param_groups)==1,'Utility migration expects the existing single Adam group')
    optimizer.param_groups[0]['params'].extend(model.utility_head.parameters())
    model.validate_parameter_count()
    return dict(version=MIGRATION,from_config=before,to_config=model.config.to_dict(),
                preserved='All compatible weights, Adam moments/steps and RNG; new head moments start empty')


def calibration_ready(model):
    receipt=model.utility_calibration
    return bool(receipt and receipt.get('reward_id')==digest(model.config.utility) and receipt.get('ready'))


def calibrate(model,optimizer,dataset,config,*,auxiliary=None,indices=None):
    from .updates import value_output
    if calibration_ready(model):return dict(required=False,receipt=model.utility_calibration)
    require(dataset.utility is not None,'Calibration requires fresh complete-game utility targets')
    indices=list(range(len(dataset.data))) if indices is None else list(indices)
    if not indices:
        return dict(required=True,reason='No critic-train rows; calibration deferred',receipt=model.utility_calibration)
    # Fixed three head-only passes. No extra games, retries or policy/value drift.
    # Keep the existing Adam state for every compatible parameter untouched.
    parameters=list(model.parameters());flags=[p.requires_grad for p in parameters]
    rates=[g['lr'] for g in optimizer.param_groups]
    for p in parameters:p.requires_grad_(False)
    for p in model.utility_head.parameters():p.requires_grad_(True)
    for g in optimizer.param_groups:g['lr']=1e-3
    model.eval();steps=0
    teacher=auxiliary.teacher if auxiliary is not None and auxiliary.seen else None
    fraction=auxiliary.options['teacher_fraction'] if teacher is not None else 0.
    teacher_seconds=0.;teacher_presentations=0;teacher_steps=0
    def teacher_error():
        predictions=torch.empty_like(teacher.utility)
        with torch.no_grad():
            for part in teacher.microbatches(range(len(teacher.data)),config['microbatch_size'],config['max_candidate_cells'],config['bucket']):
                ix,inputs,_=teacher.value_batch(part,model)
                predictions[ix]=value_output(model,inputs,teacher,ix)[-1].float()
        diff=predictions-teacher.utility
        return dict(mse=float(diff.square().mean()),mae=float(diff.abs().mean())),predictions
    def error():
        squared=absolute=0.
        with torch.no_grad():
            for part in dataset.microbatches(indices,config['microbatch_size'],config['max_candidate_cells'],config['bucket']):
                ix,inputs,_=dataset.value_batch(part,model)
                diff=value_output(model,inputs,dataset,ix)[-1].float()-dataset.utility[ix]
                squared+=float(diff.square().sum());absolute+=float(diff.abs().sum())
        n=4*len(indices)
        return dict(mse=squared/n,mae=absolute/n)
    try:
        before=error()
        ordinary_before=before
        teacher_before=teacher_after=None
        if teacher is not None:
            import time
            began=time.perf_counter();teacher_before,_=teacher_error();teacher_seconds+=time.perf_counter()-began
            before={k:(1-fraction)*before[k]+fraction*teacher_before[k] for k in before}
        for _ in range(3):
            order=[indices[j] for j in torch.randperm(len(indices)).tolist()]
            for start in range(0,len(order),config['effective_batch_size']):
                batch=order[start:start+config['effective_batch_size']];optimizer.zero_grad(set_to_none=True)
                for part in dataset.microbatches(batch,config['microbatch_size'],config['max_candidate_cells'],config['bucket']):
                    ix,inputs,_=dataset.value_batch(part,model)
                    diff=value_output(model,inputs,dataset,ix)[-1].float()-dataset.utility[ix]
                    ((1-fraction)*(diff.square().mean(-1)*dataset.critic_weights[ix]).sum()/len(batch)).backward()
                if teacher is not None:
                    began=time.perf_counter()
                    # A bounded whole teacher set alongside each already
                    # scheduled ordinary calibration step: no new steps and
                    # exactly the same mixture used by the auxiliary loss.
                    for part in teacher.microbatches(range(len(teacher.data)),config['microbatch_size'],config['max_candidate_cells'],config['bucket']):
                        ix,inputs,_=teacher.value_batch(part,model)
                        diff=value_output(model,inputs,teacher,ix)[-1].float()-teacher.utility[ix]
                        (fraction*diff.square().mean(-1).sum()/len(teacher.data)).backward()
                    teacher_presentations+=len(teacher.data);teacher_steps+=1
                    auxiliary.calibration_seen.update(range(len(teacher.data)))
                    if teacher.device.type=='cuda':torch.cuda.synchronize(teacher.device)
                    teacher_seconds+=time.perf_counter()-began
                torch.nn.utils.clip_grad_norm_(model.utility_head.parameters(),config['grad_clip'],error_if_nonfinite=True)
                optimizer.step();steps+=1
        after=error()
        ordinary_after=after
        if teacher is not None:
            began=time.perf_counter();teacher_after,predictions=teacher_error();teacher_seconds+=time.perf_counter()-began
            after={k:(1-fraction)*after[k]+fraction*teacher_after[k] for k in after}
            if auxiliary.probe:auxiliary.probe.after_calibration(predictions)
    finally:
        optimizer.zero_grad(set_to_none=True)
        for p,flag in zip(parameters,flags):p.requires_grad_(flag)
        for group,lr in zip(optimizer.param_groups,rates):group['lr']=lr
    require(all(torch.isfinite(p).all() for p in model.utility_head.parameters()),'Nonfinite calibrated utility')
    # Numerical and fit gate; more fresh ordinary data can improve a poor fit.
    ready=after['mse']<=before['mse']+1e-8
    if teacher is not None:
        # A better weighted fit must not waive the existing ordinary-return
        # readiness gate. A failed ordinary fit still defers later searches.
        ready=ready and ordinary_after['mse']<=ordinary_before['mse']+1e-8
    model.utility_calibration=dict(version=MIGRATION,reward_id=digest(model.config.utility),ready=ready,
        decisions=len(indices),games=len({dataset.data[i][2][0] for i in indices}),steps=steps,
        before=before,after=after,scope='Fresh ordinary trajectories, in-sample calibration; not a strength guarantee')
    if teacher is not None:
        model.utility_calibration.update(version='teacher-mixed-utility-calibration/1.0.0',
            scope='Fresh ordinary + recorded mixed-search/direct teacher outcomes; weighted in-sample fit, not playing strength',
            readiness_gate='Both ordinary MSE and weighted mixture MSE must not increase',
            ordinary_fraction=1-fraction,teacher_fraction=fraction,teacher_decisions=len(teacher.data),
            teacher_games=auxiliary.games,teacher_target_source=auxiliary.binding,
            ordinary_before=ordinary_before,ordinary_after=ordinary_after,teacher_before=teacher_before,teacher_after=teacher_after,
            teacher_seconds=teacher_seconds,teacher_sample_presentations=teacher_presentations,teacher_optimizer_steps=teacher_steps)
    return dict(required=True,receipt=model.utility_calibration)
