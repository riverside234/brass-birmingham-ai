"""Explicit checkpoint extension and bounded, head-only return calibration."""
import copy
import torch

from brass_ai.learning.contracts import digest,require
from brass_ai.learning.reward import specification
from brass_ai.learning.utility_model import UtilityNet,utility_config

MIGRATION='four-seat-utility-extension/1.0.0'


def reward_spec(config):return specification(config['reward'],config['insolvency']) if 'reward' in config else None


def migrate(model,optimizer,spec):
    if spec is None:return None
    if getattr(model.config,'utility',None) is not None:
        require(model.config.utility==spec,'Checkpoint reward differs; explicit reward migration required')
        return None
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


def calibrate(model,optimizer,dataset,config):
    if calibration_ready(model):return dict(required=False,receipt=model.utility_calibration)
    require(dataset.utility is not None,'Calibration requires fresh complete-game utility targets')
    # Fixed three head-only passes. No extra games, retries or policy/value drift.
    # Keep the existing Adam state for every compatible parameter untouched.
    parameters=list(model.parameters());flags=[p.requires_grad for p in parameters]
    rates=[g['lr'] for g in optimizer.param_groups]
    for p in parameters:p.requires_grad_(False)
    for p in model.utility_head.parameters():p.requires_grad_(True)
    for g in optimizer.param_groups:g['lr']=1e-3
    model.eval();steps=0
    def error():
        squared=absolute=0.
        with torch.no_grad():
            for part in dataset.microbatches(range(len(dataset.data)),config['microbatch_size'],config['max_candidate_cells'],config['bucket']):
                ix,inputs,_=dataset.batch(part)
                diff=model(*inputs,remaining=dataset.remaining[ix])[-1].float()-dataset.utility[ix]
                squared+=float(diff.square().sum());absolute+=float(diff.abs().sum())
        n=4*len(dataset.data)
        return dict(mse=squared/n,mae=absolute/n)
    before=error()
    try:
        for _ in range(3):
            order=torch.randperm(len(dataset.data)).tolist()
            for start in range(0,len(order),config['effective_batch_size']):
                batch=order[start:start+config['effective_batch_size']];optimizer.zero_grad(set_to_none=True)
                for part in dataset.microbatches(batch,config['microbatch_size'],config['max_candidate_cells'],config['bucket']):
                    ix,inputs,_=dataset.batch(part)
                    diff=model(*inputs,remaining=dataset.remaining[ix])[-1].float()-dataset.utility[ix]
                    (diff.square().mean(-1).sum()/len(batch)).backward()
                torch.nn.utils.clip_grad_norm_(model.utility_head.parameters(),config['grad_clip'],error_if_nonfinite=True)
                optimizer.step();steps+=1
        after=error()
    finally:
        optimizer.zero_grad(set_to_none=True)
        for p,flag in zip(parameters,flags):p.requires_grad_(flag)
        for group,lr in zip(optimizer.param_groups,rates):group['lr']=lr
    require(all(torch.isfinite(p).all() for p in model.utility_head.parameters()),'Nonfinite calibrated utility')
    # Numerical and fit gate; more fresh ordinary data can improve a poor fit.
    ready=after['mse']<=before['mse']+1e-8
    model.utility_calibration=dict(version=MIGRATION,reward_id=digest(model.config.utility),ready=ready,
        decisions=len(dataset.data),games=len({d[2][0] for d in dataset.data}),steps=steps,
        before=before,after=after,scope='Fresh ordinary trajectories, in-sample calibration; not a strength guarantee')
    return dict(required=True,receipt=model.utility_calibration)
