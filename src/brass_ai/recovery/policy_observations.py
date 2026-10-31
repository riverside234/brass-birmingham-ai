"""Explicit self-play retirement of style fingerprints and the anchor KL gate.

Keep the 34-column critic interface and legacy checkpoint evidence. No tensors,
optimizer moments, progress, fixed probe positions or economic baselines change.
"""
import copy
import torch
from brass_ai.learning.contracts import require

KEY='policy_observations'
CONTRACT='selfplay-policy-observations/1.0.0'
MIGRATION='neutral-fingerprints-economic-lr/1.0.0'
DEFAULT=dict(contract=CONTRACT,seat_fingerprints='neutral',anchor_kl='off',fixed_probe='pool_admission_only')


def enabled(config):return config.get(KEY)==DEFAULT


def configure(config):
    from .lr_controller import AUTO_CONTRACT,ECONOMIC_CONTRACT
    result=copy.deepcopy(config)
    require(result['ppo_lr_controller']['contract'] in (AUTO_CONTRACT,ECONOMIC_CONTRACT),
            'Economic-only LR adoption requires the calibrated/economic controller')
    require(KEY not in result or result[KEY]==DEFAULT,'Unknown policy observation settings')
    result[KEY]=copy.deepcopy(DEFAULT)
    result['ppo_lr_controller'].update(contract=ECONOMIC_CONTRACT,anchor_kl_budget=None,anchor_group_kl_budget=None)
    return result


def validate(config):
    from .lr_controller import ECONOMIC_CONTRACT
    require(config.get(KEY)==DEFAULT and config['ppo_lr_controller']['contract']==ECONOMIC_CONTRACT,
            'Neutral fingerprints require explicit economic-only LR settings')
    require('critic_data' in config and 'il_kl_limit' in config and 'opponent_pool' in config,
            'Economic-only continuation requires existing critic, health and pool provenance')
    require(config.get('entropy_control',{}).get('mode')=='off',
            'Retiring the fixed entropy diagnostic requires fixed entropy weights')


def _neutral_descriptor_inputs(module,args):
    values=args[0]
    require(values.shape[-1]==34,'Neutral fingerprint descriptor dimension differs')
    # Preserve role and age, including descriptor dropout/unknown semantics.
    # Apply at the encoder boundary for BOTH fresh and previously cached rows.
    return (torch.cat((values[...,:6],torch.zeros_like(values[...,6:])),dim=-1),*args[1:])


def install(model,config):
    if not enabled(config):return
    validate(config)
    if getattr(model,'_neutral_seat_fingerprints',False):return
    require(hasattr(model,'descriptor_encoder'),'Missing critic descriptor encoder')
    model.descriptor_encoder.register_forward_pre_hook(_neutral_descriptor_inputs)
    model._neutral_seat_fingerprints=True


def neutral_fingerprint():
    return dict(contract='neutral-seat-fingerprint/1.0.0',values=[0.]*28,source='disabled')


def advance_health(state,counts,*,generation,warmup_done):
    """Advance economic evidence only; old KL fields are archived, never read."""
    from . import il_kl_limit as old
    result=copy.deepcopy(old.validate_state(state,state['config']))
    require(generation==result['last_generation']+1,'Economic health skipped/repeated a generation')
    c=result['config']
    observation={k:counts[k] for k in old.COUNTS} if old.rates([counts]) is not None else None
    result['rolling']=(result['rolling']+[observation])[-c['rolling_iterations']:]
    if warmup_done and len(result['baseline_rows'])<c['baseline_iterations']:
        result['baseline_rows'].append(copy.deepcopy(observation))
        if len(result['baseline_rows'])==c['baseline_iterations']:
            result['baseline']=old.baseline_record(result['baseline_rows'],c)
    result['last_generation']=generation
    current=old.rates(result['rolling']) if len(result['rolling'])==c['rolling_iterations'] else None
    evidence=old.health(old.baseline_evidence(result['baseline']),current)
    gate=dict(ok=evidence['ok'],reasons=list(evidence['reasons']),health=evidence,
        anchor_distance=None,anchor_kl_enabled=False,scope='Economic health only; AnchorKL measurement, calibration and gating disabled')
    old.validate_state(result,c,generation)
    return result,dict(contract=CONTRACT,generation=generation,anchor_kl_enabled=False,
        measurement=None,decision='disabled',reason='anchor_kl_retired',health=evidence,lr_gate=gate,
        baseline=copy.deepcopy(result['baseline']),baseline_iterations=len(result['baseline_rows']),
        scope='Fresh committed ordinary-game economics; archived KL values and recovery latch never participate')
