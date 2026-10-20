"""Explicit, once-only g110 IL-KL adjustment; never a fresh-run default."""
import copy

import torch

from brass_ai.bank import read
from brass_ai.learning.contracts import digest,project_path,require
from brass_ai.step5.league import implementation_id,write_json
from .il_kl_limit import validate_state

CONTRACT='joint7-g110-il-kl-x1.5/1.0.0'
KEY='il_kl_g110_hotfix'
MODEL='6efcf125c85ae05a156e43e74346a5721aff441304721bfd5a4e4045f1b8dd0b'


def adjusted(previous):
    require(isinstance(previous,dict),'Missing committed IL-KL state')
    result=copy.deepcopy(validate_state(previous,previous['config'],110))
    result['limit']*=1.5
    # Keep cadence, health evidence, fixed probe and recovery latch intact.
    return validate_state(result,result['config'],110)


def prepare(state):
    result=copy.deepcopy(state)
    if KEY in state:
        receipt=state[KEY]
        require(receipt['contract']==CONTRACT and receipt['checkpoint']['model_id']==MODEL,
                'Unrecognized g110 IL-KL hotfix receipt')
        return result,False
    require(state['iteration']==110 and state['history'][-1]['generation']==110 and
            state['history'][-1]['model_id']==MODEL,'This one-time hotfix is pinned to the reviewed joint7 generation 110')
    old=state['il_kl_limit'];new=adjusted(old);factor=old['config']['group_factor']
    result['il_kl_limit']=new
    result[KEY]=dict(contract=CONTRACT,generation=110,checkpoint=copy.deepcopy(state['history'][-1]),
        before_digest=digest(old),after_digest=digest(new),limit_before=old['limit'],limit_after=new['limit'],
        group_limit_before=factor*old['limit'],group_limit_after=factor*new['limit'],
        reason='User-authorized one-time recovery after removed IL sale-probability health lock',
        scope='Limit only; model, both Adams, RNG, progress, probe, health/cadence, latch, population and lineage retained')
    return result,True


def verify_checkpoint(state,raw):
    """Allow only the exact recorded override until the next checkpoint commits."""
    saved=raw['training'].get('il_kl_limit');current=state['il_kl_limit']
    if saved==current:return None
    receipt=state.get(KEY,{})
    require(receipt.get('contract')==CONTRACT and receipt.get('generation')==110 and
            state['iteration']==raw['step']==110 and raw['model_id']==MODEL and
            receipt.get('checkpoint')==state['history'][-1],
            'IL-KL state differs from committed checkpoint without the exact g110 hotfix')
    expected=adjusted(saved)
    require(digest(saved)==receipt['before_digest'] and digest(expected)==receipt['after_digest'] and
            current==expected and receipt['limit_before']==saved['limit'] and receipt['limit_after']==expected['limit'] and
            receipt['group_limit_before']==2*saved['limit'] and receipt['group_limit_after']==2*expected['limit'],
            'G110 IL-KL hotfix differs from its committed checkpoint')
    return copy.deepcopy(receipt)


def run(root,run_path,output,*,apply=False,refine_ppg=False):
    from .retention import owned,run_lock,safe
    from .resume import load_resume
    from .config import insolvency_event_config,validate_config
    root,run=owned(root,run_path);destination=project_path(root,output)
    require(destination.is_relative_to(root/'configs/step5') and destination.suffix=='.json',
            'Hotfix config output must be a JSON under configs/step5')
    require((run/'state.json').is_file(),'Missing existing run state')
    with run_lock(run):
        state=read(run/'state.json');updated,changed=prepare(state)
        config=insolvency_event_config(state['signature']['config'])
        if refine_ppg:
            from .value_phase import refined_options
            config=refined_options(config)
        config=validate_config(config)
        require(config['insolvency']==dict(event_cost=.0075,max_cost=.075),'Unexpected insolvency recipe')
        if destination.exists():require(read(destination)==config,'Config output already contains different settings')
        signature=dict(state['signature'],config=config,implementation=implementation_id())
        # Verify the real checkpoint and reviewed config/source transition,
        # including pending-collection and frozen-opponent checks, before writes.
        with torch.random.fork_rng(devices=[]):
            _,raw,_=load_resume(root,run,state,signature,reviewed=True)
        verify_checkpoint(state,raw)
        verify_checkpoint(updated,raw)
        report=dict(applied=apply,already_applied=not changed,iteration=state['iteration'],
            games_done=state['games_done'],config=destination.relative_to(root).as_posix(),
            hotfix=updated[KEY],insolvency=config['insolvency'],value_phase=config['value_phase'])
        if apply:
            if not destination.exists():write_json(destination,config)
            if changed:
                backup=safe(run,run/'state-before-ilkl-g110-x1.5.json')
                if backup.exists():require(read(backup)==state,'Hotfix backup differs')
                else:write_json(backup,state)
                require(read(run/'state.json')==state,'Run advanced during IL-KL hotfix')
                write_json(run/'state.json',updated)
        return report
