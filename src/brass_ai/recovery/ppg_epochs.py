"""Deterministic phase-level epoch cap, persisted with value Adam and schedule."""
import copy

from brass_ai.learning.contracts import require


def initial(generation,config):
    return dict(contract=config['epoch_controller']['contract'],cap=config['max_epochs'],
        window=[],window_origin=0,phases=0,last_phase=generation)


def validate(state,config,*,phases=None,last_phase=None):
    gate=config['epoch_controller']
    require(type(state) is dict and set(state)==set(initial(0,config))
        and state['contract']==gate['contract']
        and all(type(state[k]) is int for k in ('cap','window_origin','phases','last_phase'))
        and gate['minimum']<=state['cap']<=gate['maximum']
        and 0<=state['window_origin']<=state['phases']
        and state['last_phase']>=0
        and (phases is None or state['phases']==phases)
        and (last_phase is None or state['last_phase']==last_phase),
        'Invalid persisted PPG epoch cap')
    rows=state['window'];n=min(gate['window_phases'],state['phases']-state['window_origin'])
    require(type(rows) is list and len(rows)==n,'Invalid PPG epoch evidence window')
    for i,row in enumerate(rows):
        require(type(row) is dict and set(row)=={'generation','cap','steps_per_epoch','planned_steps',
            'best_budget_step','late_best','early_best','kl_limited'}
            and all(type(row[k]) is int for k in ('generation','cap','steps_per_epoch','planned_steps','best_budget_step'))
            and all(type(row[k]) is bool for k in ('late_best','early_best','kl_limited'))
            and row['generation']==state['last_phase']-config['phase_interval']*(n-1-i)
            and row['cap']==state['cap'] and row['steps_per_epoch']>=0
            and row['planned_steps']==row['cap']*row['steps_per_epoch']
            and 0<=row['best_budget_step']<=row['planned_steps'],
            'Invalid PPG epoch evidence row')
    return state


def advance(state,report,config,generation):
    before=copy.deepcopy(validate(state,config));state=copy.deepcopy(before);gate=config['epoch_controller']
    require(report.get('scheduled') is True and report.get('epoch_cap_used')==state['cap']
        and generation==state['last_phase']+config['phase_interval'],
        'PPG epoch decision must follow the next scheduled phase')
    steps=report.get('steps_per_epoch',0);planned=report.get('planned_steps',0);best=report.get('best_budget_step',0)
    require(all(type(x) is int for x in (steps,planned,best)) and planned==steps*state['cap']
        and 0<=best<=planned,'Invalid PPG best-checkpoint budget position')
    limited=bool(report.get('kl_guard_tripped',False))
    # KL-limited phases request more cloning pressure, not a different epoch
    # budget. Keep them in the four-phase window without an up/down vote.
    late=bool(planned and best>0 and best>=gate['late_fraction']*planned and not limited)
    early=bool(planned and best<=steps and not limited)
    row=dict(generation=generation,cap=state['cap'],steps_per_epoch=steps,planned_steps=planned,
        best_budget_step=best,late_best=late,early_best=early,kl_limited=limited)
    state['window']=(state['window']+[row])[-gate['window_phases']:]
    evidence=copy.deepcopy(state['window']);enough=len(evidence)==gate['window_phases']
    late_count=sum(r['late_best'] for r in evidence);early_count=sum(r['early_best'] for r in evidence)
    state.update(phases=state['phases']+1,last_phase=generation)
    decision='hold';reason='window_incomplete' if not enough else 'insufficient_votes'
    if enough and late_count>=gate['qualifying_phases'] and state['cap']<gate['maximum']:
        state['cap']=min(gate['maximum'],state['cap']+1)
        decision='up';reason='best_in_final_quarter_in_three_of_four'
    elif enough and early_count>=gate['qualifying_phases'] and state['cap']>gate['minimum']:
        state['cap']=max(gate['minimum'],state['cap']-1)
        decision='down';reason='best_in_first_epoch_in_three_of_four'
    if decision!='hold':state.update(window=[],window_origin=state['phases'])
    validate(state,config)
    return state,dict(contract=gate['contract'],generation=generation,cap_used=before['cap'],
        cap_next=state['cap'],decision=decision,reason=reason,
        late_best_phases=late_count,early_best_phases=early_count,evidence=evidence,
        before=before,after=copy.deepcopy(state),
        scope='Best state position in the planned budget, excluding retry work. Three of four scheduled phases at the same cap; wait iterations are not evidence. Window resets after a cap change. KL-limited phases request beta pressure and cast no epoch vote. Bounds 1–6.')
