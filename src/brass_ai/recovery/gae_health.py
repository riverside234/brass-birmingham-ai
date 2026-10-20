"""Committed stage-health evidence, independent of mixed-table EV admission."""
import copy

from brass_ai.learning.contracts import require

CONTRACT='staged-gae-health-window/1.0.0'


def initial(generation):
    return dict(contract=CONTRACT,origin=generation,entries=[])


def validate(state,gate,generation):
    require(type(state) is dict and set(state)=={'contract','origin','entries'}
        and state['contract']==CONTRACT and type(state['origin']) is int
        and 0<=state['origin']<=generation,'Invalid GAE stage evidence')
    rows=state['entries'];start=max(state['origin']+1,generation-gate['window_iterations']+1)
    require(type(rows) is list and all(type(r) is dict for r in rows)
        and [r.get('generation') for r in rows]==list(range(start,generation+1)),
        'GAE stage evidence progress differs')
    for row in rows:
        require(set(row)=={'generation','healthy','accepted_steps'}
            and type(row['generation']) is int and type(row['healthy']) is bool
            and type(row['accepted_steps']) is int and row['accepted_steps']>=0,
            'Invalid GAE stage evidence row')
    return state


def advance(state,gate,*,generation,stage_origin,healthy,accepted_steps):
    state=copy.deepcopy(validate(state,gate,generation-1))
    require(type(healthy) is bool and type(accepted_steps) is int and accepted_steps>=0,
        'Invalid GAE stage measurements')
    state['entries']=(state['entries']+[dict(generation=generation,healthy=healthy,
        accepted_steps=accepted_steps)])[-gate['window_iterations']:]
    validate(state,gate,generation)
    rows=[r for r in state['entries'] if r['generation']>stage_origin]
    good=[r for r in rows if r['healthy']]
    steps=sum(r['accepted_steps'] for r in good)
    contributors=sum(r['accepted_steps']>0 for r in good)
    reasons=[]
    if len(rows)<gate['window_iterations']:reasons.append('observation_window_incomplete')
    if len(good)!=len(rows):reasons.append('unhealthy_iteration_in_window')
    if steps<gate['minimum_accepted_steps']:reasons.append('accepted_steps_insufficient')
    if contributors<gate['minimum_value_iterations_in_window']:reasons.append('accepted_iterations_insufficient')
    return state,dict(contract=CONTRACT,origin=state['origin'],window_start=rows[0]['generation'] if rows else None,
        window_end=generation,observed_iterations=len(rows),required_iterations=gate['window_iterations'],
        healthy_iterations=len(good),accepted_steps=steps,
        required_accepted_steps=gate['minimum_accepted_steps'],accepted_iterations=contributors,
        required_accepted_iterations=gate['minimum_value_iterations_in_window'],ready=not reasons,reasons=reasons,
        scope='Current lambda stage, committed rolling window; accepted PPG work from healthy iterations only; zero accepted steps alone do not reset health')
