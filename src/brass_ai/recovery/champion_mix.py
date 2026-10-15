"""Frozen champion exposure with a stable IL anchor and a hard two-seat cap."""
import copy
from collections import Counter

from brass_ai.bank import sha
from brass_ai.learning.contracts import digest,project_path,require
from .execution import neural_spec
from .population import ordered

CONTRACT='staged-champion-opponents/1.0.0'


def enabled(config):
    from .config import latest_model_recipe
    return not latest_model_recipe(config) and 'champion_opponents' in config


def promoted(state,previous,*,reason='promotion'):
    config=state['signature']['config']
    # Persist both quantities: changing collection size on a later resume must
    # neither stretch the original stages nor backdate the first introduction.
    state['champion_mix']=dict(contract=CONTRACT,champion_model_id=state['champion']['model_id'],
        origin_games=state['games_done'],stage_games=config['champion_opponents']['stage_iterations']*config['games_per_iteration'],
        transition_tables=0,anchor=copy.deepcopy(state['evaluation_anchor']),
        previous=copy.deepcopy(previous),reason=reason)


def initialize(root,run,state):
    if not enabled(state['signature']['config']):return False
    changed='champion_mix' not in state
    if changed:
        gate=state.get('champion_gate',{}).get('accepted_gate') or {}
        generation=gate.get('incumbent_generation',state['evaluation_anchor']['generation'])
        candidates=state.get('population',[])+[state['evaluation_anchor']]+state['history']
        previous=next((e for e in candidates if e['generation']==generation and
                       project_path(root,e['path']).is_file()),None)
        # Explicit fallback when the former incumbent predates bounded retention.
        previous=previous or state['evaluation_anchor']
        from .retention import slim_checkpoint
        previous=slim_checkpoint(root,run,previous,previous if previous.get('weights_only') else None)
        promoted(state,previous,reason='feature_introduction')
    mix=state['champion_mix']
    require(mix['contract']==CONTRACT and mix['champion_model_id']==state['champion']['model_id'] and
            mix['anchor']['model_id']==state['evaluation_anchor']['model_id'] and
            type(mix['origin_games']) is int and 0<=mix['origin_games']<=state['games_done'] and
            type(mix['stage_games']) is int and mix['stage_games']>0 and
            type(mix['transition_tables']) is int and mix['transition_tables']>=0,
            'Champion rollout does not match committed progress/anchor')
    for entry in (mix['anchor'],mix['previous']):
        path=project_path(root,entry['path'])
        require(path.is_relative_to(run) and sha(path)==entry['sha256'],'Champion rollout checkpoint changed')
    return changed


def status(state,config):
    mix=state['champion_mix'];elapsed=state['games_done']-mix['origin_games']
    require(elapsed>=0,'Champion rollout predates committed progress')
    if state['champion']['model_id']==mix['anchor']['model_id']:stage='bootstrap'
    elif config['champion_opponents']['max_champion_opponents']==1:stage='one_champion_override'
    elif elapsed<mix['stage_games']:stage='one_champion'
    elif elapsed<2*mix['stage_games']:stage='balanced_transition'
    else:stage='two_champions'
    return dict(contract=CONTRACT,stage=stage,origin_games=mix['origin_games'],
        stage_games=mix['stage_games'],elapsed_ordinary_games=elapsed,anchor=copy.deepcopy(mix['anchor']),
        max_champion_opponents=config['champion_opponents']['max_champion_opponents'])


def opponents(state,config,rng,*,transition_index=0,current_id=None):
    """None slots mean frozen legacy fallbacks, only before a distinct champion.

    The initial champion IS the IL anchor: no distinct older neural policy exists.
    Use one copy and two admitted frozen legacy policies, never three copies of
    the same champion disguised by seat roles. Promoted runs use neural opponents.
    """
    info=status(state,config);mix=state['champion_mix'];champion=state['champion'];anchor=mix['anchor']
    if info['stage']=='bootstrap':entries=[champion,None,None]
    else:
        # One of each per consecutive pair, with a deterministic shuffled order
        # so the two-champion tables do not always share a rotation parity.
        parity=int(digest(dict(champion=champion['model_id'],origin=mix['origin_games'],pair=transition_index//2))[:8],16)%2
        two=info['stage']=='two_champions' or (info['stage']=='balanced_transition' and (transition_index%2)^parity==1)
        if two:entries=[champion,champion,anchor]
        else:
            available=ordered(state.get('population',[]))
            recent={e['model_id'] for e in available[-config['population']['recent']:]}
            distinct=[e for e in available if e['model_id'] not in (champion['model_id'],anchor['model_id'],current_id)]
            previous=mix['previous']
            if previous['model_id'] not in (champion['model_id'],anchor['model_id'],current_id):older=previous
            else:older=rng.choice([e for e in distinct if e['model_id'] not in recent] or distinct or [anchor])
            entries=[champion,older,anchor]
    count=sum(e is not None and e['model_id']==champion['model_id'] for e in entries)
    require(1<=count<=config['champion_opponents']['max_champion_opponents'],'Champion opponent seat limit exceeded')
    return entries,dict(stage=info['stage'],champion_opponents=count,
        transition_index=transition_index if info['stage']=='balanced_transition' else None,
        generations=[e['generation'] if e else None for e in entries],
        model_ids=[e['model_id'] if e else None for e in entries],
        duplicate_older=(entries[1] is not None and entries[1]['model_id']==anchor['model_id'] and count==1),
        bootstrap_legacy_slots=sum(e is None for e in entries))


def policies(learner,entries,bootstrap_policies=()):
    fallback=iter(bootstrap_policies)
    require(len(bootstrap_policies)>=sum(e is None for e in entries),'Initial champion needs two frozen legacy opponents')
    return [learner]+[neural_spec(e['model_id'],.05,'neutral') if e else next(fallback) for e in entries]


def summarize(state,config,receipts):
    return dict(status(state,config),tables=len(receipts),
        tables_by_champion_count={str(n):sum(r['champion_opponents']==n for r in receipts) for n in (1,2)},
        generation_seats=dict(Counter(g for r in receipts for g in r['generations'] if g is not None)),
        transition_tables_before=state['champion_mix']['transition_tables'],
        transition_tables_added=sum(r['stage']=='balanced_transition' for r in receipts),
        duplicate_older_tables=sum(r['duplicate_older'] for r in receipts),
        bootstrap_legacy_slots=sum(r['bootstrap_legacy_slots'] for r in receipts),
        scope='Actual frozen opponent seats; only the learner supplies PPO data; bootstrap uses frozen legacy fallbacks')
