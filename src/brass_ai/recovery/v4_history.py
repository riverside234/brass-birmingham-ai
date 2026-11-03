"""Move the frozen v4 training ensemble into PFSP after committed table wins.

The rule-bot member has no neural checkpoint/probe and never supplies critic
rows. Its five admitted variants share one PFSP score, cycled by actual seats.
"""
import copy
import math
from brass_ai.bank import read
from brass_ai.learning.contracts import digest,require

KEY='v4_history'
CONTRACT='v4-history-gate/1.0.0'
MIGRATION='v4-history-pool-adoption/1.0.0'
DEFAULT=dict(contract=CONTRACT,window_iterations=10,win_share=1.,
    source='ordinary-two-learner-v4-tables',released_tables='selfplay',
    initial_sampling='match-il',manual_generation=None)


def validate(c):
    require(type(c) is dict and dict(c,manual_generation=None)==DEFAULT and
        (c['manual_generation'] is None or type(c['manual_generation']) is int and c['manual_generation']>=0),
        'Invalid v4 history gate')
    return c


def bot(member):return member.get('kind')=='v4-history'


def active(pool):return bool(pool and pool.get(KEY,{}).get('activated'))


def policies(root,config):
    from .legacy_search import legacy_spec
    profiles=sorted(read(root/'training/opponents/pool.json')['training'])
    result={p:legacy_spec(root,p,config['legacy_budget'],search_free=True,
        work_budget=config['legacy_opponent_work_budget']) for p in profiles}
    validate_policies(result)
    return result


def validate_policies(values):
    require(type(values) is dict and len(values)==5 and all(
        p.startswith('bank-3dc65253a610191d66a5/') and 'pottery-flex' not in p and
        v.get('search_free') is True and v.get('policy_id') for p,v in values.items()),
        'V4 PFSP requires the five frozen search-free training variants')


def validate_state(pool):
    gate=pool.get(KEY);members=[m for m in pool['members'] if bot(m)]
    if gate is None:
        require(not members,'V4 PFSP member without admission receipt');return
    require(set(gate)=={'config','origin','window','activated','policies'},'Invalid v4 history state')
    c=validate(gate['config']);validate_policies(gate['policies'])
    require(type(gate['origin']) is int and 0<=gate['origin']<=pool['generation'],'Invalid v4 gate origin')
    window=gate['window'];require(isinstance(window,list) and len(window)<=c['window_iterations'],'Invalid v4 gate window')
    previous=None
    for row in window:
        require(set(row)=={'generation','games','win_share_sum'} and type(row['generation']) is int and
            gate['origin']<row['generation']<=pool['generation'] and (previous is None or row['generation']==previous+1)
            and type(row['games']) is int and row['games']>0 and type(row['win_share_sum']) in (int,float)
            and math.isfinite(row['win_share_sum']) and 0<=row['win_share_sum']<=row['games'],
            'Invalid committed ordinary v4 evidence')
        previous=row['generation']
    activation=gate['activated']
    require(len(members)==int(activation is not None),'V4 activation/member mismatch')
    if activation is None:return
    require(activation['source'] in ('manual_override','measured_ordinary_tables') and
        gate['origin']<=activation['generation']<=pool['generation'] and activation['games_done']<=pool['games_done'],
        'Invalid v4 activation progress')
    if activation['source']=='manual_override':
        require(activation['generation']==c['manual_generation'] and activation.get('authority')=='user'
            and activation.get('measured_win_share') is None,'Manual v4 move is not measured qualification')
    else:
        require(len(window)==10 and all(r['win_share_sum']==r['games'] for r in window) and
            window[-1]['generation']==activation['generation'],'V4 automatic gate needs ten perfect committed iterations')
    member=members[0]
    require(member['model_id']=='v4-history-'+digest(gate['policies']) and member['generation']==activation['generation']
        and member['admitted_iteration']==activation['generation'] and member['policies']==gate['policies']
        and type(member['profile_seats']) is int and member['profile_seats']>=0
        and not {'path','sha256','probe','imported_from'}&set(member), 'Invalid frozen v4 PFSP member')


def activate(pool,*,source):
    from .pfsp_pool import seal,distribution
    gate=pool[KEY]
    require(gate['activated'] is None,'V4 history already activated')
    # This recipe has explicitly retired IL panels at startup. Do not quietly
    # change the IL gate in order to manufacture equal probabilities.
    require((pool.get('planning') or {}).get('il_gate') is not None,'V4 PFSP needs the existing IL sampling qualification')
    il=next(m for m in pool['members'] if m['generation']==0 and not bot(m))
    record=dict(source=source,generation=pool['generation'],games_done=pool['games_done'],
        initial_S=il['S'],initial_C=il['C'],measured_win_share=None if source=='manual_override' else 1.)
    if source=='manual_override':record.update(authority='user',reason='Explicit immediate move at continuation; no qualification games')
    gate['activated']=record
    pool['members'].append(dict(kind='v4-history',model_id='v4-history-'+digest(gate['policies']),
        generation=pool['generation'],admitted_iteration=pool['generation'],S=il['S'],C=il['C'],alarm_streak=0,
        policies=copy.deepcopy(gate['policies']),profile_seats=0,admission_health=None))
    seal(pool);plan=distribution(pool)
    record['initial_probability']=next(r['p'] for r in plan['members'] if r['model_id']==pool['members'][-1]['model_id'])
    require(record['initial_probability']==plan['il_probability'],'V4 initial probability differs from IL')
    seal(pool)


def migrate(pool,config,values):
    """Pure adoption; publish together with the reviewed source transition."""
    from .quality_pool import state_digest
    from .pfsp_pool import seal
    c=validate(config[KEY]);validate_policies(values);updated=copy.deepcopy(pool)
    if KEY in updated:
        require(updated[KEY]['config']==c and updated[KEY]['policies']==values,'V4 history settings changed')
        validate_state(updated);return updated,None
    updated[KEY]=dict(config=copy.deepcopy(c),origin=pool['generation'],window=[],activated=None,policies=copy.deepcopy(values))
    if c['manual_generation'] is not None:
        require(c['manual_generation']==pool['generation'],'Stale manual v4 history generation')
        activate(updated,source='manual_override')
    seal(updated);validate_state(updated)
    return updated,dict(contract=MIGRATION,from_digest=state_digest(pool),to_digest=state_digest(updated),
        config=copy.deepcopy(c),policies=copy.deepcopy(values),activation=copy.deepcopy(updated[KEY]['activated']))


def advance(pool,games,kinds,learner_policy_id):
    """Call on the next pool before its checkpoint commits; never mid-collection."""
    from .pfsp_pool import seal
    from brass_ai.learning.replay import terminal_targets
    if KEY not in pool:return None
    gate=pool[KEY]
    if not active(pool):
        wins=[]
        for game in games:
            if kinds[game['setup_group']]!='legacy':continue
            require('branch' not in game,'Branch cannot qualify v4 retirement')
            seats=[s for s,p in enumerate(game['opponents']) if p==learner_policy_id]
            require(len(seats)==2,'V4 retirement requires two learner seats')
            win=terminal_targets(game['result'])['win_share'];wins.append(sum(win[s] for s in seats))
        if wins:
            gate['window']=(gate['window']+[dict(generation=pool['generation'],games=len(wins),win_share_sum=math.fsum(wins))])[-10:]
        else:gate['window']=[]
        if len(gate['window'])==10 and all(r['win_share_sum']==r['games'] for r in gate['window']):
            activate(pool,source='measured_ordinary_tables')
    seal(pool);validate_state(pool)
    return report(pool)


def report(pool):
    if KEY not in pool:return None
    gate=pool[KEY];plan=pool['next_planning'];member=next((m for m in pool['members'] if bot(m)),None)
    return dict(contract=CONTRACT,activated=copy.deepcopy(gate['activated']),window=copy.deepcopy(gate['window']),
        next_table_counts=dict(selfplay=166,historical=154,legacy=0) if active(pool) else None,
        probability=next((r['p'] for r in plan['members'] if member and r['model_id']==member['model_id']),None),
        profile_seats=member['profile_seats'] if member else 0,
        legacy_checks='retired' if active(pool) else 'active',
        scope='Combined first-place win share of two learners on ordinary v4 tables. One-way scheduling gate; no evidence for LR/GAE/critic health. Rule-bot member uses PFSP; neural pool capacity unchanged.')
