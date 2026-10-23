"""Versioned, deterministic exploration rollout; no model or optimizer reset."""
import copy
import math
import statistics

from brass_ai.learning.contracts import digest, require
from brass_ai.step5.policy import FAMILY_WEIGHTS

GROUPS = ('canal_action', 'canal_other', 'rail_action', 'rail_other')
Q = dict(contract='family-weighted-exploration/3.0.0', action_family_weights=FAMILY_WEIGHTS,
    unknown_action_kind_weight=1, pass_only_when_alone=True, industry_subgroups=['build','develop'],
    focus_multiplier=2., other_phases='v2', snapshot_seats='match_learner')
LEGACY_ENTROPY = dict(contract='entropy-floor-controller/1.0.0', mode='apply',
    measure='raw-policy-fixed-il-probe', groups=list(GROUPS), alpha_base=.005, alpha_max=.05,
    floor_start_ratio=.9, floor_decay_per_iteration=.999, floor_min_ratio=.6,
    gain=4., max_step=1.5, relax_factor=1.25, hysteresis=1.03, median_window=3, baseline_iterations=10,
    deployment_temperature_check=dict(every_iterations=50, temperatures=[1.,.7]))
DECAY_ENTROPY = dict(LEGACY_ENTROPY, deployment_temperature_check=dict(enabled=False,every_iterations=50,temperatures=[1.,.7]))
UNIFORM_ENTROPY = dict(DECAY_ENTROPY, contract='entropy-floor-controller/2.0.0',
    floor_start_ratio=.95, floor_decay_per_iteration=1., floor_min_ratio=.95)
ENTROPY = dict(UNIFORM_ENTROPY, contract='entropy-floor-controller/3.0.0',
    floor_ratio_by_group=dict(canal_action=.95,canal_other=.90,rail_action=.95,rail_other=.90))
ANCHOR_ENTROPY = dict({k:copy.deepcopy(v) for k,v in ENTROPY.items()
    if k not in ('floor_start_ratio','floor_decay_per_iteration','floor_min_ratio')},
    contract='entropy-floor-controller/4.0.0',floor_drop_per_anchor_promotion=.02,
    floor_min_ratio_by_group=dict(canal_action=.85,canal_other=.80,rail_action=.85,rail_other=.80),
    promotion_origin=dict(generation=0,event_count=0,events_sha256=digest([])))
LEGACY_OPENING = dict(contract='opening-temperature/1.0.0', temperature=1.5, action_decisions_per_seat=2, seats='neural')
OPENING = dict(LEGACY_OPENING, contract='opening-temperature/2.0.0', temperature=2.)
LEGACY_BRANCH = dict(contract='selfplay-branching/1.0.0', tables=['selfplay'], rate_per_decision=.005,
    max_per_game=1, skip_opening_decisions=2, skip_final_rounds='rail_empty_deck_hand_le_4',
    move_distribution=dict(temperature_1=.7, temperature_2=.25, exploration_q=.05),
    exclude_played_move=True, reshuffle_hidden_deck=True, branch_move_row='excluded')
HIGH_BRANCH = dict(LEGACY_BRANCH, contract='selfplay-branching/2.0.0', rate_per_decision=.015,
    reset_opening_at_fork=True)
BRANCH = dict(HIGH_BRANCH, rate_per_decision=.005)
LOCAL_SCORING_BRANCH = dict(HIGH_BRANCH,rate_per_decision=.010)
ROLLOUT = dict(contract='exploration-rollout/1.0.0', mode='fresh',
    opening_after_iterations=0, raw_entropy_after_iterations=0, branching_after_iterations=20)
BLOCKS = ('exploration_distribution','entropy_control','opening','branching','exploration_rollout')
MIGRATION = 'staged-exploration/1.0.0'
FROZEN_MIGRATION = 'frozen-entropy-branch-openings/1.0.0'
GROUP_FLOOR_MIGRATION = 'group-entropy-floors-low-branching/1.0.0'
ANCHOR_FLOOR_MIGRATION = 'anchor-promotion-entropy-floors/1.0.0'
HIGH_BRANCH_MIGRATION = 'anchor-entropy-high-branching/1.0.0'
LOW_BRANCH_MIGRATION = 'anchor-entropy-half-percent-branching/1.0.0'
STATE = 'exploration-state/1.0.0'
ANCHOR_STATE = 'exploration-state/2.0.0'


def configure(config, staged=False):
    result=copy.deepcopy(config)
    result.update(exploration=.02, exploration_distribution=copy.deepcopy(Q),
        entropy_control=copy.deepcopy(ANCHOR_ENTROPY), opening=copy.deepcopy(OPENING),
        branching=copy.deepcopy(BRANCH), exploration_rollout=copy.deepcopy(ROLLOUT))
    if staged:
        result['exploration_rollout'].update(mode='staged',opening_after_iterations=1,raw_entropy_after_iterations=2)
    return result


def continuation_config(config):
    """Exact opt-in migration, never restart an already adopted rollout."""
    if any(b in config for b in BLOCKS):
        validate(config)
        return copy.deepcopy(config)
    require(config.get('selfplay_recipe')=='latest-model-selfplay/3.0.0' and config['exploration']==.08,
        'Staged exploration migration requires the existing .08 quality-pool recipe')
    result=configure(config,staged=True)
    # Reproduce the old signed migration exactly. Panel retirement is a
    # separate reviewed transition; fresh generation uses the new default.
    result['entropy_control']=copy.deepcopy(LEGACY_ENTROPY)
    result['opening']=copy.deepcopy(LEGACY_OPENING)
    result['branching']=copy.deepcopy(LEGACY_BRANCH)
    result['exploration']=.03  # Preserve the earlier signed staged-adoption recipe.
    return result


def frozen_config(config):
    """Explicit continuation: retain the baseline, controller state and rollout age."""
    validate(config)
    # Older source-review routes may precede exploration adoption. This change
    # does not adopt/restart their rollout or mask a later applicable route.
    if not any(b in config for b in BLOCKS):return copy.deepcopy(config)
    result=copy.deepcopy(config)
    # Keep this older signed migration exact, including its 1.5% fork rate.
    result.update(entropy_control=dict(copy.deepcopy(UNIFORM_ENTROPY),mode=config['entropy_control']['mode']),
                  opening=copy.deepcopy(OPENING),branching=copy.deepcopy(HIGH_BRANCH))
    validate(result)
    return result


def group_floor_config(config):
    """Change only group floors and fork rate; preserve opening and rollout."""
    validate(config)
    if not any(b in config for b in BLOCKS):return copy.deepcopy(config)
    # A legacy opening needs its separate, explicit T=2 adoption first.
    if config['opening']!=OPENING:return copy.deepcopy(config)
    result=copy.deepcopy(config)
    result.update(entropy_control=dict(copy.deepcopy(ENTROPY),mode=config['entropy_control']['mode']),
                  branching=copy.deepcopy(BRANCH))
    validate(result)
    return result


def anchor_floor_config(config, state=None):
    """Bind prospective promotion counting into the reviewed run signature."""
    validate(config)
    result=copy.deepcopy(config)
    if not any(b in config for b in BLOCKS):return result
    if config['opening']!=OPENING or config['branching']!=BRANCH:return result
    if config['entropy_control']['contract']==ANCHOR_ENTROPY['contract']:return result
    result['entropy_control']=dict(copy.deepcopy(ANCHOR_ENTROPY),mode=config['entropy_control']['mode'])
    if state is not None:
        events=(state.get('policy_anchor') or {}).get('events',[])
        result['entropy_control']['promotion_origin']=dict(generation=state['iteration'],
            event_count=len(events),events_sha256=digest(events))
    validate(result)
    return result


def high_branch_config(config):
    """Change only the fork probability; never reinitialize entropy or its origin."""
    validate(config)
    result=copy.deepcopy(config)
    if config.get('entropy_control',{}).get('contract')==ANCHOR_ENTROPY['contract']:
        result['branching']=copy.deepcopy(HIGH_BRANCH)
    validate(result)
    return result


def low_branch_config(config):
    """Reduce only fork probability; retain entropy evidence and fork openings."""
    validate(config)
    require(config.get('entropy_control',{}).get('contract')==ANCHOR_ENTROPY['contract'] and
        config.get('branching',{}).get('contract')==BRANCH['contract'],
        'Half-percent branching requires the current anchor-entropy recipe')
    result=copy.deepcopy(config)
    result['branching']['rate_per_decision']=.005
    validate(result)
    return result


def validate(config):
    present=[b in config for b in BLOCKS]
    if not any(present): return
    require(all(present), 'Exploration revision requires all versioned blocks')
    # This initial recipe is deliberately narrow; experiments need explicit
    # reviewed signatures rather than unbounded free-form controller options.
    current=config['opening']==OPENING and config['branching'] in (BRANCH,LOCAL_SCORING_BRANCH)
    uniform=config['opening']==OPENING and config['branching']==HIGH_BRANCH
    legacy=config['opening']==LEGACY_OPENING and config['branching']==LEGACY_BRANCH
    require(type(config['exploration']) in (int,float) and config['exploration'] in (0.,.02,.03) and config['exploration_distribution']==Q and
        (current or uniform or legacy), 'Invalid exploration revision recipe')
    e=config['entropy_control']
    require(type(e) is dict,'Invalid entropy floor controller')
    check=e.get('deployment_temperature_check',{})
    require('enabled' not in check or check['enabled'] is False,'Temperature comparisons must be explicitly disabled')
    recipes=(ENTROPY,ANCHOR_ENTROPY) if current else (UNIFORM_ENTROPY,ANCHOR_ENTROPY) if uniform else (DECAY_ENTROPY,LEGACY_ENTROPY)
    comparable=dict(e,mode='apply')
    if e.get('contract')==ANCHOR_ENTROPY['contract']:
        origin=e.get('promotion_origin')
        require(type(origin) is dict and set(origin)=={'generation','event_count','events_sha256'} and
            all(type(origin[k]) is int and origin[k]>=0 for k in ('generation','event_count')) and
            type(origin['events_sha256']) is str and len(origin['events_sha256'])==64 and
            all(c in '0123456789abcdef' for c in origin['events_sha256']) and
            (origin['event_count']!=0 or origin['events_sha256']==digest([])), 'Invalid entropy promotion origin')
        comparable['promotion_origin']=ANCHOR_ENTROPY['promotion_origin']
    require(e.get('mode') in ('off','shadow','apply') and comparable in recipes,
        'Invalid entropy floor controller')
    r=config['exploration_rollout']
    expected=dict(ROLLOUT,mode='staged',opening_after_iterations=1,raw_entropy_after_iterations=2) if r.get('mode')=='staged' else ROLLOUT
    require(r==expected, 'Invalid exploration adoption schedule')
    require(config['update']['entropy_weight']==e['alpha_base'] and 'il_kl_limit' in config and
        config.get('selfplay_recipe')=='latest-model-selfplay/3.0.0', 'Exploration requires fixed IL probe and latest-model recipe')


def group(context):
    require(context['era'] in ('CanalEra','RailEra'), 'Unknown entropy era')
    return context['era'].removesuffix('Era').lower()+('_action' if context['phase']=='action' else '_other')


def check_behavior(b, phase, *, branch_move=False):
    require(set(b)=={'kind','policy_id','probability','temperature','exploration','exploration_version',
        'component','action_decision'} and b['kind']=='model' and type(b['exploration_version']) is int and
        b['exploration_version']==3 and type(b['exploration']) in (float,int) and
        math.isfinite(b['exploration']) and 0<=b['exploration']<=.5 and
        b['component'] in (('branch_move',) if branch_move else ('policy','exploration')) and b['temperature'] in (1.,1.5,2.),
        'Invalid v3 behavior record')
    count=b['action_decision']
    require((type(count) is int and count>=0) if phase=='action' else count is None,
        'Invalid opening decision counter')


def initial(config, generation):
    anchored=config['entropy_control']['contract']==ANCHOR_ENTROPY['contract']
    return dict(contract=ANCHOR_STATE if anchored else STATE, origin=generation, generation=generation,
        **(dict(floor_anchor_promotions=0) if anchored else {}),
        baseline_rows=[], baseline=None, armed_at=None,
        groups={g:dict(alpha=config['entropy_control']['alpha_base'],window=[],cap_streak=0) for g in GROUPS})


def restore(config, state, checkpoint=None):
    if 'exploration_rollout' not in config: return None
    saved=state.get('exploration')
    if saved is None:
        require(checkpoint is None, 'Checkpoint exploration state missing from committed run')
        return initial(config,state['iteration'])
    anchored=config['entropy_control']['contract']==ANCHOR_ENTROPY['contract']
    upgraded=saved.get('contract')==ANCHOR_STATE
    fields={'contract','origin','generation','baseline_rows','baseline','armed_at','groups'}
    require(type(saved) is dict and set(saved)==fields|({'floor_anchor_promotions'} if upgraded else set()) and
        saved==checkpoint and saved['contract'] in (STATE,ANCHOR_STATE) and type(saved['generation']) is int and
        type(saved['origin']) is int and saved['generation']==state['iteration'] and
        0<=saved['origin']<=saved['generation'], 'Exploration checkpoint state differs')
    if upgraded:
        require(anchored and type(saved['floor_anchor_promotions']) is int and saved['floor_anchor_promotions']>=0,
            'Invalid entropy promotion count or unsupported return to frozen floors')
    finite=lambda x:type(x) in (int,float) and math.isfinite(x) and x>=0
    rows=saved['baseline_rows']
    require(type(rows) is list and len(rows)<=10 and
        all(type(r) is dict and set(r)==set(GROUPS) and all(finite(v) for v in r.values()) for r in rows),
        'Invalid entropy baseline evidence')
    expected={g:statistics.mean(r[g] for r in rows) for g in GROUPS} if len(rows)==10 else None
    require(saved['baseline']==expected and
        ((type(saved['armed_at']) is int and saved['origin']<saved['armed_at']<=saved['generation'])
         if expected is not None else saved['armed_at'] is None), 'Entropy baseline differs from evidence')
    require(set(saved['groups'])==set(GROUPS), 'Missing entropy controller group')
    for entry in saved['groups'].values():
        require(type(entry) is dict and set(entry)=={'alpha','window','cap_streak'} and
            finite(entry['alpha']) and .005<=entry['alpha']<=.05 and type(entry['window']) is list and
            len(entry['window'])<=3 and all(finite(v) for v in entry['window']) and
            type(entry['cap_streak']) is int and entry['cap_streak']>=0,
            'Invalid saved entropy controller')
    result=copy.deepcopy(saved)
    if anchored and not upgraded:result.update(contract=ANCHOR_STATE,floor_anchor_promotions=0)
    return sync_anchor_floor(config,result,state.get('policy_anchor'))


def sync_anchor_floor(config, state, controller):
    """Replay committed anchor receipts, including a pending pre-PPO promotion.

    The signed origin makes this restart-safe even if a promotion is committed
    to state.json before the first new entropy checkpoint is written.
    """
    if config['entropy_control']['contract']!=ANCHOR_ENTROPY['contract']:return state
    result=copy.deepcopy(state);origin=config['entropy_control']['promotion_origin']
    require(origin['generation']<=state['generation'],'Entropy promotion origin is in the future')
    events=(controller or {}).get('events',[]);start=origin['event_count']
    require(type(events) is list and len(events)>=start and digest(events[:start])==origin['events_sha256'],
        'Entropy anchor promotion history differs from its signed origin')
    selected=events[start:]
    require(all(type(e) is dict and e.get('decision')=='promote' and type(e.get('reference_generation')) is int and
        type(e.get('generation')) is int and
        0<e['reference_generation']<=e['generation'] and origin['generation']<=e['generation'] and
        e.get('evidence',{}).get('points_delta_ci95',[0])[0]>0 for e in selected),
        'Unqualified anchor event cannot lower entropy floors')
    generations=[e['reference_generation'] for e in events]
    require(generations==sorted(set(generations)), 'Anchor references must advance to distinct newer generations')
    require(type(result.get('floor_anchor_promotions')) is int and
        0<=result['floor_anchor_promotions']<=len(selected),'Entropy promotion count went backwards')
    result['floor_anchor_promotions']=len(selected)
    return result


def floor_ratios(config, state):
    e=config['entropy_control'];ratios=e.get('floor_ratio_by_group')
    if e['contract']!=ANCHOR_ENTROPY['contract']:return copy.deepcopy(ratios)
    require(state.get('contract')==ANCHOR_STATE,'Entropy promotion state must be restored before use')
    count=state['floor_anchor_promotions']
    return {g:max(e['floor_min_ratio_by_group'][g],round(ratios[g]-count*e['floor_drop_per_anchor_promotion'],12))
        for g in GROUPS}


def anchor_floor_transition(config, state, controller):
    if state is None:return state,None
    after=sync_anchor_floor(config,state,controller)
    if after.get('floor_anchor_promotions')==state.get('floor_anchor_promotions'):return after,None
    return after,dict(decision='lower_on_anchor_promotion',
        promotions_before=state['floor_anchor_promotions'],promotions_after=after['floor_anchor_promotions'],
        floor_ratio_before=floor_ratios(config,state),floor_ratio_after=floor_ratios(config,after),
        baseline_preserved=True,scope='Next entropy target; original baseline and current alpha unchanged')


def effective(config, state, generation):
    if 'exploration_rollout' not in config:
        return dict(version=2,opening=None,raw_entropy=False,branching=None,alphas=None,age=None)
    age=generation-state['origin']; r=config['exploration_rollout']
    require(age>=0, 'Exploration rollout precedes its committed origin')
    e=config['entropy_control']; raw=age>=r['raw_entropy_after_iterations']
    branch_ready=age>=r['branching_after_iterations'] and state['baseline'] is not None
    return dict(version=3,opening=config['opening'] if age>=r['opening_after_iterations'] else None,
        raw_entropy=raw,branching=config['branching'] if branch_ready else None,
        branching_status='active' if branch_ready else 'waiting_for_20_iterations' if age<r['branching_after_iterations'] else 'waiting_for_entropy_baseline',
        alphas={g:(state['groups'][g]['alpha'] if e['mode']=='apply' else e['alpha_base']) for g in GROUPS},
        age=age,origin=state['origin'])


def floors(config, state, generation):
    if state['baseline'] is None:return dict.fromkeys(GROUPS)
    require(generation>=state['armed_at'], 'Entropy floor precedes its baseline')
    e=config['entropy_control']
    if 'floor_ratio_by_group' in e:
        return {g:state['baseline'][g]*ratio for g,ratio in floor_ratios(config,state).items()}
    ratio=max(e['floor_min_ratio'],e['floor_start_ratio']*
              e['floor_decay_per_iteration']**(generation-state['armed_at']))
    return {g:state['baseline'][g]*ratio for g in GROUPS}


def advance(state, config, measurement, *, generation, warmup_done, healthy, policy_anchor=None):
    """One final-probe observation per committed iteration; no RNG consumption."""
    s=sync_anchor_floor(config,copy.deepcopy(state),policy_anchor); e=config['entropy_control']
    require(generation==s['generation']+1, 'Entropy controller skipped/repeated a generation')
    active=effective(config,s,generation-1)['raw_entropy']
    values={g:measurement.get('groups',{}).get(g,{}).get('raw_entropy') for g in GROUPS}
    valid=all(type(v) in (float,int) and math.isfinite(v) and v>=0 for v in values.values())
    if active and warmup_done and valid and s['baseline'] is None:
        s['baseline_rows'].append(values)
        if len(s['baseline_rows'])==e['baseline_iterations']:
            s['baseline']={g:statistics.mean(r[g] for r in s['baseline_rows']) for g in GROUPS}
            s['armed_at']=generation
    report={}; targets=floors(config,s,generation)
    for g in GROUPS:
        entry=s['groups'][g]; before=entry['alpha']; h=values[g]; floor=None
        if active and warmup_done and valid:
            entry['window']=(entry['window']+[h])[-e['median_window']:]
        median=statistics.median(entry['window']) if entry['window'] else None
        reason='baseline_pending'; decision='hold'
        if s['baseline'] is not None:
            floor=targets[g]
            if not active or not warmup_done or not valid:reason='inactive_or_invalid'
            elif not healthy:reason='unhealthy'
            elif e['mode']=='off':reason='disabled'
            elif generation<=s['armed_at'] or len(entry['window'])<e['median_window']:reason='arming'
            elif floor>0 and median<floor:
                entry['alpha']=min(e['alpha_max'],before*min(e['max_step'],math.exp(e['gain']*(floor-median)/floor)))
                decision='raise';reason='below_floor'
            elif median>e['hysteresis']*floor:
                entry['alpha']=max(e['alpha_base'],before/e['relax_factor']);decision='relax';reason='above_hysteresis'
            else:reason='hysteresis'
        entry['cap_streak']=entry['cap_streak']+1 if entry['alpha']>=e['alpha_max'] else 0
        alarms=[]
        if entry['cap_streak']>=10:alarms.append('alpha_at_cap_10_iterations')
        if valid and s['baseline'] and h>1.3*s['baseline'][g]:alarms.append('entropy_above_1.3_baseline')
        report[g]=dict(raw_entropy=h,normalized_entropy=measurement.get('groups',{}).get(g,{}).get('normalized_entropy'),
            baseline=s['baseline'][g] if s['baseline'] else None,floor=floor,median=median,
            alpha_before=before,alpha_after=entry['alpha'],decision=decision,reason=reason,alarms=alarms)
    s['generation']=generation
    return s,dict(contract=e['contract'],mode=e['mode'],origin=s['origin'],armed_at=s['armed_at'],
        **(dict(floor_ratio_by_group=floor_ratios(config,s)) if 'floor_ratio_by_group' in e else {}),
        **{k:copy.deepcopy(e[k]) for k in ('floor_start_ratio','floor_decay_per_iteration','floor_min_ratio',
            'floor_min_ratio_by_group','floor_drop_per_anchor_promotion','promotion_origin') if k in e},
        **(dict(floor_anchor_promotions=s['floor_anchor_promotions'],
            floor_initial_ratio_by_group=copy.deepcopy(e['floor_ratio_by_group'])) if 'floor_anchor_promotions' in s else {}),
        baseline_iterations=len(s['baseline_rows']),healthy=bool(healthy),groups=report,
        scope='Final raw policy on fixed IL probe; prospective post-warmup baseline, no retired IL action checks')
