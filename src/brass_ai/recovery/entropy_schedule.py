"""Committed-health entropy annealing; reward steps require reviewed restarts."""
import copy
import math

from brass_ai.learning.contracts import digest, require

KEY='entropy_schedule'
CONTRACT='entropy-anneal-schedule/1.1.0'
STATE='entropy-anneal-state/1.0.0'
MIGRATION='entropy-anneal-adoption/1.0.0'
GROUPS=('canal_action','canal_other','rail_action','rail_other')
DEFAULT=dict(contract=CONTRACT,mode='off',start_generation=None,reset_generation=None,
    start=dict.fromkeys(GROUPS,.005),final=dict.fromkeys(GROUPS,.001),ramp_iterations=100,
    shape='linear',advance_on='healthy-entropy-update',measure_every=5,
    insolvency_start=dict(event_cost=.02,max_cost=.075),insolvency_steps=[
        dict(at_progress=0,insolvency=dict(event_cost=.01,max_cost=.075)),
        dict(at_progress=50,insolvency=dict(event_cost=.005,max_cost=.05))])


def validate_update(update):
    c=update[KEY]
    require(type(c) is dict and set(c)==set(DEFAULT) and c['contract']==CONTRACT,
        'Invalid entropy anneal contract/fields')
    require(c['mode'] in ('off','on') and c['shape']=='linear' and c['advance_on']=='healthy-entropy-update',
        'Invalid entropy anneal mode/shape/health rule')
    require((c['start_generation'] is None if c['mode']=='off' else
        type(c['start_generation']) is int and c['start_generation']>=1) and
        (c['reset_generation'] is None or type(c['reset_generation']) is int and c['reset_generation']>=0),
        'Invalid entropy anneal start/reset generation')
    for key in ('ramp_iterations','measure_every'):
        require(type(c[key]) is int and c[key]>=1,'Invalid entropy anneal '+key)
    for name in ('start','final'):
        require(type(c[name]) is dict and set(c[name])==set(GROUPS) and all(
            type(v) in (float,int) and math.isfinite(v) and v>0 for v in c[name].values()),
            'Invalid entropy anneal '+name)
    require(c['start']==update['fixed_entropy']['weights'] and all(c['final'][g]<=c['start'][g] for g in GROUPS)
        and all(v==update['entropy_weight'] for v in c['start'].values()),
        'Entropy anneal start must match its fixed preset; final cannot exceed start')
    from .utility import event_cost_upgrade
    from brass_ai.learning.reward import specification
    before=specification(insolvency=c['insolvency_start']);previous=-1
    require(type(c['insolvency_steps']) is list and c['insolvency_steps'],'Missing planned insolvency steps')
    for i,step in enumerate(c['insolvency_steps']):
        require(type(step) is dict and set(step)=={'at_progress','insolvency'} and
            type(step['at_progress']) is int and previous<step['at_progress']<=c['ramp_iterations'] and
            (i!=0 or step['at_progress']==0),'Invalid insolvency step boundary')
        after=specification(insolvency=step['insolvency'])
        require(event_cost_upgrade(before,after),'Unreviewed planned insolvency transition')
        before=after;previous=step['at_progress']
    return c


def step_count(c,cost):
    specs=[c['insolvency_start']]+[s['insolvency'] for s in c['insolvency_steps']]
    require(cost in specs,'Applied insolvency is outside the entropy schedule')
    return specs.index(cost)


def validate(config):
    c=validate_update(config['update'])
    require(config['entropy_control']['mode']=='off' and
        config['entropy_control']['alpha_base']==config['update']['entropy_weight'],
        'Annealing requires adaptive entropy off and the unchanged start preset')
    n=step_count(c,config['insolvency'])
    require(c['mode']=='off' or n>=1,'Enable annealing together with insolvency step one')
    from .utility import reward_spec
    reward=reward_spec(config)
    require(config['search']['reward']==reward and
        ('teacher_phase' not in config or config['teacher_phase']['search']['reward']==reward),
        'Scheduled learner/search/teacher rewards differ')
    # Legacy teacher_value is dormant when teacher_enabled=false; preserve its
    # saved options instead of changing an unrelated, unreachable recipe.
    require(config.get('teacher_enabled') is False and config.get('teacher_phase',{}).get('mode','off')=='off',
        'Keep teacher fitting off during the entropy anneal experiment')
    return c


def weights(c,k,generation):
    if c['mode']=='off' or generation<c['start_generation']:return copy.deepcopy(c['start'])
    fraction=min(1.,k/c['ramp_iterations'])
    return {g:c['final'][g] if fraction==1 else c['start'][g]+(c['final'][g]-c['start'][g])*fraction for g in GROUPS}


def waiting(s):
    c=s['config'];n=s['insolvency_steps_applied']
    return bool(c['mode']=='on' and n<len(c['insolvency_steps']) and
        s['progress']>=c['insolvency_steps'][n]['at_progress'])


def seal(s):
    s['waiting_for_insolvency_step']=waiting(s)
    s['applied']=weights(s['config'],s['progress'],s['last_generation']+1)
    return s


def validate_state(s,generation):
    require(type(s) is dict and set(s)=={'contract','config','started_at','progress','held','last_generation',
        'applied','insolvency','insolvency_steps_applied','waiting_for_insolvency_step','reset_count'} and
        s['contract']==STATE and s['last_generation']==generation,'Invalid committed entropy schedule state')
    c=s['config']
    validate_update(dict(entropy_schedule=c,fixed_entropy=dict(weights=c['start']),entropy_weight=next(iter(c['start'].values()))))
    require(all(type(s[k]) is int and s[k]>=0 for k in ('progress','held','last_generation','insolvency_steps_applied','reset_count'))
        and s['progress']<=c['ramp_iterations'] and (s['started_at'] is None or type(s['started_at']) is int and s['started_at']>=1),
        'Invalid entropy schedule counters')
    require(s['insolvency_steps_applied']==step_count(c,s['insolvency']) and
        (c['mode']=='off' or s['insolvency_steps_applied']>=1 and s['started_at'] is not None) and
        s['applied']==weights(c,s['progress'],generation+1) and s['waiting_for_insolvency_step']==waiting(s),
        'Entropy schedule coefficients/reward receipt differ')
    return s


def restore(config,saved,generation):
    """Pure, repeatable adoption from the checkpoint-bound exploration state.

    Until a new iteration commits, state.json and the checkpoint keep the old
    schedule. The signed config transition is replayed without advancing k.
    """
    if KEY not in config['update']:
        require(saved is None,'Cannot silently drop a committed entropy schedule; use off')
        return None
    c=validate(config);n=step_count(c,config['insolvency'])
    if saved is None:
        require(c['reset_generation'] is None and (c['mode']=='off' and n==0 or
            c['mode']=='on' and c['start_generation']>=generation+1 and n==1),
            'Initial entropy schedule must start prospectively with its first reward step')
        return seal(dict(contract=STATE,config=copy.deepcopy(c),started_at=c['start_generation'],progress=0,held=0,
            last_generation=generation,insolvency=copy.deepcopy(config['insolvency']),insolvency_steps_applied=n,reset_count=0))
    s=copy.deepcopy(validate_state(saved,generation));old=s['config'];previous=s['insolvency_steps_applied']
    immutable=lambda x:{k:v for k,v in x.items() if k not in ('mode','start_generation','reset_generation')}
    require(immutable(old)==immutable(c),'Cannot silently alter an adopted entropy ramp or reward plan')
    reset=c['reset_generation']!=old['reset_generation']
    if reset:
        require(c['mode']=='on' and c['reset_generation']==generation and c['start_generation']>=generation+1
            and n<=1,'Explicit entropy reset requires a prospective start before the second reward step')
        s.update(progress=0,held=0,started_at=c['start_generation'],reset_count=s['reset_count']+1)
    elif c['mode']=='on' and (old['mode']=='off' or c['start_generation']!=old['start_generation']):
        require(old['mode']=='off' and c['start_generation']>=generation+1,'Entropy start is in the past or already active')
        if s['started_at'] is None:s['started_at']=c['start_generation']
    if n!=previous:
        from .utility import event_cost_upgrade,reward_spec
        before=copy.deepcopy(config);before['insolvency']=s['insolvency']
        require(abs(n-previous)==1 and event_cost_upgrade(reward_spec(before),reward_spec(config)),
            'Entropy reward receipt requires one reviewed step')
        if n>previous:require(s['progress']>=c['insolvency_steps'][n-1]['at_progress'],'Insolvency step is not due')
    s.update(config=copy.deepcopy(c),insolvency=copy.deepcopy(config['insolvency']),insolvency_steps_applied=n)
    return seal(s)


def restore_run(config,state):
    """Replay signed boundary changes if a restart preceded the next PPO commit.

    In particular, on -> off or reset -> off before any collection must keep
    the already applied reward and reset receipt. Never invent a starting step
    just because an edited config names a lower cost.
    """
    saved=(state.get('exploration') or {}).get(KEY);generation=state['iteration']
    if 'signature' in state:
        known={digest(state['signature']):state['signature']};tail=digest(state['signature']);chain=[]
        for transition in reversed(state.get('source_transitions',[])):
            if transition['iteration']==generation and transition['checkpoint']==state['history'][-1] and transition['to_signature']==tail:
                chain.append(transition);origin=transition['from_signature'];tail=digest(origin);known[tail]=origin
        for transition in reversed(chain):
            target=known[transition['to_signature']]['config']
            saved=restore(target,saved,generation)
    return restore(config,saved,generation)


def effective(config,state):
    s=validate_state(state[KEY],state['generation']);c=validate(config)
    require(c==s['config'] and config['insolvency']==s['insolvency'],'Entropy schedule has not adopted the current config')
    return copy.deepcopy(s['applied'])


def receipt(config,state):
    s=state[KEY]
    return dict(contract=CONTRACT,config_id=digest(config['update'][KEY]),generation=state['generation']+1,
        progress=s['progress'],alphas=effective(config,state))


def behavior_weights(update,behavior):
    c=validate_update(update);r=getattr(behavior,KEY,None)
    require(type(r) is dict and set(r)=={'contract','config_id','generation','progress','alphas'} and
        r['contract']==CONTRACT and r['config_id']==digest(c) and type(r['generation']) is int and r['generation']>=1
        and type(r['progress']) is int and 0<=r['progress']<=c['ramp_iterations'],
        'Missing/incompatible collected entropy schedule receipt')
    expected=weights(c,r['progress'],r['generation'])
    require(r['alphas']==expected and behavior.entropy_alphas==expected,'Collected/PPO scheduled entropy coefficients differ')
    return expected


def advance(s,config,*,generation,healthy):
    s=copy.deepcopy(validate_state(s,generation-1));validate(config)
    require(s['config']==config['update'][KEY] and s['insolvency']==config['insolvency'],'Unadopted entropy schedule')
    applied=copy.deepcopy(s['applied']);before=s['progress'];c=s['config']
    if c['mode']=='off':reason='disabled'
    elif generation<c['start_generation']:reason='before_start'
    elif waiting(s):reason='waiting_for_insolvency_step';s['held']+=1
    elif before>=c['ramp_iterations']:reason='complete'
    elif healthy:s['progress']+=1;reason='healthy_commit'
    else:s['held']+=1;reason='unhealthy'
    s['last_generation']=generation;seal(s)
    report=dict(contract=CONTRACT,mode=c['mode'],generation=generation,started_at=s['started_at'],
        progress_before=before,progress=s['progress'],ramp_iterations=c['ramp_iterations'],held=s['held'],
        healthy=bool(healthy),reason=reason,effective_alpha=applied,next_alpha=copy.deepcopy(s['applied']),
        insolvency=copy.deepcopy(s['insolvency']),insolvency_steps_applied=s['insolvency_steps_applied'],
        waiting_for_insolvency_step=s['waiting_for_insolvency_step'],
        next_insolvency_step=copy.deepcopy(c['insolvency_steps'][s['insolvency_steps_applied']])
            if s['insolvency_steps_applied']<len(c['insolvency_steps']) else None,
        scope='Loss coefficients used this iteration; progress commits with the checkpoint. Reward changes require a reviewed restart, never a live automatic edit.')
    return s,report


def measurement_due(config,generation):
    c=config['update'].get(KEY)
    return bool(c and c['mode']=='on' and generation>=c['start_generation'] and
        (generation-c['start_generation']+1)%c['measure_every']==0)


def _reward(config,cost):
    from .utility import reward_spec
    config['insolvency']=copy.deepcopy(cost);config['search']['reward']=reward_spec(config)
    if 'teacher_phase' in config:config['teacher_phase']['search']['reward']=reward_spec(config)


def configure(config,state,command,*,start=None,reset=False,reverse=False):
    result=copy.deepcopy(config);update=result['update'];g=state['iteration']
    update.setdefault(KEY,copy.deepcopy(DEFAULT));c=validate_update(update)
    # Adoption may have committed its signature but not its first new checkpoint.
    current=restore_run(result,state)
    if command=='on':
        require(not reverse,'Reverse is only valid for an insolvency step')
        requested=g+1 if start in (None,'next') else int(start)
        if c['mode']=='on' and not reset:
            require(start in (None,'next') or requested==c['start_generation'],'Schedule already on; use an explicit reset')
        else:
            require(requested>=g+1,'Entropy start generation is in the past')
            c.update(mode='on',start_generation=requested)
        if reset:c['reset_generation']=g
        if current['insolvency_steps_applied']==0:_reward(result,c['insolvency_steps'][0]['insolvency'])
    elif command=='off':
        require(not reset and not reverse and start is None,'Invalid off options')
        c.update(mode='off',start_generation=None)
    elif command=='insolvency-step':
        require(not reset and start is None and KEY in config['update'],'Start the entropy schedule before its reward steps')
        n=current['insolvency_steps_applied'];target=n-1 if reverse else n+1
        require(1<=target<=len(c['insolvency_steps']),'No listed reward step in that direction')
        if not reverse:
            require(c['mode']=='on','Turn the entropy schedule on before applying its next reward step')
            require(current['progress']>=c['insolvency_steps'][target-1]['at_progress'],'Insolvency step not due')
        _reward(result,c['insolvency_steps'][target-1]['insolvency'])
    else:raise ValueError('Unknown entropy schedule command')
    validate(result);restore(result,current,g)
    return result


def reviewed_config(config,state,target):
    """Only exact helper transformations, for the pinned source-review handler."""
    c=target['update'].get(KEY);require(c is not None,'Use off instead of removing the schedule')
    previous=config['update'].get(KEY)
    command=('on' if c['mode']=='on' else 'off')
    if previous is not None and c==previous and config['insolvency']!=target['insolvency']:command='insolvency-step'
    reset=c.get('reset_generation') is not None and c.get('reset_generation')!=(previous or {}).get('reset_generation')
    result=configure(config,state,command,start=c['start_generation'] if command=='on' else None,reset=reset,
        reverse=command=='insolvency-step' and step_count(c,target['insolvency'])<step_count(c,config['insolvency']))
    require(result==target,'Unrelated settings changed with the entropy schedule')
    return result
