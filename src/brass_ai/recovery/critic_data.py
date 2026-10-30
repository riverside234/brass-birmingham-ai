"""Scored-opponent critic admission, immutable table metadata and rollout gates."""
import copy
import math
import numpy as np

from brass_ai.learning.contracts import digest,require
from brass_ai.learning.seat_descriptors import (CONTRACT,ROLES,FAMILIES,INDUSTRIES,
    descriptor,validate,relative)

LEGACY_DEFAULT=dict(contract='critic-rows/1.0.0',learner_rows=True,snapshot_rows=True,snapshot_weight=1.,
    excluded_roles=['il','rule_bot'],holdout_fraction=.05,
    rollout=dict(mode='auto',descriptor_ev_min_gain=0.,snapshot_ev_tolerance=.02,reduced_snapshot_weight=.5))
DEFAULT=dict(copy.deepcopy(LEGACY_DEFAULT),contract='critic-rows/2.0.0',
    rollout=dict(LEGACY_DEFAULT['rollout'],reduced_snapshot_weight=0.))
FLOOR_MIGRATION='snapshot-weight-zero-floor/1.0.0'
FIXED_DEFAULT=dict(copy.deepcopy(DEFAULT),contract='critic-rows/3.0.0',
    rollout=dict(DEFAULT['rollout'],mode='fixed'))
FIXED_MIGRATION='snapshot-weight-fixed-one/1.0.0'
FIXED_STATE='snapshot-weight-fixed/1.0.0'

RECOVERY_CONTRACT='snapshot-weight-recovery/1.0.0'
RECOVERY_ITERATIONS=5
SNAPSHOT_WEIGHTS=(0.,.5,.75,1.)


def floor_config(config):
    result=copy.deepcopy(config)
    if 'critic_data' not in result:return result
    c=validate_config(result['critic_data'])
    if fixed(c):return result
    c['contract']=DEFAULT['contract'];c['rollout']['reduced_snapshot_weight']=0.
    validate_config(c)
    return result


def fixed_config(config):
    result=copy.deepcopy(config)
    if 'critic_data' in result:
        validate_config(result['critic_data'])
        result['critic_data']=copy.deepcopy(FIXED_DEFAULT)
    return result


def fixed(c):return c.get('contract')==FIXED_DEFAULT['contract']


def validate_config(c):
    require(type(c) is dict and set(c)==set(DEFAULT) and c['contract'] in (DEFAULT['contract'],LEGACY_DEFAULT['contract'],FIXED_DEFAULT['contract'])
        and c['learner_rows'] is True and type(c['snapshot_rows']) is bool
        and c['excluded_roles']==DEFAULT['excluded_roles'],'Invalid critic row contract')
    if fixed(c):
        require(c==FIXED_DEFAULT and type(c['snapshot_weight']) in (int,float)
            and type(c['holdout_fraction']) in (int,float) and type(c['rollout']) is dict
            and all(type(c['rollout'][k]) is not bool for k in c['rollout'] if k!='mode'),
            'Fixed snapshot critic requires weight 1.00 and disabled weight control')
        return c
    expected=DEFAULT if c['contract']==DEFAULT['contract'] else LEGACY_DEFAULT
    require(type(c['snapshot_weight']) in (int,float) and c['snapshot_weight'] in ((0.,.5,1.) if expected is DEFAULT else (.5,1.))
        and type(c['holdout_fraction']) in (int,float) and c['holdout_fraction']==.05,
        'Invalid critic snapshot weight/holdout')
    require(type(c['rollout']) is dict and set(c['rollout'])==set(DEFAULT['rollout'])
        and c['rollout']['mode'] in ('auto','learner_only','snapshots')
        and all(type(c['rollout'][k]) is not bool and c['rollout'][k]==v for k,v in expected['rollout'].items() if k!='mode'),
        'Invalid critic rollout settings')
    return c


def fingerprint(probe,probabilities=None,*,rule_bot=False):
    """Per-era family mass, and build-conditional industry mass, raw T=1."""
    result=np.zeros(28,dtype=np.float64);counts=[0,0];build_mass=[0.,0.]
    offsets=probe.arrays['offsets']
    if not rule_bot:
        require(probabilities is not None and len(probabilities)==offsets[-1],'Fingerprint probabilities missing')
    for i,row in enumerate(probe.rows):
        if row['phase']!='action':continue
        require('industries' in row and 'chosen' in row,'Rebuild fixed probe with chosen moves/industries for v11.2')
        era=('CanalEra','RailEra').index(row['era']);counts[era]+=1
        a,b=map(int,offsets[i:i+2]);p=np.zeros(b-a) if rule_bot else np.asarray(probabilities[a:b],dtype=float)
        if rule_bot:p[row['chosen']]=1.
        require(len(p)==len(row['kinds']) and np.isfinite(p).all() and (p>=0).all() and abs(p.sum()-1)<1e-5,
                'Fingerprint policy is not normalized')
        for j,(kind,industry) in enumerate(zip(row['kinds'],row['industries'])):
            kind=kind if kind in FAMILIES else 'other'
            result[era*8+FAMILIES.index(kind)]+=p[j]
            if kind=='build':
                require(industry in INDUSTRIES,'Missing build industry in fingerprint probe')
                result[16+era*6+INDUSTRIES.index(industry)]+=p[j];build_mass[era]+=p[j]
    for era in range(2):
        if counts[era]:result[era*8:era*8+8]/=counts[era]
        if build_mass[era]:result[16+era*6:16+era*6+6]/=build_mass[era]
    return dict(contract='policy-style-fingerprint/1.0.0',probe_hash=probe.metadata['probe_hash'],
        values=result.astype(np.float32).tolist(),action_positions=counts,build_mass=build_mass,
        source='recorded_v4_choices' if rule_bot else 'raw_policy_temperature_1',
        industry_order=list(INDUSTRIES),
        normalization='Family groups sum to 1 per observed era; industry groups conditional on build; empty groups are zero')


def planned_table(policies,rotation,current,generation,members,learner_fp,rule_fp,*,learner_policy_id,strategies=None,context_fingerprints=None):
    lookup={m['model_id']:m for m in members};seats=[];values=[]
    for seat in range(4):
        p=policies[(seat-rotation)%4];mid=p.get('model_id')
        if p.get('policy_id')==learner_policy_id:
            require(mid==current,'Learner descriptor model differs')
            role='learner';born=generation;fp=learner_fp
        elif mid in lookup:
            member=lookup[mid];born=member['generation'];role='il' if born==0 else 'snapshot';fp=member['fingerprint']
        elif p.get('kind') in ('external','legacy') or p.get('search_free'):
            role='rule_bot';born=generation;fp=rule_fp
        else:raise ValueError('Unrecognized ordinary opponent for critic descriptors')
        a=strategies['seats'][seat] if strategies else None
        if a is not None:
            from .domino_sampling import validate_assignment
            validate_assignment(a)
            require((a['training_role']=='learner')==(role=='learner') and mid==a['model_id'],
                'Strategy role/model differs from critic descriptor')
            require(context_fingerprints is not None,'Missing context-aware fingerprint provider')
            fp=context_fingerprints.get(mid,a['strategy_id'])
        require(0<=born<=generation,'Opponent from a future generation')
        seats.append(dict(role=role,generation=born,model_id=mid,policy_id=p.get('policy_id'),
                          fingerprint_id=digest(fp),**(dict(strategy=a) if a is not None else {})))
        values.append(descriptor(role,0 if role in ('learner','rule_bot') else generation-born,fp['values']))
    validate(values)
    return dict(contract=CONTRACT,generation=generation,seats=seats,stable=values,
                learner_fingerprint_id=digest(learner_fp))


def check_table(table,game,policy_id):
    require(type(table) is dict and set(table)=={'contract','generation','seats','stable','learner_fingerprint_id'}
        and table['contract']==CONTRACT and type(table['generation']) is int and table['generation']>=0
        and len(table['seats'])==4,'Missing/invalid recorded critic table')
    values=validate(table['stable'])
    for seat,record in enumerate(table['seats']):
        role=record['role'];require(role in ROLES[:-1] and record['policy_id']==game['opponents'][seat],
                                   'Recorded descriptor opponent identity differs')
        require((role=='learner')==(record['policy_id']==policy_id),'Critic learner role differs from collection')
        if 'strategy' in record:
            from .domino_sampling import validate_assignment
            a=validate_assignment(record['strategy'])
            require(a['model_id']==record['model_id'] and a['physical_seat']==seat and
                (a['training_role']=='learner')==(role=='learner'),'Critic strategy role differs')
        if hasattr(game,'diversity_assignment'):
            require(record.get('strategy')==game.diversity_assignment['seats'][seat],
                    'Critic strategy differs from collection assignment')
        require(type(record['generation']) is int and 0<=record['generation']<=table['generation'] and
                values[seat,ROLES.index(role)]==1 and
                (role!='snapshot' or 0<record['generation']<=table['generation']) and
                (role!='il' or record['generation']==0),'Invalid critic role/generation')
        expected_lag=0. if role in ('learner','rule_bot') else min(1.,math.log1p(table['generation']-record['generation'])/math.log1p(256))
        require(abs(float(values[seat,5])-expected_lag)<1e-7 and
                (role!='learner' or record['generation']==table['generation']), 'Critic generation lag differs')
        if role in ('learner','snapshot','il'):
            require(all(r['behavior']['kind']=='model' and r['teacher'] is None
                        for r in game['records'] if r['seat']==seat),'Unexpected critic behavior/search labels')
    return table


def annotate(row,encoded,table,kind):
    role=table['seats'][row['seat']]['role']
    return dict(row,_critic_role=role,_table_kind=kind,
        **(dict(_domino=table['seats'][row['seat']]['strategy']) if 'strategy' in table['seats'][row['seat']] else {}),
        _seat_descriptors=relative(table['stable'],encoded.stable_to_relative),
        _descriptor_generation=table['generation'])


def recovery_initial(generation):
    return dict(contract=RECOVERY_CONTRACT,origin=generation,healthy_streak=0,last_change_generation=None)


def initial(c):
    return dict(generation=0,snapshots_active=c['snapshot_rows'] and c['rollout']['mode'] in ('snapshots','fixed'),
        snapshot_weight=c['snapshot_weight'],learner_ev_baseline=None,activation_generation=0 if fixed(c) else None,
        previous_fingerprint=None,weight_recovery=(dict(contract=FIXED_STATE,origin=0,previous=None)
            if fixed(c) else recovery_initial(0)))


def _validate_base(s,c,generation,*,legacy=False):
    fields=set(initial(c))-({'weight_recovery'} if legacy else set())
    require(type(s) is dict and set(s)==fields and type(s['generation']) is int and s['generation']==generation
        and type(s['snapshots_active']) is bool and type(s['snapshot_weight']) in (int,float)
        and s['snapshot_weight'] in ((.5,1.) if legacy else SNAPSHOT_WEIGHTS),'Invalid critic rollout checkpoint')
    require(not s['snapshots_active'] or c['snapshot_rows'],'Unexpected snapshot activation')
    require(s['snapshot_weight']>=c['rollout']['reduced_snapshot_weight'],'Snapshot weight below configured floor')
    for k in ('learner_ev_baseline',):
        require(s[k] is None or (type(s[k]) in (int,float) and math.isfinite(s[k])),'Invalid critic rollout EV')
    require(s['activation_generation'] is None or
        type(s['activation_generation']) is int and 0<=s['activation_generation']<=generation,'Invalid critic activation age')
    return s


def validate_state(s,c,generation):
    _validate_base(s,c,generation)
    r=s['weight_recovery']
    if fixed(c):
        require(s['snapshots_active'] and s['snapshot_weight']==1. and type(r) is dict and
            set(r)=={'contract','origin','previous'} and r['contract']==FIXED_STATE and
            type(r['origin']) is int and 0<=r['origin']<=generation,'Invalid fixed snapshot weight checkpoint')
        old=r['previous']
        if old is not None:
            require(type(old) is dict and set(old)=={'config','state'} and not fixed(old['config']),
                'Invalid archived snapshot controller')
            validate_config(old['config']);validate_state(old['state'],old['config'],r['origin'])
        else:require(r['origin']==0,'Missing archived snapshot controller')
        return s
    from . import snapshot_r2
    if r.get('contract')==snapshot_r2.CONTRACT:
        require(s['learner_ev_baseline'] is None,'Legacy EV must not be reused as the R2 reference')
        snapshot_r2.validate(r,generation,s['snapshot_weight'])
        return s
    require(type(r) is dict and set(r)==set(recovery_initial(0)) and r['contract']==RECOVERY_CONTRACT
        and type(r['origin']) is int and 0<=r['origin']<=generation
        and type(r['healthy_streak']) is int and 0<=r['healthy_streak']<RECOVERY_ITERATIONS,
        'Invalid snapshot weight recovery checkpoint')
    last=r['last_change_generation']
    require(last is None or type(last) is int and r['origin']<=last<=generation,
        'Invalid snapshot weight recovery age')
    require(r['healthy_streak']<=generation-(last if last is not None else r['origin'])
        and (not r['healthy_streak'] or s['snapshots_active'] and s['snapshot_weight']<1.),
        'Invalid snapshot weight recovery streak')
    return s


def fix_weight(s,old_config,new_config,generation):
    """Explicit checkpoint transition, before any resumed critic sampling."""
    validate_config(new_config);validate_state(s,old_config,generation)
    require(fixed(new_config) and not fixed(old_config),'Invalid fixed snapshot transition')
    result=copy.deepcopy(s)
    result.update(snapshots_active=True,snapshot_weight=1.,
        activation_generation=s['activation_generation'] if s['snapshots_active'] else generation,
        weight_recovery=dict(contract=FIXED_STATE,origin=generation,
            previous=dict(config=copy.deepcopy(old_config),state=copy.deepcopy(s))))
    return validate_state(result,new_config,generation)


def migrate_state(s,c,generation):
    """Exact legacy checkpoint upgrade; no retrospective recovery evidence."""
    s=copy.deepcopy(_validate_base(s,c,generation,legacy=True))
    s['weight_recovery']=recovery_initial(generation)
    return validate_state(s,c,generation)


def learner_ev(metrics):return metrics.get('groups',{}).get('role:learner',{}).get('explained_variance')


def recovery_console(report):
    if report.get('weight_control')=='fixed':return ' snapshot_weight=fixed'
    r=report.get('weight_recovery')
    if not r:return ''
    text=f' snapshot_recovery={r["healthy_streak_next"]}/{r["required_iterations"]}:{report["decision"]}'
    if 'learner' in r:
        a=r['learner'];value=f'{a["r2"]:.4f}' if a['r2'] is not None else 'n/a'
        text+=f' snapshot_R2={value}[{a["qualifying_iterations"]}/{r["minimum_iterations"]}]'
    return text


def advance(s,c,*,generation,report,healthy,fingerprint_now,ppo_snapshot_rows=0,regime=None):
    from . import snapshot_r2
    if fixed(c):
        from .value_phase import critic_work
        s=copy.deepcopy(validate_state(s,c,generation-1))
        value_rows=(report.get('role_usage',{}).get('snapshot',{}).get('distinct_used',0)
            if report.get('optimizer_steps',0)>0 else 0)
        require(all(type(n) is int and n>=0 for n in (ppo_snapshot_rows,value_rows)),
            'Invalid retained snapshot training counts')
        observation=snapshot_r2.entry(generation,report,healthy=healthy is True,trained=True)
        evidence=snapshot_r2.pool([p for p in observation['tables'].values() if p is not None])
        previous=s['previous_fingerprint'];drift=None
        if previous is not None and fingerprint_now is not None:
            drift=float(np.mean(np.abs(np.asarray(previous)-np.asarray(fingerprint_now['values']))))
        if fingerprint_now is not None:s['previous_fingerprint']=fingerprint_now['values']
        s['generation']=generation;validate_state(s,c,generation)
        return s,dict(snapshot_rows_used=bool(ppo_snapshot_rows or value_rows),snapshot_rows_enabled=True,
            ppo_snapshot_distinct_used=ppo_snapshot_rows,value_snapshot_distinct_used=value_rows,
            critic_training=critic_work(report),snapshot_rows_next=True,weight_used=1.,weight_next=1.,
            weight_control='fixed',weight_recovery=None,decision='fixed_snapshot_weight',
            reason='configured_fixed_one_recovery_and_decay_disabled',
            learner_heldout_ev=learner_ev(report.get('heldout_after',{})),learner_r2=evidence['r2'],
            learner_mse=evidence['learner_mse'],learner_prediction_bias=evidence['prediction_bias'],
            learner_target_variance=evidence['target_variance'],learner_evidence=evidence,
            fingerprint_mean_absolute_drift=drift,
            scope='Fixed 1.00 snapshot critic loss weight; current ordinary learner holdout R2 is diagnostic only. GAE, economic and PPO/PPG guards unchanged.')
    if s['weight_recovery'].get('contract')==snapshot_r2.CONTRACT:
        return _advance_r2(s,c,generation=generation,report=report,healthy=healthy,
            fingerprint_now=fingerprint_now,ppo_snapshot_rows=ppo_snapshot_rows,regime=regime)
    return _advance_ev(s,c,generation=generation,report=report,healthy=healthy,
        fingerprint_now=fingerprint_now,ppo_snapshot_rows=ppo_snapshot_rows)


def _advance_r2(s,c,*,generation,report,healthy,fingerprint_now,ppo_snapshot_rows,regime):
    from . import snapshot_r2
    from .value_phase import critic_work
    s=copy.deepcopy(validate_state(s,c,generation-1));before=s['snapshots_active'];weight=s['snapshot_weight']
    work=critic_work(report)
    value_rows=(report.get('role_usage',{}).get('snapshot',{}).get('distinct_used',0)
                if report.get('optimizer_steps',0)>0 else 0)
    require(all(type(n) is int and n>=0 for n in (ppo_snapshot_rows,value_rows)),
        'Invalid retained snapshot training counts')
    trained=weight>0 and bool(ppo_snapshot_rows or value_rows)
    # At zero the snapshot loss is disabled. Recovery must be able to observe
    # real learner critic work, without claiming that zero-weight rows trained.
    zero_observation=weight==0 and work['optimizer_steps']>0
    health_ok=healthy is True and not report.get('kl_guard_tripped',False)
    observation=snapshot_r2.entry(generation,report,healthy=health_ok,trained=before and (trained or zero_observation))
    recovery,next_weight,decision,reason,evidence=snapshot_r2.advance(s['weight_recovery'],weight,
        generation=generation,observation=observation,current_regime=regime or snapshot_r2.regime(report,{}),
        active=before,tolerance=c['rollout']['snapshot_ev_tolerance'],floor=c['rollout']['reduced_snapshot_weight'])
    # Descriptor admission remains the existing EV/health gate. Only subsequent
    # snapshot WEIGHT decisions change to R2.
    ev=learner_ev(report.get('heldout_after',{}));gain=report.get('descriptor_gain',{}).get('learner_ev_gain')
    if not before and c['snapshot_rows'] and c['rollout']['mode']=='auto':
        if healthy and work['optimizer_steps']>0 and snapshot_r2.finite(ev) and snapshot_r2.finite(gain) and gain>=c['rollout']['descriptor_ev_min_gain']:
            s.update(snapshots_active=True,activation_generation=generation)
            recovery.update(stage_origin=generation,entries=[],healthy_streak=0)
            decision='enable_snapshots';reason='heldout_descriptor_gain_and_health'
    previous=s['previous_fingerprint'];drift=None
    if previous is not None and fingerprint_now is not None:
        drift=float(np.mean(np.abs(np.asarray(previous)-np.asarray(fingerprint_now['values']))))
    if fingerprint_now is not None:s['previous_fingerprint']=fingerprint_now['values']
    s.update(generation=generation,snapshot_weight=next_weight,weight_recovery=recovery)
    validate_state(s,c,generation)
    return s,dict(snapshot_rows_used=trained,snapshot_rows_enabled=before,critic_training=work,
        ppo_snapshot_distinct_used=ppo_snapshot_rows,value_snapshot_distinct_used=value_rows,
        zero_weight_learner_observation=zero_observation,
        snapshot_weight_floor=c['rollout']['reduced_snapshot_weight'],
        snapshot_rows_next=s['snapshots_active'],weight_used=weight,weight_next=next_weight,
        decision=decision,reason=reason,learner_heldout_ev=ev,
        learner_rolling_r2=evidence['learner']['r2'],learner_mse=evidence['learner']['learner_mse'],
        learner_prediction_bias=evidence['learner']['prediction_bias'],
        learner_target_variance=evidence['learner']['target_variance'],
        fingerprint_mean_absolute_drift=drift,weight_recovery=evidence,
        scope='Snapshot weight uses frozen-reference rolling R2; descriptor admission and GAE EV gates unchanged')


def _advance_ev(s,c,*,generation,report,healthy,fingerprint_now,ppo_snapshot_rows=0):
    """Archived v1 transition for checkpoint regression/reproduction only."""
    from .value_phase import critic_work
    work=critic_work(report)
    s=copy.deepcopy(validate_state(s,c,generation-1));before=s['snapshots_active'];weight=s['snapshot_weight']
    metrics=report.get('heldout_after',{});ev=learner_ev(metrics);gain=report.get('descriptor_gain',{}).get('learner_ev_gain')
    value_snapshot_rows=(report.get('role_usage',{}).get('snapshot',{}).get('distinct_used',0)
        if report.get('optimizer_steps',0)>0 else 0)
    require(all(type(n) is int and n>=0 for n in (ppo_snapshot_rows,value_snapshot_rows)),
        'Invalid retained snapshot training counts')
    trained=bool(ppo_snapshot_rows or value_snapshot_rows)
    decision='hold';reason='health_or_descriptor_evidence_not_ready'
    valid=lambda x:type(x) in (int,float) and math.isfinite(x)
    recovery=s['weight_recovery'];streak_before=recovery['healthy_streak'];evidence_streak=0
    baseline=s['learner_ev_baseline'];ev_recovered=valid(baseline) and valid(ev) and ev>=baseline
    health_ok=healthy is True and not report.get('kl_guard_tripped',False)
    eligible=bool(before and trained and health_ok and ev_recovered)
    # Consecutive evidence belongs to the currently used weight. A missing or
    # unhealthy iteration, reduction, or successful stage change clears it.
    recovery['healthy_streak']=0
    if not before and c['snapshot_rows'] and c['rollout']['mode']=='auto':
        if healthy and work['optimizer_steps']>0 and valid(ev) and valid(gain) and gain>=c['rollout']['descriptor_ev_min_gain']:
            s.update(snapshots_active=True,learner_ev_baseline=ev,activation_generation=generation)
            decision='enable_snapshots';reason='heldout_descriptor_gain_and_health'
    elif before:
        # Enabling the route does not create snapshot opponents. In an IL-only
        # pool there may be no such rows for many iterations. Attribute an EV
        # change to this rollout only after actual retained PPO or PPG fitting;
        # available rows or rejected PPG attempts are not training exposure.
        if not trained:reason='no_snapshot_training'
        elif valid(baseline) and valid(ev) and ev<baseline-c['rollout']['snapshot_ev_tolerance'] and weight>.5:
            s['snapshot_weight']=.5;decision='reduce_snapshot_weight';reason='learner_heldout_ev_below_phase1_baseline'
        elif weight<1.:
            if not valid(baseline) or not valid(ev):reason='snapshot_recovery_ev_unavailable'
            elif not health_ok:reason='snapshot_recovery_health_not_ready'
            elif not ev_recovered:reason='snapshot_recovery_ev_below_phase1_baseline'
            else:
                evidence_streak=streak_before+1
                if evidence_streak>=RECOVERY_ITERATIONS:
                    s['snapshot_weight']=SNAPSHOT_WEIGHTS[SNAPSHOT_WEIGHTS.index(weight)+1]
                    decision='increase_snapshot_weight';reason='sustained_healthy_learner_ev_recovery'
                else:
                    recovery['healthy_streak']=evidence_streak;reason='snapshot_recovery_streak'
        else:reason='snapshots_active'
    else:reason='configured_learner_only'
    if s['snapshot_weight']!=weight:recovery['last_change_generation']=generation
    previous=s['previous_fingerprint'];drift=None
    if previous is not None and fingerprint_now is not None:
        drift=float(np.mean(np.abs(np.asarray(previous)-np.asarray(fingerprint_now['values']))))
    if fingerprint_now is not None:s['previous_fingerprint']=fingerprint_now['values']
    s['generation']=generation
    validate_state(s,c,generation)
    return s,dict(snapshot_rows_used=trained,snapshot_rows_enabled=before,critic_training=work,
        ppo_snapshot_distinct_used=ppo_snapshot_rows,value_snapshot_distinct_used=value_snapshot_rows,
        snapshot_rows_next=s['snapshots_active'],weight_used=weight,
        weight_next=s['snapshot_weight'],decision=decision,reason=reason,learner_heldout_ev=ev,
        phase1_learner_ev=s['learner_ev_baseline'],fingerprint_mean_absolute_drift=drift,
        weight_recovery=dict(contract=RECOVERY_CONTRACT,origin=recovery['origin'],
            required_iterations=RECOVERY_ITERATIONS,healthy_streak_before=streak_before,
            evidence_streak=evidence_streak,healthy_streak_next=recovery['healthy_streak'],
            eligible=eligible,health_ok=health_ok,ev_recovered=ev_recovered,
            ev_threshold=baseline,reduce_below=baseline-c['rollout']['snapshot_ev_tolerance'] if valid(baseline) else None,
            next_weight=SNAPSHOT_WEIGHTS[SNAPSHOT_WEIGHTS.index(s['snapshot_weight'])+1] if s['snapshot_weight']<1. else None,
            last_change_generation=recovery['last_change_generation'],
            scope='Five consecutive healthy, actually trained iterations at each weight; EV at least the fixed phase-1 baseline. Changes apply next iteration; no historical backfill'),
        snapshot_effect_ev_delta=(ev-s['learner_ev_baseline'] if before and trained and valid(ev) and valid(s['learner_ev_baseline']) else None),
        scope='Actual retained PPO/PPG snapshot exposure required; counts per phase may overlap. Sequential heldout diagnostic, not a causal snapshot benefit estimate')
