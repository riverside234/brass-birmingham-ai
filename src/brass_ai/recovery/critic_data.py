"""Scored-opponent critic admission, immutable table metadata and rollout gates."""
import copy
import math
import numpy as np

from brass_ai.learning.contracts import digest,require
from brass_ai.learning.seat_descriptors import (CONTRACT,ROLES,FAMILIES,INDUSTRIES,
    descriptor,validate,relative)

DEFAULT=dict(contract='critic-rows/1.0.0',learner_rows=True,snapshot_rows=True,snapshot_weight=1.,
    excluded_roles=['il','rule_bot'],holdout_fraction=.05,
    rollout=dict(mode='auto',descriptor_ev_min_gain=0.,snapshot_ev_tolerance=.02,reduced_snapshot_weight=.5))


def validate_config(c):
    require(type(c) is dict and set(c)==set(DEFAULT) and c['contract']==DEFAULT['contract']
        and c['learner_rows'] is True and type(c['snapshot_rows']) is bool
        and c['excluded_roles']==DEFAULT['excluded_roles'],'Invalid critic row contract')
    require(type(c['snapshot_weight']) in (int,float) and c['snapshot_weight'] in (.5,1.)
        and type(c['holdout_fraction']) in (int,float) and c['holdout_fraction']==.05,
        'Invalid critic snapshot weight/holdout')
    require(type(c['rollout']) is dict and set(c['rollout'])==set(DEFAULT['rollout'])
        and c['rollout']['mode'] in ('auto','learner_only','snapshots')
        and all(c['rollout'][k]==v for k,v in DEFAULT['rollout'].items() if k!='mode'),
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


def planned_table(policies,rotation,current,generation,members,learner_fp,rule_fp,*,learner_policy_id):
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
        require(0<=born<=generation,'Opponent from a future generation')
        seats.append(dict(role=role,generation=born,model_id=mid,policy_id=p.get('policy_id'),
                          fingerprint_id=digest(fp)))
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
        _seat_descriptors=relative(table['stable'],encoded.stable_to_relative),
        _descriptor_generation=table['generation'])


def initial(c):
    return dict(generation=0,snapshots_active=c['snapshot_rows'] and c['rollout']['mode']=='snapshots',
        snapshot_weight=c['snapshot_weight'],learner_ev_baseline=None,activation_generation=None,
        previous_fingerprint=None)


def validate_state(s,c,generation):
    require(type(s) is dict and set(s)==set(initial(c)) and s['generation']==generation
        and type(s['snapshots_active']) is bool and s['snapshot_weight'] in (.5,1.),'Invalid critic rollout checkpoint')
    require(not s['snapshots_active'] or c['snapshot_rows'],'Unexpected snapshot activation')
    for k in ('learner_ev_baseline',):
        require(s[k] is None or (type(s[k]) in (int,float) and math.isfinite(s[k])),'Invalid critic rollout EV')
    require(s['activation_generation'] is None or
        type(s['activation_generation']) is int and 0<=s['activation_generation']<=generation,'Invalid critic activation age')
    return s


def learner_ev(metrics):return metrics.get('groups',{}).get('role:learner',{}).get('explained_variance')


def advance(s,c,*,generation,report,healthy,fingerprint_now):
    s=copy.deepcopy(validate_state(s,c,generation-1));before=s['snapshots_active'];weight=s['snapshot_weight']
    metrics=report.get('heldout_after',{});ev=learner_ev(metrics);gain=report.get('descriptor_gain',{}).get('learner_ev_gain')
    decision='hold';reason='health_or_descriptor_evidence_not_ready'
    valid=lambda x:type(x) in (int,float) and math.isfinite(x)
    if not before and c['snapshot_rows'] and c['rollout']['mode']=='auto':
        if healthy and report.get('optimizer_steps',0)>0 and valid(ev) and valid(gain) and gain>=c['rollout']['descriptor_ev_min_gain']:
            s.update(snapshots_active=True,learner_ev_baseline=ev,activation_generation=generation)
            decision='enable_snapshots';reason='heldout_descriptor_gain_and_health'
    elif before:
        baseline=s['learner_ev_baseline']
        if valid(baseline) and valid(ev) and ev<baseline-c['rollout']['snapshot_ev_tolerance'] and weight>.5:
            s['snapshot_weight']=.5;decision='reduce_snapshot_weight';reason='learner_heldout_ev_below_phase1_baseline'
        else:reason='snapshots_active'
    else:reason='configured_learner_only'
    previous=s['previous_fingerprint'];drift=None
    if previous is not None and fingerprint_now is not None:
        drift=float(np.mean(np.abs(np.asarray(previous)-np.asarray(fingerprint_now['values']))))
    if fingerprint_now is not None:s['previous_fingerprint']=fingerprint_now['values']
    s['generation']=generation
    return s,dict(snapshot_rows_used=before,snapshot_rows_next=s['snapshots_active'],weight_used=weight,
        weight_next=s['snapshot_weight'],decision=decision,reason=reason,learner_heldout_ev=ev,
        phase1_learner_ev=s['learner_ev_baseline'],fingerprint_mean_absolute_drift=drift,
        snapshot_effect_ev_delta=(ev-s['learner_ev_baseline'] if before and valid(ev) and valid(s['learner_ev_baseline']) else None),
        scope='Sequential heldout diagnostic; iteration distributions differ, not a causal snapshot benefit estimate')
