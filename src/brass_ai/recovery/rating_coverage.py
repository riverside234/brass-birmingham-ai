"""Qualified, balanced strong-checkpoint lineups; the PL estimator is unchanged."""
import copy
from collections import Counter

from brass_ai.learning.contracts import digest, require
from .plackett_luce import IL, V4

CONTRACT='competitive-rating-coverage/1.0.0'
DEFAULT=dict(contract=CONTRACT,win_share_above=.8,reference_setups_each=4,
    competitive_setups=24,selection='nearest-balanced-distinct')


def validate(config, rating):
    require(type(config) is dict and config==DEFAULT and rating['setups']==32 and
        rating['seat_rotations']==4 and rating['nearest_opponents']>=3,
        'Competitive rating coverage requires the versioned 32-setup protocol')
    return config


def prepare(root, state, config):
    if 'rating_coverage' not in config:return None
    validate(config['rating_coverage'],config['rating'])
    from . import panel_gates
    saved=copy.deepcopy(state['ratings'].get('coverage',dict(contract=CONTRACT,activated=None)))
    require(set(saved)=={'contract','activated'} and saved['contract']==CONTRACT,'Invalid rating-coverage state')
    active=saved['activated']
    if active is not None:
        require(set(active)=={'generation','panels'} and type(active['generation']) is int and
            0<=active['generation']<=state['iteration'] and set(active['panels'])==set(panel_gates.NAMES),
            'Invalid competitive rating activation')
        for name, evidence in active['panels'].items():
            require(evidence['generation']<=active['generation'] and evidence['win_share']>.8,
                'Competitive rating gate requires both panel win shares above 80%')
            panel_gates.verify_evidence(root,state['ratings'],name,evidence)
        return saved
    evidence=panel_gates.latest_evidence(root,state,config)
    if all(evidence[n] is not None and evidence[n]['win_share']>.8 for n in panel_gates.NAMES):
        saved['activated']=dict(generation=state['iteration'],panels=evidence)
    return saved


def lineups(rating, agents, candidate, scores, generation, anchor=None):
    """One balanced list per setup, rotated identically through all four seats."""
    c=rating['config']; seeds=rating['seeds']
    checkpoints=[a for a,e in agents.items() if a not in (IL,V4,candidate) and
        e['kind']=='model' and a in scores]
    last=max(checkpoints,key=lambda a:(agents[a]['checkpoint']['generation'],a)) if checkpoints else None
    theta=scores[last]['theta'] if last else 0.
    ordered=sorted(checkpoints,key=lambda a:(abs(scores[a]['theta']-theta),a))
    # Checkpoint generations can share weights. A model identity gets one slot.
    excluded={agents[candidate]['checkpoint']['model_id'],agents[IL]['checkpoint']['model_id']}
    available=[]; seen=set(excluded)
    required=list(dict.fromkeys(a for a in (last,anchor) if a in checkpoints))
    for a in required+ordered:
        identity=agents[a]['checkpoint']['model_id']
        if identity not in seen: available.append(a); seen.add(identity)
        if len(available)==c['nearest_opponents']:break
    # Spread reference setups every four positions, and rotate that pattern
    # through all fixed seeds over successive slots, rather than pinning IL/v4.
    offset=(generation//c['every_iterations'])%len(seeds)
    references={(offset+4*i)%len(seeds):IL if i%2==0 else V4 for i in range(8)}
    counts=Counter(); rows=[]; fallback=Counter(); position_counts=Counter()
    for i,seed in enumerate(seeds):
        base=[candidate]; reference=references.get(i)
        if reference:base.append(reference)
        needed=4-len(base);chosen=[]
        for _ in range(min(needed,len(available))):
            options=[a for a in available if a not in chosen]
            a=min(options,key=lambda a:(counts[a],digest(dict(seed=c['seed'],slot=generation,setup=seed,agent=a))))
            chosen.append(a);counts[a]+=1
        base+=chosen
        # Explicit startup/duplicate-model fallback: unique references first;
        # only duplicate a seat when there are not enough frozen identities.
        for a in (IL,V4):
            if len(base)<4 and a not in base:base.append(a);fallback[a]+=1
        while len(base)<4:
            options=available or [IL,V4]
            a=min(options,key=lambda a:(position_counts[a]+base.count(a),a))
            base.append(a);fallback[a]+=1
        position_counts.update(base[1:])
        for seat in range(4):rows.append(dict(seed=seed,seat=seat,kind='rating',agents=base))
    require(len(rows)==128,'Competitive rating changed the game budget')
    return rows,dict(contract=CONTRACT,active=True,estimated_theta=theta,nearest=available,
        required_opponents=required,previous_rated_checkpoint=last,active_kl_reference=anchor,
        reference_setup_indices={str(i):a for i,a in sorted(references.items())},
        rated_setup_appearances=dict(counts),opponent_setup_appearances=dict(position_counts),
        startup_fallback_seats=dict(fallback),main_games=128,
        rated_opponents_per_game=sum(counts.values())/len(seeds))
