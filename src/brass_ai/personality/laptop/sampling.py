"""Seat-specific once-per-game selection. JSON only; never pickle save files."""
import copy
import hashlib
import random
import math
from brass_ai.learning.contracts import require,digest
from brass_ai.personality.search.contracts import Profile
from brass_ai.personality.search.intent import IntentState
from brass_ai.personality.difficulty.ui import IntentDisplay,PUBLIC_TEXT,HINT_TEXT
from .contracts import SAMPLING,SAVE,seal,unseal


def stream(package_id,game_id,seed,seat,tag):
    require(type(seat) is int and 0<=seat<4 and all(isinstance(s,str) and s and '\n' not in s for s in (package_id,game_id,seed,tag)), 'Invalid sampling identity')
    value='\n'.join((SAMPLING,package_id,game_id,seed,str(seat),tag))+'\n'
    return hashlib.sha256(value.encode('utf-8')).hexdigest()


def choose(package,game_id,seed,seat,archetype,tier):
    require(tier in ('easy','medium','hard'),'Unknown difficulty')
    groups=package['roster']['archetypes']
    if archetype=='random':archetype=groups[int(stream(package['id'],game_id,seed,seat,'archetype')[:2],16)%16]['id']
    matches=[g for g in groups if g['id']==archetype];require(len(matches)==1,'Unknown style')
    group=matches[0];require(len(group['members'])==8,'Eight members required')
    member=group['members'][int(stream(package['id'],game_id,seed,seat,archetype)[:2],16)%8]
    return dict(seat=seat,archetype=archetype,member=member,tier=tier,
                rng_seed=stream(package['id'],game_id,seed,seat,'policy'))


def tuples(value):return tuple(tuples(v) for v in value) if isinstance(value,list) else value


def new_save(package,game_id,seed,assignments):
    require(len(assignments)<=4 and len({x['seat'] for x in assignments})==len(assignments),'Duplicate seats')
    seats=[]
    for a in sorted(assignments,key=lambda x:x['seat']):
        selected=choose(package,game_id,seed,**a)
        rng=random.Random(int(selected['rng_seed'],16))
        member=next(m for m in package['roster']['members'] if m['id']==selected['member'])
        seats.append(dict(selection=selected,rng=rng.getstate(),intent=IntentState(a['seat'],Profile.parse(member['profile'])).snapshot(),
                          recall=[],ui=None))
    return seal(dict(format=SAVE,sampling=SAMPLING,package_id=package['id'],game_id=game_id,seed=seed,
                     seats=seats,pending=None))


def restore(package,save,game_id):
    unseal(save,SAVE)
    require(save['package_id']==package['id'] and save['game_id']==game_id and save['sampling']==SAMPLING,'Save/package/game mismatch')
    require(len(save['seats'])<=4 and len({s['selection']['seat'] for s in save['seats']})==len(save['seats']),'Duplicate saved seat')
    pending=save['pending']
    if pending is not None:
        require(set(pending)=={'key','remaining_seconds'} and isinstance(pending['key'],list) and len(pending['key'])==5
            and pending['key'][0] in [s['selection']['seat'] for s in save['seats']]
            and type(pending['remaining_seconds']) in (int,float) and math.isfinite(pending['remaining_seconds'])
            and 0<=pending['remaining_seconds']<=12,'Invalid saved whole-turn allowance')
        k=pending['key'];owner=next(s['selection'] for s in save['seats'] if s['selection']['seat']==k[0])
        require(type(k[0]) is int and k[1] in ('CanalEra','RailEra')
                and type(k[2]) is int and 1<=k[2]<=8 and type(k[3]) is int and 1<=k[3]<=4
                and type(k[4]) is int and 0<=k[4]<4
                and pending['remaining_seconds']<={'easy':2.,'medium':5.,'hard':12.}[owner['tier']],
                'Invalid saved whole-turn key or tier allowance')
    for s in save['seats']:
        a=s['selection'];require(a==choose(package,game_id,save['seed'],a['seat'],a['archetype'],a['tier']),'Saved variant differs; never reroll')
        rng=random.Random();rng.setstate(tuples(s['rng']))
        member=next(m for m in package['roster']['members'] if m['id']==a['member'])
        intent=IntentState(a['seat'],Profile.parse(member['profile']));intent.restore(s['intent'])
        require(len(s['recall'])<=256 and all(set(r)=={'revision','hand'} and type(r['revision']) is int
            and isinstance(r['hand'],list) and len(r['hand'])<=10 and all(isinstance(c,str) for c in r['hand']) for r in s['recall']), 'Invalid private recall')
        if s['ui'] is not None:
            u=s['ui'];group=next(g for g in package['roster']['archetypes'] if g['id']==a['archetype'])
            require(set(u)==set(IntentDisplay(a['seat']).__dict__) and u['seat']==a['seat']
                and u['last_text'] in [None,group['description'],*PUBLIC_TEXT.values(),*HINT_TEXT.values()]
                and u['active'] in [None,*HINT_TEXT] and u['pending'] in [None,*HINT_TEXT]
                and type(u['pending_count']) is int and 0<=u['pending_count']<=256
                and u['mode'] in ('hidden','public','intent-hints'),'Invalid saved public display')
            require(u['last_turn'] is None or (isinstance(u['last_turn'],(list,tuple)) and len(u['last_turn'])==2
                and u['last_turn'][0] in ('CanalEra','RailEra') and type(u['last_turn'][1]) is int and 0<=u['last_turn'][1]<=36),
                'Invalid saved public turn')
    return copy.deepcopy(save)
