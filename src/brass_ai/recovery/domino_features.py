"""Observable predecision descriptors; sparse phase events, never actor inputs."""
import math
from collections import Counter
from functools import lru_cache
import numpy as np
from brass_ai.learning.contracts import require
from brass_ai.learning.features import feature_schema
from brass_ai.step5.policy import INDUSTRIES
from .teacher_schedule import phase_for

PHASES=('canal_early','canal_late','rail_early','rail_late')
ACTIONS=('build','develop','sell','loan','network','scout','pass')


@lru_cache(maxsize=1)
def manifest():
    schema=feature_schema();slots=list(schema['slots'])
    regions=sorted({s.split(':')[0] for s in slots})
    groups=[dict(name='action',fields=list(ACTIONS),source='choices[chosen].kind at context.phase=action',scale=1.),
        dict(name='investment',fields=[f'{k}.{i}' for k in ('installed_count','installed_level_sum','next_mat_level') for i in INDUSTRIES],
             source='own slots[].tile.{seat,type,level}; own players[].mat[].tiles[0].level (empty stack=8)',scales=[8.]*6+[32.]*6+[8.]*6),
        dict(name='realization',fields=[f'{k}.{i}' for k in ('flipped','unflipped') for i in INDUSTRIES],
             source='own slots[].tile.{seat,type,flipped}; absent tile contributes zero',scale=8.),
        dict(name='financing',fields=['cash','income','spent'],source='players[view.seat].{cash,income,spent}',scales=[100.,30.,100.]),
        dict(name='network',fields=['links']+['footprint:'+r for r in regions]+['asset:'+s for s in slots],
             source='own links[].owner/regions and own slots[].tile.seat; footprint=regions incident to own links or assets',
             scales=[16.]+[1.]*(len(regions)+len(slots)))]
    return dict(contract='brass-strategic-occupancy/1.0.0',phases=list(PHASES),industries=list(INDUSTRIES),
        groups=groups,slots=slots,regions=regions,clip=[-1.,1.],
        weighting='Each family L2 scale 1/sqrt(5*family_width), fixed before training',
        perspective='Acting seat only; labels, contexts, opponent identities and final outcomes excluded',
        missing='Required public fields fail admission; empty tile/stack explicit; terminal missing phases remain zero, no renormalization',
        occupancy='sum phase-block(x)/(4*main_action_count_in_phase); complete games only')


def width(): return sum(len(g['fields']) for g in manifest()['groups'])


def validate_manifest():
    from brass_ai.learning.contracts import CONFIG_DIR,read_json
    p=CONFIG_DIR/'domino-descriptor-1.0.0.json'
    require(p.is_file() and read_json(p)==manifest(),'Missing or incompatible DOMiNO descriptor manifest')
    return manifest()


def event(row):
    v=row['view']
    if v['context']['phase']!='action': return None
    phase=phase_for(v['context']);require(phase in PHASES,'Unknown DOMiNO phase')
    seat=v['seat'];p=v['players'][seat]
    require(p['seat']==seat,'DOMiNO public player perspective mismatch')
    action=row['choices'][row['chosen']]['kind']
    require(action in ACTIONS,'Unknown DOMiNO main action: '+action)
    own=[s for s in v['slots'] if s.get('tile') and s['tile']['seat']==seat]
    tiles=[s['tile'] for s in own];mats={m['type']:m['tiles'] for m in p['mat']}
    require(set(mats)==set(INDUSTRIES),'Missing DOMiNO public industry stack')
    links=[l for l in v['links'] if l.get('owner')==seat]
    footprint={s['id'].split(':')[0] for s in own}|{r for l in links for r in l['regions']}
    owned={s['id'] for s in own};m=manifest()
    values=[[float(action==a) for a in ACTIONS],
        [sum(t['type']==i for t in tiles) for i in INDUSTRIES]+
        [sum(t['level'] for t in tiles if t['type']==i) for i in INDUSTRIES]+
        [mats[i][0]['level'] if mats[i] else 8 for i in INDUSTRIES],
        [sum(t['type']==i and bool(t['flipped'])==flipped for t in tiles) for flipped in (True,False) for i in INDUSTRIES],
        [p[k] for k in ('cash','income','spent')],
        [len(links)]+[float(r in footprint) for r in m['regions']]+[float(s in owned) for s in m['slots']]]
    x=[]
    for group,vals in zip(m['groups'],values):
        a=np.asarray(vals,dtype=np.float64);require(np.isfinite(a).all(),'Nonfinite DOMiNO public feature')
        a=np.clip(a/np.asarray(group.get('scales',group.get('scale'))),-1,1)/math.sqrt(5*len(vals))
        x.extend(a.tolist())
    return dict(phase=PHASES.index(phase),x=x)


def occupancy(events):
    counts=Counter(e['phase'] for e in events if e is not None)
    psi=np.zeros((4,width()),dtype=np.float64)
    for e in events:
        if e is not None: psi[e['phase']]+=np.asarray(e['x'])/(4*counts[e['phase']])
    return psi.reshape(-1),[counts[p] for p in range(4)]
