"""Public own-turn trajectories. Identity/outcome metadata never enters features."""
from pathlib import Path
import numpy as np

from brass_ai.bank import sha
from brass_ai.learning.contracts import digest,require
from brass_ai.learning.replay import terminal_targets
from brass_ai.personality.contracts import INDUSTRIES,REGIONS
from brass_ai.search.common import turn_key

VERSION='public-behavior-trace/1.1.0'
MAX_TURNS=32
KINDS=('build','develop','sell','network','loan','pass','scout','coal','iron','beer','liquidate','bonus_develop','bonus_skip')
INDUSTRY=INDUSTRIES[1:]
FIELDS=('canal','rail','round','cash','cash_change','income','income_change','coal_market','iron_market','merchant_beer_used','effects_observed','era_boundary')+tuple('action_'+x for x in KINDS)+tuple(
    f'{kind}_{industry}' for kind in ('build','built_level','develop','flips','unflipped','resources','vacancies','merchant_access') for industry in INDUSTRY)+tuple('area_'+x for x in REGIONS[1:])


def schema():return dict(format=VERSION,features=list(FIELDS),max_turns=MAX_TURNS,
    excluded=['controller/style IDs','seeds','absolute seat ID','hands/deck','machine timing','final VP/win/rank'],
    temporal='Ordered own turns, both eras; public subchoices and their adjacent observed effects aggregated within a turn',
    effects='effects_observed is the fraction of own choices with a same-era successor observation; era-clear and absent terminal observations are censored')


def trajectory(game,seat):
    rows=game['records'];groups=[]
    for i,r in enumerate(rows):
        c=r['view']['context']
        if r['seat']!=seat or c['turn_owner']!=seat or c['phase']=='debt':continue
        key=turn_key(c)
        if not groups or groups[-1]['key']!=key:groups.append(dict(key=key,indexes=[]))
        groups[-1]['indexes'].append(i)
    require(0<len(groups)<=MAX_TURNS,'Invalid/oversized behavior trajectory')
    x=np.zeros((MAX_TURNS,len(FIELDS)),dtype=np.float32);mask=np.zeros(MAX_TURNS,dtype=bool)
    for t,group in enumerate(groups):
        start=group['indexes'][0]
        view=rows[start]['view'];p=view['players'][seat]
        values=dict.fromkeys(FIELDS,0.)
        values.update(canal=float(group['key'][0]=='CanalEra'),rail=float(group['key'][0]=='RailEra'),round=group['key'][1]/8,
            cash=p['cash']/100,income=p['income']/30,
            coal_market=view['coal_market']/24,iron_market=view['iron_market']/24)
        for s in view['slots']:
            tile=s.get('tile')
            if tile and tile['seat']==seat:
                values['unflipped_'+tile['type']]+=float(not tile['flipped'])
                values['resources_'+tile['type']]+=tile['resource']
            if tile is None:
                for industry in s['types']:values['vacancies_'+industry]+=1
        # Sale access is public connectivity through any player's built links;
        # it must not be inferred from hand-dependent legal build choices.
        graph={}
        for link in view['links']:
            if 'owner' in link:
                for city in link['regions']:graph.setdefault(city,set()).update(x for x in link['regions'] if x!=city)
        reachable={}
        for industry in INDUSTRY:
            seen={s['id'].split(':')[0] for s in view['slots'] if s.get('tile',{}).get('seat')==seat
                and s['tile']['type']==industry and not s['tile']['flipped']}
            pending=list(seen)
            while pending:
                for city in graph.get(pending.pop(),()):
                    if city not in seen:seen.add(city);pending.append(city)
            reachable[industry]=seen
        for merchant in view['merchants']:
            for industry in merchant['types']:
                if merchant['id'].split(':')[0] in reachable[industry]:values['merchant_access_'+industry]+=1
        for i in group['indexes']:
            r=rows[i];c=r['choices'][r['chosen']];kind=c['kind']
            # The next record is BEFORE the next action, so this transition
            # contains only the just-selected choice's public effects. Waiting
            # until our next turn instead attributes opponents' consumption to
            # us and loses flipped level-I tiles at the Canal/Rail clear.
            if i+1<len(rows):
                before=r['view'];after=rows[i+1]['view']
                if after['context']['era']==before['context']['era']:
                    values['effects_observed']+=1/len(group['indexes'])
                    values['cash_change']+=(after['players'][seat]['cash']-before['players'][seat]['cash'])/100
                    values['income_change']+=(after['players'][seat]['income']-before['players'][seat]['income'])/30
                    previous={s['id']:s.get('tile') for s in before['slots']}
                    for s in after['slots']:
                        tile=s.get('tile');old=previous.get(s['id'])
                        if tile and tile['seat']==seat and tile['flipped'] and (not old or old['id']!=tile['id'] or not old['flipped']):
                            values['flips_'+tile['type']]+=1
                else:values['era_boundary']=1.
            if kind in KINDS:values['action_'+kind]+=1
            if kind=='build':
                industry=c['industry'];values['build_'+industry]+=1
                mat=next(m['tiles'] for m in r['view']['players'][seat]['mat'] if m['type']==industry)
                values['built_level_'+industry]+=(mat[0]['level'] if mat else 0)/4
            if kind in ('develop','bonus_develop'):values['develop_'+c['industry']]+=1
            if kind=='beer' and c['target'] in {'merchant:'+m['id'] for m in r['view']['merchants']}:values['merchant_beer_used']+=1
            if kind in ('build','sell','network'):
                link=next((l for l in r['view']['links'] if l['id']==c['target']),None)
                areas=link['regions'] if link else [c['target'].split(':')[0]]
                for area in areas:
                    if 'area_'+area in values:values['area_'+area]+=1/len(areas)
        x[t]=[values[name] for name in FIELDS];mask[t]=True
    require(np.isfinite(x).all(),'Nonfinite public trajectory')
    return x,mask


def from_collection(directory,*,include_parents=False):
    from brass_ai.personality.search.collection import read
    manifest,pairs=read(directory,teacher_game=False,lazy=True);output=[]
    require(not manifest['spec']['calibration_probe'],'Readiness/calibration collections cannot fit discovery descriptors')
    require(manifest['spec'].get('discovery_id') is None or manifest['spec']['discovery_id'],'Missing discovery ID')
    for game,side in pairs:
        job=side['job'];targets=terminal_targets(game['result'])
        for seat in range(4):
            active=seat in job['neural_seats']
            if not active and not include_parents:continue
            x,mask=trajectory(game,seat)
            controller=job['controllers'][seat] if active else game['opponents'][seat]
            # The descriptor API takes ONLY arrays. These fields are retained
            # separately for grouped splits, diagnostics and own-seat quality.
            output.append(dict(values=x,mask=mask,metadata=dict(trace_id=digest(dict(collection=manifest['id'],game=game['game_id'],seat=seat)),
                game_id=game['game_id'],setup_group=game['setup_group'],split=game['split'],candidate_id=controller,
                seat=seat,active=active,quality={k:targets[k][seat] for k in ('win_share','placement','vp')},
                source=manifest['id'],pool_id=manifest['spec']['pool_id'],model_id=manifest['spec']['model_id'],
                search=manifest['spec']['search'])))
    return manifest,output


def arrays(records):
    require(records,'No behavior records')
    x=np.stack([r['values'] for r in records]);mask=np.stack([r['mask'] for r in records])
    require(x.shape==(len(records),MAX_TURNS,len(FIELDS)) and mask.shape==x.shape[:2]
        and mask.dtype==np.bool_ and np.isfinite(x).all(),'Invalid temporal features/mask')
    require(all(0<m.sum()<=MAX_TURNS and np.array_equal(m,np.arange(MAX_TURNS)<m.sum()) for m in mask),'Temporal masks must be nonempty ordered prefixes')
    return x,mask
