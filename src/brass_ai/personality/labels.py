"""Masked future PUBLIC completion labels; never future features or PPO inputs.

These labels describe realized behavior, not optimal plans or hidden intentions.
Confidence predicts each defined event within 1/2/4 own turns, including failures.
"""
import numpy as np
import torch
from torch.nn import functional as F

from brass_ai.learning.contracts import require
from .contracts import FAMILIES,INDUSTRIES,REGIONS,HORIZONS,FACTORS,compatible


def turn_key(view):
    c=view['context'];return (c['era'],c['round'],c['turn'],c['turn_owner'])


def _public(view):
    # Explicit projection: other seats' actual hands, seeds or future private
    # observations cannot enter label extraction, much less present features.
    return dict(context=view['context'],slots=view['slots'],links=view['links'],
                players=[{k:p[k] for k in ('seat','cash','income')} for p in view['players']])


def events(game):
    rows=game['records'];result=[];born={}
    def add(step,seat,family,industry='none',regions=(),origin=None):
        result.append(dict(step=step,seat=seat,family=family,industry=industry,
            regions=tuple(r for r in regions if r in REGIONS),origin=origin))
    for i in range(len(rows)-1):
        a,b=_public(rows[i]['view']),_public(rows[i+1]['view'])
        old={s['id']:s.get('tile') for s in a['slots']}
        for slot in b['slots']:
            tile=slot.get('tile');previous=old.get(slot['id'])
            if not tile:continue
            token=(slot['id'],tile['id']);same=previous and previous['id']==tile['id']
            if not same:born[token]=i
            area=(slot['id'].split(':')[0],)
            if tile['flipped'] and not (same and previous['flipped']):
                if tile['type'] in INDUSTRIES[1:4]:add(i+1,tile['seat'],'sale_chain',tile['type'],area)
                if token in born:add(i+1,tile['seat'],'industry_sequence',tile['type'],area,born[token])
            if same and previous['resource']>tile['resource'] and tile['type'] in INDUSTRIES[4:]:
                add(i+1,tile['seat'],'resource_supply',tile['type'],area)
        old_links={l['id']:l.get('owner') for l in a['links']}
        for link in b['links']:
            if 'owner' in link and old_links[link['id']]!=link['owner']:
                add(i+1,link['owner'],'network_extension',regions=link['regions'])
        chosen=rows[i]['choices'][rows[i]['chosen']];seat=rows[i]['seat']
        if chosen['kind']=='loan' and b['players'][seat]['cash']>a['players'][seat]['cash']:
            add(i+1,seat,'liquidity_recovery')
    return result


def goal_labels(game):
    """One label at the first ordinary action of each actual own turn.

    Same-era horizons include the remainder of that turn. Opponent turns do
    not advance the counter; a short era/game masks unobserved negative labels.
    The final action's unseen effect is not inferred from the final score.
    """
    rows=game['records'];observed=events(game);turns=[]
    for i,row in enumerate(rows):
        c=row['view']['context']
        if c['phase']=='debt':continue
        key=turn_key(row['view'])
        if not turns or turns[-1]['key']!=key:turns.append(dict(key=key,start=i))
    for i,t in enumerate(turns):t['end']=turns[i+1]['start'] if i+1<len(turns) else len(rows)-1
    output={}
    for t in turns:
        root=t['start'];row=rows[root];seat=row['seat'];era=t['key'][0]
        if row['view']['context']['phase']!='action' or seat!=t['key'][3]:continue
        future=[x for x in turns if x['start']>=root and x['key'][0]==era and x['key'][3]==seat][:4]
        confidence=np.zeros((len(FAMILIES),len(HORIZONS)),dtype=np.float32)
        mask=np.zeros_like(confidence,dtype=bool);found=[]
        for hi,horizon in enumerate(HORIZONS):
            end=future[min(horizon,len(future))-1]['end']
            hits=[e for e in observed if root<e['step']<=end and e['seat']==seat and
                rows[e['step']]['view']['context']['era']==era and (e['origin'] is None or e['origin']>=root)]
            full=len(future)>=horizon and end<len(rows)-1
            for fi,family in enumerate(FAMILIES):
                present=not hits if family=='flexible' else any(e['family']==family for e in hits)
                mask[fi,hi]=full or (present and family!='flexible')
                confidence[fi,hi]=float(present)
            if hi==2:found=hits
        values=dict(family=5,industry=0,region=0,horizon=2);valid={k:False for k in FACTORS}
        if found:
            first=min(e['step'] for e in found);earliest=[e for e in found if e['step']==first]
            # Several public effects may complete simultaneously; do not invent
            # a single intent family, industry or region where it is ambiguous.
            families={e['family'] for e in earliest}
            if len(families)==1:
                family=next(iter(families));values['family']=FAMILIES.index(family);valid['family']=True
                for key,choices in [('industry',{e['industry'] for e in earliest}),
                                    ('region',{r for e in earliest for r in e['regions']} or {'unspecified'})]:
                    if len(choices)==1:values[key]=FACTORS[key].index(next(iter(choices)));valid[key]=True
            count=next((i+1 for i,x in enumerate(future) if first<=x['end']),4)
            values['horizon']=next(i for i,h in enumerate(HORIZONS) if h>=count);valid['horizon']=True
        elif mask[5,2]:valid={k:True for k in FACTORS}
        output[root]=dict(values=values,mask=valid,confidence=confidence.tolist(),confidence_mask=mask.tolist(),
            provenance='Public realized events; unseen outcomes/ambiguous factors masked')
    return output


def blank():
    return dict(values={k:0 for k in FACTORS},mask={k:False for k in FACTORS},
        confidence=np.zeros((len(FAMILIES),3)).tolist(),confidence_mask=np.zeros((len(FAMILIES),3),dtype=bool).tolist())


def loss_rows(output,labels):
    """Per-row loss, so minibatching never promotes missing labels to zeros."""
    device=output['policy_logits'].device
    total=output['policy_logits'].new_zeros(len(labels));count=total.clone()
    for name,vocab in FACTORS.items():
        targets=torch.tensor([x['values'][name] for x in labels],device=device)
        present=torch.tensor([x['mask'][name] for x in labels],device=device,dtype=torch.bool)
        logits=output['goals'][name]
        if name in ('industry','region'):
            allowed=[]
            for label in labels:
                family=FAMILIES[label['values']['family']]
                require(not label['mask'][name] or label['mask']['family'],'Factor missing its family')
                allowed.append([compatible(family,item,'unspecified') if name=='industry' else
                    (item=='unspecified' if family in ('liquidity_recovery','flexible') else True) for item in vocab])
            allowed=torch.tensor(allowed,device=device)
            require(bool(allowed.gather(1,targets[:,None]).squeeze(1)[present].all()),'Incompatible supervised goal')
            logits=logits.masked_fill(~allowed,-1e9)
        total+=F.cross_entropy(logits,targets,reduction='none')*present;count+=present
    targets=torch.tensor([x['confidence'] for x in labels],device=device)
    mask=torch.tensor([x['confidence_mask'] for x in labels],device=device)
    bce=F.binary_cross_entropy_with_logits(output['confidence_logits'],targets,reduction='none')
    n=mask.sum((1,2));total+=(bce*mask).sum((1,2))/n.clamp_min(1);count+=(n>0)
    return total/count.clamp_min(1),count>0


def confidence_report(probabilities,labels):
    predicted=np.asarray(probabilities);target=np.asarray([x['confidence'] for x in labels])
    mask=np.asarray([x['confidence_mask'] for x in labels],dtype=bool)
    p,y=predicted[mask],target[mask]
    if not len(y):return dict(samples=0,brier=None,ece=None,qualified=False)
    ece=0.
    for lo in np.arange(0,1,.1):
        b=(p>=lo)&(p<lo+.1 if lo<.9 else p<=1)
        if b.any():ece+=b.mean()*abs(p[b].mean()-y[b].mean())
    return dict(samples=len(y),brier=float(((p-y)**2).mean()),ece=float(ece),
        qualified=False,note='Held-out event calibration diagnostic, not a tactical-veto or release certificate')
