"""Causal per-seat soft commitments and bounded root-only prior preferences."""
import copy
import math
import numpy as np

from brass_ai.learning.contracts import digest,require
from brass_ai.personality.contracts import GOALS,FACTORS,FAMILIES,INDUSTRIES,REGIONS,HORIZONS,compatible,context,context_vector
from brass_ai.search.common import ranked_candidates,turn_key


def public_progress(view,industry):
    seat=view['seat'];p=view['players'][seat]
    mat=next((m['tiles'] for m in p['mat'] if m['type']==industry),[])
    # Removing a low tile by development is progress, even without a new build.
    return dict(tier=min((t['level'] for t in mat),default=9),cash=p['cash'],income=p['income'],
        flipped=sum(bool(s.get('tile',{}).get('flipped')) for s in view['slots'] if s.get('tile',{}).get('seat')==seat),
        links=sum(l.get('owner')==seat for l in view['links']))


def viable(goal,view):
    if goal is None:return False
    family=goal['family'];industry=goal['industry'];region=goal['region'];seat=view['seat']
    if family=='flexible':return False
    if family=='sale_chain':return any(s.get('tile',{}).get('seat')==seat and
        s['tile']['type']==industry and not s['tile']['flipped'] and (region=='unspecified' or s['id'].split(':')[0]==region) for s in view['slots'])
    if family=='industry_sequence':return any(m['type']==industry and m['tiles'] for m in view['players'][seat]['mat']) and any(
        industry in s['types'] and (not s.get('tile') or s['tile']['seat']==seat) and
        (region=='unspecified' or s['id'].split(':')[0]==region) for s in view['slots'])
    if family=='network_extension':return any('owner' not in l and (region=='unspecified' or region in l['regions']) for l in view['links'])
    if family=='resource_supply':return any(m['type']==industry and m['tiles'] for m in view['players'][seat]['mat']) or any(
        s.get('tile',{}).get('seat')==seat and s['tile']['type']==industry and s['tile']['resource']>0 for s in view['slots'])
    return True


def proposal(response,view,profile,calibration):
    if not calibration.get('bias_ready'):return None
    logits=np.asarray(response['goals']['family'],dtype=float)
    preferences={'neutral':[0]*6,'industrialist':[.08,.1,0,0,0,0],
        'networker':[0,0,0,.1,0,0],'supplier':[0,0,.1,0,0,0],'investor':[0,.05,0,0,.1,0]}
    family=FAMILIES[int(np.argmax(logits+preferences[profile.style]))]
    if family=='flexible':return None
    candidates=[x for x in INDUSTRIES if compatible(family,x,'unspecified')]
    industry=max(candidates,key=lambda x:response['goals']['industry'][INDUSTRIES.index(x)])
    regions=['unspecified'] if family=='liquidity_recovery' else list(REGIONS)
    region=max(regions,key=lambda x:response['goals']['region'][REGIONS.index(x)])
    h=int(np.argmax(response['goals']['horizon']));fi=FAMILIES.index(family)
    logit=response['confidence_logits'][fi][h]/calibration['temperature']
    confidence=1/(1+math.exp(-float(np.clip(logit,-50,50))))
    if confidence<.55:return None
    goal=dict(schema=GOALS,family=family,industry=industry,region=region,horizon=HORIZONS[h],confidence=confidence,
        age=0,created_turn=digest(turn_key(view['context'])),revision_reason='new-proposal')
    if not viable(goal,view):
        goal['region']='unspecified'
        if not viable(goal,view):return None
    return goal


class IntentState:
    def __init__(self,seat,profile):
        self.seat=seat;self.profile=profile;self.goal=None;self.key=None;self.progress=None;self.stall=0;self.revisions=[]

    def before_turn(self,view):
        require(view['seat']==self.seat,'Private intent belongs to another seat')
        key=turn_key(view['context'])
        if self.key!=key and self.goal:
            current=public_progress(view,self.goal['industry'])
            productive=self.progress is not None and any(current[k]>self.progress[k] for k in current)
            self.stall=0 if productive else self.stall+1
            self.goal['age']=min(4,self.goal['age']+1)
            if productive:self.goal['revision_reason']='productive-progress'
            if self.key[0]!=key[0] or not viable(self.goal,view) or self.stall>=self.profile.patience or self.goal['age']>=self.goal['horizon']:
                self.revisions.append(dict(reason='expired-or-disrupted',turn=list(key)));self.goal=None;self.stall=0
        self.key=key
        return context(self.profile.style,copy.deepcopy(self.goal))

    def commit(self,response,view,calibration):
        new=proposal(response,view,self.profile,calibration)
        if new and (self.goal is None or new['confidence']>self.goal['confidence']+self.profile.hysteresis):
            self.goal=new;self.stall=0;self.revisions.append(dict(reason='new-proposal',turn=list(self.key)))
        if self.goal:self.progress=public_progress(view,self.goal['industry'])
        self.revisions=self.revisions[-16:]
        return copy.deepcopy(self.goal)

    def snapshot(self):
        return dict(seat=self.seat,profile=self.profile.to_dict(),goal=copy.deepcopy(self.goal),
            turn=list(self.key) if self.key else None,progress=self.progress,stall=self.stall,revisions=self.revisions)

    def restore(self,value):
        require(value['seat']==self.seat and value['profile']==self.profile.to_dict(),'Intent save attribution differs')
        context_vector(context(self.profile.style,value['goal']))
        require(type(value['stall']) is int and value['stall']>=0,'Invalid intent stall age')
        self.goal=copy.deepcopy(value['goal']);self.key=tuple(value['turn']) if value['turn'] else None
        self.progress=value['progress'];self.stall=value['stall'];self.revisions=value['revisions'][-16:]


def alignment(view,choices,goal):
    """One bounded alignment at the initial action root, never per payment step."""
    result=np.zeros(len(choices))
    if goal is None or view['context']['phase']!='action':return result
    family,industry,region=goal['family'],goal['industry'],goal['region']
    p=view['players'][view['seat']]
    next_tile=next((m['tiles'][0] for m in p['mat'] if m['type']==industry and m['tiles']),None)
    needs_money=next_tile is not None and p['cash']<next_tile['cost']
    for i,c in enumerate(choices):
        k=c['kind'];score=0.
        if family=='sale_chain':
            score=1. if k=='sell' else (.25 if k=='network' else 0.)
        elif family=='industry_sequence':
            score=1. if k=='build' and c.get('industry')==industry else (.5 if k=='develop' and c.get('industry')==industry else .3 if k=='loan' and needs_money else 0.)
        elif family=='resource_supply':score=1. if k=='build' and c.get('industry')==industry else .25 if k=='develop' and c.get('industry')==industry else 0.
        elif family=='network_extension':score=1. if k=='network' else .2 if k=='loan' and p['cash']<15 else 0.
        elif family=='liquidity_recovery':score=1. if k=='loan' else .5 if k=='sell' else 0.
        location=c.get('target','')
        if region!='unspecified' and location:
            link=next((l for l in view['links'] if l['id']==location),None)
            matches=region in link['regions'] if link else str(location).split(':')[0]==region
            if not matches:score*=.5
        result[i]=np.clip(score,-1,1)
    return result


def biased_prior(base,raw,choices,alignment_values,profile,confidence):
    base=np.asarray(base,dtype=float);raw=np.asarray(raw,dtype=float);g=np.asarray(alignment_values,dtype=float)
    require(base.shape==raw.shape==g.shape and np.isfinite(g).all() and np.max(np.abs(g))<=1,'Unbounded prior alignment')
    if profile.alpha==0 or confidence<=0 or not np.any(g):return base.copy()
    require(0<=confidence<=1,'Invalid goal confidence')
    from brass_ai.step5.policy import exploration_distribution
    tilted=raw*np.exp(profile.alpha*confidence*g);tilted/=tilted.sum()
    preferred=(1-profile.epsilon)*tilted+profile.epsilon*raw
    # Preserve the existing search policy's 10% grouped exploration outside
    # the bounded neural prior, including every legal industry family.
    mixed=.9*preferred+.1*exploration_distribution(choices)
    return mixed/mixed.sum()


def candidate_order(choices,base,biased,top_k,profile):
    if profile.alpha==0 or np.array_equal(base,biased):return ranked_candidates(choices,base)
    unbiased,protected=ranked_candidates(choices,base);order,protected2=ranked_candidates(choices,biased)
    reserve=max(1,math.ceil(top_k*profile.unbiased_fraction));leading=unbiased[:reserve]
    return leading+[i for i in order if i not in leading],sorted(set(protected+protected2))
