"""Cotton behavior metadata and coarse per-table statistics; no target shaping."""
import math
from collections import Counter
from brass_ai.learning.contracts import require

CONTRACT='teacher-cotton-exploration/1.0.0'
PROBABILITY=.2
CAP=3


def settings():
    return dict(contract=CONTRACT,probability=PROBABILITY,max_per_seat_era=CAP,
        long_candidates=2,short_candidates=1,pause_unflipped=True,scope='train')


def enabled(job):
    return 'cotton' in (job.get('teacher_revision') or {}).get('exploration',{})


def validate_label(label, choices):
    """Reuse already validated legal order/counts. The improved-policy target is untouched."""
    e=label.get('cotton_exploration')
    if e is None:return
    require(isinstance(e,dict) and set(e)=={'contract','targeted','probability','used_before',
        'cotton_visits','total_visits','covered_locations'} and e['contract']==CONTRACT,
        'Invalid cotton exploration metadata')
    require(label.get('explored') is True and type(e['targeted']) is bool and
        type(e['used_before']) is int and 0<=e['used_before']<CAP,'Invalid cotton exploration allowance')
    total=sum(label['counts']);cotton=0;locations=set();played=None
    for c,n in zip(choices,label['counts']):
        is_cotton=c['kind']=='build' and c.get('industry')=='CottonMill'
        if n and is_cotton:cotton+=n;locations.add(c['target'])
        if c['id']==label['played_action']:played=(n,is_cotton)
    require(total>0 and cotton>0 and played is not None and played[0]>0,
            'Cotton exploration needs visited legal actions')
    require(all(type(e[k]) is int for k in ('cotton_visits','total_visits','covered_locations')) and
        (e['cotton_visits'],e['total_visits'],e['covered_locations'])==(cotton,total,len(locations)),
        'Cotton coverage counters differ from search evidence')
    n,is_cotton=played
    probability=(1-PROBABILITY)*n/total+(PROBABILITY*n/cotton if is_cotton else 0.)
    require((not e['targeted'] or is_cotton) and type(e['probability']) in (int,float) and
        math.isfinite(e['probability']) and abs(e['probability']-probability)<=1e-12,
        'Cotton exploration behavior probability differs')


def validate_receipt(receipt, job, rows):
    """One small table check at publication, after existing row/binding admission."""
    if not enabled(job):
        require(receipt is None, 'Unexpected cotton exploration on an unbiased table')
        return
    require(isinstance(receipt,dict) and receipt.get('contract')==CONTRACT, 'Missing cotton exploration receipt')
    used=receipt['targeted_by_era_seat'];built=receipt['builds_by_era_seat']
    require(len(used)==len(built)==2 and all(len(x)==4 for x in (*used,*built)), 'Cotton receipt shape')
    observed=[[0]*4 for _ in range(2)]
    for row in rows:
        l=row['teacher'];e=l.get('cotton_exploration')
        if not e:continue
        era=0 if row['view']['context']['era']=='CanalEra' else 1;seat=l['seat']
        require(e['used_before']==observed[era][seat], 'Cotton exploration allowance sequence differs')
        observed[era][seat]+=int(e['targeted'])
    require(used==observed and all(type(n) is int and 0<=n<=CAP for a in used for n in a),
            'Cotton committed quota differs from labels')
    require(all(type(b) is int and b>=u for bs,us in zip(built,used) for b,u in zip(bs,us)),
            'Cotton actual build count differs')
    require(sum(receipt[p]['targeted_builds'] for p in ('long','short'))==sum(map(sum,used)) and
        sum(receipt[p]['actual_builds'] for p in ('long','short'))==sum(map(sum,built)), 'Cotton build totals differ')
    for profile in ('long','short'):
        c=receipt[profile]
        require(c['triggers']==c['targeted_builds']<=c['covered_decisions']<=c['eligible_decisions']<=c['legal_decisions']<=c['action_decisions']
            and c['covered_decisions']+c['no_evidence']+c['no_cotton_visits']==c['eligible_decisions'],
            'Cotton exploration counters differ')


def aggregate(metrics):
    """O(tables) reduction once per iteration, not per action or optimizer step."""
    receipts=[m.get('cpu_inference',{}).get('revision',{}).get('cotton') for m in metrics]
    receipts=[r for r in receipts if r is not None]
    if not receipts:return None
    result=dict(contract=CONTRACT,tables=len(receipts),settings=settings())
    for profile in ('long','short'):
        counts=Counter()
        for r in receipts:counts.update(r[profile])
        result[profile]=dict(counts)
    totals=Counter(result['long']);totals.update(result['short'])
    result['total']=dict(totals)
    result['trigger_rate']=totals['triggers']/totals['covered_decisions'] if totals['covered_decisions'] else None
    result['coverage_rate']=totals['covered_decisions']/totals['eligible_decisions'] if totals['eligible_decisions'] else None
    result['natural_builds']=totals['actual_builds']-totals['targeted_builds']
    return result
