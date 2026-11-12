"""Cotton behavior metadata and coarse per-table statistics; no target shaping."""
import math
from collections import Counter
from brass_ai.learning.contracts import require

CONTRACT='teacher-cotton-exploration/1.0.0'
DEVELOP_CONTRACT='teacher-cotton-exploration/2.0.0'
PROBABILITY=.2
DEVELOP_PROBABILITY=.25
CAP=3


def settings(*, develop=False):
    return dict(contract=DEVELOP_CONTRACT if develop else CONTRACT,
        probability=DEVELOP_PROBABILITY if develop else PROBABILITY,max_per_seat_era=CAP,
        long_candidates=2,short_candidates=1,pause_unflipped=True,scope='train')


def job_settings(job):
    return (job.get('teacher_revision') or {}).get('exploration', {}).get('cotton')


def enabled(job):
    return job_settings(job) is not None


def validate_label(label, choices):
    """Reuse already validated legal order/counts. The improved-policy target is untouched."""
    e=label.get('cotton_exploration')
    if e is None:return
    require(isinstance(e,dict) and set(e)=={'contract','targeted','probability','used_before',
        'cotton_visits','total_visits','covered_locations'} and e['contract'] in (CONTRACT, DEVELOP_CONTRACT),
        'Invalid cotton exploration metadata')
    require(label.get('explored') is True and type(e['targeted']) is bool and
        type(e['used_before']) is int and 0<=e['used_before']<CAP,'Invalid cotton exploration allowance')
    total=sum(label['counts']);cotton=0;locations=set();played=None
    develop = e['contract'] == DEVELOP_CONTRACT
    mixture = DEVELOP_PROBABILITY if develop else PROBABILITY
    for c,n in zip(choices,label['counts']):
        is_cotton=c.get('industry')=='CottonMill' and (c['kind']=='build' or develop and c['kind']=='develop')
        if n and is_cotton:cotton+=n;locations.add((c['kind'], c.get('target')))
        if c['id']==label['played_action']:played=(n,is_cotton)
    require(total>0 and cotton>0 and played is not None and played[0]>0,
            'Cotton exploration needs visited legal actions')
    require(all(type(e[k]) is int for k in ('cotton_visits','total_visits','covered_locations')) and
        (e['cotton_visits'],e['total_visits'],e['covered_locations'])==(cotton,total,len(locations)),
        'Cotton coverage counters differ from search evidence')
    n,is_cotton=played
    probability=(1-mixture)*n/total+(mixture*n/cotton if is_cotton else 0.)
    require((not e['targeted'] or is_cotton) and type(e['probability']) in (int,float) and
        math.isfinite(e['probability']) and abs(e['probability']-probability)<=1e-12,
        'Cotton exploration behavior probability differs')


def validate_receipt(receipt, job, rows):
    """One small table check at publication, after existing row/binding admission."""
    if not enabled(job):
        require(receipt is None, 'Unexpected cotton exploration on an unbiased table')
        return
    contract = job_settings(job)['contract']
    require(isinstance(receipt,dict) and receipt.get('contract')==contract, 'Missing cotton exploration receipt')
    used=receipt['targeted_by_era_seat'];built=receipt['builds_by_era_seat']
    developed=receipt['develops_by_era_seat'] if contract==DEVELOP_CONTRACT else [[0]*4 for _ in range(2)]
    require(len(used)==len(built)==len(developed)==2 and all(len(x)==4 for x in (*used,*built,*developed)), 'Cotton receipt shape')
    # With long-search labels only (the g1490 restart), short searches still explore and
    # spend allowance between labels: the labelled sequence may skip, never go back.
    every=(job.get('teacher_revision') or {}).get('short_labels') is True
    observed=[[0]*4 for _ in range(2)];labelled=0
    for row in rows:
        l=row['teacher'];e=l.get('cotton_exploration')
        if not e:continue
        require(e['contract']==contract, 'Cotton label and table settings differ')
        era=0 if row['view']['context']['era']=='CanalEra' else 1;seat=l['seat']
        require(e['used_before']==observed[era][seat] if every and contract==CONTRACT else e['used_before']>=observed[era][seat],
                'Cotton exploration allowance sequence differs')
        observed[era][seat]=e['used_before']+int(e['targeted']);labelled+=int(e['targeted'])
    continuation = sum(receipt[p].get('continuation_targets', 0) for p in ('long', 'short'))
    labelled_triggers = receipt['long']['triggers'] + (receipt['short']['triggers'] if every else 0)
    quota_matches = used == observed if every and contract==CONTRACT else all(
        u>=o for us,os in zip(used,observed) for u,o in zip(us,os))
    require(quota_matches and labelled==labelled_triggers and
            all(type(n) is int and 0<=n<=CAP for a in used for n in a),
            'Cotton committed quota differs from labels')
    require(all(type(b) is int and type(d) is int and min(b,d)>=0 and b+d>=u
        for bs,ds,us in zip(built,developed,used) for b,d,u in zip(bs,ds,us)), 'Cotton action counts differ')
    require(sum(receipt[p]['triggers'] for p in ('long','short')) + continuation==sum(map(sum,used)) and
        sum(receipt[p]['actual_builds'] for p in ('long','short'))==sum(map(sum,built)) and
        sum(receipt[p].get('actual_develops',0) for p in ('long','short'))==sum(map(sum,developed)),
        'Cotton action totals differ')
    for profile in ('long','short'):
        c=receipt[profile]
        d, extra = c.get('targeted_develops',0), c.get('continuation_targets',0)
        require(0<=extra<=d<=c.get('actual_develops',0) and 0<=c['targeted_builds']<=c['actual_builds'] and
            c['triggers']==c['targeted_builds']+d-extra<=c['covered_decisions']<=c['eligible_decisions']<=c['legal_decisions']<=c['action_decisions']
            and c['covered_decisions']+c['no_evidence']+c['no_cotton_visits']==c['eligible_decisions'],
            'Cotton exploration counters differ')


def aggregate(metrics):
    """O(tables) reduction once per iteration, not per action or optimizer step."""
    receipts=[m.get('cpu_inference',{}).get('revision',{}).get('cotton') for m in metrics]
    receipts=[r for r in receipts if r is not None]
    if not receipts:return None
    contract = receipts[0]['contract']
    require(all(r['contract']==contract for r in receipts), 'Mixed cotton exploration settings')
    result=dict(contract=contract,tables=len(receipts),settings=settings(develop=contract==DEVELOP_CONTRACT))
    for profile in ('long','short'):
        counts=Counter()
        for r in receipts:counts.update(r[profile])
        result[profile]=dict(counts)
    totals=Counter(result['long']);totals.update(result['short'])
    result['total']=dict(totals)
    result['trigger_rate']=totals['triggers']/totals['covered_decisions'] if totals['covered_decisions'] else None
    result['coverage_rate']=totals['covered_decisions']/totals['eligible_decisions'] if totals['eligible_decisions'] else None
    result['natural_builds']=totals['actual_builds']-totals['targeted_builds']
    if contract == DEVELOP_CONTRACT:
        result['natural_develops']=totals['actual_develops']-totals['targeted_develops']
    return result
