"""Cotton behavior metadata and coarse per-table statistics; no target shaping."""
import math
from collections import Counter
from brass_ai.learning.contracts import require
from brass_ai.step5.policy import cotton_build_available, has_unflipped_cotton

CONTRACT='teacher-cotton-exploration/1.0.0'
DEVELOP_CONTRACT='teacher-cotton-exploration/2.0.0'
MIXED_CONTRACT='teacher-cotton-exploration/3.0.0'
MIXED_25_CONTRACT='teacher-cotton-exploration/4.0.0'
SEATED_CONTRACT='teacher-cotton-exploration/5.0.0'
SINGLE_SEAT_CONTRACT='teacher-cotton-exploration/6.0.0'
SEATS='cotton_seats'
PROBABILITY=.2
DEVELOP_PROBABILITY=.25
MIXED_PROBABILITY=.3
CAP=3
MIXED_CONTRACTS=(MIXED_CONTRACT,MIXED_25_CONTRACT)
SEATED_CONTRACTS=(SEATED_CONTRACT, SINGLE_SEAT_CONTRACT)
PROBABILITIES={CONTRACT:PROBABILITY, DEVELOP_CONTRACT:DEVELOP_PROBABILITY, MIXED_CONTRACT:MIXED_PROBABILITY,
               MIXED_25_CONTRACT:DEVELOP_PROBABILITY, **dict.fromkeys(SEATED_CONTRACTS, MIXED_PROBABILITY)}


def settings(*, develop=False, mixed=False, contract=None):
    if contract is None:
        contract = MIXED_CONTRACT if mixed else (DEVELOP_CONTRACT if develop else CONTRACT)
    value = dict(contract=contract, probability=PROBABILITIES[contract], max_per_seat_era=CAP,
        long_candidates=2, short_candidates=1, pause_unflipped=True,
        scope='train-and-heldout' if contract in (*MIXED_CONTRACTS, *SEATED_CONTRACTS) else 'train')
    if contract == SINGLE_SEAT_CONTRACT:
        value.update(minimum_build_level=2, short_window='all-actions')
    return value


def job_settings(job):
    return (job.get('teacher_revision') or {}).get('exploration', {}).get('cotton')


def enabled(job):
    return job_settings(job) is not None


def validate_label(label, choices, *, view=None):
    """Reuse already validated legal order/counts. The improved-policy target is untouched."""
    e=label.get('cotton_exploration')
    if e is None:return
    require(isinstance(e,dict) and type(e.get('contract')) is str and e['contract'] in PROBABILITIES,
            'Invalid cotton exploration contract')
    single = e['contract'] == SINGLE_SEAT_CONTRACT
    fields = {'contract','targeted','probability','used_before','cotton_visits','total_visits','covered_locations'}
    require(set(e) == (fields | {'planned_action'} if single else fields),
        'Invalid cotton exploration metadata')
    require(label.get('explored') is True and type(e['targeted']) is bool and
        type(e['used_before']) is int and 0<=e['used_before']<CAP,'Invalid cotton exploration allowance')
    total=sum(label['counts']);cotton=0;locations=set();played=None
    develop = e['contract'] != CONTRACT
    mixture = PROBABILITIES[e['contract']]
    builds = True
    if single:
        require(view is not None and not has_unflipped_cotton(view, label['seat'])
                and any(c['id'] == e['planned_action'] for c in choices), 'Invalid teacher cotton eligibility or plan')
        builds = cotton_build_available(view, label['seat'])
    def targeted(c):
        return c.get('industry')=='CottonMill' and (builds and c['kind']=='build' or develop and c['kind']=='develop')
    for c,n in zip(choices,label['counts']):
        is_cotton=targeted(c)
        if n and is_cotton:cotton+=n;locations.add((c['kind'], c.get('target')))
        if c['id']==label['played_action']:played=(n,is_cotton)
    require(total>0 and cotton>0 and played is not None and (single or played[0]>0),
            'Cotton exploration needs visited legal actions')
    require(all(type(e[k]) is int for k in ('cotton_visits','total_visits','covered_locations')) and
        (e['cotton_visits'],e['total_visits'],e['covered_locations'])==(cotton,total,len(locations)),
        'Cotton coverage counters differ from search evidence')
    n,is_cotton=played
    if label.get('sampling_temperature') == 2.:
        weights = [math.sqrt(count) for count in label['counts']]
        total = sum(weights)
        cotton = sum(w for c,w in zip(choices,weights) if targeted(c))
        n = math.sqrt(n)
    planned = single and label['played_action'] == e['planned_action']
    base = float(planned) if single else n/total
    probability=(1-mixture)*base+(mixture*n/cotton if is_cotton else 0.)
    require((is_cotton and n > 0 if e['targeted'] else not single or planned) and
        type(e['probability']) in (int,float) and
        math.isfinite(e['probability']) and abs(e['probability']-probability)<=1e-12,
        'Cotton exploration behavior probability differs')


def validate_receipt(receipt, job, rows):
    """One small table check at publication, after existing row/binding admission."""
    if not enabled(job):
        require(receipt is None, 'Unexpected cotton exploration on an unbiased table')
        return
    contract = job_settings(job)['contract']
    require(isinstance(receipt,dict) and receipt.get('contract')==contract, 'Missing cotton exploration receipt')
    seats = job['teacher_revision'].get(SEATS)
    if seats is not None:
        require(receipt.get(SEATS) == seats, 'Teacher cotton receipt seats differ')
    used=receipt['targeted_by_era_seat'];built=receipt['builds_by_era_seat']
    developed=receipt['develops_by_era_seat'] if contract!=CONTRACT else [[0]*4 for _ in range(2)]
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
        require(seats is None or seat in seats, 'Cotton label from an unassigned teacher seat')
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
    if seats is not None:
        require(all(n == 0 for a in used for seat, n in enumerate(a) if seat not in seats),
                'Unassigned teacher seat used cotton exploration')
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
    receipts=[(m.get('cpu_inference') or {}).get('revision',{}).get('cotton') for m in metrics]
    receipts=[r for r in receipts if r is not None]
    if not receipts:return None
    contract = receipts[0]['contract']
    require(all(r['contract']==contract for r in receipts), 'Mixed cotton exploration settings')
    result=dict(contract=contract, tables=len(receipts), settings=settings(contract=contract))
    if contract in SEATED_CONTRACTS:
        counts = Counter(len(r[SEATS]) for r in receipts)
        result.update(one_seat_tables=counts[1], two_seat_tables=counts[2])
    for profile in ('long','short'):
        counts=Counter()
        for r in receipts:counts.update(r[profile])
        result[profile]=dict(counts)
    totals=Counter(result['long']);totals.update(result['short'])
    result['total']=dict(totals)
    result['trigger_rate']=totals['triggers']/totals['covered_decisions'] if totals['covered_decisions'] else None
    result['coverage_rate']=totals['covered_decisions']/totals['eligible_decisions'] if totals['eligible_decisions'] else None
    result['natural_builds']=totals['actual_builds']-totals['targeted_builds']
    if contract != CONTRACT:
        result['natural_develops']=totals['actual_develops']-totals['targeted_develops']
    return result
