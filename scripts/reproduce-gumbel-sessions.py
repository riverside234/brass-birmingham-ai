"""Small independent L11 oracle: frozen quota and eligible round watermarks.

This is not a Brass planner or a timing benchmark. Scripted completed returns
make the recommendation boundary observable without scheduler/CPU noise.
Completed-Q and the schedule are independently spelled out here.
"""
import argparse
import copy
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def scores(logits, value, counts, returns, actor=0, c1=50., c2=.1):
    visited = [i for i, n in enumerate(counts) if n]
    mixed = value[actor]
    if visited:
        peak = max(logits[i] for i in visited)
        mass = [math.exp(logits[i]-peak) for i in visited]
        mean = sum(w*returns[i][actor]/counts[i] for i, w in zip(visited, mass))/sum(mass)
        mixed = (mixed+sum(counts)*mean)/(1+sum(counts))
    q = [returns[i][actor]/n if n else mixed for i, n in enumerate(counts)]
    return [l+c2*(c1+max(counts))*v for l, v in zip(logits, q)]


def quota(available, samples, startup):
    samples = [n for n in samples if n > 0][-8:]
    estimate = sorted(samples)[(len(samples)-1)//4] if len(samples) >= 4 else startup
    return 0 if estimate is None else available//estimate


def reproduce(case):
    n, k = case['budget'], case['k']
    logits, value, uniforms = case['logits'], case['value'], case['uniforms']
    k = 2**int(math.log2(min(k, len(logits))))
    while k > 1 and n < k*int(math.log2(k)):
        k //= 2
    g = [-math.log(-math.log(u)) for u in uniforms]
    rank = lambda candidates, s: sorted(candidates, key=lambda i: (-s[i]-g[i], i))
    candidates = sorted(rank(range(len(logits)), logits)[:k])
    counts, returns = [0]*len(logits), [[0.]*4 for _ in logits]
    completed = rounds = remainder = 0
    snapshot = None
    selected = []
    round_done = 0
    while completed < n and completed < case['stop_after']:
        edge = candidates[round_done % len(candidates)]
        selected.append(edge)
        v = case['values'][completed % len(case['values'])]
        counts[edge] += 1
        returns[edge] = [a+b for a, b in zip(returns[edge], v)]
        completed += 1
        round_done += 1
        balanced = False
        if len(candidates) > 1:
            per_action = n//(len(candidates)*int(math.log2(k)))
            if round_done == len(candidates)*per_action:
                candidates = sorted(rank(candidates, scores(logits,value,counts,returns))[:len(candidates)//2])
                rounds += 1
                round_done = 0
                balanced = True
        else:
            remainder += 1
        if completed == n or k == 1 or balanced:
            snapshot = dict(count=counts.copy(), returns=copy.deepcopy(returns), survivors=candidates.copy(),
                            support_watermark=completed, completed_rounds=rounds,
                            remainder_simulations=remainder,
                            reason='quota_complete' if completed == n else 'single_candidate' if k == 1 else 'balanced_round')
    if snapshot:
        ranking = rank(snapshot['survivors'], scores(logits,value,snapshot['count'],snapshot['returns']))
    else:
        ranking = rank(range(len(logits)),logits)
    return dict(effective_k=k, selected=selected, completed=completed, snapshot=snapshot,
                ranking=ranking, partial_completed=completed-(snapshot['support_watermark'] if snapshot else 0))


def cases():
    base = dict(logits=[.2, -.1, .3, .4], value=[.3,.2,.1,.4], uniforms=[.3,.6,.8,.4],
                values=[[.2,.4,.3,.1],[.8,.1,.0,.1],[.1,.2,.4,.3]], k=4)
    return [dict(base, name=name, budget=n, stop_after=stop, k=k) for name,n,stop,k in [
        ('before-first-round',24,2,4), ('exact-round-boundary',24,12,4),
        ('incomplete-second-round',24,15,4), ('last-backup',24,24,4),
        ('during-remainder',23,20,4), ('remainder-complete',23,23,4),
        ('no-work',0,0,4), ('k1-interrupted',7,2,1), ('k1-no-backup',7,0,1),
        ('small-budget',2,2,4)]]


def document():
    return dict(identity='brass-gumbel-is-mcts/session-v1', cases=[dict(c, expected=reproduce(c)) for c in cases()],
                quotas=[dict(available=a, samples=s, startup=e, expected=quota(a,s,e)) for a,s,e in [
                    (3600,[],10),(3500,[5,6,7],10),(3400,[5,6,7,8],10),
                    (3000,[1,2,3,4,5,6,7,8,9,10],20),(3600,[],None),(0,[1,2,3,4],10)]])


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path);a=p.parse_args()
    payload=json.dumps(document(),indent=2)+'\n'
    if a.out: a.out.write_text(payload,encoding='utf-8')
    else: print(payload,end='')
