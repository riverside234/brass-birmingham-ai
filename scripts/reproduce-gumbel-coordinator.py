"""Independent L12 oracle: eligible snapshot pooling and reserved round slots.

Only the small L11 mathematical reference is reused. No native scheduler, tree,
or evaluator code participates in this executable specification.
"""
import argparse
import copy
import json
import math
from pathlib import Path
import runpy

ROOT = Path(__file__).resolve().parents[1]
SCORES = runpy.run_path(str(ROOT / 'scripts/reproduce-gumbel-sessions.py'))['scores']


def merge(case):
    logits, value = case['logits'], case['value']
    g = [-math.log(-math.log(u)) for u in case['uniforms']]
    count, returns, survivors = [0] * len(logits), [[0.] * 4 for _ in logits], set()
    eligible = []
    for lane, snapshot in enumerate(case['snapshots']):
        if snapshot is None:
            continue
        assert sum(snapshot['count']) == snapshot['watermark'] > 0
        eligible.append(lane)
        survivors.update(snapshot['survivors'])
        count = [a + b for a, b in zip(count, snapshot['count'])]
        returns = [[a + b for a, b in zip(x, y)] for x, y in zip(returns, snapshot['returns'])]
    scores = SCORES(logits, value, count, returns, case['actor']) if eligible else logits
    rank = sorted(survivors if eligible else range(len(logits)), key=lambda i: (-g[i]-scores[i], i))
    return dict(eligible=eligible, count=count, returns=returns, ranking=rank,
                scores=[a+b for a,b in zip(g,scores)])


def reservations(events, quota=2, candidates=(0, 1)):
    """FIFO publication, round-robin launch, no reservations in completed counts."""
    pending, completed, launched, retired = [], [], 0, 0
    trace = []
    for event in events:
        if event == 'reserve':
            if launched < quota * len(candidates):
                pending.append(candidates[launched % len(candidates)])
                launched += 1
        elif event == 'complete':
            assert pending
            completed.append(pending.pop(0))
        elif event == 'retire':
            retired += len(pending)
            pending.clear()
        else:
            raise ValueError(event)
        trace.append(dict(pending=pending.copy(), completed=completed.copy(), retired=retired,
                          balanced=len(completed) == quota*len(candidates) and not pending))
    return trace


def document():
    base = dict(logits=[0., -.2, .1, .3], value=[.3,.2,.1,.4], uniforms=[.4,.6,.7,.3], actor=0)
    a = dict(count=[3,3,3,3], returns=[[.3,0,0,0],[2.4,0,0,0],[1.2,0,0,0],[.6,0,0,0]], survivors=[1,2], watermark=12)
    b = dict(count=[4,4,4,4], returns=[[.8,0,0,0],[.4,0,0,0],[.8,0,0,0],[2.8,0,0,0]], survivors=[0,3], watermark=16)
    cases = []
    for name, snapshots in [('neither',[None,None]), ('left-only',[a,None]), ('right-only',[None,b]),
                            ('conflicting-survivors',[a,b]), ('same-survivors',[a,a])]:
        case = dict(base, name=name, snapshots=copy.deepcopy(snapshots))
        cases.append(dict(case, expected=merge(case)))
    events = ['reserve']*5 + ['complete','reserve','complete','complete','complete']
    interrupted = ['reserve','reserve','complete','retire']
    return dict(identity='brass-gumbel-is-mcts/coordinator-v1', cases=cases,
                reservations=[dict(events=e, expected=reservations(e)) for e in (events,interrupted)])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    payload = json.dumps(document(), indent=2) + '\n'
    if args.out:
        args.out.write_text(payload, encoding='utf-8')
    else:
        print(payload, end='')
