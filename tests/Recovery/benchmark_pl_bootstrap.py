"""Bounded CPU refit-only comparison of the full exported g850 rating history."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
from brass_ai.bank import read
from brass_ai.learning.contracts import SOURCE_ROOT,require
from brass_ai.recovery import plackett_luce as P


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--export',default='results/joint7-v11-2-grouped-lr-critic-g0850-20261024T181807288965Z')
    parser.add_argument('--output',required=True)
    parser.add_argument('--refits',type=int,default=200)
    args=parser.parse_args();root=SOURCE_ROOT/args.export;state=read(root/'state.json')
    archive=read(next((root/'ratings').glob('history-*.json')))
    games=list(archive['games']);missing=[]
    for slot in state['ratings']['slots']:
        path=root/'ratings/slots'/f"generation-{slot['generation']:06d}"/'games.jsonl'
        if not path.exists():
            missing.append(slot['generation']);continue
        games.extend(json.loads(line) for line in path.read_text().splitlines() if line.strip())
    games=[g for g in games if g['kind']=='rating'];agents=state['ratings']['agents']
    config=dict(state['signature']['config']['rating'],bootstrap=args.refits)
    require(len(games)==len({g['game_id'] for g in games}),'Duplicated history')
    reports={};arrays={};timings={}
    for name,kwargs in [('cold_serial',{}),('warm_serial',dict(compute=P.COMPUTE,parallel=False)),
                        ('warm_parallel4',dict(compute=P.COMPUTE))]:
        print(f'{name}: {len(games)} games, {len(agents)} agents, {args.refits} refits',flush=True)
        start=time.perf_counter();reports[name],arrays[name]=P.estimate(games,agents,config,**kwargs)
        timings[name]=time.perf_counter()-start;print(f'{name}: {timings[name]:.3f}s',flush=True)
    base=reports['cold_serial'];comparisons={}
    for name in ('warm_serial','warm_parallel4'):
        other=reports[name]
        changes=[dict(a=a['a'],b=a['b'],old=a['points_delta_ci95'],new=b['points_delta_ci95'])
            for a,b in zip(base['differences'],other['differences'])
            if (a['b_stronger'],a['a_stronger'])!=(b['b_stronger'],b['a_stronger'])]
        comparisons[name]=dict(max_theta_sample_difference=float(np.max(np.abs(arrays[name]-arrays['cold_serial']))),
            max_rating_point_difference=max(abs(base['ratings'][a]['points']-other['ratings'][a]['points']) for a in agents),
            max_marginal_CI_point_difference=max(abs(x-y) for a in agents for x,y in zip(base['ratings'][a]['points_ci95'],other['ratings'][a]['points_ci95'])),
            max_paired_CI_point_difference=max(abs(x-y) for a,b in zip(base['differences'],other['differences']) for x,y in zip(a['points_delta_ci95'],b['points_delta_ci95'])),
            pairwise_decision_changes=changes)
    # The complete paired table above includes the active anchor's comparisons.
    # Promotion itself still applies the existing newer-generation/evidence gates.
    result=dict(games=len(games),agents=len(agents),refits=args.refits,missing_export_slots=missing,seconds=timings,comparisons=comparisons,
        warm_serial_parallel_max_difference=float(np.max(np.abs(arrays['warm_serial']-arrays['warm_parallel4']))),
        reports=reports,scope='Existing exported main games only; no games/evaluations collected. Local CPU timing, not lab GPU throughput.')
    out=SOURCE_ROOT/args.output;out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='reports'},indent=2),flush=True)


if __name__=='__main__':main()
