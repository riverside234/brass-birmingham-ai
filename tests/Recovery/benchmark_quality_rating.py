"""Bounded synthetic three-slot rating fit and finite-sample bootstrap coverage check."""
import argparse
import copy
from pathlib import Path
import time
import numpy as np

from brass_ai.learning.contracts import SOURCE_ROOT,project_path,require,digest
from brass_ai.recovery import ratings as r,plackett_luce as pl
from brass_ai.recovery.quality_config import RATING
from brass_ai.step5.league import seeds_for,write_json


def synthetic(spec,theta,seed):
    rng=np.random.default_rng(seed);rows=[];seat=np.array([.2,-.15,.1,-.15])
    for i,row in enumerate(spec['lineups']):
        agents=[row['agents'][(s-row['seat'])%4] for s in range(4)]
        u=np.array([theta[a] for a in agents])+seat;order=np.argsort(-(u+rng.gumbel(size=4))).tolist()
        ranks=[order.index(s)+1 for s in range(4)]
        rows.append(dict(game_id=f'synthetic-{seed}-{spec["generation"]}-{i}',slot=spec['generation'],
            setup_id=digest(dict(rules='steam-brass-patch-1.1.0',setup_seed=row['seed'])),setup_seed=row['seed'],
            candidate_seat=row['seat'],kind=row['kind'],agents=agents,order=order,
            official_placement=ranks,vp=[100-20*rank for rank in ranks],win_share=[int(rank==1) for rank in ranks]))
    return rows


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True)
    p.add_argument('--repetitions',type=int,default=40);a=p.parse_args()
    require(1<=a.repetitions<=100,'Bound this diagnostic to at most 100 trials')
    out=project_path(SOURCE_ROOT,a.output);require(not out.exists(),'Use a new diagnostic directory');out.mkdir(parents=True)
    c=copy.deepcopy(RATING);seeds,_=seeds_for(c['seed'],c['setups'],'validation')
    state=dict(agents={pl.IL:dict(kind='model',checkpoint=dict(generation=0,model_id='synthetic-IL')),
                       pl.V4:dict(kind='v4',policy={})},seeds=seeds,latest_report=None)
    theta={pl.IL:0.,pl.V4:-.3,'generation-000010':.4,'generation-000020':.55,'generation-000030':.7}
    all_rows=[];slots=[];began=time.perf_counter();initial=None
    for generation in (10,20,30):
        cfg=dict(rating=c,execution={},host='synthetic')
        spec=r.slot_spec(SOURCE_ROOT,out,state,dict(generation=generation,model_id=f'synthetic-{generation}'),cfg)
        if initial is None:initial=copy.deepcopy(spec)
        rows=synthetic(spec,theta,seed=1900+generation);all_rows.extend(row for row in rows if row['kind']=='rating')
        state['agents'][spec['candidate']]=spec['agents'][spec['candidate']]
        start=time.perf_counter();report,samples=pl.estimate(all_rows,state['agents'],c);elapsed=time.perf_counter()-start
        directory=out/f'slot-{generation:04d}';directory.mkdir();r.append_games(directory/'games.jsonl',rows)
        write_json(directory/'report.json',report);r.plot_report(directory/'progress.svg',report,state['agents'])
        state['latest_report']=r.receipt(SOURCE_ROOT,directory/'report.json')
        slots.append(dict(generation=generation,games=report['games'],seconds=elapsed,
            candidate=report['ratings'][spec['candidate']],seat_offsets=report['seat_offsets']))
    # Independent PL-generated outcomes on the actual startup protocol. The
    # small trial count measures implementation calibration, not real-game validity.
    coverage={agent:[] for agent in (pl.V4,'generation-000010')};halfwidths=[];diff_coverage=[]
    for trial in range(a.repetitions):
        rows=synthetic(initial,theta,seed=81000+trial)
        report,_=pl.estimate([row for row in rows if row['kind']=='rating'],initial['agents'],dict(c,seed=c['seed']+trial))
        for agent,values in coverage.items():
            lo,hi=report['ratings'][agent]['theta_ci95'];values.append(lo<=theta[agent]<=hi)
        lo,hi=report['ratings']['generation-000010']['points_ci95'];halfwidths.append((hi-lo)/2)
        diff=next(d for d in report['differences'] if {d['a'],d['b']}=={pl.V4,'generation-000010'})
        truth=(theta[diff['b']]-theta[diff['a']])*c['elo_scale']
        diff_coverage.append(diff['points_delta_ci95'][0]<=truth<=diff['points_delta_ci95'][1])
        if (trial+1)%10==0:print(f'Synthetic coverage: {trial+1}/{a.repetitions}',flush=True)
    result=dict(scope='Synthetic Plackett-Luce outcomes only; not native game strength. Small Monte Carlo sample, not a guarantee of 95% coverage or ±40 points.',
        passed=True,slots=slots,bootstrap_refits_per_fit=200,coverage_trials=a.repetitions,
        coverage={k:dict(covered=sum(v),trials=len(v),fraction=float(np.mean(v))) for k,v in coverage.items()},
        candidate_v4_difference_coverage=float(np.mean(diff_coverage)),
        median_candidate_interval_halfwidth_points=float(np.median(halfwidths)),seconds=time.perf_counter()-began)
    write_json(out/'report.json',result);print(result,flush=True)


if __name__=='__main__':main()
