"""Bounded synthetic PL-seat forgetting/precision check; no game engine or training."""
import argparse
import copy
import json
from pathlib import Path
import time
import numpy as np

from brass_ai.learning.contracts import SOURCE_ROOT,project_path,require
from brass_ai.recovery import quality_pool as q,pfsp_pool as p
from brass_ai.step5.league import write_json
from test_pfsp_pool import pool,entry


def simulate(scored,seed):
    rng=np.random.default_rng(seed);value=pool(scored+1,generation=100,games=100000)
    rates=np.linspace(.55,.92,scored);forgotten=scored-3;rates[forgotten]=.89
    true=np.r_[.95,rates];errors=[];trajectory=[];before=[];half=near=None;unsafe=0
    started=time.perf_counter();estimator_seconds=0.;total_il=0;total_seats=0
    for i in range(140):
        if i==100:
            before=float(np.mean(before[-10:]));true[forgotten+1]=.5
            w=(1-true[1:])**2;target=(.8*w/w.sum()+.2/scored)[forgotten]
        plan=q.distribution(value);prob=np.r_[plan['il_share'],[(1-plan['il_share'])*r['p'] for r in plan['members']]]
        selected=rng.choice(scored+1,size=(144,2),p=prob)
        total_il+=int((selected==0).sum());total_seats+=selected.size
        utilities=np.c_[np.zeros((144,2)),np.log((1-true[selected])/true[selected])]+rng.gumbel(size=(144,4))
        wins=(utilities[:,:2,None]>utilities[:,None,2:]).transpose(0,2,1)
        rows={}
        for j,m in enumerate(value['members']):
            mask=selected==j
            rows[m['model_id']]=dict(seats=int(mask.sum()),outcomes=wins[mask].astype(float).reshape(-1).tolist())
        t=time.perf_counter()
        value,report=p.update_counts(value,rows,plan,generation=value['generation']+1,games_done=value['games_done']+320)
        estimator_seconds+=time.perf_counter()-t
        actual=q.distribution(value)['members'][forgotten]['p']
        if i<100:
            before.append(actual)
            if i>=40:errors.append(np.array([p.rate(m) for m in value['members'][1:]])-true[1:])
        else:
            elapsed=i-99;trajectory.append(actual)
            if half is None and actual>=before+.5*(target-before):half=elapsed
            if near is None and actual>=before+.9*(target-before):near=elapsed
            if value['generation']%5==0:
                # Inspect the real eviction decision on a copy of this fixed
                # pool; keep identities fixed for the estimator RMSE experiment.
                _,event=q.admission(value,entry(value['generation']),dict(ok=True),
                    {m['model_id']:.1 for m in value['members']},lambda a,b:.1)
                member=value['members'][forgotten+1]
                if p.rate(member)<np.median([p.rate(m) for m in value['members'][1:]]):
                    unsafe+=int((event.get('evicted') or {}).get('model_id')==member['model_id'])
    return dict(seed=seed,half_iterations=half,near_iterations=near,forgotten_evictions_below_median=unsafe,
        before=before,target=float(target),shares_after={str(i):trajectory[i-1] for i in (1,5,10,20,40)},
        squared_errors=np.square(errors).mean(axis=0).tolist(),realized_il_share=total_il/total_seats,
        estimator_seconds=estimator_seconds,total_seconds=time.perf_counter()-started)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True)
    args=parser.parse_args();out=project_path(SOURCE_ROOT,args.output)
    require(not out.exists(),'Use a new benchmark output directory');out.mkdir(parents=True)
    result=dict(scope='Synthetic correlated four-seat PL outcomes, 144 history tables/iteration, fixed 16/23-member scored pools, 16 seeds each. 100 stationary iterations then 40 forgetting iterations. Evictions inspected on copies, not applied. Independent diagnostic; the supplied dynamic-pool simulation is stored separately in ai-research/simulations/pfsp_sim_original.py with a maintained runner at scripts/pfsp_sim.py. Neither establishes playing strength.',cases=[])
    for n in (16,23):
        runs=[simulate(n,79100+s) for s in range(16)]
        rmse=2*np.sqrt(np.mean([r['squared_errors'] for r in runs],axis=0))
        half=float(np.median([r['half_iterations'] or 41 for r in runs]));near=float(np.median([r['near_iterations'] or 41 for r in runs]))
        case=dict(scored_members=n,il_share=1/(n+2),median_half_iterations=half,median_near_iterations=near,
            twice_rmse_hardest=float(rmse[0]),twice_rmse_easiest=float(rmse[-1]),twice_rmse_all=rmse.tolist(),
            unsafe_evictions=sum(r['forgotten_evictions_below_median'] for r in runs),
            estimator_ms_per_iteration=1000*sum(r['estimator_seconds'] for r in runs)/(16*140),runs=runs)
        case['response_check_passed']=half<=8 and near<=16 and case['unsafe_evictions']==0
        result['cases'].append(case);print({k:v for k,v in case.items() if k not in ('runs','twice_rmse_all')},flush=True)
    result['passed']=all(c['response_check_passed'] for c in result['cases'])
    write_json(out/'report.json',result);require(result['passed'],'Synthetic PFSP response check failed; see report')


if __name__=='__main__':main()
