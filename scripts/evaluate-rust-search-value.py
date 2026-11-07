#!/usr/bin/env python3
"""Explicit one-off 256-game correction head evaluation; never run by the smoke."""
import argparse
import copy
from datetime import datetime, timezone
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))


def main():
    import torch
    from brass_ai.learning.contracts import read_json,require,digest
    from brass_ai.learning.checkpoint import load_checkpoint
    from brass_ai.learning.batching import model_id
    from brass_ai.learning.search_value import binding
    from brass_ai.recovery.inference import InferenceService
    from brass_ai.recovery.teacher_phase import plan_jobs,reserved_rounds
    from brass_ai.recovery.search_value_config import adopt
    from brass_ai.recovery.search_value_training import h2h_verdict
    from brass_ai.rust_backend.lab_b045_smoke import verify_build
    from brass_ai.rust_backend.lab_contract import pin
    from brass_ai.rust_backend.lab_teacher_overlay import source_fingerprint
    from brass_ai.rust_backend.lab_teacher import TeacherCollector,calibrate
    from brass_ai.step5.league import write_json,seeds_for
    from rust_lab_results import export
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--config',type=Path,required=True)
    p.add_argument('--build',type=Path,required=True);p.add_argument('--device',default='cuda:0')
    p.add_argument('--lanes',type=int,default=48);p.add_argument('--mono',default='mono')
    args=p.parse_args()
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    source=pin(args.checkpoint);model,raw=load_checkpoint(args.checkpoint);model.eval()
    require(hasattr(model,'search_value_head'),'Head-to-head requires the trained correction checkpoint')
    receipt=raw['training'].get('utility_calibration')
    require(receipt and receipt['ready'] and receipt['reward_id']==digest(model.config.utility),'Head-to-head requires current-reward readiness')
    config=adopt(read_json(args.config));config['teacher_phase']['mode']='shadow'
    build=verify_build(read_json(args.build));identity=model_id(model)
    state=dict(iteration=raw['step'],history=[dict(generation=raw['step'],model_id=identity)])
    template=plan_jobs(config,state,model_id=identity,export_hash=source['sha256'],epoch=0,ready=True,native_source=source_fingerprint())[0][-1]
    seeds,_=seeds_for(450400,64,partition='validation');jobs=[]
    for group,seed in enumerate(seeds):
        for rotation in range(4):
            job=copy.deepcopy(template);key=digest(dict(h2h=source,seed=seed,rotation=rotation))
            job['seed']=seed;job['sampling_seed']=450400+group
            job['expected'].update(job_id=key,game_id='h2h/'+key)
            job.update(h2h=True,split='heldout',readiness=receipt,
                seat_values=[binding(1. if (s-rotation)%4 < 2 else 0.) for s in range(4)])
            job['reservations']={str(s):reserved_rounds(seed,job['sampling_seed'],s) for s in range(4)}
            jobs.append(job)
    out=ROOT/'artifacts/rust-backend'/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'-search-value-h2h')
    out.mkdir(parents=True,exist_ok=False);write_json(out/'jobs.json',jobs)
    report=dict(passed=False,games=0,production_modified=False)
    try:
        roots=[r for r in read_json(build['fixtures']['path'])['roots'] if r['view']['context']['phase']=='action' and len(r['expected']['choices'])>1]
        origins=[dict(state=r['expected']['state'],rail_keys=list(range(64))) for r in (roots[0],max(roots,key=lambda r:len(r['expected']['choices'])))]
        comparisons=[]
        with InferenceService(config['execution'],args.device) as service:
            service.register(model)
            calibration=calibrate(build['binary']['path'],build['core_source'],origins,jobs[0],service)
            write_json(out/'calibration.json',calibration)
            collector=TeacherCollector(build['binary']['path'],build['core_source'],[args.mono,build['bootstrap']['binary']['path']],
                service,calibration,lanes=args.lanes,model=model,history=state['history'])
            for start in range(0,256,args.lanes):
                batch=jobs[start:start+args.lanes];results=collector.collect(batch)
                # Persist small outcomes/provenance, not 16k state tensors.
                for result,job in zip(results,batch,strict=True):
                    collector.verifier.take(result,job,model)
                    scores=[(s['vp'],s['income'],s['cash']) for s in result['result']['seats']]
                    pairs=[float(scores[a]>scores[b])+.5*float(scores[a]==scores[b]) for a in range(4) for b in range(4)
                           if job['seat_values'][a]['w']==1 and job['seat_values'][b]['w']==0]
                    comparisons.append(sum(pairs)/4)
                write_json(out/f'games-{start:04d}.json',[dict(job_id=r['job_id'],result=r['result'],metrics=r['metrics']) for r in results])
                print(f'Head-to-head: {len(comparisons)}/256 games',flush=True)
            report=h2h_verdict(comparisons)
            report.update(production_modified=False,checkpoint=source,inference=service.report())
        require(pin(args.checkpoint)==source,'Head-to-head source checkpoint changed')
    finally:
        write_json(out/'qualification.json',report);print('Git-visible results: '+str(export(out)),flush=True)


if __name__=='__main__':main()
