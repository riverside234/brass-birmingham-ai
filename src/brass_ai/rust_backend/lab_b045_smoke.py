"""Two all-seat tables and one isolated head update on the pinned g1250 model."""
import copy
import os
from pathlib import Path
import threading

import numpy as np
import torch

from brass_ai.learning.contracts import SOURCE_ROOT, digest, read_json, require
from brass_ai.learning.batching import model_id
from brass_ai.learning.checkpoint import load_checkpoint, save_checkpoint
from brass_ai.learning.search_value import install
from brass_ai.recovery.inference import InferenceService
from brass_ai.recovery.teacher_phase import plan_jobs, readiness
from brass_ai.recovery.search_value_config import adopt
from brass_ai.recovery.search_value_phase import SearchValuePhase
from brass_ai.step5.league import write_json, implementation_id
from .lab_contract import pin
from .lab_teacher_overlay import source_fingerprint, CONTRACT
from .lab_teacher import TeacherCollector, calibrate, turn_probes, Worker
from .lab_teacher_inference import Client
from .lab_snapshot import verify_unchanged, tree_digest

CHECKPOINT = 'de427bb22e38cc5dc9d4e79151cccdb321650c02978f79bcdad746c3e6be7985'


def verify_build(build):
    require(build['contract']=='brass-lab-b045-build/1.0.0' and build['producer']==source_fingerprint()
            and build['implementation']==implementation_id(), 'B04.5 build source differs')
    for item in [build['binary'],build['fixtures'],build['bootstrap']['binary'],*build['bootstrap']['dependencies']]:
        require(pin(item['path'])==item, 'B04.5 runtime changed')
    return build


def run(args):
    out=args.out; snapshot=read_json(out/'snapshot.json'); build=verify_build(read_json(out/'build.json'))
    run=Path(snapshot['state_file']['path']).parent
    lock=read_json(run/'.active-run.json')
    require(lock['pid']==os.getppid() and os.environ.get('BRASS_B045_PARENT_LOCK')==(run/'.active-run.json').read_text(), 'Missing parent run lock')
    require(snapshot['generation']==1250 and snapshot['checkpoint']['sha256']==CHECKPOINT, 'B04.5 checkpoint pin differs')
    require(torch.device(args.device).type=='cuda' and torch.cuda.is_available(),'CUDA lab required')
    torch.set_num_threads(2); torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    model,raw=load_checkpoint(snapshot['checkpoint']['path']); model.eval()
    source_id=model_id(model)
    config=adopt(snapshot['state']['signature']['config']); config['teacher_phase']['mode']='shadow'
    config['teacher_phase']['search_value']['stage']='A'
    reward=digest(model.config.utility)
    receipt,ready=readiness(snapshot['state'].get('teacher_phase',{}).get('readiness'),raw['training'].get('utility_calibration'),reward)
    if receipt is None:
        # Explicit user authorization is limited to this SHA and diagnostic scope.
        receipt=dict(ready=True,reward_id=reward,basis='user-authorized-g1250-null-utility',calibration_measured=False)
        ready=True
    require(ready, 'Explicit false/mismatched utility readiness; no shadow dispatch')
    write_json(out/'readiness.json',dict(receipt=raw['training'].get('utility_calibration'),ready=ready,authorization=receipt))
    install(model)
    identity=model_id(model)
    state=dict(iteration=1250,history=[dict(generation=1250,model_id=identity)])
    jobs,_=plan_jobs(config,state,model_id=identity,export_hash=CHECKPOINT,epoch=0,ready=True,native_source=source_fingerprint())
    jobs=[next(j for j in jobs if j['split']==split) for split in ('train','heldout')]
    for j in jobs:j['readiness']=receipt
    write_json(out/'jobs.json',dict(jobs=jobs,source_model=source_id,diagnostic_model=identity,zero_head_added=True))
    fixtures=read_json(build['fixtures']['path'])
    roots=[r for r in fixtures['roots'] if r['view']['context']['phase']=='action' and len(r['expected']['choices'])>1]
    roots=[roots[0],max(roots,key=lambda r:len(r['expected']['choices']))]
    origins=[dict(state=r['expected']['state'],rail_keys=list(range(64))) for r in roots]
    try:
        with InferenceService(snapshot['execution'],args.device) as service:
            require(service.register(model)==identity,'Frozen model differs')
            with service.lock:service._model(identity)
            calibration=calibrate(build['binary']['path'],build['core_source'],origins,jobs[0],service)
            write_json(out/'gpu-calibration.json',calibration)
            # Same count, root, seed and model: compare v1 main values against
            # the sixth-output zero blend, not two wall-time-dependent quotas.
            old=copy.deepcopy(jobs[0])
            for key in ('seating','fast_search','search_value'): old.pop(key)
            endpoint=service.channels(1).acquire(); cancel=threading.Event()
            try:
                parity=[]
                for i,origin in enumerate(origins):
                    request=dict(contract=CONTRACT,op='calibrate',job=old,origin=origin,core_source=build['core_source'])
                    result=Worker(build['binary']['path'],request,Client(endpoint,old['expected'],cancel),cancel).collect(old['expected'],build['core_source'],calibration=True)
                    newer=next(s for s in calibration['samples'] if s['root_index']==i and not s['warmup'])
                    require(result['probe']==newer['probe'],'w=0 B04 full-profile Q/policy/recommendation/label differs')
                    parity.append(dict(root=roots[i]['id'],identical=True))
                write_json(out/'zero-blend-parity.json',parity)
            finally:endpoint.close()
            timed={profile:turn_probes(build['binary']['path'],build['core_source'],origins,jobs[0],service,calibration,
                      model,state['history'],profile=profile) for profile in ('full','fast')}
            write_json(out/'timed-probes.json',timed)
            bootstrap=[args.mono,build['bootstrap']['binary']['path']]
            backend=TeacherCollector(build['binary']['path'],build['core_source'],bootstrap,service,calibration,
                lanes=2,model=model,history=state['history'])
            try:
                tables=backend.collect(jobs)
            except BaseException:
                write_json(out/'collection-failure.json',dict(failures=backend.failures))
                raise
            finally:
                write_json(out/'inference.json',service.report())
            for table,job in zip(tables,jobs,strict=True):
                accepted,_=backend.verifier.take(table,job,model)
                require(accepted,'Full-profile labels absent')
            write_json(out/'tables.json',tables)
            write_json(out/'inference.json',service.report())
            require(service.error is None,'Inference service failed')
        # Bind Adam only after the diagnostic learner reaches its final device.
        # Device conversion may replace Parameter objects on some torch paths.
        model.to(args.device)
        phase=SearchValuePhase(out/'value-buffer',model,config['teacher_phase']['search_value'],reward)
        try:phase.admit(tables,jobs,1250)
        finally:write_json(out/'value-admission.json',phase.report)
        rows=phase.buffer.rows(1250)
        require(len(rows)==64 and len(phase.buffer.rows(1250,'heldout'))==64,'Two-table value volume differs')
        before={k:tree_digest(v) for k,v in model.state_dict().items()}
        update=phase.companion.head_only(rows,generation=1250,steps=1)
        after={k:tree_digest(v) for k,v in model.state_dict().items()}
        changed=[k for k in before if before[k]!=after[k]]
        require(changed and all(k.startswith('search_value_head.') for k in changed),'Stage A changed ordinary weights')
        save_checkpoint(out/'diagnostic-head.pt',model,step=1250,training={'search_value':phase.payload()})
        restored,saved=load_checkpoint(out/'diagnostic-head.pt')
        require(model_id(restored)==model_id(model),'Search head checkpoint roundtrip differs')
        SearchValuePhase(out/'value-buffer',restored,config['teacher_phase']['search_value'],reward,saved=saved['training']['search_value'])
        write_json(out/'head-update.json',dict(update=update,changed=changed,restored=True,source_checkpoint_unchanged=True))
        return dict(passed=True,lab_qualification='Passed for exact g1250 source/model/device',new_games=2,
                    diagnostic_head_steps=1,ordinary_optimizer_steps=0,production_modified=False,
                    labels=sum(t['metrics']['labels'] for t in tables),value_rows=128)
    finally: verify_unchanged(snapshot)


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--device',default='cuda:0');p.add_argument('--mono',default='mono')
    a=p.parse_args();write_json(a.out/'lab-checks.json',run(a))
