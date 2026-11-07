"""Bounded B07 process-pool comparison and two independent all-seat shadow passes."""
import copy
import json
from pathlib import Path
import time
from unittest.mock import patch

from brass_ai.learning.contracts import SOURCE_ROOT,require,digest
from brass_ai.step5.league import write_json


def optimized_native(root,jobs,service,execution,build,output,mono,meter,producer,snapshot):
    from brass_ai.rust_backend.lab_pool import NativePool
    from brass_ai.rust_backend import lab_admission as admission
    jobs=copy.deepcopy(jobs)
    for job in jobs:job['path']=str(output/'plan'/('job-'+str(job['index'])+'.jsonl.gz'))
    build=dict(build,_receipt_sha256=producer['build_sha256'])
    with NativePool(root,service,execution,build,snapshot['state']['signature']['config'],
                    snapshot['signature_sha256'],mono=mono,scope='diagnostic') as pool:
        began=time.perf_counter()
        with meter.phase('collection_with_family_validation_and_sealing'):records=pool.collect(jobs)
        elapsed=time.perf_counter()-began;scheduler=pool.last_collection_report
    with meter.phase('admission_and_first_read'):
        games=[admission.read(r['path'],r['native_admission'],allow_diagnostic=True) for r in records]
    with meter.phase('admitted_replay_warm_read'):
        for r in records:admission.read(r['path'],r['native_admission'],allow_diagnostic=True)
    return games,dict(collection_seconds=elapsed,records=records,scheduler=scheduler,
        lanes=scheduler['lanes'],metrics=[m for r in records for m in r['native_metrics']])


def ordinary(args,snapshot,build,reference):
    import rust_lab_b06_ordinary as baseline
    with patch.object(baseline,'native',optimized_native):report=baseline.run(args,snapshot,build,reference)
    report['comparison_scope']='Current reference vs process-isolated native; same three frozen job strata and natural branches'
    return report


def canary(args,snapshot,build,inputs,index):
    import torch
    from brass_ai.bank import sha
    from brass_ai.learning.checkpoint import load_checkpoint,save_checkpoint
    from brass_ai.learning.batching import model_id
    from brass_ai.learning.search_value import install
    from brass_ai.recovery.inference import InferenceService
    from brass_ai.recovery.teacher_phase import plan_jobs,readiness
    from brass_ai.recovery.teacher_phase_config import buffer_identity
    from brass_ai.recovery.teacher_buffer import TeacherBuffer
    from brass_ai.recovery.search_value_config import adopt
    from brass_ai.recovery.search_value_phase import SearchValuePhase
    from brass_ai.rust_backend.lab_teacher import TeacherCollector,calibrate
    from brass_ai.rust_backend.lab_teacher_overlay import source_fingerprint
    from brass_ai.rust_backend.lab_b045_smoke import CHECKPOINT
    out=args.out/('canary-'+str(index));out.mkdir()
    model,raw=load_checkpoint(snapshot['checkpoint']['path']);model.eval()
    config=adopt(snapshot['state']['signature']['config']);config['teacher_phase']['mode']='shadow'
    config['teacher_phase']['search_value']['stage']='A';reward=digest(model.config.utility)
    receipt,ready=readiness(snapshot['state'].get('teacher_phase',{}).get('readiness'),raw['training'].get('utility_calibration'),reward)
    if receipt is None and snapshot['generation']==1250 and snapshot['checkpoint']['sha256']==CHECKPOINT:
        receipt=dict(ready=True,reward_id=reward,basis='user-authorized-g1250-null-utility',calibration_measured=False);ready=True
    if not ready:return dict(passed=False,deferred=True,reason='Task utility readiness unavailable; teacher scope deferred',tables=0)
    install(model);identity=model_id(model);generation=snapshot['generation']
    state=dict(iteration=generation,history=[dict(generation=generation,model_id=identity)])
    jobs,_=plan_jobs(config,state,model_id=identity,export_hash=snapshot['checkpoint']['sha256'],
                     epoch=index,ready=True,native_source=source_fingerprint())
    jobs=[next(j for j in jobs if j['split']==s) for s in ('train','heldout')]
    for job in jobs:job['readiness']=receipt
    write_json(out/'jobs.json',jobs)
    fixtures=json.loads((inputs/'fixtures.json').read_text())
    roots=[r for r in fixtures['roots'] if r['view']['context']['phase']=='action' and len(r['expected']['choices'])>1]
    roots=[roots[0],max(roots,key=lambda r:len(r['expected']['choices']))]
    origins=[dict(state=r['expected']['state'],rail_keys=list(range(64))) for r in roots]
    buffer=TeacherBuffer(SOURCE_ROOT,out/'policy',buffer_identity(config))
    with InferenceService(snapshot['execution'],args.device) as service:
        require(service.register(model)==identity,'Canary model changed')
        with service.lock:service._model(identity)
        calibration=calibrate(build['binary']['path'],build['core_source'],origins,jobs[0],service)
        write_json(out/'calibration.json',calibration)
        collector=TeacherCollector(build['binary']['path'],build['core_source'],
            [args.mono,build['bootstrap']['binary']['path']],service,calibration,lanes=2,model=model,history=state['history'])
        try:tables=collector.collect(jobs)
        except BaseException:
            write_json(out/'failure.json',dict(failures=collector.failures));raise
        finally:write_json(out/'inference.json',service.report())
        require(len(tables)==2,'Dropped/extra canary table')
        policy_sealing={}
        for job,table in zip(jobs,tables,strict=True):
            began=time.perf_counter()
            rows,rejected=collector.verifier.take(table,job,model)
            require(rows,'Missing train/heldout labels')
            buffer.publish(generation,job['split'],rows,model_id=identity,reward_id=reward)
            policy_sealing[job['expected']['job_id']]=time.perf_counter()-began
        require(service.error is None,'Canary inference failed')
    write_json(out/'tables.json',tables)
    reload=TeacherBuffer(SOURCE_ROOT,out/'policy',buffer_identity(config),receipts=buffer.pending)
    for split in ('train','heldout'):
        require(digest([(age,r.row) for age,r in reload.rows(generation,split)])==
                digest([(age,r.row) for age,r in buffer.rows(generation,split)]),'Canary buffer reload differs')
    model.to(args.device)
    values=SearchValuePhase(out/'value',model,config['teacher_phase']['search_value'],reward)
    # Each final turn pays its own policy + value seal, plus native post-work.
    # Keep the user-selected 0.5s fast / unchanged 1s full caps; never charge another lane's work.
    try:values.admit(tables,jobs,generation,post_spent=policy_sealing)
    finally:write_json(out/'admission.json',dict(policy_sealing_seconds=policy_sealing,value=values.report))
    # Shadow canary validates buffer/checkpoint/optimizer restore; does not enable
    # teacher PPO or calibrate production beta from these two tables.
    path=out/'shadow.pt';save_checkpoint(path,model,step=generation,training={'search_value':values.payload()})
    restored,saved=load_checkpoint(path)
    require(model_id(restored)==model_id(model),'Canary checkpoint reload differs')
    SearchValuePhase(out/'value',restored,config['teacher_phase']['search_value'],reward,saved=saved['training']['search_value'])
    return dict(passed=True,pass_index=index,tables=2,checkpoint_sha256=sha(path),buffer_reloaded=True,
        checkpoint_reloaded=True,ordinary_updates=0,teacher_optimizer_steps=0,blend_weight=0,
        readiness=receipt,labels=sum(t['metrics']['labels'] for t in tables),production_modified=False,
        policy_sealing_seconds=policy_sealing,value_admission=values.report)
