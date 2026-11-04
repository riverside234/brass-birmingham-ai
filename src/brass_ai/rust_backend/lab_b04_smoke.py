"""Exactly two reserved-schedule shadow tables; no optimizer or production write."""
import copy
import json
import os
from pathlib import Path
import time

import torch
from brass_ai.learning.contracts import SOURCE_ROOT,require,digest,read_json
from brass_ai.learning.checkpoint import load_checkpoint
from brass_ai.recovery.inference import InferenceService
from brass_ai.recovery.teacher_phase import TeacherPhase,readiness
from brass_ai.recovery.teacher_phase_config import adopt
from brass_ai.step5.league import implementation_id,write_json
from .lab_contract import pin,sha
from .lab_snapshot import verify_unchanged
from .lab_b03_smoke import entry as b02_entry
from .lab_teacher import TeacherCollector,calibrate
from .lab_teacher_overlay import source_fingerprint

G570_RUN = 'artifacts/step5/joint7-v11-2-g1290-trained-restart'
G570_CHECKPOINT = '06626c1fb7d0b91f64d939f9fbb3db7c5f3aa072331eb3d773f101e729117ab9'
G570_MODEL = '18f42a48933c18095d96f1cc64b836d3f9aac92361a4e4ee7c423830d9a11a91'


def entry(b02,b03,run,device):
    snapshot,_,_,_=b02_entry(b02,run,device)
    r=read_json(b03)
    require(r.get('contract')=='brass-lab-b03/1.0.0' and r.get('passed') is True
        and r.get('lab_qualification')=='Passed for this exact committed checkpoint'
        and r['method']=='search_v1' and r['v205_enabled'] is False,'B04 requires passing lab B03')
    if r['transfer']['implementation']!=implementation_id():
        from .lab_prerequisite_reuse import verify_b03
        verify_b03(b03,r['transfer']['implementation'])
    lab=r['lab'];require(lab['passed'] is True and lab['new_games']==0 and lab['optimizer_steps']==0,'Invalid B03 scope')
    for k in ('preflight','restoration','replay','diagnostic_fork'):
        require(pin(lab[k]['path'])==lab[k],'Changed B03 '+k)
    preflight=read_json(lab['preflight']['path'])
    require(preflight['snapshot']['state_file']==snapshot['state_file'] and
            preflight['snapshot']['checkpoint']==snapshot['checkpoint'],'B03 belongs to another committed state')
    return snapshot


def latest_readiness(root,state,raw,reward_id,*,checkpoint=None):
    """Most recent non-null committed receipt, never skip an unknown gap/false.

    Adoption may precede a saved teacher phase. Inspect retained checkpoints in
    descending order only while consecutive lineage entries remain available.
    Missing history means deferred, except for the explicit user-authorized
    null-receipt assumption on the exact g570 diagnostic checkpoint below.
    """
    current=raw['training'].get('utility_calibration')
    saved=state.get('teacher_phase',{}).get('readiness')
    if current is not None or saved is not None:
        receipt,ready=readiness(saved,current,reward_id)
        return receipt,ready,dict(kind='committed-current-or-phase',generation=state['iteration'])
    if checkpoint is not None and checkpoint.get('sha256')==G570_CHECKPOINT:
        latest=state['history'][-1]
        require(state['iteration']==raw['step']==latest['generation']==570
                and raw['model_id']==latest['model_id']==G570_MODEL
                and latest['sha256']==G570_CHECKPOINT
                and latest['path']==G570_RUN+'/generation-0570.pt'
                and raw['training']['fresh_run']==G570_RUN
                and raw['model_config'].get('utility') is not None
                and digest(raw['model_config']['utility'])==reward_id,
                'G570 null-readiness assumption checkpoint/model/run/reward differs')
        # This is an explicit diagnostic assumption, not a fitted calibration
        # receipt. Preserve the real null in reports and checkpoint metadata.
        return None,True,dict(kind='user-authorized-g570-null-utility',generation=570,
            checkpoint_sha256=G570_CHECKPOINT,model_id=G570_MODEL,reward_id=reward_id,
            scope='B04 shadow qualification only',calibration_measured=False)
    for item in reversed(state['history'][:-1]):
        path=(Path(root)/item['path']).resolve()
        require(path.is_relative_to(Path(root).resolve()),'Readiness checkpoint escaped project')
        if not path.is_file():return None,False,dict(reason='Latest non-null readiness cannot be proved across retired checkpoint',generation=item['generation'])
        require(sha(path)==item['sha256'],'Readiness checkpoint changed')
        previous=torch.load(path,map_location='cpu',weights_only=True)
        require(previous['step']==item['generation'] and previous['model_id']==item['model_id'],'Readiness lineage differs')
        if previous['model_config'].get('utility') != raw['model_config'].get('utility'):
            return None,False,dict(reason='Reward boundary before a current-reward receipt')
        receipt=previous['training'].get('utility_calibration')
        if receipt is not None:
            receipt,ready=readiness(None,receipt,reward_id)
            return receipt,ready,dict(kind='retained-lineage-checkpoint',file=pin(path),generation=item['generation'])
    return None,False,dict(reason='No committed current-reward readiness receipt')


def run(args):
    root=SOURCE_ROOT;out=args.out;build=read_json(args.build)
    snapshot=entry(args.b02,args.b03,args.run,args.device)
    require(snapshot['generation']==args.expected_generation and snapshot['checkpoint']['sha256']==args.checkpoint_sha256,
            'Expected generation/checkpoint changed')
    parent=json.loads(os.environ.get('BRASS_B04_PARENT_LOCK','{}'))
    require(parent.get('pid')==os.getppid() and read_json(args.run/'.active-run.json')==parent,'B04 child requires parent run lock')
    for item in (build['binary'],build['bootstrap']['binary'],*build['bootstrap']['dependencies']):
        require(pin(item['path'])==item,'B04 runtime changed')
    require(build['producer']==source_fingerprint() and build['implementation']==implementation_id(),'B04 build source differs')
    require(torch.device(args.device).type=='cuda' and torch.cuda.is_available(),'B04 lab smoke requires actual CUDA')
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    model,raw=load_checkpoint(snapshot['checkpoint']['path'],device='cpu');model.eval()
    config=adopt(snapshot['state']['signature']['config']);config['teacher_phase']['mode']='shadow'
    reward=digest(config['teacher_phase']['search']['reward'])
    receipt,ready,evidence=latest_readiness(root,snapshot['state'],raw,reward,checkpoint=snapshot['checkpoint'])
    write_json(out/'readiness.json',dict(receipt=receipt,ready=ready,evidence=evidence,model_calibration_unchanged=True))
    if not ready:
        return dict(passed=False,deferred=True,reason='Latest current-reward lab readiness is not ready/unavailable; no teacher tables dispatched',
            readiness=pin(out/'readiness.json'),new_games=0,optimizer_steps=0,production_modified=False)
    state=dict(iteration=snapshot['generation'],history=copy.deepcopy(snapshot['state']['history']))
    if 'teacher_seed_cursor' in snapshot['state']:state['teacher_seed_cursor']=snapshot['state']['teacher_seed_cursor']
    phase=TeacherPhase(root,out/'diagnostic-phase',config,state,model)
    phase.receipt=(receipt if receipt is not None else
        dict(ready=True,reward_id=reward,basis=copy.deepcopy(evidence)))
    phase.ready=ready
    full=phase.begin(state,model,export_hash=snapshot['checkpoint']['sha256'],epoch=0,native_source=source_fingerprint())
    jobs=[next(j for j in full if j['split']==split) for split in ('train','heldout')]
    phase.jobs=jobs
    write_json(out/'frozen-teacher-jobs.json',dict(jobs=jobs,full_plan_digest=digest(full),planned_tables=len(full),
        diagnostic_subset=True,production_mode=snapshot['state']['signature']['config'].get('teacher_phase',{}).get('mode','absent'),
        diagnostic_mode='shadow',ordinary_seed_cursor_unchanged=True))
    b02=read_json(args.b02);b02_build=read_json(b02['build']['path'])
    fixtures=read_json(Path(b02_build['b01']['path']).parent/'inputs/fixtures.json')
    roots=[r for r in fixtures['roots'] if r['view']['context']['phase']=='action' and len(r['expected']['choices'])>1]
    require(roots,'Missing admitted action roots for GPU calibration')
    # One typical and one wide permitted root. No setup/phase resampling.
    roots=[roots[0],max(roots,key=lambda r:len(r['expected']['choices']))]
    origins=[dict(state=r['expected']['state'],rail_keys=list(range(64))) for r in roots]
    began=time.monotonic()
    try:
        with InferenceService(snapshot['execution'],args.device) as service:
            require(service.register(model)==snapshot['model_id'],'Frozen teacher model differs')
            # Initial CUDA model loading is outside the measured trajectory work.
            with service.lock:service._model(snapshot['model_id'])
            calibration=calibrate(build['binary']['path'],build['core_source'],origins,jobs[0],service)
            calibration['roots']=[r['id'] for r in roots];write_json(out/'gpu-calibration.json',calibration)
            from .lab_teacher import turn_probes
            measured=turn_probes(build['binary']['path'],build['core_source'],origins,jobs[0],service,calibration,
                model,snapshot['state']['history'])
            write_json(out/'timed-turns.json',measured)
            bootstrap=[args.mono,build['bootstrap']['binary']['path']]
            backend=TeacherCollector(build['binary']['path'],build['core_source'],bootstrap,service,calibration,lanes=2,
                model=model,history=snapshot['state']['history'])
            results=backend.collect(jobs);write_json(out/'native-tables.json',results)
            service_report=service.report();require(service.error is None,'Teacher inference service failed')
        # Audit the actual previous table-batched verifier outside the measured
        # collection. Its output must match the per-turn verified cache exactly.
        comparison=backend.verifier.compare_previous_barrier(results)
        model.to(args.device);phase.admit_collection(results,model,verifier=backend.verifier)
        for item,result in zip(comparison['tables'],results,strict=True):
            item['verified_reuse_seconds']=result['metrics']['verified_reuse_seconds']
            item['barrier_reduction_seconds']=item['previous_barrier_seconds']-item['verified_reuse_seconds']
        write_json(out/'verification-comparison.json',comparison)
        phase.prepare(model);report=phase.finish(model,{})
        require(report['collection']['labels']>0,'No completed label evidence; B04 remains unqualified')
        write_json(out/'teacher-report.json',report);write_json(out/'inference-report.json',service_report)
        verify_unchanged(snapshot)
        return dict(passed=True,readiness=pin(out/'readiness.json'),calibration=pin(out/'gpu-calibration.json'),
            validation=pin(out/'verification-comparison.json'),timing=pin(out/'timed-turns.json'),tables=pin(out/'native-tables.json'),jobs=pin(out/'frozen-teacher-jobs.json'),report=pin(out/'teacher-report.json'),
            inference=pin(out/'inference-report.json'),new_games=2,optimizer_steps=0,production_modified=False,
            elapsed_seconds=time.monotonic()-began,scope='Two separate full teacher tables, one train and one heldout; 4s/K16, one session/thread per table, two concurrent lanes. Missing label kinds supplemented by authored tests, no extra games.')
    finally:verify_unchanged(snapshot)


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--b02',type=Path,required=True)
    p.add_argument('--b03',type=Path,required=True);p.add_argument('--build',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--expected-generation',type=int,required=True);p.add_argument('--checkpoint-sha256',required=True)
    p.add_argument('--device',default='cuda:0');p.add_argument('--mono',default='mono');a=p.parse_args()
    write_json(a.out/'lab-checks.json',run(a))
