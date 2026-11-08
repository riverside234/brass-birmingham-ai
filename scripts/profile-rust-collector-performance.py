#!/usr/bin/env python3
"""At most three frozen checkpoint families per arm; no training/teacher search."""
import argparse
import copy
from contextlib import contextmanager
from datetime import datetime,timezone
import importlib.util
import os
from pathlib import Path
import sys
import time
import traceback
import uuid

ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]


@contextmanager
def frozen_run(root,run,generation,device):
    import torch
    from brass_ai.bank import read
    from brass_ai.learning.contracts import require
    from brass_ai.recovery.retention import run_lock
    from brass_ai.rust_backend.lab_snapshot import capture,verify_unchanged
    with run_lock(run):
        require(read(run/'state.json')['iteration']==generation,'Stale matched-profile generation')
        target=torch.device(device)
        snapshot,model,raw=capture(root,run,devices=[target.index or 0] if target.type=='cuda' else [])
        del model,raw
        try:yield snapshot
        except BaseException as error:
            try:verify_unchanged(snapshot)
            except Exception as changed:error.add_note('Snapshot verification also failed: '+str(changed))
            raise
        else:verify_unchanged(snapshot)


def arguments():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',default='artifacts/step5/joint7-v11-2-g1290-trained-restart')
    p.add_argument('--expected-generation',type=int,required=True)
    p.add_argument('--device',default='cuda');p.add_argument('--mono',default='mono')
    p.add_argument('--out',type=Path)
    p.add_argument('--arm',choices=('reference','rust'),help=argparse.SUPPRESS)
    return p.parse_args()


def collect_arm(args,out):
    import torch
    from brass_ai.bank import read
    from brass_ai.learning.contracts import digest,require
    from brass_ai.learning.checkpoint import load_checkpoint
    from brass_ai.recovery.execution import ActorPool
    from brass_ai.recovery.host_runtime import local_scoring_host,training_host
    from brass_ai.recovery.inference import InferenceService
    from brass_ai.recovery.replay import read_shards
    from brass_ai.rust_backend import lab_admission as A
    from brass_ai.rust_backend.lab_backend import check_build
    from brass_ai.rust_backend.lab_perf_runtime import resolve,CONTRACT
    from brass_ai.rust_backend.lab_jobs import plan
    from brass_ai.rust_backend.lab_pool import NativePool
    from brass_ai.step5.league import write_json
    from rust_lab_b06_ordinary import freeze_jobs,prepared_rows
    from rust_lab_b06_metrics import Meter,measurement_preflight,work
    snapshot=read(out/'snapshot.json');config=snapshot['state']['signature']['config'];execution=snapshot['execution']
    directory=out/args.arm;directory.mkdir()
    environment=measurement_preflight(args.device)
    class TimedReference(ActorPool):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs);self.startup_seconds=0.
        def start(self,*args,**kwargs):
            began=time.perf_counter()
            try:return super().start(*args,**kwargs)
            finally:self.startup_seconds+=time.perf_counter()-began
    with Meter(args.device) as meter:
        with meter.phase('checkpoint_load'):
            model,raw=load_checkpoint(snapshot['checkpoint']['path']);model.eval()
        with InferenceService(execution,args.device) as service:
            with meter.phase('planning_and_model_registration'):
                planned=plan(ROOT,args.run,snapshot,model,raw,service,directory/'plan');jobs=freeze_jobs(planned)
                write_json(directory/'frozen.json',dict(jobs=jobs,ids=[digest({k:v for k,v in j.items() if k!='path'}) for j in jobs]))
            with meter.phase('model_preload'):
                for identity in sorted({p['model_id'] for j in jobs for p in j['policies'] if p['kind']=='neural'}):
                    with service.lock:service._model(identity)
            began=time.perf_counter()
            if args.arm=='reference':
                for j in jobs:j['path']=str(directory/'replay'/('game-'+str(j['index'])+'.jsonl.gz'))
                host=local_scoring_host(ROOT) if any('local_scoring' in j['provenance'] for j in jobs) else training_host(ROOT)
                with TimedReference(ROOT,host,service,execution,mono=args.mono) as pool:
                    records=pool.collect(jobs);scheduler=pool.last_collection_report;startup=pool.startup_seconds
                    shutdown_began=time.perf_counter()
                shutdown=time.perf_counter()-shutdown_began
            else:
                original=check_build(snapshot['state']['native_backend']['build']);build=resolve(ROOT,original)
                require(build.get('contract')==CONTRACT,'Prepare the new ordinary adapter before matched profiling')
                with NativePool(ROOT,service,execution,build,config,snapshot['signature_sha256'],mono=args.mono,scope='diagnostic') as pool:
                    records=pool.collect(jobs);scheduler=pool.last_collection_report
                startup=scheduler['startup_seconds'];shutdown=scheduler['shutdown_seconds']
            collection=time.perf_counter()-began
            began=time.perf_counter()
            if args.arm=='reference':
                games=read_shards([r['path'] for r in records],receipts={r['path']:r['features'] for r in records if r.get('features')})
            else:games=[A.read(r['path'],r['native_admission'],allow_diagnostic=True) for r in records]
            by_id={g['game_id']:g for g in games};games=[by_id[j['stable_game_id']] for j in jobs]
            first_read=time.perf_counter()-began;began=time.perf_counter()
            prepared=[prepared_rows([g],config,j['provenance']['learner_behavior']['policy_id']) for g,j in zip(games,jobs,strict=True)]
            prepare=time.perf_counter()-began
            service_report=service.report()
        result=dict(passed=True,arm=args.arm,checkpoint=snapshot['checkpoint'],generation=snapshot['generation'],
            model_id=snapshot['model_id'],config_sha256=snapshot['config_sha256'],teacher_mode=config.get('teacher_phase',{}).get('mode','off'),
            teacher_search_executed=False,records=records,work=work(games),scheduler=scheduler,
            startup_seconds=startup,shutdown_seconds=shutdown,collection_including_startup_shutdown_seconds=collection,
            startup_scope=('process launch only; asynchronous imports remain in collection' if args.arm=='reference' else 'full ready barrier'),
            reference_host_startup_worker_seconds=(sum(r.get('metrics',{}).get('host.startup',{}).get('seconds',0.) for r in records) if args.arm=='reference' else None),
            collection_without_measured_startup_shutdown_seconds=max(0.,collection-startup-shutdown),replay_read_seconds=first_read,
            preparation_seconds=prepare,accepted_work_seconds=collection+first_read+prepare,
            prepared_rows=sum(map(len,prepared)),inference=service_report,measurement=meter.report(),environment=environment)
    write_json(directory/'report.json',result)


def main():
    args=arguments()
    if args.arm:
        collect_arm(args,args.out.resolve());return
    from brass_ai.bank import read
    from brass_ai.learning.contracts import require
    from brass_ai.recovery.retention import owned
    from brass_ai.rust_backend.lab_replay_compat import execution
    from brass_ai.step5.league import write_json
    from rust_lab_results import export
    out=(args.out or ROOT/'artifacts/rust-backend'/('collector-matched-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')+'-'+uuid.uuid4().hex[:8])).resolve()
    require(out.is_relative_to(ROOT/'artifacts/rust-backend') and not out.exists(),'Use a new owned artifacts/rust-backend output')
    out.mkdir(parents=True);report=dict(passed=False,scope='At most three frozen ordinary families per arm; actual checkpoint; no updates/teacher search')
    try:
        root,run=owned(ROOT,args.run)
        with frozen_run(root,run,args.expected_generation,args.device) as snapshot:
            from brass_ai.rust_backend.lab_backend import validate_state,check_build
            from brass_ai.rust_backend.lab_perf_runtime import resolve,CONTRACT
            backend=validate_state(snapshot['state'])
            require(backend is not None and backend['mode']=='rust','Matched profile needs the adopted Rust run')
            build=resolve(ROOT,check_build(backend['build']))
            require(build.get('contract')==CONTRACT,'Run prepare-rust-collector-performance.py before matched profiling')
            write_json(out/'snapshot.json',snapshot);report['execution']=execution()
            spec=importlib.util.spec_from_file_location('pi_bounded_command',ROOT/'scripts/qualify-rust-lab.py')
            base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
            env=dict(base.python_environment(),BRASS_COLLECTOR_VALIDATION='profile')
            for arm in ('reference','rust'):
                base.command([sys.executable,'-B',__file__,'--arm',arm,'--out',str(out),'--run',args.run,
                    '--expected-generation',str(args.expected_generation),'--device',args.device,'--mono',args.mono],
                    cwd=ROOT,env=env,log=out/(arm+'.log'),timeout=120)
            from brass_ai.recovery.replay import read_shards
            from brass_ai.rust_backend.lab_admission import read as read_native
            from rust_lab_b06_ordinary import prepared_rows
            from rust_lab_b06_metrics import compare_games,compare_rows
            arms={a:read(out/a/'report.json') for a in ('reference','rust')}
            left=read(out/'reference/frozen.json');right=read(out/'rust/frozen.json')
            require(left['ids']==right['ids'],'Planner did not preserve identical frozen jobs')
            refs=arms['reference']['records'];games={'reference':read_shards([r['path'] for r in refs],receipts={r['path']:r['features'] for r in refs if r.get('features')}),
                'rust':[read_native(r['path'],r['native_admission'],allow_diagnostic=True) for r in arms['rust']['records']]}
            by_arm={a:{g['game_id']:g for g in gs} for a,gs in games.items()};config=snapshot['state']['signature']['config']
            games={a:[by_arm[a][j['stable_game_id']] for j in left['jobs']] for a in games}
            matched=compare_games(games['reference'],games['rust'])
            for j in left['jobs']:
                compare_rows(*(prepared_rows([by_arm[a][j['stable_game_id']]],config,j['provenance']['learner_behavior']['policy_id']) for a in ('reference','rust')))
            report.update(arms=arms,matched=matched,generation=snapshot['generation'],checkpoint=snapshot['checkpoint'],
                training_updates=0,teacher_search_executed=False)
        report.update(passed=True,source_unchanged=True)
    except BaseException as error:
        report.update(passed=False,error=str(error),traceback=traceback.format_exc());raise
    finally:
        write_json(out/'qualification.json',report)
        try:print('Git-visible results: '+str(export(out)),flush=True)
        except Exception as error:
            if report['passed']:raise
            print('Result export also failed: '+str(error),file=sys.stderr,flush=True)
        print('Original lab evidence: '+str(out),flush=True)


if __name__=='__main__':main()
