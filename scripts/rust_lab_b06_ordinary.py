"""Bounded actual g1250 ordinary collection: current ActorPool versus native lanes."""
from collections import Counter
import copy
import queue
import threading
import time

import torch

from brass_ai.learning.contracts import SOURCE_ROOT, digest, require
from brass_ai.learning.checkpoint import load_checkpoint
from brass_ai.recovery.inference import InferenceService
from brass_ai.recovery.execution import ActorPool
from brass_ai.recovery.replay import read_shards
from brass_ai.rust_backend.lab_collection import collect_lane, frozen_id, route
from brass_ai.rust_backend.lab_families import FamilyStore, collect_families
from brass_ai.rust_backend.lab_transport import Pipe
from brass_ai.rust_backend.lab_jobs import plan
from brass_ai.rust_backend.lab_b02_smoke import reference_part, preflight_reference
from brass_ai.rust_backend.lab_b03_smoke import prepared_rows
from brass_ai.rust_backend import lab_admission as admission
from brass_ai.step5.league import write_json
from rust_lab_b06_metrics import Meter, choose_jobs, work, compare_games, compare_rows, performance


def freeze_jobs(planned):
    jobs=choose_jobs(planned['jobs'])
    for job in jobs:
        # Same additive identity in both backends. Policy/job/branch draws never change.
        job.setdefault('stable_game_id',digest(dict(job=frozen_id(job),seed=job['seed'])))
    return jobs


def native(root, jobs, service, execution, build, output, mono, meter, producer, snapshot):
    lanes=min(execution['cpu_budget']-execution['cpu_reserve'],
              execution['processes']*execution['hosts_per_process'],len(jobs)*execution['hosts_per_process'])
    require(lanes>0,'No native worker capacity')
    lease=service.channels(lanes);owned=[lease.acquire() for _ in range(lanes)]
    endpoints=queue.Queue();stop=threading.Event();failure_lock=threading.Lock();first_failure=[]
    def fail(error):
        with failure_lock:
            if not first_failure:first_failure.append((error,error.__traceback__))
            stop.set()
    for endpoint in owned: endpoint.timeout=30.;endpoints.put(endpoint)
    bootstrap=[mono,build['bootstrap']['binary']['path']]
    def one(job,continuation,submit):
        endpoint=endpoints.get(timeout=30)
        try:
            with Pipe(bootstrap,binary=False) as draws:
                return collect_lane(root,build['collector']['path'],draws,job,endpoint,execution,
                                    continuation=continuation,submit_branch=submit,cancel=stop)
        except BaseException as error:fail(error);raise
        finally:endpoints.put(endpoint)
    def fallback(job,continuation,submit):
        require(continuation is None,'Unexpected fallback continuation')
        endpoint=endpoints.get(timeout=30)
        try:return reference_part(root,job,endpoint,execution,mono)
        except BaseException as error:fail(error);raise
        finally:endpoints.put(endpoint)
    class MeasuredStore(FamilyStore):
        def publish(self,*args,**kwargs):
            with meter.phase('family_validation_and_sealing_nested_in_collection'):
                return super().publish(*args,**kwargs)
    store=MeasuredStore(output/'families',producer)
    try:
        began=time.perf_counter()
        with meter.phase('collection_with_family_validation_and_sealing'):
            families,scheduler=collect_families(jobs,store,one,fallback,lanes=lanes,cancel=stop,
                                               model_cohorts=execution.get('model_cohorts',False))
        elapsed=time.perf_counter()-began
    except BaseException:
        if first_failure:
            error,tb=first_failure[0];raise error.with_traceback(tb)
        raise
    finally:
        for endpoint in owned:endpoint.close()
    games=[];records=[]
    with meter.phase('admission_and_first_read'):
        for job,family in zip(jobs,families,strict=True):
            # Diagnostic scope cannot train; separately enforce the unmodified natural trigger.
            admission.branch_trigger(family,job,False)
            binding=admission.binding(job,producer,generation=snapshot['generation'],model_id=snapshot['model_id'],
                                      config=snapshot['state']['signature']['config'],scope='diagnostic')
            record=admission.publish(output/'replay'/(frozen_id(job)+'.jsonl.gz'),family,binding)
            games.append(admission.read(record['path'],record['native_admission'],allow_diagnostic=True));records.append(record)
    with meter.phase('admitted_replay_warm_read'):
        for record in records:admission.read(record['path'],record['native_admission'],allow_diagnostic=True)
    metrics=[dict(job=j['index'],role=role,route=route(j)[0],timings=part['timings'],
                  native_cost=part.get('native_receipt',{}).get('cost'),
                  buffer_bytes=part.get('native_receipt',{}).get('buffer_bytes'))
             for j,f in zip(jobs,families,strict=True) for role,part in [('parent',f['parent']),('branch',f['child'])] if part]
    return games,dict(collection_seconds=elapsed,records=records,scheduler=scheduler,lanes=lanes,metrics=metrics)


def reference(root,jobs,service,execution,output,mono,meter,snapshot):
    actual=copy.deepcopy(jobs)
    for job in actual:job['path']=str(output/'replay'/('game-'+str(job['index'])+'.jsonl.gz'))
    from brass_ai.recovery.host_runtime import local_scoring_host, training_host
    host=local_scoring_host(root) if any('local_scoring' in j['provenance'] for j in actual) else training_host(root)
    with ActorPool(root,host,service,execution,mono=mono) as pool:
        began=time.perf_counter()
        with meter.phase('collection_with_family_validation_and_sealing'):
            records=pool.collect(actual)
        elapsed=time.perf_counter()-began
        scheduler=pool.last_collection_report
        active=len(pool.processes)*pool.lanes
    receipts={r['path']:r.get('features') for r in records if r.get('features')}
    with meter.phase('admission_and_first_read'):
        games=read_shards([r['path'] for r in records],receipts=receipts)
    with meter.phase('admitted_replay_warm_read'):
        read_shards([r['path'] for r in records],receipts=receipts)
    by_id={g['game_id']:g for g in games}
    return [by_id[j['stable_game_id']] for j in jobs],dict(collection_seconds=elapsed,records=records,
        scheduler=scheduler,lanes=active,metrics=[r.get('metrics',{}) for r in records])


def run(args,snapshot,build,reference_info):
    root=SOURCE_ROOT;output=args.out/'ordinary';output.mkdir()
    execution=snapshot['execution'];config=snapshot['state']['signature']['config']
    arms={};all_games={};all_rows={};frozen=None
    for arm in ('reference','rust'):
        directory=output/arm;directory.mkdir()
        print('B06 ordinary: '+arm,flush=True)
        with Meter(args.device) as meter:
            with meter.phase('cpu_checkpoint_load'):
                model,raw=load_checkpoint(snapshot['checkpoint']['path']);model.eval()
            with InferenceService(execution,args.device) as service:
                with meter.phase('planning_and_model_registration'):
                    planned=plan(root,args.run,snapshot,model,raw,service,directory/'plan')
                    jobs=freeze_jobs(planned)
                if frozen is None:
                    frozen=[frozen_id(j) for j in jobs]
                    write_json(output/'frozen-jobs.json',dict(jobs=jobs,full_plan_hash=digest(planned['jobs']),
                        composition=planned['composition'],subset_by='first actual (kind,V4 presence) stratum',
                        branch_override=False,execution=execution,teacher='off'))
                require(frozen==[frozen_id(j) for j in jobs],'Comparison replanned different jobs')
                with meter.phase('reference_policy_preflight'):
                    preflight_reference(root,jobs,execution,args.mono)
                with meter.phase('gpu_model_load'):
                    for identity in sorted({p['model_id'] for j in jobs for p in j['policies'] if p['kind']=='neural'}):
                        with service.lock:service._model(identity)
                before=copy.deepcopy(service.report())
                if arm=='reference':games,report=reference(root,jobs,service,execution,directory,args.mono,meter,snapshot)
                else:
                    producer=dict(build_sha256=reference_info['ordinary_build_sha256'],b01_sha256=build['b01']['sha256'],
                                  signature=snapshot['signature_sha256'],profile='search_v1',v205_enabled=False)
                    games,report=native(root,jobs,service,execution,build,directory,args.mono,meter,producer,snapshot)
                after=copy.deepcopy(service.report());require(service.error is None,'Ordinary GPU service failed')
                from brass_ai.recovery.telemetry import iteration_utilization
                report['inference']=iteration_utilization(before,after,report['records'],report['collection_seconds'],len(games),execution)
                report['service']=after
            rows=[]
            for pass_index in range(2):
                with meter.phase('preparation_first' if pass_index==0 else 'preparation_warm'):
                    current=[prepared_rows([g],config,j['provenance']['learner_behavior']['policy_id']) for g,j in zip(games,jobs,strict=True)]
                if pass_index==0:rows=current
                else:
                    for a,b in zip(rows,current,strict=True):compare_rows(a,b)
            report.update(work=work(games),measurement=meter.report(),cache_scope='Cold worker processes; preloaded GPU models; second replay read/preparation is warm',
                          accepted_work_seconds=report['collection_seconds']+sum(meter.phases['admission_and_first_read'])+sum(meter.phases['preparation_first']))
            write_json(directory/'report.json',report)
            arms[arm]=report;all_games[arm]=games;all_rows[arm]=rows
            del model,raw
    prepared_count=sum(compare_rows(a,b) for a,b in zip(all_rows['reference'],all_rows['rust'],strict=True))
    matched=compare_games(all_games['reference'],all_games['rust'])
    result=dict(passed=True,arms=arms,matched=matched,prepared_rows=prepared_count,
        performance=performance(arms['reference'],arms['rust'],matched['equal']),
        natural_branch_observed=bool(matched['work']['branches']),
        scope='At most three parent families per backend; existing planner/current recipe, natural forks, no production admission',
        coverage=dict(strata=[dict(kind=j['provenance']['kind'],route=route(j)[0]) for j in jobs],
                      full_plan_composition=planned['composition'],worker_cap=execution['cpu_budget']-execution['cpu_reserve']))
    write_json(output/'comparison.json',result)
    return result
