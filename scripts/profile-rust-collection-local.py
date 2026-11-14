"""Matched, reduced CPU collection probe using native workers and real replay.

This uses fixture weights, smaller search quotas and a local CPU allocation.
It exercises collection, admission and recovery boundaries without training.
"""
import argparse
import json
import multiprocessing
from pathlib import Path
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]


def plan(count, teacher_count, model, anchor):
    from brass_ai.learning.batching import model_id
    from brass_ai.learning.contracts import digest
    from brass_ai.recovery import ordinary_cotton, training_revision_g1491 as revision
    from brass_ai.recovery.execution import neural_spec
    from brass_ai.recovery.exploration import THREE_T2_OPENING
    from brass_ai.recovery.opening_objective import T2_THREE
    from brass_ai.recovery.search_value_config import fast_profile
    from brass_ai.recovery.teacher_cotton import SEATED_CONTRACT
    from brass_ai.recovery.teacher_phase import plan_jobs
    from brass_ai.step5.league import seeds_for
    from brass_ai.step5.policy import cotton_seats
    from brass_ai.rust_backend.lab_teacher_overlay import source_fingerprint
    from test_b025 import config

    identity = model_id(model)
    learner = neural_spec(identity, 0., 'neutral', version=3, opening=THREE_T2_OPENING)
    opponent = neural_spec(model_id(anchor), 0., 'neutral', version=3, opening=THREE_T2_OPENING)
    options = config()
    ordinary = ordinary_cotton.configure(dict(update=dict(opening_objective=T2_THREE),
        table_plan={}, reward=options.get('reward'), insolvency=options.get('insolvency')), probability=.10)
    signature = digest(ordinary)
    jobs = []
    for index, seed in enumerate(seeds_for(10, count)[0]):
        historic = index % 10 in (0, 1, 2)
        kind = 'historical' if historic else 'selfplay'
        sampling_seed = 31 + index
        proof = dict(kind=kind, purpose='training', ordinary_game_index=index, sampling_seed=sampling_seed,
                     spec=signature, exploration=dict(age=10), ppo_opening_objective=T2_THREE,
                     learner_behavior=dict(model_id=identity, policy_id=learner['policy_id']))
        if not historic:
            assignment = ordinary_cotton.assignment(index, sampling_seed, probability=.10)
            if assignment is not None:
                proof[ordinary_cotton.KEY] = assignment
        jobs.append(dict(index=index, seed=seed, sampling_seed=sampling_seed, stable_game_id=f'local-{index}',
                         policies=[learner]*3 + [opponent if historic else learner], provenance=proof))
    options['teacher_phase']['mode'] = 'shadow'
    teacher, _ = plan_jobs(options, dict(iteration=10), model_id=identity, export_hash='a'*64,
                           epoch=1, ready=True, native_source=source_fingerprint())
    from brass_ai.learning.search_value import binding
    opening = dict(revision.EXPLORATION, game_opening=revision.GAME_OPENING)
    marker = revision._mixed_tables(dict(exploration=opening), 0, seated=True)['exploration']
    for index, job in enumerate(teacher[:teacher_count]):
        biased = index % 2 == 0
        seats = cotton_seats(job['sampling_seed'], 1 + index//2 % 2, SEATED_CONTRACT) if biased else []
        job.update(readiness=dict(ready=True,reward_id=job['expected']['reward_id']), seating='all-seat-search',
                   fast_search=fast_profile(job['search']),
                   search_value=binding(0.), teacher_revision=revision._job_block(
                       marker, split=job['split'], short_labels=False, cotton_table=biased, seats=seats))
    return ordinary, signature, jobs, teacher[:teacher_count]


def old_teacher_check(verifier):
    """The previous single-table policy lock and one policy check per table."""
    from concurrent.futures import Future
    class Serial:
        def __init__(self, value, count, *, pending):
            self.value, self.metrics = value, {}
        def __enter__(self):
            return self
        def __exit__(self, *_):
            pass
        def reserve(self):
            pass
        def release(self):
            pass
        def submit(self, key):
            self.value.settle(key)
            future = Future()
            future.set_result(None)
            return future
    from unittest.mock import patch
    return patch.object(verifier, 'TableLogitBatcher', Serial)


def prepare_digest(data):
    from hashlib import sha256
    import numpy as np
    from brass_ai.learning.contracts import digest
    def plain(value):
        if isinstance(value,np.ndarray):return value.tolist()
        if isinstance(value,np.generic):return value.item()
        if isinstance(value,dict):return {key:plain(val) for key,val in value.items()}
        if isinstance(value,(list,tuple)):return [plain(val) for val in value]
        return value
    output = sha256()
    for row, target, identity, encoded in data:
        output.update(digest(plain((row,target,identity))).encode())
        for array in (encoded.state, encoded.actions, encoded.stable_to_relative):
            output.update(memoryview(np.ascontiguousarray(array)).cast('B'))
    return output.hexdigest()


def run_arm(args, output, binaries, common, *, candidate):
    from contextlib import nullcontext
    from brass_ai.learning.batching import model_id
    from brass_ai.learning.contracts import digest
    from brass_ai.recovery.collector_runtime import cpu_block, scope
    from brass_ai.recovery.ingress import reader_pool, StreamingCollection
    from brass_ai.recovery.resources import CPUAdmission
    from brass_ai.recovery.utility import reward_spec
    from brass_ai.rust_backend import lab_teacher_finalization
    from brass_ai.rust_backend.lab_contract import pin
    from brass_ai.rust_backend.lab_cpu_runtime import CorePlan
    from brass_ai.rust_backend.lab_pool import NativePool
    from brass_ai.rust_backend.lab_teacher import TeacherCollector, count_calibration
    from brass_ai.rust_backend.lab_shadow import TableCache, TeacherRun

    output.mkdir()
    model, bundles, runtime, options, signature, jobs, teachers = common
    jobs = [dict(job, path=str(output / f'game-{job["index"]}.gz')) for job in jobs]
    kinds = {digest(dict(rules='steam-brass-patch-1.1.0',setup_seed=j['seed'])):j['provenance']['kind'] for j in jobs}
    learner = jobs[0]['provenance']['learner_behavior']['policy_id']
    collector = binaries/'ordinary/brass-lab-collector.exe'
    worker = binaries/'teacher/brass-lab-teacher.exe'
    build = dict(bootstrap=dict(binary=pin(args.bootstrap)), collector=pin(collector),
                 _receipt_sha256=pin(collector)['sha256'], b01=dict(sha256='c'*64))
    core = (binaries/'teacher/core.txt').read_text().strip()
    budget = dict(mode='count',full=args.full,fast=args.fast,full_k=8,fast_k=8,c_scale=.15,fast_c_scale=.5)
    inference = dict(contract=runtime['contract'], runtime=runtime['runtime'], bundle=bundles[model_id(model)], budget=budget)
    execution = dict(processes=args.ordinary_lanes,hosts_per_process=1,worker_threads=1,
                     max_decisions=5000,batch_size=32,max_candidate_cells=8192)
    gate = CPUAdmission(multiprocessing.get_context('spawn'),args.cpu_slots,0)
    service = SimpleNamespace(cpu=gate, error=None, device='cpu')
    cores = CorePlan(())
    teacher = TeacherCollector(worker,core,[str(args.bootstrap)],service,count_calibration(core,teachers[0]),
                               lanes=args.teacher_lanes,model=model,history=[dict(generation=10,model_id=model_id(model))],
                               inference=inference,core_plan=cores)
    table_cache = TableCache(output,output/'teacher',teachers)
    use_candidate = nullcontext() if candidate else old_teacher_check(lab_teacher_finalization)
    accepted = []
    begin = time.perf_counter()
    with scope(dict(collector_validation='production_fast',replay_bindings=True,
                    native_inference=cpu_block(workers=args.cpu_slots))), use_candidate, \
            reader_pool(args.readers,admission=gate) as readers:
        def admit(tables):
            accepted.extend(teacher.verifier.take(result,job,model)[0]
                            for result,job in zip(tables,teachers,strict=True))
        run = TeacherRun(teacher,cores,lambda:teacher.collect(teachers,on_result=table_cache.publish),admit)
        with StreamingCollection(readers,args.readers,learner,kinds,options.get('insolvency'),
                                 reward_spec(options)) as stream:
            try:
                with NativePool(ROOT,service,execution,build,options,signature,cpu=runtime,core_plan=cores) as ordinary:
                    kwargs = dict(ingress=stream) if candidate else dict(on_records=stream.submit)
                    records = ordinary.collect(jobs,**kwargs)
                    ordinary_done = time.perf_counter()
                run.finish()
                ingress = {}
                games, data, diagnostics = stream.finish(records,ingress)
            except BaseException:
                gate.abort()
                run.abort()
                raise
    elapsed = time.perf_counter() - begin
    result = dict(candidate=candidate, seconds=elapsed, ordinary_seconds=ordinary_done-begin,
                  games=len(games), rows=len(data), data_sha256=prepare_digest(data), teacher_games=len(accepted),
                  teacher_rows=sum(map(len,accepted)), teacher_labels=digest([
                      [row.label for row in table] for table in accepted]), ingress=ingress, diagnostics=digest(diagnostics),
                  teacher_finalization=teacher.finalization, ordinary=ordinary.last_collection_report,
                  binaries=[pin(collector),pin(worker),pin(args.bootstrap)])
    (output/'result.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(output.name,elapsed,len(data),result['teacher_rows'],flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ('before','after','bootstrap','runtime','out'):
        parser.add_argument('--'+arg,type=Path,required=True)
    for key, default in (('ordinary-games',12),('teacher-games',4),('ordinary-lanes',3),('teacher-lanes',2),
                         ('cpu-slots',8),('readers',2),('full',64),('fast',24),('repetitions',2)):
        parser.add_argument('--'+key,type=int,default=default)
    args=parser.parse_args()
    for key in ('before','after','bootstrap','runtime','out'):
        setattr(args,key,getattr(args,key).resolve())
    if not 0 < args.teacher_lanes <= args.teacher_games <= 32 or min(
            args.ordinary_games,args.ordinary_lanes,args.cpu_slots,args.readers,args.repetitions) <= 0:
        parser.error('Use positive job, lane and repetition counts, with at most 32 fixture tables')
    args.out.mkdir(parents=True,exist_ok=False)
    import torch
    torch.manual_seed(890)
    torch.set_num_threads(1)
    from brass_ai.learning.batching import model_id
    from brass_ai.rust_backend.lab_contract import pin
    from brass_ai.rust_backend.lab_cpu_models import ensure
    from brass_ai.recovery.collector_runtime import CPU_CONTRACT
    from test_cpu_inference import headed_model, anchor_model
    model, anchor = headed_model(), anchor_model()
    models = {model_id(model):model,model_id(anchor):anchor}
    bundles, _ = ensure(args.out,models,models)
    library = pin(args.runtime)
    runtime = dict(contract=CPU_CONTRACT,runtime=dict(path=library['path'],sha256=library['sha256']),
                   bundles=bundles,exported_bundles=0,model_cache=4,pinning='none')
    options, signature, jobs, teachers = plan(args.ordinary_games,args.teacher_games,model,anchor)
    (args.out/'jobs.json').write_text(json.dumps(dict(ordinary=jobs,teacher=teachers),indent=2),encoding='utf-8')
    common = model, bundles, runtime, options, signature, jobs, teachers
    (args.out/'protocol.json').write_text(json.dumps(dict(
        settings={k:(str(v) if isinstance(v,Path) else v) for k,v in vars(args).items()},
        jobs=pin(args.out/'jobs.json'), bundles=bundles, runtime=runtime['runtime'],
        fixture_weights='Deterministically initialized test learner and historical anchor',
        timing='Game dispatch through teacher table publication, ordinary replay ingress, and worker retirement'),indent=2),
        encoding='utf-8')
    for repeat in range(args.repetitions):
        results={}
        for candidate in (False,True) if repeat % 2 == 0 else (True,False):
            name = 'after' if candidate else 'before'
            results[candidate]=run_arm(args,args.out/f'{repeat}-{name}',getattr(args,name),common,candidate=candidate)
        for key in ('games','rows','data_sha256','teacher_games','teacher_rows','teacher_labels','diagnostics'):
            assert results[False][key]==results[True][key], (key,results[False][key],results[True][key])


if __name__=='__main__':
    main()
