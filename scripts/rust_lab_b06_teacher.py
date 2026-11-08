"""B06 fixed-root search and frozen-label overhead; no new teacher games."""
from concurrent.futures import ThreadPoolExecutor
import copy
import gzip
import json
import threading
import time

import torch

from brass_ai.learning.batching import model_id
from brass_ai.learning.checkpoint import load_checkpoint
from brass_ai.learning.contracts import digest, require
from brass_ai.learning.search_value import install
from brass_ai.recovery.inference import InferenceService
from brass_ai.rust_backend.lab_teacher import Worker, calibrate
from brass_ai.rust_backend.lab_teacher_inference import Client
from brass_ai.rust_backend.lab_teacher_finalization import TurnVerifier
from brass_ai.rust_backend.lab_teacher_overlay import CONTRACT
from brass_ai.step5.league import write_json
from rust_lab_b06_metrics import Meter, distribution


def summarize_turns(results):
    turns=[t for r in results for t in r['metrics']['turns']]
    searches=[s for r in results for s in r['metrics']['searches']]
    def total(native,probe):
        values=[r['metrics'].get(native,r.get(probe)) for r in results]
        return sum(values) if all(v is not None for v in values) else None
    gpu_ns=total('gpu_wait_ns','gpu_wait_ns')
    encoding_ns=total('encoding_ns','encoding_ns')
    return dict(turns=len(turns),labels=sum(r['metrics']['labels'] for r in results),
        completed_trajectories=sum(s['completed'] for s in searches),
        completed_rounds=sum(s['rounds'] for s in searches),effective_k=distribution([s['effective_k'] for s in searches]),
        timings={k:distribution([t[k]/1e9 for t in turns if k in t]) for k in
                 ('search_ns','native_post_ns','post_search_ns','selection_ns','labels_ns','commit_ns','verification_ipc_ns','elapsed_ns')},
        inference_requests=total('requests','inference_requests'),
        gpu_wait_seconds=gpu_ns/1e9 if gpu_ns is not None else None,
        encoding_seconds=encoding_ns/1e9 if encoding_ns is not None else None,
        input_bytes=total('input_bytes','input_bytes'))


def probe_schedule():
    # Each pair has exactly two independent one-thread sessions. Each root sees
    # both arms first/second once; there is no shared tree or quota adaptation.
    return [('policy_only',0),('all_seat_w0',0),('all_seat_w0',1),('policy_only',1)]


def probes(args,snapshot,build,model,jobs,fixtures):
    roots=[r for r in fixtures['roots'] if r['view']['context']['phase']=='action' and len(r['expected']['choices'])>1]
    require(len(roots)>=2,'Need typical and complex permitted roots')
    typical=roots[0];wide=max(roots,key=lambda r:len(r['expected']['choices']))
    if wide['id']==typical['id']:typical=next(r for r in roots if r['id']!=wide['id'])
    roots=[typical,wide];origins=[dict(state=r['expected']['state'],rail_keys=list(range(64))) for r in roots]
    variants=dict(all_seat_w0=copy.deepcopy(jobs[0]),policy_only=copy.deepcopy(jobs[0]))
    for key in ('seating','fast_search','search_value'):variants['policy_only'].pop(key)
    history=[dict(generation=1250,model_id=model_id(model))];results=[]
    with Meter(args.device) as meter, InferenceService(snapshot['execution'],args.device) as service:
        require(service.register(model)==model_id(model),'Probe model changed')
        with meter.phase('gpu_model_load'):
            with service.lock:service._model(model_id(model))
        with meter.phase('calibration'):
            calibration=calibrate(build['binary']['path'],build['core_source'],origins,variants['all_seat_w0'],service)
        write_json(args.out/'teacher-calibration.json',calibration)
        # Fixed-count zero-blend output parity is distinct from wall-clock speed.
        endpoint=service.channels(1).acquire();cancel=threading.Event()
        try:
            for i,origin in enumerate(origins):
                job=variants['policy_only'];client=Client(endpoint,job['expected'],cancel)
                old=Worker(build['binary']['path'],dict(contract=CONTRACT,op='calibrate',job=job,
                    origin=origin,core_source=build['core_source']),client,cancel).collect(job['expected'],build['core_source'],calibration=True)
                expected=next(s for s in calibration['samples'] if s['root_index']==i and not s['warmup'])
                require(old['probe']==expected['probe'],'Matched zero-blend count probe differs')
        finally:endpoint.close()
        before=copy.deepcopy(service.report())
        for arm,repetition in probe_schedule():
            print(f'B06 search: {arm}, pair {repetition+1}',flush=True)
            lease=service.channels(2);owned=[lease.acquire(),lease.acquire()];cancel=threading.Event()
            first_failure=[];failure_lock=threading.Lock()
            def one(index):
                job=variants[arm];verifier=TurnVerifier(model,history,[job],cancel,diagnostic=True)
                class MeasuredClient(Client):
                    input_bytes=0
                    def evaluate(self,meta,payload):
                        self.input_bytes+=len(payload)
                        return super().evaluate(meta,payload)
                client=MeasuredClient(owned[index],job['expected'],cancel,job.get('search_value'))
                request=dict(contract=CONTRACT,op='turn_probe',job=job,origin=origins[index],
                    core_source=build['core_source'],probe_profile='full',estimate_ns=calibration['estimate_ns'],calibration_id=digest(calibration))
                worker=Worker(build['binary']['path'],request,client,cancel)
                try:
                    result=worker.collect(job['expected'],build['core_source'],verifier=verifier,turn_probe=True)
                    result.update(arm=arm,repetition=repetition,root=roots[index]['id'],
                        legal_candidates=len(roots[index]['expected']['choices']),
                        verification=verifier.cache[job['expected']['job_id']]['timings'],python_gpu_wait_seconds=client.wait_seconds,
                        gpu_wait_ns=client.wait_seconds*1e9,inference_requests=client.requests,input_bytes=client.input_bytes)
                    return result
                except BaseException as error:
                    with failure_lock:
                        if not first_failure:first_failure.append((error,error.__traceback__))
                        cancel.set()
                    write_json(args.out/f'probe-failure-{arm}-{repetition}-{index}.json',worker.failure or dict(error='probe failed'))
                    raise
            try:
                with meter.phase('matched_search_pair'),ThreadPoolExecutor(max_workers=2) as pool:
                    pair=list(pool.map(one,range(2)))
                results.extend(pair)
                write_json(args.out/'teacher-probes.json',dict(results=results,complete=len(results)==8))
            except BaseException:
                if first_failure:
                    error,tb=first_failure[0];raise error.with_traceback(tb)
                raise
            finally:
                for endpoint in owned:endpoint.close()
        after=copy.deepcopy(service.report());require(service.error is None,'Search inference failure')
        from brass_ai.recovery.telemetry import iteration_utilization
        inference=iteration_utilization(before,after,[],sum(meter.phases['matched_search_pair']),0,snapshot['execution'])
        report=dict(passed=True,schedule=probe_schedule(),sessions=2,threads_per_session=1,k=16,
            search_seconds=4,post_seconds=1,failures=0,arms={a:summarize_turns([r for r in results if r['arm']==a]) for a in variants},
            inference=inference,measurement=meter.report(),
            scope='Eight full-profile fixed-root turns; same v1/count parity/calibration/model. Policy-only versus all-seat zero-blend export path, not complete-table schedule speedup')
    write_json(args.out/'teacher-search.json',report)
    return report


def replay_stream(model,history,job,table):
    """Replay identical recorded rows through the actual streamed verifier.

    Native handoff bounds are retained; verification has no hard deadline.
    This is an offline validator timing,
    never a claim to recollect or republish those historical turns.
    """
    from brass_ai.recovery.teacher_phase_config import TIMING
    verifier=TurnVerifier(model,history,[job],threading.Event());offset=0
    turns=table['metrics']['turns'];checks=table['metrics']['verification']
    require(len(turns)==64,'Incomplete frozen all-seat verification evidence')
    sparse=all('label_rows' in turn for turn in turns)
    require(sparse or len(checks)==64,'Missing frozen per-turn row coverage')
    by_index={check['index']:check for check in checks}
    require(len(by_index)==len(checks),'Duplicate verification samples')
    for index,turn in enumerate(turns,1):
        check=by_index.get(index)
        require(turn['index']==index and (sparse or check is not None),'Recorded turn/check index differs')
        count=turn['label_rows'] if sparse else check['rows']
        require(type(count) is int and 0<=count<=2 and (check is None or check['rows']==count),'Recorded row coverage differs')
        rows=table['rows'][offset:offset+count];offset+=count
        allowance=(500_000_000 if turn['profile']=='fast' else 1_000_000_000)-turn['native_post_ns']
        if check is not None and 'allowance_seconds' in check:
            allowance=min(int(check['allowance_seconds']*1e9),allowance)
        meta=dict(job=job['expected'],timing=TIMING,index=turn['index'],profile=turn['profile'],
            remaining_ns=allowance,native_post_ns=turn['native_post_ns'],search_ns=turn['search_ns'],rows=rows)
        verifier.verify(meta,time.monotonic())
    require(offset==len(table['rows']),'Unconsumed frozen labels')
    began=time.perf_counter();accepted,rejected=verifier.take(copy.deepcopy(table),job,model)
    return accepted,rejected,time.perf_counter()-began


def frozen_overhead(args,snapshot,model,jobs,tables):
    from brass_ai.rust_backend.lab_admission import admit_teacher_rows
    from brass_ai.recovery.teacher_buffer import TeacherBuffer
    from brass_ai.recovery.search_value_config import adopt
    from brass_ai.recovery.teacher_phase_config import buffer_identity
    from brass_ai.recovery.search_value_phase import SearchValuePhase
    from brass_ai.recovery.teacher_term import forward_policy, ACTOR_PREFIXES
    history=[dict(generation=1250,model_id=model_id(model))];comparisons=[];accepted=[]
    with Meter(args.device) as meter:
        for i,(job,table) in enumerate(zip(jobs,tables,strict=True)):
            outputs={};elapsed={};reuse=None
            for arm in (('batch','stream') if i%2==0 else ('stream','batch')):
                began=time.perf_counter()
                if arm=='batch':rows,rejected=admit_teacher_rows(table['rows'],job=job,model=model,history=history)
                else:rows,rejected,reuse=replay_stream(model,history,job,table)
                elapsed[arm]=time.perf_counter()-began
                outputs[arm]=dict(rows=digest([r.row for r in rows]),rejected=rejected)
            require(outputs['batch']==outputs['stream'],'Batch/stream/reuse admission differs')
            comparisons.append(dict(job_id=job['expected']['job_id'],rows=len(rows),seconds=elapsed,reuse_seconds=reuse,equal=True))
            accepted.append(rows)
        config=adopt(snapshot['state']['signature']['config']);config['teacher_phase']['mode']='shadow'
        config['teacher_phase']['search_value']['stage']='A'
        buffer=TeacherBuffer(args.out.parents[2],args.out/'teacher-buffer',buffer_identity(config))
        with meter.phase('policy_buffer_seal'):
            for job,rows in zip(jobs,accepted,strict=True):
                buffer.publish(1250,job['split'],rows,model_id=job['expected']['model_id'],reward_id=job['expected']['reward_id'])
        with meter.phase('policy_buffer_warm_read'):buffer.rows(1250)
        restored=TeacherBuffer(args.out.parents[2],args.out/'teacher-buffer',buffer_identity(config),receipts=buffer.pending)
        with meter.phase('policy_buffer_cold_read'):restored.rows(1250)
        model.to(args.device)
        phase=SearchValuePhase(args.out/'value-buffer',model,config['teacher_phase']['search_value'],jobs[0]['expected']['reward_id'])
        with meter.phase('value_buffer_seal'):
            # Independent sealing is timed, without a final-turn verification cap.
            phase.admit(tables,jobs,1250)
        with meter.phase('value_buffer_read'):values=phase.buffer.rows(1250)
        with meter.phase('search_value_head_one_step'):head=phase.companion.head_only(values,generation=1250,steps=1)
        rows=next(r for j,r in zip(jobs,accepted,strict=True) if j['split']=='train')[:64]
        require(rows,'Missing frozen teacher training labels')
        parameters=tuple(p for n,p in model.named_parameters() if n.startswith(ACTOR_PREFIXES) and p.requires_grad)
        model.eval()
        for _ in range(2):
            with meter.phase('teacher_policy_KL_forward_backward'):
                logits=forward_policy(model,rows).float();target=torch.zeros_like(logits)
                for i,row in enumerate(rows):target[i,:len(row.label['target'])]=target.new_tensor(row.label['target'])
                logp=logits.log_softmax(-1).masked_fill(~torch.isfinite(logits),0.)
                loss=(torch.special.xlogy(target,target)-target*logp).sum(-1).mean()
                gradients=torch.autograd.grad(loss,parameters,allow_unused=True)
                require(torch.isfinite(loss).item() and all(g is None or torch.isfinite(g).all().item() for g in gradients),'Nonfinite teacher gradients')
        report=dict(passed=True,validation=comparisons,measurement=meter.report(),head_update=head,
            policy_rows=len(rows),value_rows=len(values),new_teacher_tables=0,
            scope='Frozen g1250 labels; actual model KL gradient kernel, buffers and one isolated head step. No ordinary Adam step or production beta calibration',
            historical_all_seat=dict(reused=True,table_wall_seconds=[t['metrics']['elapsed_ns']/1e9 for t in tables],
                lane_seconds=sum(t['metrics']['elapsed_ns']/1e9 for t in tables),summary=summarize_turns(tables),
                scope='Original B04.5 two-table timing; not a fresh run or a matched g570 policy-only table control'))
    write_json(args.out/'teacher-overhead.json',report)
    return report


def run(args,snapshot,build,inputs):
    model,_=load_checkpoint(snapshot['checkpoint']['path']);model.eval();install(model)
    jobs=json.loads((inputs/'jobs.json').read_text())['jobs']
    tables=json.loads(gzip.decompress((inputs/'tables.json.gz').read_bytes()))
    require(all(j['expected']['model_id']==model_id(model) for j in jobs),'Frozen g1250 diagnostic model differs')
    require({j['split'] for j in jobs}=={'train','heldout'} and len(jobs)==len(tables)==2,'Frozen table scope differs')
    search=probes(args,snapshot,build,model,jobs,json.loads((inputs/'fixtures.json').read_text()))
    overhead=frozen_overhead(args,snapshot,model,jobs,tables)
    overhead['historical_all_seat']['inference_service']=json.loads((inputs/'all-seat-inference.json').read_text())
    overhead['historical_all_seat']['value_admission']=json.loads((inputs/'all-seat-value-admission.json').read_text())
    write_json(args.out/'teacher-overhead.json',overhead)
    return dict(passed=True,search=search,overhead=overhead)
