"""Two-owner native inference sweep. Encoder/ORT microbench, not a search test."""
import argparse
import json
from pathlib import Path
import statistics
import sys
import time
ROOT=Path(__file__).resolve().parents[1]


def benchmark(a):
    import psutil
    from brass_ai.native.protocol import NativeWorker
    from brass_ai.native.inference import NativeEvaluator
    from brass_ai.native.simulation import NativeSpielGame
    fixtures={name:json.loads((a.fixtures/(name+'.json')).read_text()) for name in ('initial','dense','complex')}
    output=[]
    with NativeWorker(a.worker,timeout=60) as w:
        process=psutil.Process(w.process.pid);baseline_threads=process.num_threads()
        es=[NativeEvaluator(w,s,a.manifest) for s in (0,1)]
        loaded=[e.loaded for e in es];loaded_threads=process.num_threads();loaded_rss=process.memory_info().rss
        games=[NativeSpielGame(worker=w,session=s) for s in (0,1)]
        states=[{n:games[s].import_state(dto) for n,dto in fixtures.items()} for s in (0,1)]
        ctx=dict(model_id=es[0].manifest['model_id'],reward_id=es[0].manifest['reward_id'],profile_id='neutral',style=[0.]*4,intent=[0.]*8,
                 remaining=[0.,.01,.05,.075],max_cost=.075,stable_descriptors=None)
        scenarios={'initial':['initial']*4,'dense':['dense']*4,'complex':['complex']*4,'mixed':['initial','complex','dense','complex']}
        thread_before={t.id:t.user_time+t.system_time for t in process.threads()};start_all=time.perf_counter()
        for scenario,names in scenarios.items():
            for cap in (1,2,4):
                for flush_us in (0,500,1000,2000):
                    raw=[];wall_start=time.perf_counter()
                    for iteration in range(a.iterations):
                        for s,e in enumerate(es):
                            req=[]
                            for i,name in enumerate(names):
                                state=states[s][name]
                                # Simulator emits the permitted actor view natively;
                                # only identity fields below cross this diagnostic boundary.
                                view={'seat':state.context['actor'],'context':state.context}
                                req.append(e.request(view,state.legal,ctx,request_id=f'{scenario}-{iteration}-{i}',state=state))
                            e.submit(req,budget_ms=4000,cap=cap,pending_limit=4,flush_us=flush_us)
                        for e in es:
                            r=e.wait();assert r['state']=='complete',r
                            raw.append(dict(session=e.session,owner=r['owner'],**r['stats']))
                    wall=time.perf_counter()-wall_start
                    overlap_pairs=sum(min(x['work_finished_ns'],y['work_finished_ns'])>max(x['work_started_ns'],y['work_started_ns']) for x,y in zip(raw[::2],raw[1::2]))
                    output.append(dict(scenario=scenario,cap=cap,flush_us=flush_us,raw=raw,wall_seconds=wall,overlap_pairs=overlap_pairs,
                        positions=sum(r['positions'] for r in raw),nn_ns=sum(r['inference_ns'] for r in raw),
                        mean_nn_ms_per_position=sum(r['inference_ns'] for r in raw)/sum(r['positions'] for r in raw)/1e6,
                        max_batch=max(r['max_batch'] for r in raw),padded_cells=sum(r['padded_cells'] for r in raw),
                        valid_cells=sum(r['valid_cells'] for r in raw)))
            print(json.dumps(dict(completed=scenario,configurations=len(output))),flush=True)
        elapsed=time.perf_counter()-start_all
        thread_cpu={t.id:t.user_time+t.system_time-thread_before.get(t.id,0) for t in process.threads()}
        result=dict(status='passed',sessions=2,threads_per_session=1,baseline_process_threads=baseline_threads,
            loaded_process_threads=loaded_threads,final_process_threads=process.num_threads(),loaded_rss_bytes=loaded_rss,
            final_rss_bytes=process.memory_info().rss,thread_cpu_seconds=thread_cpu,wall_seconds=elapsed,
            loaded=loaded,iterations=a.iterations,overlap_pairs=sum(r['overlap_pairs'] for r in output),rows=output,
            scope='Native encoder and same-process ORT under two owners; diagnostic control included in wall time. No search/trajectory/strength claim.',
            policy_selected=dict(cap=1,pending_limit=1,flush_us=0),retuning='Defer full-trajectory choice to L14')
        for game in games:game.close()
    a.report.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('rows','loaded')},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for n in ('reference','worker','manifest','fixtures','report'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--iterations',type=int,default=15);a=p.parse_args()
    sys.path[:0]=[str(a.reference.resolve()/'src')]+[str(ROOT/d) for d in ('artifacts/r2-pythondeps','artifacts/r4-pythondeps','artifacts/r0-pythonlibs','artifacts/r0-validation-deps','artifacts/benchmark-deps')]
    benchmark(a)
