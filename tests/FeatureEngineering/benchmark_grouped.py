"""Alternating pinned-v4/grouped-v5 CPU diagnostics, not release qualification.

Each arm runs in its own persistent, source-pinned Python process. Only one arm
performs measured work at a time. ORT uses the existing two-thread laptop service.
"""
import argparse
from contextlib import redirect_stdout
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import traceback


def summary(values):
    ordered=sorted(values)
    return dict(n=len(values),median_ms=statistics.median(values),
        p95_ms=ordered[int(.95*(len(ordered)-1))],minimum_ms=min(values),maximum_ms=max(values))


def worker(args):
    start=time.perf_counter();sys.path.insert(0,str(args.runtime/'src'))
    with redirect_stdout(sys.stderr):
        import numpy as np
        import torch
        from dataclasses import replace
        import random
        from brass_ai.learning.checkpoint import load_checkpoint
        from brass_ai.learning.batching import collate
        from brass_ai.learning.contracts import compatibility,digest
        from brass_ai.recovery.encoding import PreparedEncoder
        from brass_ai.learning.reward import CostLedger
        from brass_ai.personality.contracts import context
        from brass_ai.personality.search.inference import PersonalityRemote,PersonalityRequest
        from brass_ai.personality.search.contracts import Profile,search_config
        from brass_ai.personality.search.engine import SearchSession
        from brass_ai.personality.difficulty.collection import TrackedChannels
        from brass_ai.personality.laptop.inference import PortableInference
        from brass_ai.personality.laptop.benchmark import processor_name
        from brass_ai.search.common import turn_key
        from brass_ai.searchlab.spiel import BrassSpielGame,action_id
        torch.set_num_threads(2)
        rows=json.loads(args.inputs.read_text(encoding='utf-8'))
        model,_=load_checkpoint(args.checkpoint);model.eval()
        receipt=json.loads((args.graphs/'receipt.json').read_text(encoding='utf-8'))
        entries=[{k:e[k] for k in ('path','model_id','actor')} for e in receipt['models'].values()]
        actor=next(e for e in entries if e['actor']);reference=next(e for e in entries if not e['actor'])
        service=PortableInference(args.graphs,entries,'ort-cpu');service.__enter__()
        channels=TrackedChannels(service.channels(8))
        remote=PersonalityRemote(channels.acquire(),actor['model_id'])
        reward=service.graphs[reference['model_id']].meta['config']['utility']
        config=replace(search_config(reward,seconds=2.,workers=2),cpu_role='laptop').validate()
        session=SearchSession(args.root,service,channels,actor['model_id'],[reference['model_id']],
            config,[Profile(alpha=0.)]*4,dict(bias_ready=False,temperature=1.,scope='Timing only; no goal qualification'))
        game=BrassSpielGame(root=args.root,fast=True)
        encoder=PreparedEncoder()
        encoded=[encoder.encode(r['view'],r['choices']) for r in rows]
        encoded_hash=hashlib.sha256(b''.join(e.state.tobytes()+e.actions.tobytes()+e.stable_to_relative.tobytes()
            for e in encoded)).hexdigest()
        pools=dict(small=[i for i,r in enumerate(rows) if len(r['choices'])<=6],
                   large=[i for i,r in enumerate(rows) if len(r['choices'])>=300],
                   rail_ordinary=[i for i,r in enumerate(rows) if r['view']['context']['era']=='RailEra'
                                  and r['view']['context']['phase']=='action'])
        batches=[]
        for size in (1,2,8):
            for name,indices in pools.items():
                selected=[indices[i%len(indices)] for i in range(size)]
                batch=collate([encoded[i] for i in selected]);cap=reward['insolvency']['max_cost']
                inputs=dict(zip(('states','actions','legal_mask','stable_to_relative'),[t.numpy() for t in batch]))
                inputs['remaining']=np.full((size,4),cap,np.float32)
                batches.append((name,size,selected,batch,inputs))
    ready=dict(status='ready',cold_process_seconds=time.perf_counter()-start,processor=processor_name(),
        compatibility=compatibility(),config=model.config.to_dict(),parameters=model.parameter_count,
        encoded_sha256=encoded_hash,backend='ort-cpu',threads=2,workers=2)
    print(json.dumps(ready),flush=True)

    def timed(call,repeat=7,warm=2):
        for _ in range(warm):call()
        times=[]
        for _ in range(repeat):
            began=time.perf_counter();call();times.append((time.perf_counter()-began)*1000)
        return summary(times)

    def micro():
        measurements={}
        for mode in ('cold_encoder','persistent_miss','warm_hit'):
            enc=PreparedEncoder(cache_size=256 if mode=='warm_hit' else 0)
            for row in rows:enc.encode(row['view'],row['choices'])
            samples=[]
            for _ in range(3):
                for row in rows:
                    began=time.perf_counter()
                    selected=PreparedEncoder(cache_size=0) if mode=='cold_encoder' else enc
                    selected.encode(row['view'],row['choices'])
                    samples.append((time.perf_counter()-began)*1000)
            measurements[mode]=summary(samples)
        cases=[];graph=service.graphs[reference['model_id']]
        for name,size,selected,batch,inputs in batches:
            remaining=torch.from_numpy(inputs['remaining'])
            with torch.inference_mode():
                torch_result=timed(lambda:model(*batch,remaining=remaining))
            ort_result=timed(lambda:graph.run(inputs))
            def request():
                requests=[PersonalityRequest(f'bench-{i}',actor['model_id'],rows[index]['view'],rows[index]['choices'],
                    time.monotonic()+60,personality=context(),remaining=tuple(map(float,inputs['remaining'][i])))
                    for i,index in enumerate(selected)]
                responses=remote.predict(requests)
                assert all(r['status']=='ok' for r in responses)
            remote.encoder.cache.clear();began=time.perf_counter();request();first=(time.perf_counter()-began)*1000
            cases.append(dict(size_class=name,batch=size,candidates=inputs['actions'].shape[1],
                torch_forward=torch_result,ort_forward=ort_result,
                complete_first_request_ms=first,complete_warm_request=timed(request,repeat=5)))
        return dict(encoding=measurements,cases=cases,inference=service.report())

    def whole_turn(command):
        seconds=command['seconds'];index=command['case'];seed=command['seed']
        row=rows[index];view=row['view'];seat=view['seat'];key=turn_key(view['context'])
        session.finish_turn();session.config=replace(config,seconds=seconds)
        for w in session.workers:w.config=session.config
        session.begin(seed,seed)
        state=game.from_observation(view,seed);assert state.legal==row['choices']
        ledger=CostLedger(reward);ledger.observe(view);rng=random.Random(seed)
        before=dict(session.telemetry);start=time.monotonic();qualified=0;finished=0;deadlines=[]
        result_status='turn_complete'
        for _ in range(256):
            if state.is_terminal():break
            if state.is_chance_node():result_status='chance_boundary';break
            if state.current_player()!=seat or turn_key(state.context)!=key:break
            live=state.views[seat];choices=state.legal
            chosen,label,info=session.select(live,choices,rng,ledger)
            if session.turn:deadlines.append(session.turn['deadline'])
            assert 0<=chosen<len(choices)
            qualified+=int(label is not None);choice=choices[chosen]
            ledger.chosen(live,choice);state.apply_action(action_id(choice['id']))
            finished+=int(choice['kind']=='finish')
            if not state.is_terminal() and not state.is_chance_node():ledger.observe(state.views[seat])
        else:raise ValueError('Whole-turn benchmark exceeded its bounded continuation cap')
        elapsed=time.monotonic()-start
        delta={k:session.telemetry[k]-before[k] for k in before}
        assert len(set(deadlines))<=1,'A second action or subchoice reset the shared clock'
        session.finish_turn()
        return dict(case=index,era=view['context']['era'],root_candidates=len(row['choices']),
            seconds=seconds,elapsed_seconds=elapsed,metrics=delta,qualified_labels=qualified,
            completed_actions=finished,turn_actions_complete=finished>=view['context']['actions_remaining'],
            status=result_status,shared_deadline=True,seed=seed,
            completed_simulations_per_second=delta['completed']/max(elapsed,1e-9),
            search_config=session.config.to_dict())

    try:
        for line in sys.stdin:
            command=json.loads(line)
            if command['operation']=='close':break
            try:
                with redirect_stdout(sys.stderr):
                    result=micro() if command['operation']=='micro' else whole_turn(command)
                print(json.dumps(dict(ok=True,result=result)),flush=True)
            except Exception as error:
                traceback.print_exc(file=sys.stderr)
                print(json.dumps(dict(ok=False,error=f'{type(error).__name__}: {error}')),flush=True)
    finally:
        session.close();game.close();channels.close();service.__exit__()


def orchestrate(args):
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=False)
    children={};logs={};report=dict(format='grouped-projection-benchmark/1.0.0',cold={},micro=[],turns=[])
    def send(label,command):
        child=children[label];child.stdin.write(json.dumps(command)+'\n');child.stdin.flush()
        line=child.stdout.readline()
        if not line:raise RuntimeError(f'{label} worker exited; inspect its log')
        reply=json.loads(line)
        if not reply['ok']:raise RuntimeError(f'{label}: {reply["error"]}')
        return reply['result']
    try:
        for label,runtime,graphs,checkpoint in (
            ('baseline',args.baseline_runtime,args.baseline_graphs,args.baseline_checkpoint),
            ('grouped',args.root,args.grouped_graphs,args.grouped_checkpoint)):
            logs[label]=(output/(label+'.log')).open('w',encoding='utf-8')
            env=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
            command=[sys.executable,str(Path(__file__).resolve()),'--worker','--runtime',str(runtime.resolve()),
                '--graphs',str(graphs.resolve()),'--checkpoint',str(checkpoint.resolve()),
                '--root',str(args.root.resolve()),'--inputs',str(args.inputs.resolve())]
            began=time.monotonic();child=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=subprocess.PIPE,
                stderr=logs[label],text=True,encoding='utf-8',env=env)
            children[label]=child;line=child.stdout.readline()
            if not line:raise RuntimeError(f'{label} did not start; inspect its log')
            ready=json.loads(line);assert ready['status']=='ready';ready['launch_seconds']=time.monotonic()-began
            report['cold'][label]=ready
        assert report['cold']['baseline']['encoded_sha256']==report['cold']['grouped']['encoded_sha256']
        assert report['cold']['baseline']['config']['architecture']=='brass-four-seat-mlp-resnet/4.0.0'
        assert report['cold']['baseline']['config']['residual_blocks']==5
        assert report['cold']['grouped']['config']['residual_blocks']==6
        for block in range(4):
            for label in (('baseline','grouped') if block%2==0 else ('grouped','baseline')):
                report['micro'].append(dict(block=block,arm=label,**send(label,dict(operation='micro'))))
            print(f'Paired micro block {block+1}/4',flush=True)
        # Both eras, an ordinary two-action turn, and the existing full difficulty caps.
        cases=(13,7)
        for repeat in range(2):
            for case in cases:
                for tier,seconds in enumerate((2.,5.,12.)):
                    order=('baseline','grouped') if (repeat+case+tier)%2==0 else ('grouped','baseline')
                    seed=57731+repeat*101+case
                    for label in order:
                        result=send(label,dict(operation='turn',case=case,seconds=seconds,seed=seed))
                        report['turns'].append(dict(arm=label,repeat=repeat,**result))
                    (output/'partial.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
                    print(f'Paired turn: repeat={repeat+1}/2 case={case} budget={seconds}s',flush=True)
        report.update(inputs_sha256=hashlib.sha256(args.inputs.read_bytes()).hexdigest(),backend='ort-cpu',
            tensor_threads=2,search_workers=2,
            scope='Local CPU diagnostics, alternating AB/BA arms. Cold is one fresh process per arm with OS cache uncontrolled. '
                  'Four micro blocks; two whole-turn repeats per era/budget. Different trained architecture weights and an '
                  'untrained alpha-zero goal branch; root observations and sampling seeds match. Search work can depend '
                  'on resulting policy/tree paths. Not a strength comparison, release qualification or a causal speed proof.')
        (output/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    finally:
        for label,child in children.items():
            if child.poll() is None:
                try:child.stdin.write('{"operation":"close"}\n');child.stdin.flush();child.wait(timeout=30)
                except Exception:child.terminate();child.wait(timeout=10)
            logs[label].close()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--worker',action='store_true');p.add_argument('--runtime',type=Path)
    p.add_argument('--graphs',type=Path);p.add_argument('--checkpoint',type=Path)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--inputs',type=Path,required=True)
    p.add_argument('--output',type=Path)
    for name in ('baseline-runtime','baseline-graphs','baseline-checkpoint','grouped-graphs','grouped-checkpoint'):
        p.add_argument('--'+name,type=Path)
    args=p.parse_args()
    if args.worker:worker(args)
    else:orchestrate(args)


if __name__=='__main__':main()
