"""Isolated initialized-model CPU deployment experiment; never a training package.

Uses the production named ONNX inference queue and complete-turn SearchSession.
The W352 model exists only here: production checkpoint/config admission is not
relaxed. Search clocks are always shared across workers and both turn actions.
"""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack,contextmanager,nullcontext
import copy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import platform
import random
import statistics
import time
from unittest.mock import Mock,patch

import numpy as np
import torch
from torch import nn

from brass_ai.learning.batching import collate
from brass_ai.learning.contracts import SOURCE_ROOT,digest,require
from brass_ai.learning.features import Encoder
from brass_ai.learning.model import ModelConfig,GroupedStateEncoder,dense_multiply_adds
from brass_ai.learning.reward import CostLedger,specification
from brass_ai.learning.utility_model import UtilityNet,utility_config
from brass_ai.personality.contracts import CONTEXT_DIM,context,context_vector
from brass_ai.personality.model import PersonalityConfig,import_parent
from brass_ai.personality.laptop.export import Named,Graph,BASE_INPUTS,BASE_OUTPUTS,GOAL_OUTPUTS,compare,samples
from brass_ai.personality.laptop.inference import PortableInference,execution
from brass_ai.personality.search.contracts import Profile,search_config
from brass_ai.personality.search.engine import SearchSession
from brass_ai.personality.difficulty.collection import TrackedChannels
from brass_ai.recovery.encoding import PreparedEncoder
from brass_ai.recovery.inference import InferenceService,Endpoint,RemoteEvaluator,share_encoder_cache
from brass_ai.search.common import turn_key
from brass_ai.searchlab.spiel import BrassSpielGame,action_id
from fe_fixtures import benchmark_positions
from profile_grouped import process_memory


WIDTHS=(32,64,48,256,288,48)
SEED=610352


def write(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,indent=2),encoding='utf-8')


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def summary(values):
    ordered=sorted(values)
    return dict(n=len(values),median_ms=statistics.median(values),
        p95_ms=ordered[int(.95*(len(values)-1))],samples_ms=values)


def build_models():
    """Reuse current math, but keep the unsupported wider config benchmark-only."""
    torch.manual_seed(SEED)
    v9=UtilityNet(utility_config(ModelConfig.preset(),specification())).eval()
    actor,_=import_parent(v9);actor.eval()
    # Keep the unchanged action encoder and head widths, initialize wider layers
    # normally. These are two untrained networks, not a weight migration.
    torch.manual_seed(SEED+1)
    bigger=copy.deepcopy(v9)
    bigger.config=replace(v9.config,hidden=352,projection_widths=WIDTHS,
        architecture='benchmark-only/w352-six-blocks/1')
    bigger.state_encoder=GroupedStateEncoder(bigger.config)
    bigger.policy_scorer=nn.Sequential(nn.Linear(352+128,128),nn.GELU(),nn.Linear(128,1))
    bigger.outcomes=nn.Sequential(nn.Linear(352,128),nn.GELU(),nn.Linear(128,12))
    bigger.add_utility_head();bigger.eval()
    big_actor=copy.deepcopy(actor);big_actor.core=copy.deepcopy(bigger)
    big_actor.config=PersonalityConfig(bigger.config.to_dict())
    big_actor.context_projection=nn.Linear(CONTEXT_DIM,352,bias=False)
    nn.init.zeros_(big_actor.context_projection.weight)
    big_actor.goal_trunk=nn.Sequential(nn.Linear(352+CONTEXT_DIM,128),nn.GELU());big_actor.eval()
    require(v9.parameter_count==2015937 and bigger.parameter_count==2647761,'Architecture parameter count drift')
    require(bigger.state_encoder.fusion[0].in_features==748 and len(bigger.state_encoder.backbone)==7,'Wrong W352 layout')
    # Normal checkpoint/export validation MUST continue to reject the experiment.
    try:bigger.config.validate()
    except ValueError:pass
    else:raise AssertionError('Experimental architecture leaked into production admission')
    return {'v9':(v9,actor),'w352':(bigger,big_actor)}


class BenchmarkGraph(Graph):
    """Explicit test-only loader; never emits a production deploy receipt."""
    def __init__(self,path,meta,threads=2):
        import onnxruntime as ort
        require(meta['format']=='isolated-capacity-graph/1' and sha(path)==meta['sha256'],'Benchmark graph changed')
        self.path=path;self.meta=meta;self.backend='ort-cpu';self.inputs=meta['inputs'];self.outputs=meta['outputs']
        self.calls=0;self.seconds=0.;self.shapes=Counter();self.version=ort.__version__
        self.hardware='CPUExecutionProvider'
        options=ort.SessionOptions();options.intra_op_num_threads=threads;options.inter_op_num_threads=1
        options.execution_mode=ort.ExecutionMode.ORT_SEQUENTIAL
        options.add_session_config_entry('session.intra_op.allow_spinning','0')
        self.engine=ort.InferenceSession(str(path),options,providers=['CPUExecutionProvider'])
        require([x.name for x in self.engine.get_inputs()]==self.inputs,'Graph input order differs')
        require([x.name for x in self.engine.get_outputs()]==self.outputs,'Graph output order differs')

    def run(self,inputs):
        self.shapes[(len(inputs['states']),inputs['actions'].shape[1])]+=1
        return super().run(inputs)


def export_model(model,path,actor):
    config=(model.core.config if actor else model.config).to_dict()
    inputs=BASE_INPUTS+(['personality'] if actor else [])
    outputs=BASE_OUTPUTS+(GOAL_OUTPUTS if actor else [])
    batches=samples(config,actor);wrapper=Named(model).eval()
    dynamic={name:{0:'batch'} for name in inputs+outputs}
    for name in ('actions','legal_mask','policy_logits'):dynamic[name][1]='candidates'
    with torch.inference_mode():
        torch.onnx.export(wrapper,tuple(torch.from_numpy(batches[1][k]) for k in inputs),str(path),
            input_names=inputs,output_names=outputs,dynamic_axes=dynamic,opset_version=17,dynamo=False)
    meta=dict(format='isolated-capacity-graph/1',config=config,actor=actor,inputs=inputs,outputs=outputs,
        sha256=sha(path),bytes=path.stat().st_size,parameters=model.parameter_count,
        model_id=digest(dict(config=config,actor=actor,sha256=sha(path))),macs=dense_multiply_adds(model))
    graph=BenchmarkGraph(path,meta);errors={k:0. for k in outputs}
    invariance={k:0. for k in outputs}
    for batch in batches:
        with torch.inference_mode():values=wrapper(*(torch.from_numpy(batch[k]) for k in inputs))
        expected={k:v.numpy() for k,v in zip(outputs,values)}
        actual=graph.run(batch)
        for k,v in compare(expected,actual,batch['legal_mask']).items():errors[k]=max(errors[k],v)
        # Same row alone vs in a differently sized batch: only roundoff allowed.
        for i in range(len(batch['states'])):
            one=graph.run({k:v[i:i+1] for k,v in batch.items()})
            for k,v in compare({k:v[i:i+1] for k,v in actual.items()},one,batch['legal_mask'][i:i+1]).items():
                invariance[k]=max(invariance[k],v)
    meta['parity']=dict(passed=True,torch_ort_max_abs=errors,batch_invariance_max_abs=invariance)
    return meta,graph


def encoder_probe(rows):
    oracle=Encoder();expected=[oracle.encode(r['view'],r['choices'],[0.]*4,[0.]*8) for r in rows]
    def verify(got,want):
        require(got.choice_ids==want.choice_ids,'Choice IDs changed')
        for field in ('state','actions','stable_to_relative'):
            np.testing.assert_array_equal(getattr(got,field),getattr(want,field))
            require(not getattr(got,field).flags.writeable,'Mutable cached features')
    modes={k:[] for k in ('new_encoder','persistent_miss','warm_hit')}
    roots={}
    for block in range(3):
        for mode in (list(modes) if block%2==0 else list(reversed(modes))):
            persistent=PreparedEncoder(cache_size=256 if mode=='warm_hit' else 0)
            for row in rows:persistent.encode(row['view'],row['choices'],[0.]*4,[0.]*8)
            for row,want in zip(rows,expected):
                began=time.perf_counter();encoder=PreparedEncoder(cache_size=0) if mode=='new_encoder' else persistent
                got=encoder.encode(row['view'],row['choices'],[0.]*4,[0.]*8)
                modes[mode].append(1000*(time.perf_counter()-began));verify(got,want)
        for workers in (2,3):
            for shared in ((False,True) if block%2==0 else (True,False)):
                key=f'{workers}-workers/'+('shared' if shared else 'separate')
                record=roots.setdefault(key,dict(samples_ms=[],cache_counts=[]))
                endpoints=[Endpoint(Mock(),Mock(),i) for i in range(1+2*workers)]
                if shared:share_encoder_cache(endpoints)
                clients=[RemoteEvaluator(e,'a' if i%2 else 'b') for i,e in enumerate(endpoints)]
                with ThreadPoolExecutor(max_workers=workers) as pool:
                    list(pool.map(lambda i:i,range(workers)))
                    for row,want in zip(rows,expected):
                        def encode(client):return client.encoder.encode(row['view'],row['choices'],[0.]*4,[0.]*8)
                        began=time.perf_counter();values=[encode(clients[0])]
                        values+=list(pool.map(encode,clients[1::2]))
                        record['samples_ms'].append(1000*(time.perf_counter()-began))
                        for got in values:verify(got,want)
                caches={id(c.encoder.cache):c.encoder.cache for c in clients}
                record['cache_counts'].append({k:sum(c.report()[k] for c in caches.values()) for k in ('warm_hit','miss','capacity','retained')})
                for endpoint in endpoints:endpoint.close()
    return dict(exact_feature_parity=True,micro={k:summary(v) for k,v in modes.items()},
        coordinator_workers={k:dict(latency=summary(v['samples_ms']),cache_counts=v['cache_counts']) for k,v in roots.items()})


def prepare(out):
    require(not out.exists(),'Use a fresh isolated benchmark output directory')
    out.mkdir(parents=True);(out/'models').mkdir()
    rows=benchmark_positions();write(out/'inputs.json',rows)
    encoded=[PreparedEncoder().encode(r['view'],r['choices']) for r in rows]
    report=dict(format='deployment-capacity-benchmark/1',seed=SEED,seconds_per_turn=12.,
        scope='Initialized weights only. Timing/coverage, not playing strength. No production checkpoint or config is changed.',
        cpu=platform.processor(),platform=platform.platform(),python=platform.python_version(),torch=torch.__version__,
        tensor_threads=2,inputs_sha256=sha(out/'inputs.json'),models={},micro=[],encoder=encoder_probe(rows))
    models=build_models()
    for name,(core,actor) in models.items():
        graph_set={};metadata={}
        for head,model in (('utility',core),('actor',actor)):
            meta,graph=export_model(model,out/'models'/f'{name}-{head}.onnx',head=='actor')
            metadata[head]=meta;graph_set[head]=graph
        report['models'][name]=metadata
        for count in (1,2,3,4,6,8,16):
            for case in ('rail-action','settled-next-round'):
                idx=next(i for i,r in enumerate(rows) if r['name']==case)
                batch=dict(zip(BASE_INPUTS[:4],[t.numpy() for t in collate([encoded[idx]]*count)]))
                batch['remaining']=np.full((count,4),.075,np.float32)
                for head,model in (('utility',core),('actor',actor)):
                    inputs=dict(batch)
                    if head=='actor':inputs['personality']=np.stack([context_vector(context())]*count).astype('float32')
                    graph=graph_set[head];wrapper=Named(model);tensors=tuple(torch.from_numpy(inputs[k]) for k in graph.inputs)
                    calls=dict(ort=lambda:graph.run(inputs),torch=lambda:wrapper(*tensors))
                    measured={k:[] for k in calls}
                    with torch.inference_mode():
                        for call in calls.values():
                            for _ in range(3):call()
                        for block in range(3):
                            for backend in (list(calls) if block%2==0 else list(reversed(calls))):
                                for _ in range(7):
                                    began=time.perf_counter();calls[backend]()
                                    measured[backend].append(1000*(time.perf_counter()-began))
                    report['micro'].append(dict(model=name,head=head,batch=count,candidates=len(rows[idx]['choices']),
                        case=case,**{k:summary(v) for k,v in measured.items()}))
        print(f'Prepared {name}: utility={core.parameter_count}, actor={actor.parameter_count}',flush=True)
        write(out/'prepared.json',report)
    report['memory']=process_memory();write(out/'prepared.json',report)


class BenchmarkService(PortableInference):
    def __init__(self,out,metadata,workers):
        config=dict(execution(),cpu_budget=workers+2,cpu_reserve=2)
        InferenceService.__init__(self,config,'cpu')
        self.graphs={};self.parity=[]
        for name,heads in metadata.items():
            for head,meta in heads.items():
                self.graphs[meta['model_id']]=BenchmarkGraph(out/'models'/f'{name}-{head}.onnx',meta)


def cache_snapshot(session):
    clients=[session.base.evaluator]
    for w in session.workers:clients.extend([w.base.evaluator,*[p.evaluator for p in w.rollouts]])
    caches={id(c.encoder.cache):c.encoder.cache for c in clients}
    return {k:sum(c.report()[k] for c in caches.values()) for k in ('warm_hit','miss','evictions','capacity','retained')}


def run_turn(session,service,game,row,seed):
    session.finish_turn();session.begin(seed,seed)
    clients=[session.base.evaluator]
    for w in session.workers:clients.extend([w.base.evaluator,*[p.evaluator for p in w.rollouts]])
    # Repeated fixture seeds must not give later batch/model conditions free
    # cross-turn full-feature hits. Within-turn reuse remains fully enabled.
    for cache in {id(c.encoder.cache):c.encoder.cache for c in clients}.values():cache.clear()
    view=row['view'];seat=view['seat'];key=turn_key(view['context'])
    state=game.from_observation(view,seed);require(state.legal==row['choices'],'Reconstructed legality differs')
    ledger=CostLedger(specification());ledger.observe(view);rng=random.Random(seed)
    before=dict(session.telemetry);inference=dict(service.metrics);cache=cache_snapshot(session)
    shapes={k:Counter(g.shapes) for k,g in service.graphs.items()}
    phases=copy.deepcopy(service.timings.report())
    began=time.monotonic();qualified=0;finished=0;deadlines=[];decisions=[];status='turn_complete'
    for _ in range(256):
        if state.is_terminal():break
        if state.is_chance_node():status='chance_boundary';break
        if state.current_player()!=seat or turn_key(state.context)!=key:break
        live=state.views[seat];choices=state.legal
        for w in session.workers:w.last_statistics=None
        chosen,label,detail=session.select(live,choices,rng,ledger)
        if session.turn:deadlines.append(session.turn['deadline'])
        require(0<=chosen<len(choices),'Illegal search choice')
        qualified+=int(label is not None);choice=choices[chosen]
        stats=[w.last_statistics for w in session.workers if w.last_statistics] if detail.get('searched') else []
        decisions.append(dict(phase=live['context']['phase'],choices=len(choices),selected=choice,
            qualified=label is not None,detail=detail,
            attempted=sum(s['batching'].get('attempted',0) for s in stats),
            completed=sum(s['completed'] for s in stats),interrupted=sum(s['interrupted'] for s in stats),
            root_visits={cid:sum(s['root'].get(cid,{}).get('visits',0) for s in stats) for cid in
                sorted({cid for s in stats for cid in s['root']})},
            batching=[s['batching'] for s in stats]))
        ledger.chosen(live,choice);state.apply_action(action_id(choice['id']));finished+=int(choice['kind']=='finish')
        if not state.is_terminal() and not state.is_chance_node():ledger.observe(state.views[seat])
    else:raise ValueError('Whole-turn continuation cap exceeded')
    elapsed=time.monotonic()-began
    require(len(set(deadlines))<=1,'Turn deadline reset between actions or continuations')
    final_cache=cache_snapshot(session);hist=Counter()
    for ident,graph in service.graphs.items():
        for shape,n in (graph.shapes-shapes[ident]).items():hist[shape[0]]+=n
    final_phases=service.timings.report()
    result=dict(case=row['name'],seed=seed,elapsed_seconds=elapsed,budget_seconds=12.,
        cache_start='Full-feature LRU empty for each turn; persistent builders/action templates warm.',
        metrics={k:session.telemetry[k]-v for k,v in before.items()},
        inference={k:service.metrics[k]-v for k,v in inference.items() if not k.startswith('max_')},
        timing_seconds={k:v['seconds']-phases.get(k,{}).get('seconds',0.) for k,v in final_phases.items()},
        cache={k:final_cache[k]-cache[k] if k in ('warm_hit','miss','evictions') else final_cache[k] for k in final_cache},
        batch_histogram=dict(sorted(hist.items())),decisions=decisions,qualified_labels=qualified,
        insufficient_coverage=sum(d['detail'].get('reason')=='insufficient_coverage' for d in decisions),
        completed_actions=finished,turn_actions_complete=finished>=view['context']['actions_remaining'],
        shared_deadline=True,status=status,config=session.config.to_dict())
    session.finish_turn();return result


def search(out):
    require(not (out/'search.json').exists(),'Do not overwrite a completed search experiment')
    prepared=json.loads((out/'prepared.json').read_text(encoding='utf-8'))
    require(sha(out/'inputs.json')==prepared['inputs_sha256'],'Fixture corpus changed')
    rows=json.loads((out/'inputs.json').read_text(encoding='utf-8'))
    cases=[next(r for r in rows if r['name']==case) for case in ('settled-next-round','rail-action')]
    report=dict(format='deployment-capacity-search/1',pilot=[],comparison=[],cache_ablation=[],selected={})
    def record(kind,value):
        report[kind].append(value);write(out/'search-partial.json',report)
        m=value['metrics']
        print(f"{kind}: {value['model']} W={value['workers']} batch={value['leaf_batch']} {value['case']} "
            f"completed={m['completed']} interrupted={m['interrupted']} qualified={value['qualified_labels']} "
            f"elapsed={value['elapsed_seconds']:.3f}s cache={value['cache']['warm_hit']}/{value['cache']['miss']}",flush=True)
    # Each worker count gets its own admission capacity; no idle third worker in
    # the two-worker condition. Native host construction is outside turn clocks.
    for workers in (2,3):
        with BenchmarkService(out,prepared['models'],workers) as service:
            game=BrassSpielGame(root=SOURCE_ROOT,fast=True)
            try:
                @contextmanager
                def active(name,batch,shared=True):
                    meta=prepared['models'][name]
                    channels=TrackedChannels(service.channels(1+2*workers))
                    config=replace(search_config(specification(),seconds=12.,workers=workers),cpu_role='laptop',
                        pending_paths=batch,batch_size=batch).validate()
                    session=None
                    try:
                        with nullcontext() if shared else patch('brass_ai.personality.search.engine.share_encoder_cache',return_value=None):
                            session=SearchSession(SOURCE_ROOT,service,channels,meta['actor']['model_id'],[meta['utility']['model_id']],
                                config,[Profile(alpha=0.)]*4,dict(bias_ready=False,temperature=1.,scope='Initialized timing only'))
                        yield session
                    finally:
                        if session is not None:session.close()
                        channels.close()
                with active('v9',4) as session:
                    for batch in ((1,2,4) if workers==2 else (4,2,1)):
                        session.config=replace(session.config,pending_paths=batch,batch_size=batch)
                        for w in session.workers:w.config=session.config
                        service.config.update(batch_size=workers*batch,flush_ms=1.)
                        for i,row in enumerate(cases):
                            record('pilot',dict(model='v9',workers=workers,leaf_batch=batch,
                                **run_turn(session,service,game,row,891000+i)))
                pilots=[r for r in report['pilot'] if r['workers']==workers]
                scores={b:(sum(r['metrics']['completed'] for r in pilots if r['leaf_batch']==b),
                    sum(r['qualified_labels'] for r in pilots if r['leaf_batch']==b),-b) for b in (1,2,4)}
                batch=max(scores,key=scores.get)
                report['selected'][str(workers)]=dict(leaf_batch=batch,service_batch_cap=workers*batch,
                    flush_ms=1.,pilot_scores={str(k):list(v) for k,v in scores.items()},
                    rule='Most completed simulations, then qualified labels, then smaller leaf batch; two pilot roots only.')
                service.config.update(batch_size=workers*batch,flush_ms=1.)
                with ExitStack() as stack:
                    sessions={name:stack.enter_context(active(name,batch)) for name in ('v9','w352')}
                    for repeat in range(2):
                        for name in (('v9','w352') if repeat==0 else ('w352','v9')):
                            for i,row in enumerate(cases):
                                record('comparison',dict(model=name,workers=workers,leaf_batch=batch,repeat=repeat,
                                    **run_turn(sessions[name],service,game,row,892000+repeat*100+i)))
                # Test-local former cache ownership, identical current math.
                with active('v9',batch,shared=False) as session:
                    for i,row in enumerate(cases):
                        record('cache_ablation',dict(model='v9',workers=workers,leaf_batch=batch,cache_mode='former_separate',
                            **run_turn(session,service,game,row,892000+i)))
            finally:game.close()
            require(service.error is None,'Inference service failed')
    write(out/'search.json',report)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--phase',choices=('all','prepare','search'),default='all')
    parser.add_argument('--delete-models',action='store_true')
    args=parser.parse_args();out=args.output.resolve()
    require(out.is_relative_to(SOURCE_ROOT/'artifacts'),'Benchmark outputs must be inside artifacts')
    torch.set_num_threads(2)
    if args.phase in ('prepare','all'):prepare(out)
    if args.phase in ('search','all'):search(out)
    if args.delete_models:
        require((out/'search.json').is_file(),'Keep failed experiment files until reviewed')
        deleted=[]
        for file in sorted((out/'models').iterdir()):
            require(file.is_file() and file.suffix=='.onnx' and file.resolve().parent==out/'models','Unexpected model output')
            deleted.append(dict(path=file.name,sha256=sha(file),bytes=file.stat().st_size));file.unlink()
        write(out/'model-cleanup.json',dict(deleted=deleted,remaining=list(p.name for p in (out/'models').iterdir())))


if __name__=='__main__':main()
