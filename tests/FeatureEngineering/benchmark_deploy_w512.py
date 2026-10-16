"""Isolated W448x6/W512x8 CPU search comparison on identical complex roots.

Initialized weights only. Production architectures/configuration are untouched.
The 12-second turn clock is shared across workers, actions and continuations.
"""
import argparse
from contextlib import ExitStack
import copy
from dataclasses import replace
import json
from pathlib import Path
import platform
import shutil
import time

import numpy as np
import torch
from torch import nn

from benchmark_deploy_capacity import (BASE_INPUTS, BenchmarkService, Named,
    collate, compare, context, context_vector, encoder_probe, export_model,
    run_turn, sha, summary, write)
from benchmark_deploy_transport import Probe, active_session, cleanup
from benchmark_deploy_w448 import models as previous_models, PINNED
from brass_ai.learning.contracts import SOURCE_ROOT, require
from brass_ai.learning.model import ResidualBlock
from brass_ai.personality.contracts import CONTEXT_DIM
from brass_ai.personality.model import PersonalityConfig
from brass_ai.recovery.encoding import PreparedEncoder
from brass_ai.searchlab.spiel import BrassSpielGame
from profile_grouped import process_memory


FIXTURES = SOURCE_ROOT/'artifacts/feature-engineering/deploy-w352-w448-12s-20261016'
SOURCES = (*PINNED, 'tests/FeatureEngineering/benchmark_deploy_w448.py',
    'tests/FeatureEngineering/benchmark_deploy_w512.py',
    'training/runtime/throughput/search/runtime.json')


def models():
    core, actor = previous_models()['w448']
    torch.manual_seed(610512)
    bigger = copy.deepcopy(core)
    bigger.config = replace(core.config, hidden=512, residual_blocks=8,
        architecture='benchmark-only/w512-eight-blocks/1')
    bigger.state_encoder.fusion = nn.Sequential(nn.Linear(748,512), nn.GELU())
    bigger.state_encoder.backbone = nn.Sequential(
        *[ResidualBlock(512) for _ in range(8)], nn.LayerNorm(512))
    bigger.policy_scorer = nn.Sequential(nn.Linear(512+128,128), nn.GELU(), nn.Linear(128,1))
    bigger.outcomes = nn.Sequential(nn.Linear(512,128), nn.GELU(), nn.Linear(128,12))
    bigger.add_utility_head()
    bigger_actor = copy.deepcopy(actor)
    bigger_actor.core = copy.deepcopy(bigger)
    bigger_actor.config = PersonalityConfig(bigger.config.to_dict())
    bigger_actor.context_projection = nn.Linear(CONTEXT_DIM,512,bias=False)
    nn.init.zeros_(bigger_actor.context_projection.weight)
    bigger_actor.goal_trunk = nn.Sequential(nn.Linear(512+CONTEXT_DIM,128),nn.GELU())
    for a,b in zip(core.state_encoder.projections.parameters(),bigger.state_encoder.projections.parameters()):
        require(torch.equal(a,b),'Shared group projection changed')
    for a,b in zip(core.action_encoder.parameters(),bigger.action_encoder.parameters()):
        require(torch.equal(a,b),'Shared action encoder changed')
    require((core.parameter_count,bigger.parameter_count,bigger_actor.parameter_count)==
        (3671409,5529393,5631210),'Parameter ledger differs')
    require(len(bigger.state_encoder.backbone)==9,'W512 must have eight residual blocks')
    for model in (core,actor,bigger,bigger_actor):
        try:model.config.validate()
        except ValueError:pass
        else:raise AssertionError('Experimental model leaked into production admission')
    return {'w448':(core.eval(),actor.eval()),'w512':(bigger.eval(),bigger_actor.eval())}


def prepare(out):
    require(not out.exists(),'Use a new isolated output directory')
    (out/'models').mkdir(parents=True)
    old=json.loads((FIXTURES/'prepared.json').read_text(encoding='utf-8'))
    require(sha(FIXTURES/'inputs.json')==old['inputs_sha256'],'Earlier fixture bytes changed')
    for filename in ('inputs.json','fixture-game.json'):
        shutil.copyfile(FIXTURES/filename,out/filename)
    rows=json.loads((out/'inputs.json').read_text(encoding='utf-8'))
    require(len(rows)==6,'Expected the same six prespecified roots')
    game=BrassSpielGame(root=SOURCE_ROOT,fast=True)
    try:
        for i,row in enumerate(rows):
            state=game.from_observation(row['view'],915000+i)
            require(state.legal==row['choices'],'Reconstructed legal choices differ')
            print(f"Fixture {row['name']}: {len(row['choices'])} root choices",flush=True)
    finally:game.close()
    report=dict(format='isolated-w448-w512-deployment/1',
        scope='Initialized weights. Throughput and coverage only; not trained-model strength.',
        platform=platform.platform(),cpu=platform.processor(),python=platform.python_version(),
        torch=str(torch.__version__),seconds_per_turn=12.,tensor_threads=2,
        leaf_batch=1,service_batch_cap='worker count',flush_ms=1.,
        source_fixture=str(FIXTURES.relative_to(SOURCE_ROOT)),inputs_sha256=sha(out/'inputs.json'),
        fixture_game_sha256=sha(out/'fixture-game.json'),
        sources={p:sha(SOURCE_ROOT/p) for p in SOURCES},models={},micro=[],encoder=encoder_probe(rows))
    model_set=models();graphs={}
    for name,pair in model_set.items():
        report['models'][name]={}
        for head,model in zip(('utility','actor'),pair):
            meta,graph=export_model(model,out/'models'/f'{name}-{head}.onnx',head=='actor')
            report['models'][name][head]=meta;graphs[(name,head)]=graph
        print(f'Prepared {name}: utility={pair[0].parameter_count}, actor={pair[1].parameter_count}; parity passed',flush=True)
        write(out/'prepared.json',report)
    encoder=PreparedEncoder()
    for row in rows:
        encoded=encoder.encode(row['view'],row['choices'])
        for batch_size in (1,2,3):
            inputs=dict(zip(BASE_INPUTS[:4],[t.numpy() for t in collate([encoded]*batch_size)]))
            inputs['remaining']=np.full((batch_size,4),.075,np.float32)
            for head_index,head in enumerate(('utility','actor')):
                feed=dict(inputs)
                if head=='actor':feed['personality']=np.stack([context_vector(context())]*batch_size).astype('float32')
                measured={name:[] for name in model_set}
                for name in model_set:
                    graph=graphs[(name,head)]
                    with torch.inference_mode():
                        actual=graph.run(feed)
                        expected=Named(model_set[name][head_index])(*(torch.from_numpy(feed[k]) for k in graph.inputs))
                        compare({k:v.numpy() for k,v in zip(graph.outputs,expected)},actual,feed['legal_mask'])
                    one=graph.run({k:v[:1] for k,v in feed.items()})
                    compare({k:v[:1] for k,v in actual.items()},one,feed['legal_mask'][:1])
                    for _ in range(5):graph.run(feed)
                for repeat in range(30):
                    for name in (('w448','w512') if repeat%2==0 else ('w512','w448')):
                        began=time.perf_counter();graphs[(name,head)].run(feed)
                        measured[name].append(1000*(time.perf_counter()-began))
                report['micro'].append(dict(case=row['name'],head=head,batch=batch_size,
                    candidates=len(row['choices']),parity=True,models={k:summary(v) for k,v in measured.items()}))
    report['memory']=process_memory();report['fixture_and_real_input_parity_passed']=True
    write(out/'prepared.json',report)
    print('Preparation finished: four exports and 72 real-input model/head/batch parity checks',flush=True)


def search(out,repeats):
    require(not (out/'search.json').exists(),'Do not overwrite completed evidence')
    prepared=json.loads((out/'prepared.json').read_text(encoding='utf-8'))
    require(sha(out/'inputs.json')==prepared['inputs_sha256'],'Fixture inputs changed')
    require(all(sha(SOURCE_ROOT/p)==h for p,h in prepared['sources'].items()),'Pinned sources changed during preparation')
    rows=json.loads((out/'inputs.json').read_text(encoding='utf-8'))
    result=dict(format='w448-w512-12s-search/1',scope=prepared['scope'],comparison=[],
        repeats=repeats,timing_scope='Worker timings overlap; never sum them as whole-turn wall time.',
        sampling='Six fixed roots, paired seeds per repeat, counterbalanced model order, 2/3 workers.')
    started=time.monotonic()
    for workers in (2,3):
        with BenchmarkService(out,prepared['models'],workers) as service:
            service.config.update(batch_size=workers,flush_ms=1.)
            game=BrassSpielGame(root=SOURCE_ROOT,fast=True)
            try:
                with ExitStack() as stack:
                    sessions={name:stack.enter_context(active_session(out,service,prepared['models'],name,workers,1,True))
                        for name in ('w448','w512')}
                    for repeat in range(repeats):
                        for i,row in enumerate(rows):
                            names=('w448','w512') if (i+workers+repeat)%2==0 else ('w512','w448')
                            for name in names:
                                require(time.monotonic()-started<900,'Bounded search matrix exceeded 15 minutes')
                                probe=Probe()
                                with probe.active():value=run_turn(sessions[name],service,game,row,916000+repeat*100+i)
                                root=value['decisions'][0]
                                value.update(model=name,workers=workers,repeat=repeat,leaf_batch=1,fast=True,
                                    profile=dict(probe.totals),first_root=dict(choices=root['choices'],
                                    visited=sum(n>0 for n in root['root_visits'].values()),completed=root['completed'],
                                    attempted=root['attempted'],interrupted=root['interrupted']))
                                result['comparison'].append(value);write(out/'search-partial.json',result)
                                m=value['metrics']
                                print(f"{len(result['comparison'])}/{24*repeats}: {name} W={workers} repeat={repeat} {row['name']}: "
                                    f"completed={m['completed']} coverage={value['first_root']['visited']}/{root['choices']} "
                                    f"labels={value['qualified_labels']} seconds={value['elapsed_seconds']:.3f}",flush=True)
                                require(value['shared_deadline'] and value['turn_actions_complete'],'Turn completion/deadline check failed')
            finally:game.close()
            require(service.error is None,'Inference service failed')
    require(len(result['comparison'])==24*repeats,'Incomplete benchmark matrix')
    result['source_unchanged']=all(sha(SOURCE_ROOT/p)==h for p,h in prepared['sources'].items())
    require(result['source_unchanged'],'Pinned sources changed during benchmark')
    result['search_matrix_seconds']=time.monotonic()-started
    write(out/'search.json',result)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--phase',choices=('prepare','search','all'),default='all')
    parser.add_argument('--repeats',type=int,default=2)
    parser.add_argument('--delete-models',action='store_true')
    args=parser.parse_args();out=args.output.resolve()
    require(out.is_relative_to(SOURCE_ROOT/'artifacts/feature-engineering'),'Use an isolated benchmark directory')
    require(1<=args.repeats<=2,'Keep the experiment bounded to one or two paired repetitions')
    torch.set_num_threads(2)
    if args.phase in ('prepare','all'):prepare(out)
    if args.phase in ('search','all'):search(out,args.repeats)
    shutil.copyfile(__file__,out/'executed-benchmark.py')
    if args.delete_models:cleanup(out)


if __name__=='__main__':main()
