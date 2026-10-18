"""Small paired CPU NN search benchmark; never reads/writes training checkpoints.

Uses the current medium architecture at deterministic random initialization.
The same export, roots, worker seeds and search recipe are used in both modes.
"""
import argparse
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
import importlib.util
import json
from pathlib import Path
import random
import statistics
import sys
from threading import Event
import time

import numpy as np
import torch

from brass_ai.learning.model import ModelConfig
from brass_ai.learning.reward import specification
from brass_ai.learning.utility_model import UtilityNet, utility_config
from brass_ai.personality.laptop.export import export
from brass_ai.personality.laptop.inference import PortableInference
from brass_ai.recovery.inference import RemoteEvaluator, share_encoder_cache
from brass_ai.search.common import SearchConfig
from brass_ai.search.ismcts import MaskedISMCTS, MaskedSearchPolicy
from brass_ai.step5.policy import inference_deadline

ROOT=Path(__file__).resolve().parents[2]


def read(path):return json.loads(path.read_text(encoding='utf-8-sig'))
def write(path, value):path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--baseline-inference',type=Path,required=True)
    parser.add_argument('--model-export',type=Path)
    parser.add_argument('--seconds',type=float,default=3.)
    parser.add_argument('--repeats',type=int,default=2)
    args=parser.parse_args();out=args.output.resolve()
    assert out.is_relative_to(ROOT/'artifacts') and not out.exists()
    assert 0<args.seconds<=12 and 1<=args.repeats<=2 and args.seconds*args.repeats<=16
    out.mkdir(parents=True);started=time.monotonic();torch.set_num_threads(2);torch.manual_seed(20261018)
    graph_path=args.model_export.resolve() if args.model_export else out/'utility.onnx'
    if args.model_export:
        # Graph revalidates bytes, architecture, feature identity and parity;
        # this is reuse of the same isolated export, never a metadata relabel.
        metadata=read(graph_path.with_suffix('.json'))
    else:
        model=UtilityNet(utility_config(ModelConfig.preset('medium'),specification())).eval()
        metadata=export(model,graph_path);del model
    print(json.dumps(dict(event='exported',model_id=metadata['model_id'])),flush=True)
    name='brass_ai.personality.laptop._parallel_baseline'
    spec=importlib.util.spec_from_file_location(name,args.baseline_inference.resolve())
    old=importlib.util.module_from_spec(spec);sys.modules[name]=old;spec.loader.exec_module(old)
    fixtures=read(ROOT/'artifacts/v4-search-vs-fast-2s2w-20261017/final/positions.json')
    positions=[fixtures[i] for i in (0,3,5,7)]
    config=SearchConfig(seconds=args.seconds,workers=2,cpu_role='laptop',pending_paths=2,batch_size=2,
                        reward=specification(),contract='masked-ismcts/1.2.0',utility_scale=1.2)
    entry=dict(path=graph_path.name,model_id=metadata['model_id']);runtimes={};reports={};records=[]
    with ExitStack() as stack:
        for label,constructor in (('baseline',old.PortableInference),('candidate',PortableInference)):
            cold=time.monotonic();service=stack.enter_context(constructor(graph_path.parent,[entry]))
            channels=service.channels(2);endpoints=[channels.acquire() for _ in range(2)]
            for e in endpoints:stack.callback(e.close)
            cache=share_encoder_cache(endpoints);workers=[]
            for i,e in enumerate(endpoints):
                policy=MaskedSearchPolicy(None,ROOT,config=config,evaluator=RemoteEvaluator(e,entry['model_id']),fast_host=True)
                stack.callback(policy.close);workers.append(policy)
            runtimes[label]=dict(service=service,workers=workers,cache=cache,cold_seconds=time.monotonic()-cold)
        pool=stack.enter_context(ThreadPoolExecutor(max_workers=2))
        # Actual encoded roots and full named predictions, over the same private
        # queue/slot transport as search. Both modes see all legal candidates.
        micro={};predictions={}
        for label,runtime in runtimes.items():
            def requests(index):
                policy=runtime['workers'][index].base;values=[]
                for _ in range(3):
                    for j,row in enumerate(positions):
                        values.append(policy.predict([(row['view'],row['choices'],random.Random(910+j))])[0])
                return values
            # Warm encoders/sessions outside the measured requests.
            list(pool.map(requests,range(2)))
            begin=time.monotonic();predictions[label]=list(pool.map(requests,range(2)))
            micro[label]=dict(seconds=time.monotonic()-begin,positions=24)
        for a,b in zip(predictions['baseline'],predictions['candidate']):
            for (pa,ra),(pb,rb) in zip(a,b):
                np.testing.assert_allclose(pa,pb,rtol=2e-5,atol=2e-6)
                for key in ('win_share','placement','vp','utility'):
                    np.testing.assert_allclose(ra[key],rb[key],rtol=2e-5,atol=2e-6)
        print(json.dumps(dict(event='queue_parity_passed',micro=micro)),flush=True)
        for position,row in enumerate(positions):
            for repeat in range(args.repeats):
                order=list(runtimes) if (position+repeat)%2==0 else list(reversed(runtimes))
                for label in order:
                    assert time.monotonic()-started<210, 'Bounded benchmark deadline'
                    runtime=runtimes[label];begin=time.monotonic();deadline=begin+args.seconds
                    def search(index):
                        worker=runtime['workers'][index];rng=random.Random(25000+index+position*10)
                        bot=MaskedISMCTS(worker.game,worker.base,worker.rollouts,config,row['view']['seat'],
                            worker.context_features(row['view'],rng),rng,deadline-.025,[],{},OrderedDict(),
                            worker.simulation_games,Event(),index,
                            dict(spent=[0.]*4))
                        bot.deadline_admission=label=='candidate'
                        with inference_deadline([worker.base]+worker.rollouts,deadline-.025):
                            result=bot.run(row['view'],row['choices'])
                        assert result['opponent_nodes']==0
                        assert result['batching']['attempted']==result['completed']+result['interrupted']
                        assert sum(e['visits'] for e in result['root'].values())==result['completed']
                        assert set(result['root'])<={c['id'] for c in row['choices']}
                        assert all(np.isfinite(e['returns']).all() for e in result['root'].values())
                        return result
                    # map runs both workers before collecting either result.
                    items=list(pool.map(search,range(2)));elapsed=time.monotonic()-begin
                    records.append(dict(position=position,phase=('canal_early','canal_late','rail_early','rail_late')[position],
                        repeat=repeat,variant=label,seconds=elapsed,workers=items))
                    print(json.dumps(dict(position=position,repeat=repeat,variant=label,seconds=elapsed,
                        completed=sum(x['completed'] for x in items),interrupted=sum(x['interrupted'] for x in items),
                        tail_stops=sum(x['batching']['deadline_stops'] for x in items))),flush=True)
        for label,runtime in runtimes.items():
            reports[label]=dict(inference=runtime['service'].report(),encoding=runtime['cache'].report(),
                                cold_seconds=runtime['cold_seconds'],parity=runtime['service'].parity)
    summaries={}
    for label in runtimes:
        rows=[r for r in records if r['variant']==label];items=[w for r in rows for w in r['workers']]
        summaries[label]=dict(decisions=len(rows),completed=sum(w['completed'] for w in items),
            interrupted=sum(w['interrupted'] for w in items),
            deadline_stops=sum(w['batching']['deadline_stops'] for w in items),
            mean_seconds=statistics.mean(r['seconds'] for r in rows),
            root_alternatives=sum(len(set(k for w in r['workers'] for k in w['root'])) for r in rows),
            supported_plans=sum(w['revisited_plans'] for w in items))
    write(out/'report.json',dict(scope=__doc__,architecture=metadata['config'],model_id=metadata['model_id'],
        settings=config.to_dict(),summaries=summaries,micro=micro,reports=reports,records=records,
        total_seconds=time.monotonic()-started))
    print(json.dumps(summaries),flush=True)


if __name__=='__main__':main()
