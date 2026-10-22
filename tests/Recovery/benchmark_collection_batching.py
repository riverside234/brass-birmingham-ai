"""Bounded matched native collection, same policies/seeds/CPU cap in every arm."""
import argparse
import copy
from pathlib import Path
import time

import torch

from brass_ai.bank import read
from brass_ai.learning.contracts import SOURCE_ROOT,require
from brass_ai.learning.model import ModelConfig
from brass_ai.learning.utility_model import UtilityNet,utility_config
from brass_ai.recovery.execution import ActorPool,neural_spec
from brass_ai.recovery.inference import InferenceService
from brass_ai.recovery.inference_batches import ModelBatches
from brass_ai.recovery.legacy_search import legacy_spec
from brass_ai.recovery.replay import read_shards
from brass_ai.recovery.telemetry import iteration_utilization,environment
from brass_ai.recovery.utility import reward_spec
from brass_ai.step5.league import seeds_for,write_json
from benchmark_single_gpu import measured
from smoke_branch_scheduling import signature,compare


VARIANTS={'baseline':(2.,1.,False),'padding3':(3.,1.,False),
          'padding4':(4.,1.,False),'wait2':(3.,2.,False),'cohorts':(3.,1.,True),
          'cohorts2':(2.,1.,True)}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',default='cpu');p.add_argument('--games',type=int,default=12)
    p.add_argument('--processes',type=int,default=4);p.add_argument('--lanes',type=int,default=2)
    p.add_argument('--cpu-budget',type=int,default=12);p.add_argument('--threads',type=int,default=2)
    p.add_argument('--repeats',type=int,default=1)
    p.add_argument('--variants',nargs='+',choices=tuple(VARIANTS),default=['baseline','padding3','cohorts'])
    a=p.parse_args();out=a.output.resolve()
    require(out.is_relative_to(SOURCE_ROOT/'artifacts') and not out.exists(),'Use a new isolated artifacts directory')
    require(4<=a.games<=32 and 1<=a.repeats<=3 and a.processes*a.lanes<=a.cpu_budget-2,'Keep collection bounded within CPU budget')
    out.mkdir(parents=True);torch.set_num_threads(a.threads);torch.manual_seed(47083)
    cfg=read(SOURCE_ROOT/'configs/step5/joint7-v11-2-grouped-lr-critic-ppo-only-g460.json')
    ex=dict(cfg['execution'],processes=a.processes,hosts_per_process=a.lanes,heavy_processes=1,
        cpu_budget=a.cpu_budget,cpu_reserve=2,worker_threads=1,shard_games=1)
    models=[UtilityNet(utility_config(ModelConfig.preset('medium'),reward_spec(cfg))).eval() for _ in range(4)]
    profiles=sorted(read(SOURCE_ROOT/'training/opponents/pool.json')['training'])
    v4=[legacy_spec(SOURCE_ROOT,s,cfg['legacy_budget'],search_free=True,
                   work_budget=cfg['legacy_opponent_work_budget']) for s in profiles]
    seeds=seeds_for(47083,a.games)[0];references=None;arms=[]
    with InferenceService(ex,a.device) as service,ActorPool(SOURCE_ROOT,cfg['host'],service,ex) as pool:
        identities=[service.register(m) for m in models]
        for identity in identities:service._model(identity)
        def jobs(name,n=None):
            result=[]
            for i,seed in enumerate(seeds[:n]):
                kind='legacy' if i==a.games-1 else ('selfplay' if i%2==0 else 'historical')
                learner=neural_spec(identities[0],cfg['exploration'],'mixed',version=3,opening=cfg['opening'])
                history=[neural_spec(identities[1+(i+j)%3],cfg['exploration'],'neutral',version=3,
                                    opening=cfg['opening']) for j in range(2)]
                policies=[learner]*4 if kind=='selfplay' else [learner]*2+(history if kind=='historical' else [v4[i%len(v4)],v4[(i+1)%len(v4)]])
                result.append(dict(index=i,seed=seed,sampling_seed=seed*7,stable_game_id='batch-bench-'+str(i),
                    rotation=i%4,policies=policies,focuses={str(s):'neutral' for s in range(4)},
                    path=str(out/name/f'game-{i}.jsonl.gz'),
                    provenance=dict(producer='recovery/2.0.0',purpose='training',kind=kind,
                        exploration=dict(version=3,opening=cfg['opening'],branching=cfg['branching'] if kind=='selfplay' else None)),
                    **(dict(branching=cfg['branching']) if kind=='selfplay' else {})))
            return result
        pool.collect(jobs('warm',min(a.games,a.processes*a.lanes)))
        for repeat in range(a.repeats):
            modes=a.variants if repeat%2==0 else list(reversed(a.variants))
            for name in modes:
                ratio,wait,cohorts=VARIANTS[name]
                ex.update(padding_ratio=ratio,flush_ms=wait,model_cohorts=cohorts)
                service.scheduler=ModelBatches(ex['batch_size'],ex['max_candidate_cells'],ratio)
                pool.duration_estimates={} # identical starting duration knowledge in all arms
                if torch.device(a.device).type=='cuda':torch.cuda.reset_peak_memory_stats(a.device)
                before=copy.deepcopy(service.report());resource={}
                with measured(resource):records=pool.collect(jobs(name+'-'+str(repeat)))
                after=copy.deepcopy(service.report());elapsed=resource['elapsed_seconds']
                # Scheduler was reset for this arm; histogram/model counters remain cumulative.
                before['batching']={}
                metrics=iteration_utilization(before,after,records,elapsed,a.games,ex)
                parents={};children={}
                for record in records:
                    path=record['path']
                    for game in read_shards([path],receipts={path:record.get('features')}):
                        parents[game['game_id']]=signature(game)
                        for child in game.get('branches',[]):children[child['game_id']]=signature(child)
                if references is None:references=(parents,children)
                error=max(compare(references[0],parents),compare(references[1],children))
                arm=dict(variant=name,repeat=repeat,collection_seconds=elapsed,games_per_second=a.games/elapsed,
                    resources=resource,utilization=metrics,memory=after['memory'],scheduling=pool.last_collection_report,
                    game_choices_results_identical=True,max_probability_error=error,branches=len(children),
                    models=identities,execution=copy.deepcopy(ex))
                arms.append(arm);write_json(out/'report.json',dict(environment=environment(a.device),arms=arms,
                    seeds=seeds,model_parameters=models[0].parameter_count,games=a.games,
                    scope='Random current-size frozen models; mixed self-play/history/direct-v4 native games and branches. Same seeds/identities/rotation/CPU cap, warmed actors. No PPO or production checkpoint changes.'))
                print(name,round(elapsed,3),'s batch',round(metrics['mean_inference_batch'],3),flush=True)


if __name__=='__main__':main()
