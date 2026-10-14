"""Bounded, isolated Step 5 measurements; never resume or modify a training run.

Ordinary mode measures complete native games, replay preparation, a disposable
PPO update and checkpoint write. Teacher/opponent modes measure real root
decisions from a supplied replay, with configured clocks and worker counts.
Neither workload is a complete production iteration or a strength evaluation.
"""
import argparse
from collections import Counter, defaultdict
import copy
import importlib
import json
import os
from pathlib import Path
import random
import sys
import time
import weakref

SOURCE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(SOURCE/'src'))


BASELINE_MODULES=('learning.batching','recovery.inputs','recovery.encoding','recovery.replay')


def baseline_inputs(path):
    # Explicit local A/B experiment only. Spawned actors apply the same saved
    # modules; the source checkout is never swapped under live workers.
    for name in BASELINE_MODULES:
        file=Path(path)/Path(name.replace('.','/')).with_suffix('.py')
        module=importlib.import_module('brass_ai.'+name)
        exec(compile(file.read_text(encoding='utf-8'),str(file),'exec'),module.__dict__)


if os.environ.get('BRASS_BENCHMARK_INPUT_BASELINE'):
    baseline_inputs(os.environ['BRASS_BENCHMARK_INPUT_BASELINE'])


def main():
    main_started=time.perf_counter()
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--config',default='configs/step5/joint7-g150-gate72.json')
    parser.add_argument('--output',required=True)
    parser.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    parser.add_argument('--checkpoint',help='Read-only frozen checkpoint; otherwise use seeded small fixture weights')
    parser.add_argument('--opponent-checkpoint',action='append',default=[])
    parser.add_argument('--mode',choices=['ordinary','teacher','opponent'],default='ordinary')
    parser.add_argument('--games',type=int,default=2)
    parser.add_argument('--rounds',type=int,default=2)
    parser.add_argument('--actors',type=int,default=1)
    parser.add_argument('--lanes',type=int,default=1)
    parser.add_argument('--seed',type=int,default=936211)
    parser.add_argument('--replay',help='Search probe: authoritative complete-game shard (read only)')
    parser.add_argument('--phases',nargs='+',default=['canal_early','rail_late'])
    parser.add_argument('--profile',action='store_true',help='Short CPU/CUDA traces; use separately from wall-time comparisons')
    parser.add_argument('--baseline-inputs',type=Path,help='Saved brass_ai directory with batching, inputs, encoding and replay modules')
    args=parser.parse_args()
    if args.baseline_inputs:
        os.environ['BRASS_BENCHMARK_INPUT_BASELINE']=str(args.baseline_inputs.resolve())
        baseline_inputs(args.baseline_inputs)
    import torch
    from brass_ai.bank import read,sha
    from brass_ai.learning.checkpoint import load_checkpoint,save_checkpoint
    from brass_ai.learning.contracts import project_path,require,digest
    from brass_ai.learning.model import ModelConfig
    from brass_ai.learning.utility_model import UtilityNet,utility_config
    from brass_ai.recovery.config import validate_config
    from brass_ai.recovery.execution import ActorPool,neural_spec,make_policy
    from brass_ai.recovery.inference import InferenceService
    from brass_ai.recovery.ingress import load_collection
    from brass_ai.recovery.replay import read_shards
    from brass_ai.recovery.telemetry import environment,iteration_utilization,Timings
    from brass_ai.recovery.updates import ppo_update
    from brass_ai.step5.league import seeds_for,write_json,implementation_id
    require(args.root.is_absolute(),'Pass an absolute --root')
    root=args.root.resolve();out=project_path(root,args.output)
    require(out.is_relative_to(root/'artifacts') and not out.exists(),'Use a new output directory under artifacts')
    require(1<=args.games<=320 and 1<=args.rounds<=8 and args.actors>0 and args.lanes>0,'Unbounded benchmark')
    cfg=read(project_path(root,args.config));validate_config(cfg)
    execution=dict(cfg['execution'],processes=args.actors,hosts_per_process=args.lanes)
    require(args.actors*args.lanes<=execution['cpu_budget']-execution['cpu_reserve'],'Actors exceed shared CPU slots')
    torch.set_num_threads(cfg['threads']);torch.manual_seed(63621)
    out.mkdir(parents=True)
    if args.checkpoint:
        checkpoint=project_path(root,args.checkpoint);model,raw=load_checkpoint(checkpoint,device=args.device)
        origin=dict(path=str(checkpoint),sha256=sha(checkpoint),model_id=raw['model_id'])
    else:
        model=UtilityNet(utility_config(ModelConfig.preset('small'),cfg['search']['reward'])).to(args.device)
        origin=dict(fixture=True,seed=63621,scope='Authored small fixture; not trained lab weights')
    report=dict(format='step5-performance-probe/1.0.0',mode=args.mode,config=cfg,execution=execution,
                source=implementation_id(),baseline_inputs=str(args.baseline_inputs) if args.baseline_inputs else None,
                origin=origin,seed=args.seed,driver_sha256=sha(Path(__file__)),
                environment=environment(args.device),rounds=[],profile=args.profile,
                scope='Isolated component workload, not complete production iteration or strength evidence')
    if args.baseline_inputs:
        report['baseline_hashes']={name:sha(args.baseline_inputs/Path(name.replace('.','/')).with_suffix('.py'))
                                   for name in BASELINE_MODULES}

    class MeasuredService(InferenceService):
        def __init__(self,*a,**kw):
            super().__init__(*a,**kw);self.model_metrics=defaultdict(lambda:dict(requests=0,candidates=0,batches=Counter()))
            self.hooked=weakref.WeakSet()
        def _model(self,identity):
            result=super()._model(identity)
            if result not in self.hooked:
                self.hooked.add(result)
                def hook(module,values):self.model_metrics[identity]['batches'][str(values[0].shape[0])]+=1
                result.register_forward_pre_hook(hook)
            return result
        def _evaluate(self,items):
            for _,frame in items:
                row=self.model_metrics[frame['model']];row['requests']+=1
                if frame['encoded'] is not None:row['candidates']+=len(frame['encoded'].choice_ids)
            return super()._evaluate(items)
        def report(self):
            return dict(super().report(),by_model=copy.deepcopy(dict(self.model_metrics)),
                        histogram_scope='Padded model forward calls; oversized candidate chunks bypass this hook')

    began=time.perf_counter()
    report['initialization_seconds']=began-main_started
    with MeasuredService(execution,args.device,profile=out/'profile-inference' if args.profile else None) as service:
        identity=service.register(model)
        opponents=[service.register_checkpoint(project_path(root,p)) for p in args.opponent_checkpoint]
        specs=[neural_spec(m,cfg['exploration'],'neutral') for m in [identity]+opponents]
        if args.mode=='ordinary':
            with ActorPool(root,cfg['host'],service,execution) as pool:
                def collect(label,seeds):
                    directory=out/label;directory.mkdir()
                    jobs=[dict(index=i,seed=seed,rotation=0,sampling_seed=7632+i,policies=[specs[s%len(specs)] for s in range(4)],
                               path=str(directory/f'game-{i}.jsonl.gz'),provenance=dict(producer='step5-performance-probe',
                               purpose='training',kind='selfplay',seed=seed,sampling_seed=7632+i,
                               spec=digest(dict(benchmark=report['origin'],baseline=report.get('baseline_hashes'),label=label))))
                          for i,seed in enumerate(seeds)]
                    before=service.report();start=time.perf_counter();records=pool.collect(jobs);wall=time.perf_counter()-start
                    return dict(collection_seconds=wall,records=records,seeds=seeds,inference=service.report(),before=before,
                                utilization=iteration_utilization(before,service.report(),records,wall,len(jobs),execution))
                warm_seeds,_=seeds_for(args.seed-1000,1)
                report['warmup']=collect('warmup',warm_seeds)
                all_seeds,_=seeds_for(args.seed,args.games*args.rounds)
                for index in range(args.rounds):
                    result=collect(f'round-{index}',all_seeds[index*args.games:(index+1)*args.games])
                    result['round']=index;report['rounds'].append(result)
                    write_json(out/'report.json',report)
        else:
            require(args.replay is not None,'Search probe needs --replay')
            from brass_ai.recovery.teacher_schedule import CONTRACT,PHASES,reserved_rounds
            from brass_ai.learning.replay import validate_teacher
            from brass_ai.learning.reward import CostLedger
            require(len(args.phases)==len(set(args.phases)) and set(args.phases)<={p[0] for p in PHASES},
                    'Use each known phase at most once')
            # Normal authoritative replay ingress; its signed provenance is not
            # modified. The generated probes are independent diagnostic labels.
            games=read_shards([project_path(root,args.replay)],partition='train')
            channels=service.channels(1+execution['search_workers_per_turn'])
            endpoints=[channels.acquire() for _ in range(1+execution['search_workers_per_turn'])]
            spec=dict(kind='masked_ismcts',model_id=identity,opponents=[identity]+opponents,search=cfg['search'],
                      focus='neutral',cpu_role=cfg['legacy_budget']['role'],teacher_schedule=CONTRACT,
                      teacher_work_budget=cfg['teacher_work_budget'])
            if args.mode=='teacher':
                policy=make_policy(spec,root,endpoints,execution,None)
            else:
                from brass_ai.recovery.legacy_search import LabLegacyPolicy
                parent=next(p for p in read(root/'training/opponents/pool.json')['training'] if 'balanced' in p)
                policy=LabLegacyPolicy(root,parent,cfg['legacy_budget'],work_budget=cfg['legacy_opponent_work_budget'])
                report['opponent']=dict(parent=parent,policy_id=policy.policy_id)
            try:
                for phase in args.phases:
                    _,era,lo,hi=next(p for p in PHASES if p[0]==phase)
                    game=next(g for g in games if any(r['view']['context']['era']==era and lo<=r['view']['context']['round']<=hi and
                              r['view']['context']['phase']=='action' and len(r['choices'])>1 for r in g['records']))
                    row=next(r for r in game['records'] if r['view']['context']['era']==era and lo<=r['view']['context']['round']<=hi and
                             r['view']['context']['phase']=='action' and len(r['choices'])>1)
                    view,choices=row['view'],row['choices'];ledger=CostLedger(cfg['search']['reward'])
                    for r in game['records']:
                        ledger.observe(r['view'])
                        if r is row:break
                        ledger.chosen(r['view'],r['choices'][r['chosen']])
                    if args.mode=='teacher':policy.set_reward_context(ledger)
                    for repeat in range(args.rounds):
                        rng=random.Random(args.seed+repeat)
                        # Select a schedule seed which actually reserves this
                        # observed round; no clock or qualification is overridden.
                        sampling=next(s for s in range(10000) if reserved_rounds(args.seed,s,view['seat'])[phase]==view['context']['round'])
                        if args.mode=='teacher':policy.begin_teacher_game(rng,view['seat'],args.seed,sampling)
                        before=service.report();timings=Timings()
                        with service.cpu.acquire(execution['search_workers_per_turn'],timings):
                            start=time.monotonic()
                            if args.mode=='teacher':policy.admit_teacher_turn(view['context'],rng)
                            else:policy.admit_legacy_turn(view['context'],rng)
                            chosen,probability,label=policy.select_batch([(view,choices,rng)])[0]
                            wall=time.monotonic()-start
                        require(0<=chosen<len(choices),'Illegal probe action')
                        if label is not None:validate_teacher(label,len(choices))
                        if args.mode=='teacher':policy.finish_teacher_turn(rng)
                        native=[g.host.recovery_metrics.report() for g in getattr(policy,'simulation_games',[])
                                if hasattr(g.host,'recovery_metrics')]
                        report['rounds'].append(dict(phase=phase,repeat=repeat,seconds=wall,chosen=choices[chosen]['id'],
                            qualified=label is not None,teacher=label,
                            statistics=policy.last_statistics if args.mode=='teacher' else copy.deepcopy(policy.telemetry),
                            schedule=policy.teacher_report(rng) if args.mode=='teacher' else None,
                            native_cumulative=native,cpu_admission=timings.report(),inference=service.report(),before=before))
                        write_json(out/'report.json',report)
            finally:
                policy.close()
                for endpoint in endpoints:endpoint.close()
        report['collection_lifetime_seconds']=time.perf_counter()-began
    # No overlapping process-wide CUDA profilers. Inference exits before the
    # disposable learner measurements begin; neither is included in collection.
    if args.mode=='ordinary':
        for result in report['rounds']:
            records=result['records']
            behavior=type('Behavior',(),dict(evaluator=type('Identity',(),dict(model_id=identity))(),exploration=cfg['exploration'],
                          temperature=1.,policy_id=specs[0]['policy_id']))()
            learner=copy.deepcopy(model);optimizer=torch.optim.AdamW(learner.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
            start=time.perf_counter();telemetry={}
            rules=read(SOURCE/'configs/learning/host-1.0.0.json')['rules']
            kinds={digest(dict(rules=rules,setup_seed=seed)):'selfplay' for seed in result['seeds']}
            games,data,diagnostics=load_collection(records,behavior.policy_id,kinds,
                                                  workers=1,telemetry=telemetry,insolvency=cfg['insolvency'],reward=cfg['search']['reward'])
            result['prepare_seconds']=time.perf_counter()-start;result['prepare']=telemetry
            start=time.perf_counter()
            result['ppo']=ppo_update(learner,optimizer,games,behavior,cfg['update'],learner_data=data,
                                    insolvency=cfg['insolvency'],reward=cfg['search']['reward'],
                                    profile=out/f'profile-ppo-{result["round"]}' if args.profile else None)
            if args.device=='cuda':torch.cuda.synchronize()
            result['ppo_seconds']=time.perf_counter()-start
            start=time.perf_counter();result['updated_model_id']=save_checkpoint(out/f'probe-{result["round"]}.pt',learner,optimizer,
                training=dict(mode='performance-probe-only',source_checkpoint=origin))
            result['checkpoint_seconds']=time.perf_counter()-start
            result['component_total_seconds']=sum(result[k] for k in ('collection_seconds','prepare_seconds','ppo_seconds','checkpoint_seconds'))
            # Parity inspection must follow the measured cold prepare; reading
            # first would prime its validation cache and understate that phase.
            start=time.perf_counter()
            games=read_shards(sorted({r['path'] for r in records}),receipts={r['path']:r.get('features') for r in records})
            result['diagnostic_replay_read_seconds']=time.perf_counter()-start
            result['trajectory_digest']=digest([dict(setup=g['setup_group'],result=g['result'],records=[
                {k:r[k] for k in ('seat','view','choices','chosen','style','intent','behavior')} for r in g['records']]) for g in games])
            result['decisions']=sum(len(g['records']) for g in games)
    report['probe_seconds']=time.perf_counter()-began
    report['total_command_seconds']=time.perf_counter()-main_started
    report['command_timing_scope']='main entry through completed measurements; excludes interpreter startup and final report write'
    write_json(out/'report.json',report)
    print(json.dumps(dict(output=str(out/'report.json'),mode=args.mode,total_seconds=report['total_command_seconds'],
                         rounds=[{k:v for k,v in r.items() if k in ('round','phase','repeat','seconds','qualified',
                             'collection_seconds','prepare_seconds','ppo_seconds','checkpoint_seconds','component_total_seconds','trajectory_digest')}
                                 for r in report['rounds']]),indent=2))


if __name__=='__main__':main()
