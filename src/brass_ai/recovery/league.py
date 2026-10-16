"""Recovery league: immutable rounds, BC-compatible weights, persistent curriculum counters."""
from collections import Counter
import copy
from contextlib import ExitStack
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import time

import torch

from brass_ai.bank import admission,read,sha
from brass_ai.learning.checkpoint import load_checkpoint,save_checkpoint,resume_optimizer
from brass_ai.learning.contracts import compatibility,digest,project_path,require
from brass_ai.learning.paths import bound_reference
from brass_ai.learning.model import ModelConfig
from brass_ai.step5.league import write_json,seeds_for,implementation_id
from .config import validate_config
from .demonstrations import load_dataset
from .diagnostics import summarize
from .execution import ActorPool,neural_spec
from .inference import InferenceService
from .ingress import load_collection,reader_pool
from .telemetry import environment,iteration_utilization
from .updates import ppo_update,rehearsal_dataset
from .teacher_collection import collect_due
from .curriculum import activate_boost,boost_window,boost_report,curriculum_report
from .utility import migrate,reward_spec


class Behavior:
    def __init__(self,identity,exploration):
        self.evaluator=type('Identity',(),{'model_id':identity})()
        self.exploration,self.temperature=exploration,1.0
        self.policy_id=neural_spec(identity,exploration,'mixed')['policy_id']


def train(root,run_path,config,*,iterations=1,device='cpu',initial=None,mono=None,from_run=None,imitation_run=None,resume_reviewed_source=False,inference_devices=None,expected_generation=None,fresh=False):
    from .retention import owned,run_lock
    _,run=owned(root,run_path)
    with run_lock(run):
        require(not fresh or not (run/'state.json').exists(),'--fresh requires a new run; existing progress will not be overwritten')
        require_start_generation(run,expected_generation)
        return _train(root,run_path,config,iterations=iterations,device=device,initial=initial,mono=mono,from_run=from_run,imitation_run=imitation_run,resume_reviewed_source=resume_reviewed_source,inference_devices=inference_devices)


def require_start_generation(run,expected):
    if expected is None:return
    require(type(expected) is int and expected>=0,'Invalid expected generation')
    path=run/'state.json'
    require(path.is_file() and read(path)['iteration']==expected,
            f'Expected committed generation {expected}; refusing to start from a different checkpoint')


def _train(root,run_path,config,*,iterations=1,device='cpu',initial=None,mono=None,from_run=None,imitation_run=None,resume_reviewed_source=False,inference_devices=None):
    from .config import time_budget_config,champion_mix_config,latest_model_recipe
    from .inference import validate_inference_devices
    inference_devices=validate_inference_devices(inference_devices,device)
    config=champion_mix_config(time_budget_config(config))
    root=Path(root).resolve();run=project_path(root,run_path);validate_config(config)
    from brass_ai.learning.model import ARCHITECTURE,PARAMETER_LIMIT
    require((config['architecture'],config['parameter_limit'])==(ARCHITECTURE,PARAMETER_LIMIT),
            'Fresh grouped training requires an explicitly configured version-11 model and 3M limit; use feature-engineering.py configure')
    require(iterations>0 and not(initial and from_run),'Invalid recovery initialization')
    require(initial is None and from_run is None,'Fresh residual workflow forbids previous checkpoint/run imports')
    bc_model=bc_origin=None
    if latest_model_recipe(config):
        require(imitation_run is not None,'Latest-model recipe requires the pinned completed --imitation-run')
    if imitation_run is not None:
        require(config['imitation']['manifest'] is None,'Use --imitation-run OR in-run imitation, not both')
        from .demonstrations import completed_imitation
        bc_model,bc_origin=completed_imitation(root,imitation_run)
        if latest_model_recipe(config):
            bound_reference(root,bc_origin,config['imitation_checkpoint'],('bc_run','path','manifest'))
        require(project_path(root,imitation_run)!=run,'Self-play needs its own run directory')
        require(bc_model.config==ModelConfig.preset(config['model_size']),'BC and self-play architecture differ')
        config=copy.deepcopy(config)
        if config['rehearsal_manifest'] is None:config['rehearsal_manifest']=bc_origin['manifest']
        require(project_path(root,config['rehearsal_manifest'])==project_path(root,bc_origin['manifest']),
                'Self-play rehearsal must use the completed imitation dataset')
    revised=True
    torch.set_num_threads(config['threads'])
    pool=read(root/'training/opponents/pool.json');profiles=sorted(pool['training'])
    for p in profiles: admission(root,p,'training')
    from .legacy_search import legacy_spec
    teacher_specs={p:legacy_spec(root,p,config.get('legacy_budget')) for p in profiles}
    # Keep the admitted demonstration/teacher bindings stable for existing IL
    # origins. The optional RL-only variant is bound by config AND each job's
    # policy spec; it does not relabel the qualified frozen bank or old data.
    opponent_specs={p:legacy_spec(root,p,config.get('legacy_budget'),
        work_budget=config.get('legacy_opponent_work_budget')) for p in profiles}
    fast_specs={p:legacy_spec(root,p,config['legacy_budget'],search_free=True,
        work_budget=config['legacy_opponent_work_budget']) for p in profiles}
    signature=dict(config=config,compatibility=compatibility(),implementation=implementation_id(),legacy_pool=pool['training'],teachers=teacher_specs)
    if bc_origin:signature['imitation_source']=bc_origin
    state_path=run/'state.json'
    source_changed=False
    require(not resume_reviewed_source or state_path.exists(),'--resume-reviewed-source needs an existing same-run state.json')
    if state_path.exists():
        state=read(state_path)
        if bc_origin:
            signature['imitation_source']=bound_reference(root,bc_origin,state['signature'].get('imitation_source'),
                ('bc_run','path','manifest'))
        require(initial is None and from_run is None,'Omit initialization when resuming')
        from .resume import load_resume
        model,raw,source_changed=load_resume(root,run,state,signature,device=device,reviewed=resume_reviewed_source)
    else:
        # Before the first league commit, interrupted BC may have only its own
        # identity-bound progress receipt. Restart it from the same random seed;
        # never import partially trained or previous-run neural weights.
        allowed={'.active-run.json','imitation-progress.json'} if config['imitation']['manifest'] else {'.active-run.json'}
        require(not run.exists() or all(p.name in allowed for p in run.iterdir()),'New recovery directory must be empty')
        if revised: torch.manual_seed(config['seed'])
        from brass_ai.learning.model import FourSeatNet
        require(initial is None and from_run is None,'Fresh training rejects --initial and --from-run; resume this same new run instead')
        origin=None;history=[];previous_games=0;cursor=config['seed']
        model=FourSeatNet(ModelConfig.preset(config['model_size'])).to(device)
        model.validate_parameter_count()
        state=dict(format='step5-recovery-league/2.2.0',signature=signature,iteration=0,games_done=0,
            previous_games=0,next_seed=cursor,history=[],origin=dict(initialization='random',seed=config['seed'],parameters=model.parameter_count),
            next_search_games=config['search_ready_games'],context_counter=0,evaluations=[],search_seconds=0.,ordinary_seconds=0.)
        from .demonstrations import fresh_imitation
        if bc_origin:
            model.load_state_dict(bc_model.state_dict())
            state['origin'].update(initialization='fresh-heuristic-imitation',**bc_origin)
            state['imitation']=dict(enabled=True,source=bc_origin,optimizer_reset_before_ppo=True,
                                    note='Completed standalone fresh imitation; no second BC pass')
            print('Self-play starts from the completed imitation candidate; PPO optimizer is new',flush=True)
        else:
            state['imitation']=fresh_imitation(root,model,config['imitation'],config['update'],run,device,workers=config['audit_workers'],warmup=config.get('learning_rate_warmup'))
        # A new recipe never inherits the old full-epoch optimizer moments.
        optimizer=torch.optim.AdamW(model.parameters(),lr=config['learning_rate'],weight_decay=config['weight_decay'])
        if 'learning_rate_warmup' in config:
            from .learning_rate import configure
            warmup=config['learning_rate_warmup']
            configure(optimizer,config['learning_rate'],warmup['selfplay_steps'],warmup['start_factor'],'selfplay')
        init_path=run/'initial.pt'
        save_checkpoint(init_path,model,optimizer,training=dict(mode='step5-fresh-residual-initial',parent=None,initialization=state['origin']['initialization'],fresh_run=run.relative_to(root).as_posix(),imitation=state['imitation']))
        raw=load_checkpoint(init_path,device=device)[1]
        state['history']=[dict(path=init_path.relative_to(root).as_posix(),sha256=sha(init_path),
            model_id=raw['model_id'],generation=0)]
        state['evaluation_anchor']=state['history'][0]
        activate_boost(state,config)
        if revised:
            from .retention import update_population
            update_population(root,run,state,config['population'])
        write_json(state_path,state)
    optimizer=torch.optim.AdamW(model.parameters(),lr=config['learning_rate'],weight_decay=config['weight_decay'])
    cuda_resume=resume_optimizer(raw,optimizer,allow_cuda_topology_change=True)
    from .resume import apply_optimizer_config
    optimizer_resume=apply_optimizer_config(optimizer,config)
    if cuda_resume and cuda_resume['saved_devices']!=cuda_resume['visible_devices']:
        print(f'CUDA resume: {cuda_resume["saved_devices"]} saved / {cuda_resume["visible_devices"]} visible GPUs; '
              f'restored {cuda_resume["restored_devices"]} matching logical-device RNG streams (learner stays on cuda:0)',flush=True)
    reward=reward_spec(config)
    migration=migrate(model,optimizer,reward)
    if migration:
        print('Combined reward migration: compatible weights/Adam/RNG preserved; utility head awaits fresh-rollout calibration',flush=True)
    if source_changed:
        from .resume import commit_transition
        state=commit_transition(run,state,signature)
    from .champion import initialize as initialize_champion,enabled as champion_enabled,selfplay_policies
    from . import champion_mix
    if champion_enabled(config):initialize_champion(root,run,state)
    window=boost_window(state,config)
    if window:
        print(f'Searched-v4 recovery window: ordinary games {window["start_games"]} to {window["until_games"]}; '
              f'{max(0,window["until_games"]-state["games_done"])} remaining',flush=True)
    needs_rehearsal=config['rehearsal_manifest'] and config['rehearsal_weight'] and state['games_done']<config['rehearsal_until']
    rehearsal=load_dataset(root,config['rehearsal_manifest'],'train',workers=config['audit_workers'],progress=True,compact=True)[0] if needs_rehearsal else None
    if rehearsal:
        prepared_at=time.monotonic();rehearsal=rehearsal_dataset(rehearsal,device)
        print(f'Rehearsal tensors prepared once: {len(rehearsal.data)} rows in {time.monotonic()-prepared_at:.1f}s; reused until rehearsal expires',flush=True)
    summaries=[]
    if revised: schedule_evaluations(root,run,state,device,mono,inference_devices=inference_devices)
    with InferenceService(config['execution'],device,profile=run/'profile-collection' if config['profile_steps'] else None,
                          profile_steps=config['profile_steps'],inference_devices=inference_devices) as service,reader_pool(config['audit_workers'],admission=service.cpu) as readers:
        with ActorPool(root,config['host'],service,config['execution'],mono=mono) as ordinary,ActorPool(root,config['host'],service,config['execution'],mono=mono,heavy=True) as heavy:
            from .champion import schedule as schedule_champion
            # Resume an interrupted gate for the committed candidate before
            # advancing the learner or writing a new gate specification.
            if champion_enabled(config):
                with torch.random.fork_rng(devices=[]):schedule_champion(root,run,state,service,ordinary)
            for _ in range(iterations):
                iteration=state['iteration'];began=time.monotonic();directory=run/f'iteration-{iteration:04d}'
                inference_before=service.report()
                current=service.register(model);behavior=Behavior(current,config['exploration'])
                from .ordinary import plan_collection,sampling_report
                plan=plan_collection(root,directory,state,config,signature,service,current,
                                     profiles,opponent_specs,fast_specs,window)
                jobs=plan['jobs'];kind_by_group=plan['kind_by_group'];composition=plan['composition']
                cursor=plan['cursor'];context_counter=plan['context_counter'];incumbent=plan['incumbent']
                historical_generations=plan['historical_generations'];mix_receipts=plan['mix_receipts']
                transition_added=plan['transition_added']
                records=[]
                normal=[j for j in jobs if j['provenance']['legacy_mode']!='searched']
                legacy=[j for j in jobs if j['provenance']['legacy_mode']=='searched']
                from .ingress import StreamingCollection
                with StreamingCollection(readers,config['audit_workers'],behavior.policy_id,kind_by_group,
                                         config.get('insolvency'),reward) as prepared_collection:
                    if revised and config['overlap_pools'] and normal and legacy:
                        with ThreadPoolExecutor(max_workers=2) as coordinators:
                            pending=[coordinators.submit(ordinary.collect,normal,on_records=prepared_collection.submit),
                                     coordinators.submit(heavy.collect,legacy,on_records=prepared_collection.submit)]
                            try:
                                for future in as_completed(pending):records+=future.result()
                            except BaseException:
                                # Signal before coordinator/executor joins. A
                                # failed actor may have stranded a CPU ticket.
                                service.cpu.abort()
                                for future in pending:future.cancel()
                                raise
                    else:
                        if normal: records+=ordinary.collect(normal,on_records=prepared_collection.submit)
                        if legacy: records+=heavy.collect(legacy,on_records=prepared_collection.submit)
                    load_began=time.monotonic();ingress_metrics={}
                    games,learner_data,diagnostics=prepared_collection.finish(records,ingress_metrics)
                collected=time.monotonic()
                sampling=sampling_report(games,learner_data,behavior.policy_id,kind_by_group,config)
                weight=config['rehearsal_weight']*max(0,1-state['games_done']/config['rehearsal_until'])
                update=ppo_update(model,optimizer,games,behavior,config['update'],
                    profile=directory/'profile-ppo' if config['profile_steps'] else None,
                    profile_steps=config['profile_steps'],
                    rehearsal=rehearsal,rehearsal_weight=weight,learner_data=learner_data,insolvency=config.get('insolvency'),reward=reward,
                    sample_groups={g['game_id']:kind_by_group[g['setup_group']] for g in games})
                updated=time.monotonic();done=state['games_done']+len(games)
                ordinary_inference=service.report()
                if done>=config['rehearsal_until']:rehearsal=None
                ordinary_seconds=state.get('ordinary_seconds',0.)+updated-began
                def recalibrate_teacher_utility(auxiliary=None):
                    # Policy distillation moves the shared encoder. Refit only
                    # its utility head against this iteration's fresh ordinary
                    # returns and the explicitly opted-in teacher mixture.
                    from .updates import Dataset
                    from .utility import calibrate
                    return calibrate(model,optimizer,Dataset(learner_data,device,policy=False),config['update'],auxiliary=auxiliary)
                from .teacher_collection import due_collections
                from .review_metrics import teacher_drift_probe
                teaching_probe=(teacher_drift_probe(model,learner_data,behavior,config['update'])
                    if due_collections(done,state['next_search_games'],config) else None)
                search,next_search,cursor,search_seconds=collect_due(root,run,directory,config,state,signature,
                    model,optimizer,service,heavy,done=done,cursor=cursor,ordinary_seconds=ordinary_seconds,
                    ordinary_game_seconds=(collected-began)/max(1,len(games)),
                    recalibrate=recalibrate_teacher_utility if reward is not None else None,
                    ordinary_data=learner_data,ordinary_policy_id=behavior.policy_id)
                # Search policy updates share the trunk/Adam but occur outside
                # the PPO KL guard. Measure their TOTAL drift from collection.
                # No resampling or RNG consumption, and no teacher PPO targets.
                if search.get('updated'):
                    from .updates import Dataset,policy_measure
                    from .telemetry import Timings
                    drift_data=Dataset(learner_data,device)
                    meaningful=[i for i,forced in enumerate(drift_data.forced_cpu) if not forced]
                    model.eval()
                    search['post_teacher_sampled_kl']=policy_measure(model,drift_data,meaningful,
                        behavior,config['update'],Timings())
                    search['post_teacher_kl_scope']='Total drift from frozen ordinary behavior after PPO and teaching; all nonforced learner rows'
                    del drift_data
                    if teaching_probe is not None:
                        search['teaching_only_categorical_probe']=dict(teaching_probe.measure(model),
                            scope='Exact KL(post-PPO behavior || post-teaching behavior); bounded proportional era/phase probe, RNG-neutral; diagnostic only')
                del teaching_probe
                del learner_data
                report=dict(format='step5-recovery-report/'+'2.2.0',iteration=iteration,composition=dict(composition),
                    sampling=sampling,curriculum=curriculum_report(state['games_done'],done,config) if latest_model_recipe(config) else None,
                    reward_migration=migration if not summaries else None,
                    optimizer_resume=optimizer_resume if not summaries else None,
                    legacy_boost=boost_report(state['games_done'],done,config,window),
                    actor=update,search=search,**diagnostics,incoming_model_id=current,incoming_policy_id=behavior.policy_id,
                    selfplay_incumbent=incumbent,
                    selfplay_opponents=champion_mix.summarize(state,config,mix_receipts) if champion_mix.enabled(config) else None,
                    games_done=done,cumulative_games=state['previous_games']+done,
                    timings=dict(collection_seconds=load_began-began,ppo_seconds=updated-collected,
                        replay_load_seconds=collected-load_began,
                        search_and_distill_seconds=time.monotonic()-updated),execution=records,
                    inference=service.report(),inference_scope='cumulative for this invocation',
                    legacy_modes=dict(Counter(j['provenance']['legacy_mode'] for j in jobs if j['provenance']['legacy_mode'])),
                    historical_opponents=dict(generation_seats=dict(sorted(historical_generations.items())),
                        pool_generations=[e['generation'] for e in state.get('population',[])],
                        sampling='25% initial/IL, 25% recent, 50% older reservoir; unavailable strata fall back'),
                    environment=environment(device),strength='requires fresh post-update evaluation')
                report['utilization']=iteration_utilization(inference_before,ordinary_inference,records,
                    load_began-began,len(games),config['execution'])
                report['utilization']['replay_ingress']=ingress_metrics
                output=run/f'generation-{iteration+1:04d}.pt'
                checkpoint_began=time.monotonic()
                identity=save_checkpoint(output,model,optimizer,step=iteration+1,
                    training=dict(mode='step5-population-ppo',recipe='nonforced-actor-ppo/2.1.0' if revised else 'effective-minibatch-ppo/2.0.0',
                        parent=current,signature=digest(signature),fresh_run=run.relative_to(root).as_posix(),initialization='random-lineage',
                        evidence=dict(iteration=iteration,games_done=done,report=(directory/'report.json').relative_to(root).as_posix(),
                                      update_digest=digest(update))))
                report['post_update_model_id']=identity
                report['timings']['checkpoint_seconds']=time.monotonic()-checkpoint_began
                write_json(directory/'report.json',report)
                entry=dict(path=output.relative_to(root).as_posix(),sha256=sha(output),model_id=identity,generation=iteration+1)
                state.update(iteration=iteration+1,games_done=done,next_seed=cursor,context_counter=context_counter,
                    history=state['history']+[entry],next_search_games=next_search,
                    search_seconds=search_seconds,ordinary_seconds=ordinary_seconds)
                if champion_mix.enabled(config):state['champion_mix']['transition_tables']+=transition_added
                state['recent_training']=(state.get('recent_training',[])+[dict(iteration=iteration,games_done=done,
                    sampling=sampling,curriculum=report['curriculum'],
                    learning_rate=config['learning_rate'],learning_rate_schedule=update.get('learning_rate_schedule'),
                    actor={k:v for k,v in update.items() if k not in ('timings','memory')},
                    search={k:v for k,v in search.items() if k not in ('collection','updates')})])[-2:]
                if revised:
                    from .retention import update_population
                    update_population(root,run,state,config['population'])
                write_json(state_path,state)
                if revised:
                    evaluation_began=time.monotonic()
                    from .champion import schedule as schedule_champion
                    if champion_enabled(config):
                        with torch.random.fork_rng(devices=[]):
                            report['champion_gate']=schedule_champion(root,run,state,service,ordinary)
                    schedule_evaluations(root,run,state,device,mono,inference_devices=inference_devices)
                    report['timings']['evaluation_seconds']=time.monotonic()-evaluation_began
                    from .retention import prune
                    cleanup_began=time.monotonic()
                    if config['retention']['enabled']:
                        cleanup=prune(root,run,config['retention'],apply=True)
                        report['cleanup']={k:cleanup[k] for k in ('reclaimable_bytes','permanently_deleted_files','permanently_deleted_bytes')}
                    report['timings']['cleanup_seconds']=time.monotonic()-cleanup_began
                    service.trim([e['model_id'] for e in state['population']]+[identity]+
                                 ([state['champion']['model_id']] if incumbent else [])+
                                 ([state['champion_mix'][k]['model_id'] for k in ('anchor','previous')] if champion_mix.enabled(config) else []))
                report['timings']['iteration_seconds']=time.monotonic()-began
                write_json(directory/'report.json',report)
                summaries.append(report);summaries=summaries[-2:]
                print(f'Recovery generation {iteration+1}: {done} games, {update["optimizer_steps"]} PPO steps; '
                      f'collect={load_began-began:.1f}s prepare={collected-load_began:.1f}s PPO={updated-collected:.1f}s '
                      f'teacher={report["timings"]["search_and_distill_seconds"]:.1f}s '
                      f'eval={report["timings"].get("evaluation_seconds",0):.1f}s cleanup={report["timings"].get("cleanup_seconds",0):.1f}s; '
                      f'total={report["timings"]["iteration_seconds"]:.1f}s; '
                      f'NN batch mean={report["utilization"]["mean_inference_batch"]:.2f}; '
                      f'LR={update["learning_rates"][0]:.1e} KL={update["epochs"][-1]["post_update_kl"]:.5f} '
                      f'value={update["epochs"][-1]["value"]:.4f}; '
                      f'insolvency events={update["insolvency"].get("events",0)} '
                      f'mean cost={update["insolvency"].get("mean_cost_per_trajectory",0):.4f}; '
                      f'teacher rows={search.get("teacher_rows",0)} steps={search.get("optimizer_steps",0)}',flush=True)
    return dict(run=str(run),completed_iteration=state['iteration'],checkpoint=state['history'][-1],
                best=state.get('champion'),summaries=summaries,
                summary_scope='latest two iterations; bounded compact older metrics in retention-history.json')


def schedule_evaluations(root,run,state,device,mono,*,inference_devices=None):
    """Committed game-count schedule; evaluation never advances the train cursor."""
    from brass_ai.step5.evaluation import evaluation_config
    from .evaluation import evaluate
    from .resume import evaluation_directory
    config=state['signature']['config'];schedule=config['evaluation_schedule']
    if not schedule['enabled']: return
    for name,interval,lineups in [('monitor',schedule['monitor_every_games'],['recent-neural']),
                                ('broader',schedule['broader_every_games'],['mixed-neural','mixed-v3'])]:
        slot=state['games_done']//interval
        if slot==0 or any(e['panel']==name and e['slot']>=slot for e in state['evaluations']): continue
        candidate=state['history'][-1];available=state['population']+[state['evaluation_anchor']]
        refs=list({e['model_id']:e for e in available if e['model_id']!=candidate['model_id']}.values())
        # Defer broader panels until two independently frozen neural versions exist.
        if len(refs)<(2 if name=='broader' else 1): continue
        # A four-game alarm against a moving near-neighbour can hide regression
        # from IL. Keep its size/frequency, but compare against the fixed anchor.
        refs=[state['evaluation_anchor']] if name=='monitor' else refs[-2:]
        if refs[0]['model_id']==candidate['model_id']:continue
        cfg=evaluation_config(config,setups=schedule[name+'_setups'],seed=schedule['seed']+slot*1000,
            lineups=lineups,include_legacy=name=='broader',panel='development',use_search=False)
        directory=evaluation_directory(root,run,state,f'{name}-{slot:04d}')
        result=evaluate(root,directory,candidate['path'],[r['path'] for r in refs],cfg,device=device,mono=mono,inference_devices=inference_devices)
        require(result['failed_games']==0,'Incomplete scheduled evaluation')
        state['evaluations'].append(dict(panel=name,slot=slot,games_done=state['games_done'],candidate=candidate,
            references=refs,games=result['games'],spec_sha256=sha(directory/'spec.json'),
            report=directory.relative_to(root).as_posix()+'/report.json',report_sha256=sha(directory/'report.json')))
        write_json(run/'state.json',state)
