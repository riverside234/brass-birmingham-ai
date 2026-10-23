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
from . import critic_data


class Behavior:
    def __init__(self,identity,exploration,settings=None):
        self.evaluator=type('Identity',(),{'model_id':identity})()
        self.exploration,self.temperature=exploration,1.0
        self.exploration_version=(settings or {}).get('version',2)
        self.opening=(settings or {}).get('opening')
        self.raw_entropy=(settings or {}).get('raw_entropy',False)
        self.entropy_alphas=(settings or {}).get('alphas')
        self.policy_id=neural_spec(identity,exploration,'mixed',version=self.exploration_version,opening=self.opening)['policy_id']


def finish_training_diagnostics(model,phases,local_scoring):
    """Materialize the small report before releasing its iteration datasets."""
    report=None
    if local_scoring is not None:
        report=local_scoring.report()
        if local_scoring.active:
            require(phases is not None and phases.critic_dataset is not None,
                    'Local scoring diagnostics require the live critic dataset before release')
            report['heldout']=local_scoring.diagnostics(model,phases.critic_dataset,phases.heldout_ids)
    if phases is not None:phases.release()
    return report


def train(root,run_path,config,*,iterations=1,device='cpu',initial=None,mono=None,from_run=None,imitation_run=None,resume_reviewed_source=False,inference_devices=None,expected_generation=None,fresh=False,il_kl_probe_from_run=None):
    from .retention import owned,run_lock
    _,run=owned(root,run_path)
    with run_lock(run):
        require(not fresh or not (run/'state.json').exists(),'--fresh requires a new run; existing progress will not be overwritten')
        require_start_generation(run,expected_generation)
        if 'rehearsal_sampling' in config:
            from .rehearsal_window import cleanup
            cleanup(root,run)
        return _train(root,run_path,config,iterations=iterations,device=device,initial=initial,mono=mono,from_run=from_run,imitation_run=imitation_run,resume_reviewed_source=resume_reviewed_source,inference_devices=inference_devices,il_kl_probe_from_run=il_kl_probe_from_run)


def require_start_generation(run,expected):
    if expected is None:return
    require(type(expected) is int and expected>=0,'Invalid expected generation')
    path=run/'state.json'
    require(path.is_file() and read(path)['iteration']==expected,
            f'Expected committed generation {expected}; refusing to start from a different checkpoint')


def _train(root,run_path,config,*,iterations=1,device='cpu',initial=None,mono=None,from_run=None,imitation_run=None,resume_reviewed_source=False,inference_devices=None,il_kl_probe_from_run=None):
    from .config import time_budget_config,champion_mix_config,latest_model_recipe,quality_selfplay_recipe
    from .gae import evidence_config
    from .inference import validate_inference_devices
    inference_devices=validate_inference_devices(inference_devices,device)
    config=evidence_config(champion_mix_config(time_budget_config(config)))
    root=Path(root).resolve();run=project_path(root,run_path);validate_config(config)
    quality=quality_selfplay_recipe(config)
    if quality and config['rating']['enabled']:
        from .ratings import check_dependencies
        check_dependencies()
        try:import matplotlib
        except ImportError as error:raise ValueError('Strength plots need matplotlib; install requirements-step5.txt') from error
    from brass_ai.learning.model import ARCHITECTURE,PARAMETER_LIMIT
    require((config['architecture'],config['parameter_limit'])==(ARCHITECTURE,PARAMETER_LIMIT),
            f'Fresh grouped training requires {ARCHITECTURE} and parameter limit {PARAMETER_LIMIT}; use feature-engineering.py configure')
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
    from .config import v4_search_enabled
    teacher_specs={p:legacy_spec(root,p,config.get('legacy_budget')) for p in profiles}
    # Keep the admitted demonstration/teacher bindings stable for existing IL
    # origins. The optional RL-only variant is bound by config AND each job's
    # policy spec; it does not relabel the qualified frozen bank or old data.
    opponent_specs={p:legacy_spec(root,p,config.get('legacy_budget'),
        search_free=not v4_search_enabled(config),
        work_budget=config.get('legacy_opponent_work_budget')) for p in profiles}
    fast_specs={p:legacy_spec(root,p,config['legacy_budget'],search_free=True,
        work_budget=config['legacy_opponent_work_budget']) for p in profiles}
    signature=dict(config=config,compatibility=compatibility(),implementation=implementation_id(),legacy_pool=pool['training'],teachers=teacher_specs)
    if bc_origin:signature['imitation_source']=bc_origin
    state_path=run/'state.json'
    require(il_kl_probe_from_run is None or (not state_path.exists() and bc_origin is not None and 'il_kl_limit' in config),
            '--il-kl-probe-from-run is only for a new self-play run from completed IL')
    fixed_probe=None
    from . import policy_anchor
    anchor_values=None
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
        if 'il_kl_limit' in config:allowed.add('il-kl')
        if config.get('value_phase',{}).get('enabled') is False:
            allowed.update(('.ppg-value-cache','.ppg-eight-cache'))
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
        if 'ppo_lr_controller' in config:
            from .lr_controller import configure as configure_controller
            configure_controller(optimizer,config['ppo_lr_controller'])
        if 'il_kl_limit' in config:
            from .il_kl_probe import load as load_probe
            from .il_kl_limit import initial_state
            if il_kl_probe_from_run is not None:
                from .il_kl_probe import import_probe
                fixed_probe,receipt=import_probe(root,run,il_kl_probe_from_run,bc_origin,model,config['il_kl_limit'],config['update'])
            else:
                fixed_probe,receipt=load_probe(root,run,bc_origin,model,config['il_kl_limit'],config['update'],config['audit_workers'])
            initial_measure=fixed_probe.measure(model,config['update'])
            require(abs(initial_measure['kl_mean'])<=1e-10,
                    'Generation-zero IL probe must reproduce the original IL policy')
            state['il_kl_limit']=initial_state(config['il_kl_limit'],receipt)
            write_json(run/'il-kl/initial-report.json',dict(generation=0,measurement=initial_measure,
                limit=config['il_kl_limit']['initial_limit'],
                group_limit=config['il_kl_limit']['group_factor']*config['il_kl_limit']['initial_limit'],probe=receipt))
        from . import exploration
        if 'exploration_rollout' in config:state['exploration']=exploration.initial(config,0)
        init_path=run/'initial.pt'
        if quality:
            from . import quality_pool
            from brass_ai.learning.batching import model_id
            anchor=dict(path=init_path.relative_to(root).as_posix(),sha256='pending',model_id=model_id(model),generation=0)
            probe_ref=dict(fixed_probe.metadata['cache'],probe_hash=fixed_probe.metadata['probe_hash'])
            state['quality_pool']=quality_pool.initial(anchor,config['opponent_pool'],probe_ref)
            if 'critic_data' in config:
                from .critic_data import fingerprint
                state['quality_pool']['members'][0]['fingerprint']=fingerprint(fixed_probe,fixed_probe.arrays['reference'])
        save_checkpoint(init_path,model,optimizer,training=dict(mode='step5-fresh-residual-initial',parent=None,initialization=state['origin']['initialization'],fresh_run=run.relative_to(root).as_posix(),imitation=state['imitation'],
            **({'il_kl_limit':state['il_kl_limit']} if 'il_kl_limit' in config else {}),
            **({'exploration':state['exploration']} if 'exploration' in state else {}),
            **({'quality_pool_digest':quality_pool.state_digest(state['quality_pool'])} if quality else {})))
        raw=load_checkpoint(init_path,device=device)[1]
        state['history']=[dict(path=init_path.relative_to(root).as_posix(),sha256=sha(init_path),
            model_id=raw['model_id'],generation=0)]
        state['evaluation_anchor']=state['history'][0]
        if quality:state['quality_pool']['members'][0].update(state['evaluation_anchor'])
        activate_boost(state,config)
        if revised:
            from .retention import update_population
            update_population(root,run,state,config['population'])
        if quality:
            from .ratings import initialize as initialize_ratings
            state['ratings']=initialize_ratings(root,run,state,config)
        write_json(state_path,state)
    optimizer=torch.optim.AdamW(model.parameters(),lr=config['learning_rate'],weight_decay=config['weight_decay'])
    cuda_resume=resume_optimizer(raw,optimizer,allow_cuda_topology_change=True)
    from .resume import apply_optimizer_config
    optimizer_resume=apply_optimizer_config(optimizer,config)
    from . import exploration
    exploration_state=exploration.restore(config,state,raw['training'].get('exploration'))
    pool_migration=resumed_pool=None
    if quality:
        from . import quality_pool
        from .resume import prepare_opponent_pool
        resumed_pool,pool_migration=prepare_opponent_pool(root,run,state,raw,config,source_changed)
    anchor_model=None
    if 'il_kl_limit' in config:
        from .il_kl_limit import validate_state
        from .il_kl_probe import load as load_probe
        validate_state(state.get('il_kl_limit'),config['il_kl_limit'],state['iteration'])
        from .g110_hotfix import verify_checkpoint
        if 'policy_anchor' in state:
            policy_anchor.verify_checkpoint(state,raw)
            restored_hotfix=None
        else:restored_hotfix=verify_checkpoint(state,raw)
        if restored_hotfix is not None:
            optimizer_resume['il_kl_hotfix']=restored_hotfix
            print(f'One-time IL-KL hotfix: limit={restored_hotfix["limit_after"]:.8f}, '
                  f'group_limit={restored_hotfix["group_limit_after"]:.8f}; recovery latch preserved',flush=True)
        if fixed_probe is None:
            fixed_probe,_=load_probe(root,run,bc_origin,bc_model,config['il_kl_limit'],config['update'],
                config['audit_workers'],receipt=state['il_kl_limit']['probe'])
    elif config.get('ppo_lr_controller',{}).get('mode','off')!='off':
        anchor=state['evaluation_anchor'];path=project_path(root,anchor['path'])
        require(sha(path)==anchor['sha256'],'LR health anchor checkpoint changed')
        with torch.random.fork_rng(devices=[]):
            anchor_model,anchor_raw=load_checkpoint(path,device=device)
        require(anchor_raw['model_id']==anchor['model_id'],'LR health anchor identity differs')
        anchor_model.eval().requires_grad_(False)
    if cuda_resume and cuda_resume['saved_devices']!=cuda_resume['visible_devices']:
        print(f'CUDA resume: {cuda_resume["saved_devices"]} saved / {cuda_resume["visible_devices"]} visible GPUs; '
              f'restored {cuda_resume["restored_devices"]} matching logical-device RNG streams (learner stays on cuda:0)',flush=True)
    reward=reward_spec(config)
    migration=migrate(model,optimizer,reward)
    phases=None
    from .local_scoring_adoption import prepare as prepare_local,cleanup as cleanup_local
    phase_raw,local_adoption,local_adopting=prepare_local(root,run,state,raw,config,reviewed=source_changed)
    local_scoring=None
    if 'local_scoring_auxiliary' in config:
        from .local_scoring_auxiliary import Companion
        local_scoring=Companion(model,config['local_scoring_auxiliary'],raw=raw,generation=state['iteration'],adopting=local_adopting)
    if 'advantage' in config:
        from .actor_value import TrainingPhases
        phases=TrainingPhases(model,config,raw=phase_raw,generation=state['iteration'],root=root,run=run)
        phases.local_scoring=local_scoring
        if raw is not None and state['iteration']>0:
            from .snapshot_r2_hotfix import restore as restore_snapshot_r2
            restore_snapshot_r2(state,raw,phases)
    if quality:
        from .ratings import validate_frozen_reference
        validate_frozen_reference(root,state['ratings'])
    if migration:
        print('Combined reward migration: compatible weights/Adam/RNG preserved; value targets and fitting follow the adopted reward/training recipe',flush=True)
    if phases is not None and phases.snapshot_fixed_migration:
        change=phases.snapshot_fixed_migration
        print(f'Snapshot critic weight: {change["weight_before"]:.2f} -> 1.00 fixed; '
              'recovery and decay disabled, prior controller archived',flush=True)
    if source_changed:
        from .resume import commit_transition
        if local_adoption is not None:
            from .local_scoring_adoption import archive_metadata
            local_adoption=archive_metadata(run,local_adoption,raw)
        state=commit_transition(run,state,signature,opponent_pool=resumed_pool,pool_migration=pool_migration,local_scoring_adoption=local_adoption)
    local_cleanup=cleanup_local(root,run,state)
    if local_cleanup:
        print(f'Local scoring adopted at g{local_cleanup["new_origin"]}: retired {local_cleanup["cleanup"]["deleted_files"]} PPG cache files / {local_cleanup["cleanup"]["deleted_bytes"]:,} bytes; next phase g{local_cleanup["next_phase"]}',flush=True)
    if phases is not None:
        cache_cleanup=phases.startup(state['iteration'])
        if cache_cleanup is not None:
            status='PPG fresh reset' if cache_cleanup.get('fresh_ppg') else 'PPG disabled'
            following=(f'next phase g{cache_cleanup["next_phase"]}' if cache_cleanup.get('fresh_ppg') else
                       'no auxiliary replay will be collected')
            print(f'{status}: startup removed {len(cache_cleanup["deleted_files"])} cache files '
                  f'({cache_cleanup["deleted_bytes"]:,} bytes); {following}',flush=True)
    from .champion import initialize as initialize_champion,enabled as champion_enabled,selfplay_policies
    from . import champion_mix
    if champion_enabled(config):initialize_champion(root,run,state)
    window=boost_window(state,config)
    if window:
        print(f'Searched-v4 recovery window: ordinary games {window["start_games"]} to {window["until_games"]}; '
              f'{max(0,window["until_games"]-state["games_done"])} remaining',flush=True)
    from .rehearsal import schedule as rehearsal_schedule
    needs_rehearsal=rehearsal_schedule(config,state['games_done'],state['iteration'])['weight']>0
    rehearsal_index=None;rehearsal=None
    if needs_rehearsal and 'rehearsal_sampling' in config:
        from .rehearsal_window import TrainIndex,window as rehearsal_window
        rehearsal_index=TrainIndex(root,config['rehearsal_manifest'])
        print(f'Rehearsal lightweight index: {rehearsal_index.rows} train decisions, {rehearsal_index.games} games; '
              f'at most {config["rehearsal_sampling"]["rows_per_iteration"]} rows sampled per iteration',flush=True)
    elif needs_rehearsal:
        rehearsal=load_dataset(root,config['rehearsal_manifest'],'train',workers=config['audit_workers'],progress=True,compact=True)[0]
    if rehearsal:
        prepared_at=time.monotonic();rehearsal=rehearsal_dataset(rehearsal,device)
        print(f'Rehearsal tensors prepared once: {len(rehearsal.data)} rows in {time.monotonic()-prepared_at:.1f}s; reused until rehearsal expires',flush=True)
    summaries=[]
    if revised: schedule_evaluations(root,run,state,device,mono,inference_devices=inference_devices)
    if quality:
        from .ratings import schedule as schedule_ratings
        with torch.random.fork_rng(devices=[]):schedule_ratings(root,run,state,config,device=device,mono=mono,inference_devices=inference_devices)
        from .temperature_rating import schedule as schedule_temperature
        with torch.random.fork_rng(devices=[]):schedule_temperature(root,run,state,config,device=device,mono=mono,inference_devices=inference_devices)
    if fixed_probe is not None and 'policy_anchor' in config:
        anchor_values,anchor_transition=policy_anchor.prepare(root,run,state,config,fixed_probe,model)
        policy_anchor.console(anchor_transition)
        exploration_state,floor_transition=exploration.anchor_floor_transition(config,exploration_state,state.get('policy_anchor'))
        if floor_transition:print(f'Entropy floor promotion: {floor_transition["floor_ratio_after"]}; original baseline preserved',flush=True)
    from .ppo_preparation import PreparationPool
    collection_host=config['host']
    if local_scoring is not None and local_scoring.active:
        from .host_runtime import local_scoring_host
        collection_host=local_scoring_host(root)
    with InferenceService(config['execution'],device,profile=run/'profile-collection' if config['profile_steps'] else None,
                          profile_steps=config['profile_steps'],inference_devices=inference_devices) as service,reader_pool(config['audit_workers'],admission=service.cpu) as readers:
        with ActorPool(root,collection_host,service,config['execution'],mono=mono) as ordinary,ActorPool(root,collection_host,service,config['execution'],mono=mono,heavy=True) as heavy,PreparationPool(config['update'].get('preparation_workers',0)) as preparation_pool:
            from .champion import schedule as schedule_champion
            # Resume an interrupted gate for the committed candidate before
            # advancing the learner or writing a new gate specification.
            if champion_enabled(config):
                with torch.random.fork_rng(devices=[]):schedule_champion(root,run,state,service,ordinary)
            for _ in range(iterations):
                iteration=state['iteration'];began=time.monotonic();directory=run/f'iteration-{iteration:04d}'
                from . import v4_share
                share=v4_share.prepare(root,state,config)
                if share is not None and state.get('v4_share_gate')!=share:
                    # Freeze the decision before any ordinary replay is made;
                    # interrupted collections reproduce their exact table plan.
                    state['v4_share_gate']=share
                    write_json(state_path,state)
                inference_before=service.report()
                exploration_settings=exploration.effective(config,exploration_state,iteration)
                current=service.register(model);behavior=Behavior(current,config['exploration'],exploration_settings)
                fingerprints=None
                if 'critic_data' in config:
                    from .critic_data import fingerprint
                    fingerprints=dict(learner=fingerprint(fixed_probe,fixed_probe.probabilities(model,config['update'])),
                        rule_bot=fingerprint(fixed_probe,rule_bot=True))
                    phases.fingerprint=fingerprints['learner']
                from .ordinary import plan_collection,sampling_report
                plan=plan_collection(root,directory,state,config,signature,service,current,
                                     profiles,opponent_specs,fast_specs,window,fingerprints=fingerprints,exploration=exploration_settings)
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
                if normal and ordinary.last_collection_report is not None:
                    diagnostics['exploration_metrics']['branching']['scheduling']=ordinary.last_collection_report
                all_games=games
                sample_groups={g['game_id']:kind_by_group[g['setup_group']] for g in all_games}
                games=[g for g in all_games if 'branch' not in g]
                require(len(games)==config['games_per_iteration'],'Ordinary game count differs')
                if 'critic_data' in config:
                    require(all('_seat_descriptors' in d[0] and d[0].get('_descriptor_generation')==iteration for d in learner_data),
                            'Ordinary collection missing current-generation seat descriptors')
                    phases.snapshot_data=[d for d in learner_data if d[0]['_critic_role']=='snapshot']
                    learner_data=[d for d in learner_data if d[0]['_critic_role']=='learner']
                if phases is not None:phases.collection(records,behavior.policy_id,kind_by_group,current)
                collected=time.monotonic()
                sampling=sampling_report(games,[d for d in learner_data if not d[0].get('_branch')],behavior.policy_id,kind_by_group,config,
                    legacy_profiles={spec['policy_id']:p for specs in (opponent_specs,fast_specs) for p,spec in specs.items()})
                if exploration_settings['version']==3:
                    sampling['exploration_regime']={k:exploration_settings[k] for k in ('version','opening','raw_entropy','branching')}
                next_quality=quality_report=None
                if quality:
                    next_quality,quality_report=quality_pool.update_scores(state['quality_pool'],games,
                        plan['quality_tables'],plan['quality_planning'],generation=iteration+1,
                        games_done=state['games_done']+len(games))
                    if pool_migration and not summaries:quality_report['migration']=pool_migration
                rehearsal_status=rehearsal_schedule(config,state['games_done'],iteration)
                weight=rehearsal_status['weight']
                sampling_finished=time.monotonic();rehearsal_seconds=0.
                with ExitStack() as rehearsal_scope:
                    active_rehearsal=rehearsal
                    if rehearsal_index is not None and weight:
                        active_rehearsal=rehearsal_scope.enter_context(rehearsal_window(root,run,rehearsal_index,
                            config['rehearsal_sampling'],seed=config['seed'],iteration=iteration,device=device,
                            workers=config['audit_workers'],executor=readers))
                    ppo_began=time.monotonic();rehearsal_seconds=ppo_began-sampling_finished
                    update=ppo_update(model,optimizer,all_games,behavior,config['update'],
                        profile=directory/'profile-ppo' if config['profile_steps'] else None,
                        profile_steps=config['profile_steps'],
                        rehearsal=active_rehearsal,rehearsal_weight=weight,learner_data=learner_data,insolvency=config.get('insolvency'),reward=reward,
                        sample_groups=sample_groups,anchor_model=anchor_model,phases=phases,preparation_pool=preparation_pool)
                    if rehearsal_index is not None and weight:update['rehearsal_sampling']=active_rehearsal.report
                    ppo_ended=time.monotonic()
                rehearsal_cleaned=time.monotonic()
                active_rehearsal=None
                next_limit=limit_report=None
                if fixed_probe is not None:
                    from .il_kl_limit import advance
                    counts=diagnostics.get('by_composition',{}).get('selfplay',{}).get('health_counts')
                    if counts and counts['all_learner_tables']!=counts['tables']:counts=None
                    final_measurement=fixed_probe.measure(model,config['update'],reference=anchor_values)
                    next_limit,limit_report=advance(state['il_kl_limit'],final_measurement,
                        counts,generation=iteration+1,warmup_done=update['controller_warmup_done'])
                    policy_anchor.annotate(limit_report,state)
                from .lr_controller import finish_iteration
                active_anchor=(state.get('policy_anchor',{}).get('active') or {}).get('checkpoint',state['evaluation_anchor'])
                update['lr_controller']=finish_iteration(optimizer,update,config,sampling,games,behavior.policy_id,
                    kind_by_group,active_anchor,calibrated_anchor=limit_report['lr_gate'] if limit_report else None)
                ppo_finished=time.monotonic()
                value_report=phases.fit(model,preparation_pool=preparation_pool) if phases is not None else None
                updated=time.monotonic();done=state['games_done']+len(games)
                ordinary_inference=service.report()
                if rehearsal_schedule(config,done,iteration+1)['weight']==0:rehearsal=None
                ordinary_seconds=state.get('ordinary_seconds',0.)+updated-began
                def recalibrate_teacher_utility(auxiliary=None):
                    # Policy distillation moves the shared encoder. Refit only
                    # its utility head against this iteration's fresh ordinary
                    # returns and the explicitly opted-in teacher mixture.
                    from .updates import Dataset
                    from .utility import calibrate
                    return calibrate(model,optimizer,Dataset(learner_data,device,policy=False),config['update'],auxiliary=auxiliary,
                                     indices=phases.learner_train_ids if phases is not None else None)
                from .teacher_collection import due_collections
                from .review_metrics import teacher_drift_probe
                teaching_probe=(teacher_drift_probe(model,learner_data,behavior,config['update'])
                    if due_collections(done,state['next_search_games'],config) else None)
                search,next_search,cursor,search_seconds=collect_due(root,run,directory,config,state,signature,
                    model,optimizer,service,heavy,done=done,cursor=cursor,ordinary_seconds=ordinary_seconds,
                    ordinary_game_seconds=(collected-began)/max(1,len(games)),
                    recalibrate=recalibrate_teacher_utility if reward is not None else None,
                    ordinary_data=([learner_data[i] for i in phases.learner_train_ids] if phases is not None else learner_data),
                    ordinary_policy_id=behavior.policy_id)
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
                teaching_finished=time.monotonic()
                weights_changed=bool((value_report or {}).get('optimizer_steps') or search.get('updated'))
                # The LR decision above retains its exact post-PPO signal.
                # Final limits, quality admission and checkpoint probes include
                # shared-trunk movement from the separate value/teacher phases.
                if fixed_probe is not None and weights_changed:
                    final_measurement=fixed_probe.measure(model,config['update'],reference=anchor_values)
                    next_limit,limit_report=advance(state['il_kl_limit'],final_measurement,
                        counts,generation=iteration+1,warmup_done=update['controller_warmup_done'])
                    policy_anchor.annotate(limit_report,state)
                if phases is not None:
                    from .gae import rating_evidence
                    final_probe=(phases.ppo_probe.measure(model,anchor=anchor_model) if weights_changed
                                 else update['categorical_probe'])
                    update['final_categorical_probe']=dict(final_probe,reused_post_ppo=not weights_changed,
                        scope='Final checkpoint weights after PPO, value phase and any teaching; reference is collection behavior')
                    phases.finish(generation=iteration+1,update=update,
                        anchor_gate=limit_report['lr_gate'] if limit_report else (update.get('lr_controller') or {}).get('health'),
                        rating=rating_evidence(root,state,config['advantage']['gate']),exploration=exploration_settings)
                local_scoring_report=finish_training_diagnostics(model,phases,local_scoring)
                entropy_report=None
                if exploration_state is not None:
                    healthy=bool(limit_report and limit_report['health'].get('ok') is True and
                        update['optimizer_steps']>0 and not update['kl_early_stop'] and
                        not (value_report or {}).get('kl_guard_tripped',False) and
                        update['categorical_probe']['categorical_kl']<=1.5*config['update']['target_kl'])
                    exploration_state,entropy_report=exploration.advance(exploration_state,config,final_measurement,
                        generation=iteration+1,warmup_done=update['controller_warmup_done'],healthy=healthy,
                        policy_anchor=state.get('policy_anchor'))
                del learner_data
                output=run/f'generation-{iteration+1:04d}.pt'
                if quality:
                    from brass_ai.learning.batching import model_id
                    pending_entry=dict(path=output.relative_to(root).as_posix(),sha256='pending',
                        model_id=model_id(model),generation=iteration+1)
                    next_quality,pool_admission=quality_pool.finish(root,run,next_quality,model,pending_entry,
                        fixed_probe,config['update'],limit_report)
                    quality_pool.finish_report(quality_report,next_quality,pool_admission)
                diagnostics_finished=time.monotonic()
                report=dict(format='step5-recovery-report/'+'2.2.0',iteration=iteration,composition=dict(composition),
                    v4_share_gate=v4_share.report(share,composition),
                    panel_gates=copy.deepcopy(state.get('ratings',{}).get('panel_gates')),
                    policy_anchor=copy.deepcopy(state.get('policy_anchor')),
                    rating_coverage=copy.deepcopy(state.get('ratings',{}).get('coverage')),
                    exploration_rollout=exploration_settings,entropy_control=entropy_report,
                    quality_pool=quality_report,
                    il_kl_limit=limit_report,rehearsal_schedule=rehearsal_status,
                    sampling=sampling,curriculum=curriculum_report(state['games_done'],done,config) if latest_model_recipe(config) else None,
                    reward_migration=migration if not summaries else None,
                    optimizer_resume=optimizer_resume if not summaries else None,
                    legacy_boost=boost_report(state['games_done'],done,config,window),
                    actor=update,value_phase=value_report,search=search,**diagnostics,incoming_model_id=current,incoming_policy_id=behavior.policy_id,
                    seat_fingerprints=fingerprints,
                    selfplay_incumbent=incumbent,
                    selfplay_opponents=champion_mix.summarize(state,config,mix_receipts) if champion_mix.enabled(config) else None,
                    games_done=done,cumulative_games=state['previous_games']+done,
                    timings=dict(collection_seconds=load_began-began,
                        sampling_seconds=sampling_finished-collected,
                        rehearsal_prepare_seconds=rehearsal_seconds,ppo_seconds=ppo_ended-ppo_began,
                        rehearsal_cleanup_seconds=rehearsal_cleaned-ppo_ended,
                        post_ppo_diagnostics_seconds=ppo_finished-rehearsal_cleaned,
                        value_phase_seconds=updated-ppo_finished,
                        replay_load_seconds=collected-load_began,
                        search_and_distill_seconds=teaching_finished-updated,
                        final_diagnostics_seconds=diagnostics_finished-teaching_finished),execution=records,
                    inference=service.report(),inference_scope='cumulative for this invocation',
                    legacy_modes=dict(Counter(j['provenance']['legacy_mode'] for j in jobs if j['provenance']['legacy_mode'])),
                    historical_opponents=dict(generation_seats=dict(sorted(historical_generations.items())),
                        pool_generations=[e['generation'] for e in state.get('population',[])],
                        sampling=(quality_pool.sampling_description(next_quality)
                                  if quality else '25% initial/IL, 25% recent, 50% older reservoir; unavailable strata fall back')),
                    environment=environment(device),strength='requires fresh post-update evaluation')
                if share is not None:
                    report['curriculum'].update(effective_tables=dict(composition),
                        v4_share_reduced=share['activated'] is not None)
                report['utilization']=iteration_utilization(inference_before,ordinary_inference,records,
                    load_began-began,len(games),config['execution'])
                report['utilization']['replay_ingress']=ingress_metrics
                from .review_metrics import gate_diagnostics
                report['gate_diagnostics']=gate_diagnostics(update,value_report,limit_report,quality_report)
                if local_scoring_report is not None:
                    report['local_scoring_auxiliary']=local_scoring_report
                if state.get('local_scoring_adoption'):report['local_scoring_adoption']=state['local_scoring_adoption']
                checkpoint_began=time.monotonic()
                anchor_commit=policy_anchor.checkpoint_state(state)
                identity=save_checkpoint(output,model,optimizer,step=iteration+1,
                    training=dict(mode='step5-population-ppo',recipe=update['format'],
                        parent=current,signature=digest(signature),fresh_run=run.relative_to(root).as_posix(),initialization='random-lineage',
                        evidence=dict(iteration=iteration,games_done=done,report=(directory/'report.json').relative_to(root).as_posix(),
                                      update_digest=digest(update)),
                        **({'il_kl_limit':next_limit} if next_limit is not None else {}),
                        **({'policy_anchor':anchor_commit} if anchor_commit is not None else {}),
                        **({'exploration':exploration_state} if exploration_state is not None else {}),
                        **({'v4_share_gate':share} if share is not None else {}),
                        **({'actor_value_phases':phases.payload()} if phases is not None else {}),
                        **({'local_scoring_auxiliary':local_scoring.payload(iteration+1)} if local_scoring is not None else {}),
                        **({'quality_pool_digest':quality_pool.state_digest(next_quality)} if quality else {})))
                report['post_update_model_id']=identity
                report['timings']['checkpoint_seconds']=time.monotonic()-checkpoint_began
                write_json(directory/'report.json',report)
                entry=dict(path=output.relative_to(root).as_posix(),sha256=sha(output),model_id=identity,generation=iteration+1)
                state.update(iteration=iteration+1,games_done=done,next_seed=cursor,context_counter=context_counter,
                    history=state['history']+[entry],next_search_games=next_search,
                    search_seconds=search_seconds,ordinary_seconds=ordinary_seconds)
                if next_limit is not None:state['il_kl_limit']=next_limit
                if anchor_commit is not None:state['policy_anchor']=anchor_commit
                if exploration_state is not None:state['exploration']=exploration_state
                if quality:
                    for member in next_quality['members']:
                        if member['generation']==iteration+1:member.update(entry)
                    state['quality_pool']=next_quality
                if champion_mix.enabled(config):state['champion_mix']['transition_tables']+=transition_added
                state['recent_training']=(state.get('recent_training',[])+[dict(iteration=iteration,games_done=done,
                    v4_share_gate=report['v4_share_gate'],
                    panel_gates=report['panel_gates'],
                    policy_anchor=report['policy_anchor'],rating_coverage=report['rating_coverage'],
                    exploration_rollout=exploration_settings,entropy_control=entropy_report,
                    exploration_metrics=diagnostics.get('exploration_metrics'),
                    quality_pool=quality_report,
                    il_kl_limit=limit_report,rehearsal_schedule=rehearsal_status,
                    value_phase=value_report,
                    local_scoring_auxiliary=report.get('local_scoring_auxiliary'),
                    health_counts=diagnostics.get('health_counts'),
                    health_by_composition={k:v.get('health_counts') for k,v in diagnostics.get('by_composition',{}).items()},
                    sampling=sampling,curriculum=report['curriculum'],
                    learning_rate=config['learning_rate'],learning_rate_schedule=update.get('learning_rate_schedule'),
                    actor={k:v for k,v in update.items() if k not in ('timings','memory')},
                    search={k:v for k,v in search.items() if k not in ('collection','updates')})])[-2:]
                if revised:
                    from .retention import update_population
                    update_population(root,run,state,config['population'])
                write_json(state_path,state)
                if local_scoring is not None:local_scoring.committed(iteration+1)
                if phases is not None:phases.committed(iteration+1)
                if revised:
                    evaluation_began=time.monotonic()
                    from .champion import schedule as schedule_champion
                    if champion_enabled(config):
                        with torch.random.fork_rng(devices=[]):
                            report['champion_gate']=schedule_champion(root,run,state,service,ordinary)
                    schedule_evaluations(root,run,state,device,mono,inference_devices=inference_devices)
                    if quality:
                        with torch.random.fork_rng(devices=[]):
                            report['strength_rating']=schedule_ratings(root,run,state,config,device=device,mono=mono,inference_devices=inference_devices)
                            report['temperature_rating']=schedule_temperature(root,run,state,config,device=device,mono=mono,inference_devices=inference_devices)
                        if fixed_probe is not None and 'policy_anchor' in config:
                            anchor_values,anchor_transition=policy_anchor.prepare(root,run,state,config,fixed_probe,model)
                            policy_anchor.console(anchor_transition)
                            report['anchor_transition']=anchor_transition
                            exploration_state,floor_transition=exploration.anchor_floor_transition(config,exploration_state,state.get('policy_anchor'))
                            if floor_transition:report['entropy_floor_transition']=floor_transition
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
                control=update.get('lr_controller')
                controller_line=(f' LRmult={control["multiplier_after"]:.3f} {control["mode"]}:{control["decision"]}' if control else '')
                if share is not None:
                    controller_line+=f' v4={composition["legacy"]}/{len(games)}[{"reduced" if share["activated"] else "base"}]'
                if entropy_report:
                    controller_line+=f' q=3 eps={config["exploration"]:.2f} explore_age={exploration_settings["age"]}'
                    controller_line+=' entropy='+','.join(f'{g}:{v["alpha_after"]:.4f}' for g,v in entropy_report['groups'].items())
                    if 'floor_ratio_by_group' in entropy_report:
                        controller_line+=' Hfloor='+','.join(f'{g}:{v:.2f}' for g,v in entropy_report['floor_ratio_by_group'].items())+'*baseline'
                        controller_line+=('(anchor-promotions='+str(entropy_report['floor_anchor_promotions'])+')'
                            if 'floor_anchor_promotions' in entropy_report else '(fixed)')
                        if report.get('entropy_floor_transition'):
                            controller_line+=' next_Hfloor='+','.join(f'{g}:{v:.2f}' for g,v in report['entropy_floor_transition']['floor_ratio_after'].items())
                    elif entropy_report['floor_decay_per_iteration']==1.:
                        controller_line+=f' Hfloor={entropy_report["floor_start_ratio"]:.2f}*baseline(fixed)'
                if phases is not None:
                    gate=update['gae']['gate']
                    controller_line+=f' GAE={gate["lambda_used"]:.2f}->{gate["lambda_next"]:.2f} value_steps={value_report["optimizer_steps"]}'
                    controller_line+=f'/{value_report["attempted_steps"]} value_stop={value_report["early_stop_reason"] or "none"}'
                    controller_line+=(f' value_lr={value_report["learning_rate"]:.2g}' if value_report['enabled'] else ' PPG=off')
                    if (value_report.get('learning_rate_source') or {}).get('source')=='ppo-last-applied':
                        controller_line+=f' value_peak={value_report["peak_learning_rate"]:.2g}[PPO]'
                    if value_report.get('beta_mode')=='adaptive-up-down':
                        controller_line+=f' PPGbeta={value_report["beta_clone_final"]:g}->{value_report["beta_clone_next"]:g}'
                    if update.get('value_gradient',{}).get('mode')=='detach-encoder':controller_line+=' critic=PPO-heads/PPG-shared'
                    if update.get('value_preservation'):
                        preservation=update['value_preservation']
                        controller_line+=f' critic=preserve Vkeep={preservation["end_weighted_objective"]:.5f} Vdrift={preservation["utility_probe_rms"]:.5f}'
                    if value_report.get('schedule'):
                        controller_line+=(f' PPG={"phase" if value_report["scheduled"] else "waiting"}'
                            f' next=g{value_report["schedule"]["next_phase"]}'
                            f' ages={len(value_report["buffer"]["ages"])}'
                            f' budget={value_report.get("planned_steps",0)}steps'
                            f' epochs={value_report["epoch_controller"]["cap_used"]}->{value_report["epoch_controller"]["cap_next"]}'
                            f' epoch_gate={value_report["epoch_controller"]["decision"]}')
                    if value_report.get('acceptance_steps'):
                        controller_line+=f' value_intervals={value_report["accepted_intervals"]}/{len(value_report["interval_reports"])}'
                    stage=gate.get('stage_evidence')
                    if stage:
                        controller_line+=(f' GAE-health={gate["healthy_streak"]}/{stage["required_iterations"]}'
                            f' {stage.get("source","ppg").upper()}-critic-window={stage["accepted_steps"]}steps/{stage["accepted_iterations"]}iters')
                    split=value_report.get('split') or {}
                    v4_holdout=split.get('by_table',{}).get('legacy')
                    if v4_holdout:controller_line+=f' v4_holdout={v4_holdout["heldout_games"]}/{v4_holdout["games"]}'
                    if phases.buffer:
                        controller_line+=f' value_rows={value_report["distinct_current_rows_used"]}+{value_report["distinct_previous_rows_used"]}'
                    if phases.critic_state is not None:
                        controller_line+=f' snapshots={int(phases.critic_state["snapshots_active"])}@{phases.critic_state["snapshot_weight"]:.2f}'
                        controller_line+=critic_data.recovery_console(value_report.get('critic_rollout',{}))
                        controller_line+=' GAE-mixed='+','.join(f'{k}:{v["lambda_next"]:.2f}[{v["streak"]}/10]'
                            +(f'({v["evidence"]["mixed"]["games"]}g/{v["evidence"]["minimum_games"]},'
                              f'{v["evidence"]["contributing_iterations"]}i)' if 'evidence' in v else '')
                            for k,v in gate['table_gate'].items())
                if limit_report:
                    controller_line+=(f' AnchorKL[g{limit_report["anchor_reference"]["generation"]}]='
                        f'{limit_report["measurement"]["kl_mean"]:.4f}/{limit_report["limit_after"]:.4f}'
                        f' OrigILKL={limit_report["original_il_measurement"]["kl_mean"]:.4f}'
                        f' group_limit={limit_report["group_limit_after"]:.4f} {limit_report["decision"]}'
                        f' LRraise={int(limit_report["lr_increases_permitted_by_anchor"])}')
                    transition=report.get('anchor_transition') or {}
                    if transition.get('decision')=='promote':
                        controller_line+=f' next_anchor=g{transition["reference_generation"]}'
                    checks=report['gate_diagnostics']
                    controller_line+=f' health={"ok" if checks["health_ok"] else ",".join(checks["health_reasons"])}'
                if quality_report:
                    controller_line+=f' history={quality_report["size"]}/{quality_report["max_members"]} {quality_report["admission"]["reason"]}'
                    controller_line+=quality_pool.pfsp_pool.console(quality_report)
                from .review_metrics import mixed_ev_console
                controller_line+=mixed_ev_console(report['gate_diagnostics'])
                if report.get('local_scoring_auxiliary'):
                    aux=report['local_scoring_auxiliary']
                    controller_line+=(f' local={aux["mode"]}@{aux["shared_gradient_scale"]:.2f}'
                        f' rows={aux["retained"]["rows"]}/{aux["available"]["rows"]}'
                        f' head_steps={aux["companion_steps"]["ppo"]}+{aux["companion_steps"]["ppg"]}')
                print(f'Recovery generation {iteration+1}: {done} games, {update["optimizer_steps"]} PPO steps; '
                      f'collect={load_began-began:.1f}s prepare={collected-load_began:.1f}s '
                      f'rehearsal={rehearsal_seconds:.1f}s PPO={report["timings"]["ppo_seconds"]:.1f}s '
                      f'value_phase={report["timings"]["value_phase_seconds"]:.1f}s '
                      f'teacher={report["timings"]["search_and_distill_seconds"]:.1f}s '
                      f'diagnostics={report["timings"]["post_ppo_diagnostics_seconds"]+report["timings"]["final_diagnostics_seconds"]:.1f}s '
                      f'eval={report["timings"].get("evaluation_seconds",0):.1f}s cleanup={report["timings"].get("cleanup_seconds",0):.1f}s; '
                      f'total={report["timings"]["iteration_seconds"]:.1f}s; '
                      f'NN batch mean={report["utilization"]["mean_inference_batch"]:.2f}; '
                      f'LR={update["learning_rates"][0]:.1e}{controller_line} '
                      f'{"KLT1" if "opening_objective" in update else "KLcat"}={update.get("categorical_probe",{}).get("categorical_kl",0.):.5f} '
                      + (f'KLbehavior={update["categorical_probe"]["behavior_categorical_kl"]:.5f} ' if 'opening_objective' in update else '') +
                      f'KLsample={update["epochs"][-1]["post_update_kl"]:.5f} '
                      f'KLstop={update.get("kl_guard",{}).get("stop_reason") or "none"} '
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
