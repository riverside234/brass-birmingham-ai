"""Bounded independent teacher collections after an ordinary PPO update."""
import random
import time

from brass_ai.bank import sha
from brass_ai.learning.contracts import digest,project_path,require
from brass_ai.step5.league import seeds_for
from .population import rollout_opponents
from .replay import read_shards
from .teacher_schedule import CONTRACT,PHASES,collection_report,estimate_collection,search_allowance
from .updates import distill_update


def due_collections(done,next_search,config):
    if not config.get('teacher_enabled',True):return 0
    interval=config['search_every_games']
    return min(config.get('search_collections_per_iteration',1),
        1+(done-next_search)//interval) if interval and done>=next_search else 0


def summarize(batches):
    if not batches:return dict(updated=False,reason='Persistent search schedule not due',collection_count=0)
    totals={k:sum(b.get(k,0) for b in batches) for k in
        ('optimizer_steps','sample_presentations','teacher_rows','teacher_setups','requested_steps',
         'distinct_labels_used','unused_qualified_labels')}
    totals.update(updated=any(b['updated'] for b in batches),collection_count=len(batches),
        max_observed_reuse=max(b.get('max_observed_reuse',0) for b in batches),
        value_loss_training=any(b.get('value_loss_training',False) for b in batches),
        samples_by_phase={p:sum(b['samples_by_phase'][p] for b in batches) for p,_,_,_ in PHASES},
        collection=[r for b in batches for r in b['collection']],
        updates=[r for b in batches for r in b.get('updates',[])],
        collections=[{k:v for k,v in b.items() if k not in ('collection','updates')} for b in batches])
    totals['timings']={key:sum(b.get('timings',{}).get(key,0.) for b in batches)
                      for key in ('setup_seconds','collection_seconds','prepare_seconds','distill_seconds',
                                  'post_distill_calibration_seconds','total_seconds')}
    for key in ('used_samples_by_phase','unused_samples_by_phase','sample_presentations_by_phase'):
        totals[key]={p:sum(b.get(key,{}).get(p,0) for b in batches) for p,_,_,_ in PHASES}
    for key in ('quality','learning_rates'):
        if key in batches[-1]:totals[key]=batches[-1][key]
    first=batches[0]['teaching']
    totals['teaching']=dict(first,phases={p:{k:sum(b['teaching']['phases'][p][k] for b in batches)
        for k in first['phases'][p]} for p in first['phases']})
    fitted=[b for b in batches if 'fit_before' in b]
    if fitted:
        for field in ('fit_before','fit_after'):
            totals[field]={k:sum(b[field][k]*b['teacher_rows'] for b in fitted)/sum(b['teacher_rows'] for b in fitted)
                for k in fitted[0][field]}
        totals['fit_scope']='Sample-weighted separate collection fits; not one fixed evaluation set or held-out strength'
    auxiliary=[b['teacher_value'] for b in batches if b.get('teacher_value',{}).get('enabled')]
    if auxiliary:
        totals['teacher_value']=dict(enabled=True,collections=len(auxiliary),
            **{k:sum(a[k] for a in auxiliary) for k in ('distinct_completed_games','valid_value_positions',
                'selected_positions','distinct_positions_trained','unused_positions','sample_presentations',
                'auxiliary_training_seconds','preparation_seconds')},
            phases={p:{k:sum(a['phases'][p][k] for a in auxiliary) for k in auxiliary[0]['phases'][p]} for p,_,_,_ in PHASES},
            scope='Totals across distinct collections; per-collection fixed-probe stages below')
        for key in ('auxiliary_training_seconds','teacher_value_diagnostics_seconds','teacher_value_prepare_seconds'):
            totals['timings'][key]=sum(b.get('timings',{}).get(key,0.) for b in batches)
    return totals


def collect_due(root,run,directory,config,state,signature,model,optimizer,service,heavy,*,
                done,cursor,ordinary_seconds,ordinary_game_seconds,recalibrate=None,ordinary_data=None,ordinary_policy_id=None):
    require(recalibrate is None or callable(recalibrate),'Invalid ordinary-data calibration callback')
    next_search=state['next_search_games'];search_seconds=state.get('search_seconds',0.)
    if not config.get('teacher_enabled',True):
        return (dict(enabled=False,updated=False,reason='Teacher search and distillation disabled by configuration',
                    collection_count=0,optimizer_steps=0,teacher_rows=0,
                    teacher_value=dict(enabled=False,reason='No teacher collection while teacher_enabled=false')),
                    next_search,cursor,search_seconds)
    estimate=estimate_collection(config,ordinary_game_seconds);batches=[];deferred_reason=None
    # This is an admission cost estimate, never a thinking-time limit. After
    # the first collection, use its measured total cost plus 25% headroom.
    # Bind it to the recipe so changed time/count contracts cannot reuse it.
    recipe=digest(dict(work=config.get('teacher_work_budget'),search=config['search'],
        games=config['search_games'],execution=config['execution'],teacher_value=config.get('teacher_value')))
    from .config import teacher_value_config
    auxiliary_config=teacher_value_config(config.get('teacher_value'))
    if auxiliary_config['enabled']:
        require(ordinary_data and ordinary_policy_id and recalibrate is not None,
                'Teacher value requires this iteration\'s ordinary learner data and matching calibration')
    previous_cost=state.get('teacher_work_cost')
    if previous_cost and previous_cost['recipe']==recipe:
        estimate=max(estimate,1.25*previous_cost['seconds'])
    allowance_before=search_allowance(ordinary_seconds,search_seconds,config['search_wall_fraction'])
    due=due_collections(done,next_search,config)
    from .config import search_workers_per_turn
    workers=search_workers_per_turn(config['execution'],teacher=True)
    parallel_games=min(config['search_games'],config['execution']['heavy_processes'])
    def budget_report(deferred):
        return dict(wall_fraction=config['search_wall_fraction'],available_seconds=allowance_before,
            search_workers_per_game=workers,parallel_game_capacity=parallel_games,
            total_search_worker_capacity=parallel_games*workers,
            wall_fraction_enabled=config['search_wall_fraction'] is not None,
            remaining_seconds=search_allowance(ordinary_seconds,search_seconds,config['search_wall_fraction']),
            estimated_collection_seconds=estimate,admitted=bool(batches),due_collections=due,
            max_collections=config.get('search_collections_per_iteration',1),deferred=deferred)
    if getattr(model.config,'utility',None) is not None:
        from .utility import calibration_ready
        require(model.config.utility==config['search'].get('reward'),'Teacher/model reward differs')
        if not calibration_ready(model):
            return dict(updated=False,reason='Utility head calibration not ready; collect fresh ordinary rollouts first',
                        collection_count=0,budget=budget_report(True)),next_search,cursor,search_seconds
    for batch_index in range(due):
        # A successful policy distillation can invalidate the shared critic.
        # Re-check before reserving seeds, time or the next scheduled allocation.
        if getattr(model.config,'utility',None) is not None and not calibration_ready(model):
            deferred_reason='Utility calibration invalidated by distillation; postpone remaining collections until fresh ordinary PPO calibration'
            break
        available=search_allowance(ordinary_seconds,search_seconds,config['search_wall_fraction'])
        if available is not None and estimate>available:
            deferred_reason='Search wall-time allowance exceeded; postpone remaining bounded collections'
            break
        began=time.monotonic();identity=service.register(model)
        auxiliary_proof=None
        if auxiliary_config['enabled']:
            from .teacher_value import provenance as value_provenance
            auxiliary_proof=value_provenance(identity,model.config.utility,auxiliary_config)
        rng=random.Random(digest(dict(seed=config['seed'],done=done,threshold=next_search)))
        rollout_entries=rollout_opponents(state.get('population',state['history']),identity,
            state['evaluation_anchor'],rng,config['population']['recent'])
        # Keep the accepted incumbent represented even after it leaves the
        # general history reservoir. Labels still come from the current learner.
        if state.get('champion') and state['champion']['model_id']!=identity:
            rollout_entries=list({e['model_id']:e for e in rollout_entries+[state['champion']]}.values())
        for e in rollout_entries:
            path=project_path(root,e['path']);require(path.is_relative_to(run) and sha(path)==e['sha256'],'Imported/changed rollout model')
            if e['model_id'] not in service.models:service.register_checkpoint(path)
        rollout_ids=list(dict.fromkeys([identity]+[e['model_id'] for e in rollout_entries]))
        jobs=[]
        for i in range(config['search_games']):
            chosen,cursor=seeds_for(cursor,1);seed=chosen[0]
            index=batch_index*config['search_games']+i
            spec=dict(kind='masked_ismcts',model_id=identity,opponents=rollout_ids,search=config['search'],focus='neutral',
                cpu_role=config.get('legacy_budget',{}).get('role','reference'),teacher_schedule=CONTRACT)
            if config.get('teacher_work_budget') is not None:
                spec['teacher_work_budget']=config['teacher_work_budget']
            provenance=dict(producer='recovery/2.0.0',purpose='training',kind='masked-ismcts-teacher',
                seed=seed,sampling_seed=config['seed']*13+done+index,spec=digest(signature))
            if auxiliary_proof is not None:provenance['teacher_value']=auxiliary_proof
            jobs.append(dict(index=index,seed=seed,rotation=0,sampling_seed=provenance['sampling_seed'],
                policies=[spec]*4,path=str(directory/f'search-{index:06d}.jsonl.gz'),provenance=provenance))
        work=config.get('teacher_work_budget')
        from .config import teacher_time_seconds
        budget_label=(f'{ {p:teacher_time_seconds(work,p) for p,_,_,_ in PHASES} } seconds per shared whole turn' if work and work['mode']=='time' else
                      f'whole-turn simulation attempts {work["simulations_per_phase"]}, '
                      f'{work.get("watchdog_seconds",config["search"]["seconds"]):g}s stall watchdog' if work else
                      f'{config["search"]["seconds"]:g}s per turn')
        print(f'Teacher collection {batch_index+1}/{due}: due at {next_search} ordinary games, '
              f'{len(jobs)} games, {workers} search workers/game ({parallel_games*workers} total capacity), {budget_label}',flush=True)
        collection_started=time.monotonic()
        records=heavy.collect(jobs)
        collected=time.monotonic()
        games=read_shards([r['path'] for r in records],partition='train',receipts={r['path']:r.get('features') for r in records})
        teaching=collection_report(records,games)
        auxiliary=None
        if auxiliary_config['enabled']:
            from .teacher_value import TeacherValue,admit_collection
            bindings=admit_collection(games,records,jobs,auxiliary_proof)
            auxiliary=TeacherValue(model,games,ordinary_data,auxiliary_config,config['update'],bindings,
                                   ordinary_policy_id=ordinary_policy_id)
        prepared=time.monotonic()
        batch=distill_update(model,optimizer,games,config['update'],steps=config['distill_steps'],
                            weight=config['distill_weight'],quality=config.get('distillation'),
                            **(dict(auxiliary=auxiliary) if auxiliary is not None else {}))
        distilled=time.monotonic()
        if batch['updated'] and getattr(model.config,'utility',None) is not None and not calibration_ready(model) and recalibrate is not None:
            # The default refits on fresh ordinary data. The explicit auxiliary
            # opt-in adds its admitted teacher targets with matching weights.
            # Another due collection still requires a successful readiness fit.
            calibration_started=time.monotonic()
            batch['post_distill_calibration']=recalibrate(auxiliary) if auxiliary is not None else recalibrate()
            batch['post_distill_calibration_seconds']=time.monotonic()-calibration_started
        if auxiliary is not None:batch['teacher_value']=auxiliary.report()
        elapsed=time.monotonic()-began;search_seconds+=elapsed
        if not any(r.get('cached') for r in records):
            state['teacher_work_cost']=dict(recipe=recipe,seconds=elapsed)
            estimate=max(estimate,1.25*elapsed)
        batch.update(collection=records,teaching=teaching,due_games=next_search,elapsed_seconds=elapsed,
            timings=dict(setup_seconds=collection_started-began,collection_seconds=collected-collection_started,
                         prepare_seconds=prepared-collected,distill_seconds=distilled-prepared,
                         post_distill_calibration_seconds=batch.get('post_distill_calibration_seconds',0.),
                         total_seconds=elapsed),
            teacher_model_id=identity,rollout_generations=[e['generation'] for e in rollout_entries])
        if auxiliary is not None:
            batch['timings'].update(auxiliary_training_seconds=auxiliary.training_seconds,
                teacher_value_prepare_seconds=auxiliary.preparation_seconds,
                teacher_value_diagnostics_seconds=auxiliary.probe.seconds if auxiliary.probe else 0.)
        batches.append(batch)
        # Failed/insufficient teaching still consumes this collection's allocation.
        # Never run extra games or retry to manufacture the minimum label count.
        next_search+=config['search_every_games']
    report=summarize(batches)
    report.setdefault('teacher_value',dict(enabled=auxiliary_config['enabled'],
        reason='No teacher value collection updated' if auxiliary_config['enabled'] else 'Opt-in auxiliary value disabled'))
    if deferred_reason:report['reason']=deferred_reason
    report['budget']=budget_report(bool(deferred_reason))
    return report,next_search,cursor,search_seconds
