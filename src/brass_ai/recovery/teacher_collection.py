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
    interval=config['search_every_games']
    return min(config.get('search_collections_per_iteration',1),
        1+(done-next_search)//interval) if interval and done>=next_search else 0


def summarize(batches):
    if not batches:return dict(updated=False,reason='Persistent search schedule not due',collection_count=0)
    totals={k:sum(b.get(k,0) for b in batches) for k in
        ('optimizer_steps','sample_presentations','teacher_rows','teacher_setups','requested_steps')}
    totals.update(updated=any(b['updated'] for b in batches),collection_count=len(batches),
        max_observed_reuse=max(b.get('max_observed_reuse',0) for b in batches),value_loss_training=False,
        samples_by_phase={p:sum(b['samples_by_phase'][p] for b in batches) for p,_,_,_ in PHASES},
        collection=[r for b in batches for r in b['collection']],
        updates=[r for b in batches for r in b.get('updates',[])],
        collections=[{k:v for k,v in b.items() if k not in ('collection','updates')} for b in batches])
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
    return totals


def collect_due(root,run,directory,config,state,signature,model,optimizer,service,heavy,*,
                done,cursor,ordinary_seconds,ordinary_game_seconds,recalibrate=None):
    require(recalibrate is None or callable(recalibrate),'Invalid ordinary-data calibration callback')
    next_search=state['next_search_games'];search_seconds=state.get('search_seconds',0.)
    estimate=estimate_collection(config,ordinary_game_seconds);batches=[];deferred_reason=None
    allowance_before=search_allowance(ordinary_seconds,search_seconds,config['search_wall_fraction'])
    due=due_collections(done,next_search,config)
    if getattr(model.config,'utility',None) is not None:
        from .utility import calibration_ready
        require(model.config.utility==config['search'].get('reward'),'Teacher/model reward differs')
        if not calibration_ready(model):
            return dict(updated=False,reason='Utility head calibration not ready; collect fresh ordinary rollouts first',collection_count=0),next_search,cursor,search_seconds
    for batch_index in range(due):
        # A successful policy distillation can invalidate the shared critic.
        # Re-check before reserving seeds, time or the next scheduled allocation.
        if getattr(model.config,'utility',None) is not None and not calibration_ready(model):
            deferred_reason='Utility calibration invalidated by distillation; postpone remaining collections until fresh ordinary PPO calibration'
            break
        available=search_allowance(ordinary_seconds,search_seconds,config['search_wall_fraction'])
        if estimate>available:
            deferred_reason='Search wall-time allowance exceeded; postpone remaining bounded collections'
            break
        began=time.monotonic();identity=service.register(model)
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
            provenance=dict(producer='recovery/2.0.0',purpose='training',kind='masked-ismcts-teacher',
                seed=seed,sampling_seed=config['seed']*13+done+index,spec=digest(signature))
            jobs.append(dict(index=index,seed=seed,rotation=0,sampling_seed=provenance['sampling_seed'],
                policies=[spec]*4,path=str(directory/f'search-{index:06d}.jsonl.gz'),provenance=provenance))
        print(f'Teacher collection {batch_index+1}/{due}: due at {next_search} ordinary games, '
              f'{len(jobs)} games, {config["search"]["seconds"]:g}s per turn',flush=True)
        records=heavy.collect(jobs)
        games=read_shards([r['path'] for r in records],partition='train',receipts={r['path']:r.get('features') for r in records})
        teaching=collection_report(records,games)
        batch=distill_update(model,optimizer,games,config['update'],steps=config['distill_steps'],
                            weight=config['distill_weight'],quality=config.get('distillation'))
        if batch['updated'] and getattr(model.config,'utility',None) is not None and not calibration_ready(model) and recalibrate is not None:
            # The caller supplies only this iteration's fresh ordinary learner
            # targets, never teacher terminal outcomes. This bounded fit keeps
            # another due collection/checkpoint usable without a stale critic.
            calibration_started=time.monotonic()
            batch['post_distill_calibration']=recalibrate()
            batch['post_distill_calibration_seconds']=time.monotonic()-calibration_started
        elapsed=time.monotonic()-began;search_seconds+=elapsed
        batch.update(collection=records,teaching=teaching,due_games=next_search,elapsed_seconds=elapsed,
            teacher_model_id=identity,rollout_generations=[e['generation'] for e in rollout_entries])
        batches.append(batch)
        # Failed/insufficient teaching still consumes this collection's allocation.
        # Never run extra games or retry to manufacture the minimum label count.
        next_search+=config['search_every_games']
    report=summarize(batches)
    if deferred_reason:report['reason']=deferred_reason
    report['budget']=dict(wall_fraction=config['search_wall_fraction'],available_seconds=allowance_before,
        remaining_seconds=search_allowance(ordinary_seconds,search_seconds,config['search_wall_fraction']),
        estimated_collection_seconds=estimate,admitted=bool(batches),due_collections=due,
        max_collections=config.get('search_collections_per_iteration',1),deferred=bool(deferred_reason))
    return report,next_search,cursor,search_seconds
