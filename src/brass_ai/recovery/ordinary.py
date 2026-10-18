"""Frozen ordinary collection plans and explicit learner sample attribution."""
from collections import Counter
import random

from brass_ai.bank import sha
from brass_ai.learning.contracts import digest,project_path,require
from brass_ai.step5.league import seeds_for
from brass_ai.step5.policy import INDUSTRIES
from .config import table_rotation,latest_model_recipe,learner_seats_per_table,quality_selfplay_recipe
from .curriculum import training_table,searched_legacy
from .execution import neural_spec
from .population import sample_opponents
from .champion import enabled as champion_enabled,selfplay_policies
from . import champion_mix


def plan_collection(root,directory,state,config,signature,service,current,profiles,opponent_specs,fast_specs,window,*,fingerprints=None):
    incumbent=state.get('champion') if champion_enabled(config) else None
    if incumbent:
        path=project_path(root,incumbent['path'])
        require(sha(path)==incumbent['sha256'],'Champion checkpoint changed')
        if incumbent['model_id'] not in service.models:service.register_checkpoint(path)
    snapshots={}
    verified_snapshots=set()
    for entry in state.get('population',state['history']):
        snapshots[entry['model_id']]=entry
    jobs=[];kind_by_group={};composition=Counter();cursor=state['next_seed'];context_counter=state['context_counter']
    historical_generations=Counter();mix_receipts=[];transition_added=0
    quality=quality_selfplay_recipe(config);quality_tables={};quality_planning=None
    if quality:
        from . import quality_pool
        quality_pool.validate_state(state['quality_pool'],config['opponent_pool'],state['iteration'])
        quality_planning=quality_pool.distribution(state['quality_pool'])
    for i in range(config['games_per_iteration']):
        index=state['games_done']+i;kind,boosted=training_table(index,config,window)
        selected,cursor=seeds_for(cursor,1);seed=selected[0]
        rng=random.Random(digest(dict(seed=config['seed'],index=index)))
        learner=neural_spec(current,config['exploration'],'mixed')
        learner_count=learner_seats_per_table(config,kind)
        mix_receipt=None;quality_receipts=[]
        if kind=='selfplay' and champion_mix.enabled(config):
            entries,mix_receipt=champion_mix.opponents(state,config,rng,current_id=current,
                transition_index=state['champion_mix']['transition_tables']+transition_added)
            for entry in entries:
                if entry is None:continue
                path=project_path(root,entry['path'])
                if entry['model_id'] not in verified_snapshots:
                    require(sha(path)==entry['sha256'],'Champion rollout checkpoint changed')
                    verified_snapshots.add(entry['model_id'])
                if entry['model_id'] not in service.models:service.register_checkpoint(path)
            policies=champion_mix.policies(learner,entries,[fast_specs[p] for p in profiles[:2]])
            mix_receipts.append(mix_receipt)
            transition_added+=int(mix_receipt['stage']=='balanced_transition')
        elif kind=='selfplay': policies=selfplay_policies(learner,incumbent) if incumbent else [learner]*4
        elif kind=='historical':
            if quality:
                draws=[quality_pool.draw(state['quality_pool'],quality_planning,rng) for _ in range(2)]
                candidates=[d[0] for d in draws];quality_receipts=[d[1] for d in draws]
            else:
                candidates=sample_opponents(snapshots.values(),current,state['evaluation_anchor'],rng,config['population']['recent'],count=4-learner_count)
            opponents=[]
            for entry in candidates:
                historical_generations[entry['generation']]+=1
                path=project_path(root,entry['path'])
                if entry['model_id'] not in verified_snapshots:
                    require(sha(path)==entry['sha256'],'Historical checkpoint changed')
                    verified_snapshots.add(entry['model_id'])
                if entry['model_id'] not in service.models: service.register_checkpoint(path)
                opponents.append(neural_spec(entry['model_id'],.05,'neutral'))
            policies=[learner]*learner_count+opponents
        else:
            # Rotate all profiles across the collection instead of fixing three for an entire chunk.
            selected_specs=opponent_specs if searched_legacy(index,config,boosted) else fast_specs
            policies=[learner]*learner_count+[selected_specs[profiles[(index*3+s)%len(profiles)]] for s in range(4-learner_count)]
        rotation=table_rotation(index);focuses={}
        if quality and learner_count==2:
            pair=quality_pool.learner_pair(index,kind,config['table_plan']['quotas'])
            other=iter(policies[learner_count:]);physical=[learner if s in pair else next(other) for s in range(4)]
            policies=[physical[(s+rotation)%4] for s in range(4)]
            for seat,receipt in zip((s for s in range(4) if s not in pair),quality_receipts):receipt['seat']=seat
        for seat in range(4):
            if policies[(seat-rotation)%4]==learner:
                point=context_counter%12;context_counter+=1
                focuses[str(seat)]='neutral' if point%2==0 else INDUSTRIES[point//2]
        provenance=dict(producer='recovery/2.0.0',purpose='training',kind=kind,
            ordinary_game_index=index,seed=seed,sampling_seed=config['seed']*7+index,spec=digest(signature),
            policies=[p.get('policy_id',p.get('kind')) for p in policies],rotation=rotation,
            learner_behavior=dict(model_id=current,policy_id=learner['policy_id'],
                                  exploration=config['exploration'],temperature=1.0,focus='mixed'))
        provenance['legacy_mode']='fast' if any(p.get('search_free') for p in policies) else ('searched' if kind=='legacy' else None)
        if 'critic_data' in config:
            from .critic_data import planned_table
            require(fingerprints is not None,'Ordinary table requires incoming-policy fingerprints')
            provenance['critic_table']=planned_table(policies,rotation,current,state['iteration'],
                state['quality_pool']['members'],fingerprints['learner'],fingerprints['rule_bot'],
                learner_policy_id=learner['policy_id'])
        if mix_receipt is not None:provenance['champion_rollout']=mix_receipt
        setup=digest(dict(rules='steam-brass-patch-1.1.0',setup_seed=seed))
        if quality_receipts:
            provenance['quality_opponents']=quality_receipts
            quality_tables[setup]=dict(index=index,learner_seats=list(pair),opponents=quality_receipts,
                policy_ids=[policies[(s-rotation)%4]['policy_id'] for s in range(4)])
        jobs.append(dict(index=i,seed=seed,rotation=rotation,sampling_seed=config['seed']*7+index,
            policies=policies,focuses=focuses,path=str(directory/f'actor-{i:06d}.jsonl.gz'),provenance=provenance))
        kind_by_group[setup]=kind
        composition[kind]+=1
    return dict(jobs=jobs,kind_by_group=kind_by_group,composition=composition,cursor=cursor,
        context_counter=context_counter,historical_generations=historical_generations,
        mix_receipts=mix_receipts,transition_added=transition_added,incumbent=incumbent,
        quality_planning=quality_planning,quality_tables=quality_tables)


def sampling_report(games,data,policy_id,kinds,config):
    """Check actual completed game seats separately from decision eligibility."""
    by_kind={k:dict(tables=0,learner_trajectories=0,learner_decisions=0,actor_eligible_decisions=0,forced_decisions=0)
             for k in ('selfplay','historical','legacy')}
    groups={};trajectories=set()
    for game in games:
        kind=kinds[game['setup_group']];groups[game['game_id']]=kind
        seats=[s for s,p in enumerate(game['opponents']) if p==policy_id]
        if latest_model_recipe(config):
            require(len(seats)==learner_seats_per_table(config,kind),'Ordinary table learner attribution differs')
        trajectories.update((game['game_id'],s) for s in seats)
        by_kind[kind]['tables']+=1;by_kind[kind]['learner_trajectories']+=len(seats)
    require({d[2] for d in data}==trajectories,'Missing or external ordinary player trajectories')
    for row,_,identity,_ in data:
        require(row['behavior']['policy_id']==policy_id and row['behavior']['kind']=='model',
                'Frozen opponent action in learner sampling report')
        counts=by_kind[groups[identity[0]]];counts['learner_decisions']+=1
        counts['actor_eligible_decisions' if len(row['choices'])>1 else 'forced_decisions']+=1
    totals={key:sum(c[key] for c in by_kind.values()) for key in next(iter(by_kind.values()))}
    for counts in by_kind.values():
        for source,target in (('tables','table_fraction'),('learner_trajectories','trajectory_fraction'),
                              ('actor_eligible_decisions','actor_eligible_fraction')):
            counts[target]=counts[source]/totals[source] if totals[source] else 0.
    return dict(**totals,by_composition=by_kind,
        behavior=dict(policy_id=policy_id,exploration=config['exploration'],temperature=1.0,focus='mixed'),
        scope='Actual completed tables and learner seats; forced decisions are critic-only. Table fractions do not define PPO sample fractions.')


def sample_usage(groups,eligible,actor_used,critic_used,actor_presentations,critic_presentations):
    population=Counter(groups);actor_population=Counter(groups[i] for i in eligible)
    used=Counter(groups[i] for i in actor_used);critic=Counter(groups[i] for i in critic_used)
    by_kind={k:dict(learner_decisions=population[k],actor_eligible_decisions=actor_population[k],
        actor_distinct_used=used[k],actor_unused_eligible=actor_population[k]-used[k],
        actor_presentations=actor_presentations[k],critic_distinct_used=critic[k],
        critic_presentations=critic_presentations[k]) for k in sorted(population)}
    totals={key:sum(row[key] for row in by_kind.values()) for key in next(iter(by_kind.values()))}
    for row in by_kind.values():
        row['actor_presentation_fraction']=row['actor_presentations']/max(1,totals['actor_presentations'])
    return dict(**totals,by_composition=by_kind,
        scope='Actual PPO optimizer samples, excluding utility calibration, rehearsal and teacher distillation; presentations include epoch reuse')
