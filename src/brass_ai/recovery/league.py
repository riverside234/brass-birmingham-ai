"""Recovery league: immutable rounds, BC-compatible weights, persistent curriculum counters."""
from collections import Counter
import copy
from contextlib import ExitStack
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import random
import time

import torch

from brass_ai.bank import admission,read,sha
from brass_ai.learning.checkpoint import load_checkpoint,save_checkpoint,resume_optimizer
from brass_ai.learning.contracts import compatibility,digest,project_path,require
from brass_ai.learning.model import ModelConfig
from .replay import read_shards
from brass_ai.step5.league import write_json,seeds_for,implementation_id
from brass_ai.step5.policy import INDUSTRIES
from .config import validate_config,table_kind
from .demonstrations import load_dataset
from .diagnostics import summarize
from .execution import ActorPool,neural_spec
from .inference import InferenceService
from .telemetry import environment
from .updates import ppo_update,distill_update


class Behavior:
    def __init__(self,identity,exploration):
        self.evaluator=type('Identity',(),{'model_id':identity})()
        self.exploration,self.temperature=exploration,1.0
        self.policy_id=neural_spec(identity,exploration,'mixed')['policy_id']


def train(root,run_path,config,*,iterations=1,device='cpu',initial=None,mono=None,from_run=None,imitation_run=None):
    from .retention import owned,run_lock
    _,run=owned(root,run_path)
    with run_lock(run):
        return _train(root,run_path,config,iterations=iterations,device=device,initial=initial,mono=mono,from_run=from_run,imitation_run=imitation_run)


def _train(root,run_path,config,*,iterations=1,device='cpu',initial=None,mono=None,from_run=None,imitation_run=None):
    root=Path(root).resolve();run=project_path(root,run_path);validate_config(config)
    require(iterations>0 and not(initial and from_run),'Invalid recovery initialization')
    require(initial is None and from_run is None,'Fresh residual workflow forbids previous checkpoint/run imports')
    bc_model=bc_origin=None
    if imitation_run is not None:
        require(config['imitation']['manifest'] is None,'Use --imitation-run OR in-run imitation, not both')
        from .demonstrations import completed_imitation
        bc_model,bc_origin=completed_imitation(root,imitation_run)
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
    fast_specs={p:legacy_spec(root,p,config['legacy_budget'],search_free=True) for p in profiles}
    signature=dict(config=config,compatibility=compatibility(),implementation=implementation_id(),legacy_pool=pool['training'],teachers=teacher_specs)
    if bc_origin:signature['imitation_source']=bc_origin
    state_path=run/'state.json'
    if state_path.exists():
        state=read(state_path)
        require(state['format']=='step5-recovery-league/'+'2.2.0' and state['signature']==signature,
            'Run configuration/source changed; use a fresh recovery directory')
        require(initial is None and from_run is None,'Omit initialization when resuming')
        entry=state['history'][-1];require(sha(project_path(root,entry['path']))==entry['sha256'],'Snapshot changed')
        model,raw=load_checkpoint(project_path(root,entry['path']),device=device)
        require(model.config==ModelConfig.preset(config['model_size']) and raw['training'].get('fresh_run')==run.relative_to(root).as_posix(),
                'Resume must use this new run\'s residual checkpoint')
        require(all(project_path(root,e['path']).is_relative_to(run) for e in state['history']),'Population contains an imported checkpoint')
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
            state['imitation']=fresh_imitation(root,model,config['imitation'],config['update'],run,device,workers=config['audit_workers'])
        # A new recipe never inherits the old full-epoch optimizer moments.
        optimizer=torch.optim.AdamW(model.parameters(),lr=config['learning_rate'],weight_decay=config['weight_decay'])
        init_path=run/'initial.pt'
        save_checkpoint(init_path,model,optimizer,training=dict(mode='step5-fresh-residual-initial',parent=None,initialization='random',fresh_run=run.relative_to(root).as_posix(),imitation=state['imitation']))
        raw=load_checkpoint(init_path,device=device)[1]
        state['history']=[dict(path=init_path.relative_to(root).as_posix(),sha256=sha(init_path),
            model_id=raw['model_id'],generation=0)]
        state['evaluation_anchor']=state['history'][0]
        if revised:
            from .retention import update_population
            update_population(root,run,state,config['population'])
        write_json(state_path,state)
    optimizer=torch.optim.AdamW(model.parameters(),lr=config['learning_rate'],weight_decay=config['weight_decay'])
    resume_optimizer(raw,optimizer)
    rehearsal=load_dataset(root,config['rehearsal_manifest'],'train',workers=config['audit_workers'],progress=True,compact=True)[0] if config['rehearsal_manifest'] else None
    summaries=[]
    if revised: schedule_evaluations(root,run,state,device,mono)
    with InferenceService(config['execution'],device,profile=run/'profile-collection' if config['profile_steps'] else None,
                          profile_steps=config['profile_steps']) as service:
        with ActorPool(root,config['host'],service,config['execution'],mono=mono) as ordinary,ActorPool(root,config['host'],service,config['execution'],mono=mono,heavy=True) as heavy:
            for _ in range(iterations):
                iteration=state['iteration'];began=time.monotonic();directory=run/f'iteration-{iteration:04d}'
                current=service.register(model);behavior=Behavior(current,config['exploration'])
                snapshots={}
                verified_snapshots=set()
                for entry in state.get('population',state['history']):
                    snapshots[entry['model_id']]=entry
                jobs=[];kind_by_group={};composition=Counter();cursor=state['next_seed'];context_counter=state['context_counter']
                for i in range(config['games_per_iteration']):
                    index=state['games_done']+i;kind=table_kind(index,config['curriculum'])
                    selected,cursor=seeds_for(cursor,1);seed=selected[0]
                    rng=random.Random(digest(dict(seed=config['seed'],index=index)))
                    learner=neural_spec(current,config['exploration'],'mixed')
                    if kind=='selfplay': policies=[learner]*4
                    elif kind=='historical':
                        candidates=list(snapshots.values())
                        if rng.random()<.5: candidates=candidates[-8:]
                        opponents=[]
                        for seat in range(3):
                            entry=rng.choice(candidates);path=project_path(root,entry['path'])
                            if entry['model_id'] not in verified_snapshots:
                                require(sha(path)==entry['sha256'],'Historical checkpoint changed')
                                verified_snapshots.add(entry['model_id'])
                            if entry['model_id'] not in service.models: service.register_checkpoint(path)
                            opponents.append(neural_spec(entry['model_id'],.05,'neutral'))
                        policies=[learner]+opponents
                    else:
                        # Rotate all profiles across the collection instead of fixing three for an entire chunk.
                        selected_specs=fast_specs if index<config['legacy_opponents']['search_free_until_games'] else teacher_specs
                        policies=[learner]+[selected_specs[profiles[(index*3+s)%len(profiles)]] for s in range(3)]
                    rotation=index%4;focuses={}
                    for seat in range(4):
                        if policies[(seat-rotation)%4]==learner:
                            point=context_counter%12;context_counter+=1
                            focuses[str(seat)]='neutral' if point%2==0 else INDUSTRIES[point//2]
                    provenance=dict(producer='recovery/2.0.0',purpose='training',kind=kind,
                        seed=seed,sampling_seed=config['seed']*7+index,spec=digest(signature),
                        policies=[p.get('policy_id',p.get('kind')) for p in policies],rotation=rotation)
                    provenance['legacy_mode']='fast' if any(p.get('search_free') for p in policies) else ('searched' if kind=='legacy' else None)
                    jobs.append(dict(index=i,seed=seed,rotation=rotation,sampling_seed=config['seed']*7+index,
                        policies=policies,focuses=focuses,path=str(directory/f'actor-{i:06d}.jsonl.gz'),provenance=provenance))
                    kind_by_group[digest(dict(rules='steam-brass-patch-1.1.0',setup_seed=seed))]=kind
                    composition[kind]+=1
                records=[]
                normal=[j for j in jobs if j['provenance']['legacy_mode']!='searched']
                legacy=[j for j in jobs if j['provenance']['legacy_mode']=='searched']
                if revised and config['overlap_pools'] and normal and legacy:
                    with ThreadPoolExecutor(max_workers=2) as coordinators:
                        pending=[coordinators.submit(ordinary.collect,normal),coordinators.submit(heavy.collect,legacy)]
                        for future in pending: records+=future.result()
                else:
                    if normal: records+=ordinary.collect(normal)
                    if legacy: records+=heavy.collect(legacy)
                load_began=time.monotonic()
                games=read_shards([r['path'] for r in sorted(records,key=lambda r:r['job'])],partition='train',
                    receipts={r['path']:r.get('features') for r in records})
                collected=time.monotonic()
                diagnostics=summarize(games,{behavior.policy_id},kind_by_group)
                weight=config['rehearsal_weight']*max(0,1-state['games_done']/config['rehearsal_until'])
                update=ppo_update(model,optimizer,games,behavior,config['update'],
                    profile=directory/'profile-ppo' if config['profile_steps'] else None,
                    profile_steps=config['profile_steps'],
                    rehearsal=rehearsal,rehearsal_weight=weight)
                updated=time.monotonic();done=state['games_done']+len(games)
                search=dict(updated=False,reason='Persistent search schedule not due')
                next_search=state['next_search_games']
                ordinary_seconds=state.get('ordinary_seconds',0.)+updated-began
                search_seconds=state.get('search_seconds',0.)
                estimate=config['search_games']*(config['search']['seconds']*4*(config['search']['max_searches_per_seat'] or 64)+
                    (collected-began)/max(1,len(games)))
                search_budget_ok=not revised or search_seconds+estimate<=ordinary_seconds*config['search_wall_fraction']/(1-config['search_wall_fraction'])
                if config['search_every_games'] and done>=next_search and search_budget_ok:
                    identity=service.register(model);search_jobs=[]
                    rollout_entries=list({e['model_id']:e for e in state.get('population',state['history'])}.values())[-3:]
                    for e in rollout_entries:
                        path=project_path(root,e['path']);require(path.is_relative_to(run) and sha(path)==e['sha256'],'Imported/changed rollout model')
                        if e['model_id'] not in service.models:service.register_checkpoint(path)
                    rollout_ids=list(dict.fromkeys([identity]+[e['model_id'] for e in rollout_entries]))
                    for i in range(config['search_games']):
                        chosen,cursor=seeds_for(cursor,1);seed=chosen[0]
                        spec=dict(kind='masked_ismcts',model_id=identity,opponents=rollout_ids,search=config['search'],focus='neutral',cpu_role=config.get('legacy_budget',{}).get('role','reference'))
                        provenance=dict(producer='recovery/2.0.0',purpose='training',kind='masked-ismcts-teacher',
                            seed=seed,sampling_seed=config['seed']*13+done+i,spec=digest(signature))
                        search_jobs.append(dict(index=i,seed=seed,rotation=0,sampling_seed=provenance['sampling_seed'],
                            policies=[spec]*4,path=str(directory/f'search-{i:06d}.jsonl.gz'),provenance=provenance))
                    search_records=heavy.collect(search_jobs)
                    searched=read_shards([r['path'] for r in search_records],partition='train',receipts={r['path']:r.get('features') for r in search_records})
                    search=distill_update(model,optimizer,searched,config['update'],steps=config['distill_steps'],weight=config['distill_weight'])
                    search['collection']=search_records
                    while next_search<=done: next_search+=config['search_every_games']
                elif not search_budget_ok: search=dict(updated=False,reason='Search wall-time allowance exceeded; postpone until ordinary collection catches up')
                search_seconds+=time.monotonic()-updated if search.get('collection') else 0.
                report=dict(format='step5-recovery-report/'+'2.2.0',iteration=iteration,composition=dict(composition),
                    actor=update,search=search,**diagnostics,incoming_model_id=current,
                    games_done=done,cumulative_games=state['previous_games']+done,
                    timings=dict(collection_seconds=collected-began,ppo_seconds=updated-collected,
                        replay_load_seconds=collected-load_began,
                        search_and_distill_seconds=time.monotonic()-updated),execution=records,
                    inference=service.report(),inference_scope='cumulative for this invocation',
                    legacy_modes=dict(Counter(j['provenance']['legacy_mode'] for j in jobs if j['provenance']['legacy_mode'])),
                    environment=environment(device),strength='requires fresh post-update evaluation')
                output=run/f'generation-{iteration+1:04d}.pt'
                checkpoint_began=time.monotonic()
                identity=save_checkpoint(output,model,optimizer,step=iteration+1,
                    training=dict(mode='step5-population-ppo',recipe='nonforced-actor-ppo/2.1.0' if revised else 'effective-minibatch-ppo/2.0.0',
                        parent=current,signature=digest(signature),fresh_run=run.relative_to(root).as_posix(),initialization='random-lineage',evidence=report))
                report['post_update_model_id']=identity
                report['timings']['checkpoint_seconds']=time.monotonic()-checkpoint_began
                write_json(directory/'report.json',report)
                entry=dict(path=output.relative_to(root).as_posix(),sha256=sha(output),model_id=identity,generation=iteration+1)
                state.update(iteration=iteration+1,games_done=done,next_seed=cursor,context_counter=context_counter,
                    history=state['history']+[entry],next_search_games=next_search,
                    search_seconds=search_seconds,ordinary_seconds=ordinary_seconds)
                if revised:
                    from .retention import update_population
                    update_population(root,run,state,config['population'])
                write_json(state_path,state);summaries.append(report)
                if revised:
                    schedule_evaluations(root,run,state,device,mono)
                    from .retention import prune
                    if config['retention']['enabled']: prune(root,run,config['retention'],apply=True)
                    service.trim([e['model_id'] for e in state['population']]+[identity])
                print(f'Recovery generation {iteration+1}: {done} games, {update["optimizer_steps"]} PPO steps',flush=True)
    return dict(run=str(run),completed_iteration=state['iteration'],checkpoint=state['history'][-1],summaries=summaries)


def schedule_evaluations(root,run,state,device,mono):
    """Committed game-count schedule; evaluation never advances the train cursor."""
    from brass_ai.step5.evaluation import evaluation_config
    from .evaluation import evaluate
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
        refs=refs[-2:]
        cfg=evaluation_config(config,setups=schedule[name+'_setups'],seed=schedule['seed']+slot*1000,
            lineups=lineups,include_legacy=name=='broader',panel='development',use_search=False)
        directory=run/'evaluations'/f'{name}-{slot:04d}'
        result=evaluate(root,directory,candidate['path'],[r['path'] for r in refs],cfg,device=device,mono=mono)
        require(result['failed_games']==0,'Incomplete scheduled evaluation')
        state['evaluations'].append(dict(panel=name,slot=slot,games_done=state['games_done'],candidate=candidate,
            references=refs,games=result['games'],spec_sha256=sha(directory/'spec.json'),
            report=directory.relative_to(root).as_posix()+'/report.json',report_sha256=sha(directory/'report.json')))
        write_json(run/'state.json',state)
