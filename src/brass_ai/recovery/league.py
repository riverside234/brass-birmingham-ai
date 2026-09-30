"""Recovery league: immutable rounds, BC-compatible weights, persistent curriculum counters."""
from collections import Counter
from contextlib import ExitStack
from pathlib import Path
import random
import time

import torch

from brass_ai.bank import admission,read,sha
from brass_ai.learning.checkpoint import load_checkpoint,save_checkpoint,resume_optimizer
from brass_ai.learning.contracts import compatibility,digest,project_path,require
from brass_ai.learning.model import ModelConfig
from brass_ai.learning.replay import read_shards
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


def train(root,run_path,config,*,iterations=1,device='cpu',initial=None,mono=None,from_run=None):
    root=Path(root).resolve();run=project_path(root,run_path);validate_config(config)
    require(iterations>0 and not(initial and from_run),'Invalid recovery initialization')
    torch.set_num_threads(config['threads'])
    pool=read(root/'training/opponents/pool.json');profiles=sorted(pool['training'])
    for p in profiles: admission(root,p,'training')
    signature=dict(config=config,compatibility=compatibility(),implementation=implementation_id(),legacy_pool=pool['training'])
    state_path=run/'state.json'
    if state_path.exists():
        state=read(state_path)
        require(state['format']=='step5-recovery-league/2.0.0' and state['signature']==signature,
            'Run configuration/source changed; use a fresh recovery directory')
        require(initial is None and from_run is None,'Omit initialization when resuming')
        entry=state['history'][-1];require(sha(project_path(root,entry['path']))==entry['sha256'],'Snapshot changed')
        model,raw=load_checkpoint(project_path(root,entry['path']),device=device)
    else:
        require(not run.exists() or not any(run.iterdir()),'New recovery directory must be empty')
        require(initial or from_run,'Initialize recovery from a compatible BC/trained checkpoint')
        origin=None;history=[];previous_games=0;cursor=config['seed']
        if from_run:
            parent=read(project_path(root,from_run)/'state.json');entry=parent['history'][-1]
            require(sha(project_path(root,entry['path']))==entry['sha256'],'Parent snapshot changed')
            initial=entry['path'];history=parent['history'];previous_games=parent['games_done']
            cursor=parent['next_seed'];origin=dict(run=str(from_run),games=previous_games,checkpoint=entry)
        model,raw=load_checkpoint(project_path(root,initial),device=device)
        require(model.config==ModelConfig.preset(config['model_size']),'Recovery model size differs')
        entry=dict(path=str(Path(initial).relative_to(root).as_posix()) if Path(initial).is_absolute() else initial,
            sha256=sha(project_path(root,initial)),model_id=raw['model_id'],generation=0)
        history=history or [entry]
        state=dict(format='step5-recovery-league/2.0.0',signature=signature,iteration=0,games_done=0,
            previous_games=previous_games,next_seed=cursor,history=history,origin=origin,
            next_search_games=config['search_ready_games'],context_counter=0)
        # A new recipe never inherits the old full-epoch optimizer moments.
        optimizer=torch.optim.AdamW(model.parameters(),lr=config['learning_rate'],weight_decay=config['weight_decay'])
        init_path=run/'initial.pt'
        save_checkpoint(init_path,model,optimizer,training=dict(mode='step5-recovery-initial',parent=raw['model_id']))
        raw=load_checkpoint(init_path,device=device)[1]
        state['history']=history+[dict(path=init_path.relative_to(root).as_posix(),sha256=sha(init_path),
            model_id=raw['model_id'],generation=0)]
        write_json(state_path,state)
    optimizer=torch.optim.AdamW(model.parameters(),lr=config['learning_rate'],weight_decay=config['weight_decay'])
    resume_optimizer(raw,optimizer)
    rehearsal=load_dataset(root,config['rehearsal_manifest'],'train')[0] if config['rehearsal_manifest'] else None
    summaries=[]
    with InferenceService(config['execution'],device,profile=run/'profile-collection' if config['profile_steps'] else None,
                          profile_steps=config['profile_steps']) as service:
        with ActorPool(root,config['host'],service,config['execution'],mono=mono) as ordinary,ActorPool(root,config['host'],service,config['execution'],mono=mono,heavy=True) as heavy:
            for _ in range(iterations):
                iteration=state['iteration'];began=time.monotonic();directory=run/f'iteration-{iteration:04d}'
                current=service.register(model);behavior=Behavior(current,config['exploration'])
                snapshots={}
                verified_snapshots=set()
                for entry in state['history']:
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
                        policies=[learner]+[dict(kind='legacy',policy_id=profiles[(index*3+s)%len(profiles)],purpose='training') for s in range(3)]
                    rotation=index%4;focuses={}
                    for seat in range(4):
                        if policies[(seat-rotation)%4]==learner:
                            point=context_counter%12;context_counter+=1
                            focuses[str(seat)]='neutral' if point%2==0 else INDUSTRIES[point//2]
                    provenance=dict(producer='recovery/2.0.0',purpose='training',kind=kind,
                        seed=seed,sampling_seed=config['seed']*7+index,spec=digest(signature),
                        policies=[p.get('policy_id',p.get('kind')) for p in policies],rotation=rotation)
                    jobs.append(dict(index=i,seed=seed,rotation=rotation,sampling_seed=config['seed']*7+index,
                        policies=policies,focuses=focuses,path=str(directory/f'actor-{i:06d}.jsonl.gz'),provenance=provenance))
                    kind_by_group[digest(dict(rules='steam-brass-patch-1.1.0',setup_seed=seed))]=kind
                    composition[kind]+=1
                records=[]
                normal=[j for j in jobs if j['provenance']['kind']!='legacy'];legacy=[j for j in jobs if j['provenance']['kind']=='legacy']
                if normal: records+=ordinary.collect(normal)
                if legacy: records+=heavy.collect(legacy)
                games=read_shards([r['path'] for r in sorted(records,key=lambda r:r['job'])],partition='train')
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
                if config['search_every_games'] and done>=next_search:
                    identity=service.register(model);search_jobs=[]
                    for i in range(config['search_games']):
                        chosen,cursor=seeds_for(cursor,1);seed=chosen[0]
                        spec=dict(kind='s2',model_id=identity,search=config['search'],focus='neutral')
                        provenance=dict(producer='recovery/2.0.0',purpose='training',kind='s2-teacher',
                            seed=seed,sampling_seed=config['seed']*13+done+i,spec=digest(signature))
                        search_jobs.append(dict(index=i,seed=seed,rotation=0,sampling_seed=provenance['sampling_seed'],
                            policies=[spec]*4,path=str(directory/f'search-{i:06d}.jsonl.gz'),provenance=provenance))
                    search_records=heavy.collect(search_jobs)
                    searched=read_shards([r['path'] for r in search_records],partition='train')
                    search=distill_update(model,optimizer,searched,config['update'],steps=config['distill_steps'],weight=config['distill_weight'])
                    search['collection']=search_records
                    while next_search<=done: next_search+=config['search_every_games']
                report=dict(format='step5-recovery-report/2.0.0',iteration=iteration,composition=dict(composition),
                    actor=update,search=search,**diagnostics,incoming_model_id=current,
                    games_done=done,cumulative_games=state['previous_games']+done,
                    timings=dict(collection_seconds=collected-began,ppo_seconds=updated-collected,
                        search_and_distill_seconds=time.monotonic()-updated),execution=records,
                    inference=service.report(),inference_scope='cumulative for this invocation',
                    environment=environment(device),strength='requires fresh post-update evaluation')
                output=run/f'generation-{iteration+1:04d}.pt'
                checkpoint_began=time.monotonic()
                identity=save_checkpoint(output,model,optimizer,step=iteration+1,
                    training=dict(mode='step5-population-ppo',recipe='effective-minibatch-ppo/2.0.0',
                        parent=current,signature=digest(signature),evidence=report))
                report['post_update_model_id']=identity
                report['timings']['checkpoint_seconds']=time.monotonic()-checkpoint_began
                write_json(directory/'report.json',report)
                entry=dict(path=output.relative_to(root).as_posix(),sha256=sha(output),model_id=identity,generation=iteration+1)
                state.update(iteration=iteration+1,games_done=done,next_seed=cursor,context_counter=context_counter,
                    history=state['history']+[entry],next_search_games=next_search)
                write_json(state_path,state);summaries.append(report)
                print(f'Recovery generation {iteration+1}: {done} games, {update["optimizer_steps"]} PPO steps',flush=True)
    return dict(run=str(run),completed_iteration=state['iteration'],checkpoint=state['history'][-1],summaries=summaries)
