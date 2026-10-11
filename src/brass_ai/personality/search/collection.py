"""Batched frozen-controller tables with unchanged base replay and P2 sidecars."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import random
import time
import uuid

from brass_ai.bank import sha
from brass_ai.host import RulesHost
from brass_ai.learning.batching import model_id as base_id
from brass_ai.learning.contracts import digest,project_path,read_json,require
from brass_ai.learning.replay import split_for,terminal_targets
from brass_ai.learning.reward import CostLedger
from brass_ai.personality.contracts import local_file
from brass_ai.recovery.inference import RemoteEvaluator
from brass_ai.recovery.replay import write_shard,read_shards
from brass_ai.recovery.retention import owned
from brass_ai.step5.league import write_json
from brass_ai.step5.policy import NeuralPolicy
from .contracts import TRACES,Profile,identity,controller
from .engine import SearchSession,validate_personality_teacher
from .inference import SharedInference
from .readiness import load


def collect(root,bundle,output,jobs,config,*,tables=4,device='cpu',mono=None,teacher_game=False,decision_mode='searched',_calibration=None):
    root,out=owned(root,output)
    require(not out.exists() and jobs and 1<=tables<=24 and len(jobs)<=4096,'Use a new bounded collection directory')
    require(len({j['index'] for j in jobs})==len(jobs),'Duplicate collection jobs')
    require(decision_mode in ('searched','direct') and (not teacher_game or decision_mode=='searched'),'Teachers require searched decisions')
    if _calibration is None:model,parents,ready,handoff,pool_id=load(root,bundle)
    else:
        require(not teacher_game and decision_mode=='direct','Uncalibrated models may only collect direct readiness data')
        model,parents,ready,handoff,pool_id=_calibration
        require(ready['scope']=='direct-readiness-probe' and not ready['utility_ready'],'Invalid calibration-only input')
    require(config.reward==ready['reward'],'Collection search reward changed')
    execution=dict(handoff['signature']['config']['execution'],processes=tables,hosts_per_process=1)
    spec=dict(format=TRACES,contract=identity(),ready_id=ready['id'],model_id=ready['model']['model_id'],
        bundle=project_path(root,bundle).relative_to(root).as_posix(),pool_id=pool_id,parent_incumbent_id=base_id(parents[0]),
        search=config.to_dict(),teacher_game=teacher_game,jobs=jobs,tables=tables,decision_mode=decision_mode,calibration_probe=_calibration is not None)
    out.mkdir(parents=True);write_json(out/'spec.json',spec);began=time.monotonic()
    service=SharedInference(execution,device=device);ident=service.register(model);parent_ids=[service.register(m) for m in parents]
    require(ident==spec['model_id'],'Frozen collection model changed')
    channel_count=2+config.workers*(1+len(parent_ids)) if decision_mode=='searched' else 2
    channels=[service.channels(channel_count) for _ in range(min(tables,len(jobs)))]
    def run_table(index):
        lane=channels[index];receipts=[]
        session=SearchSession(root,service,lane,ident,parent_ids,config,[Profile()]*4,ready['goal_calibration'],mono=mono,search_enabled=decision_mode=='searched')
        opponent_endpoint=lane.acquire()
        policies={p:NeuralPolicy(None,evaluator=RemoteEvaluator(opponent_endpoint,p),focus='neutral',exploration=.05) for p in parent_ids}
        try:
            with RulesHost(project_path(root,handoff['signature']['config']['host']),mono=mono) as host:
                for job in jobs[index::len(channels)]:
                    profiles=[Profile.parse(p) for p in job['profiles']]
                    require(len(profiles)==4 and set(job['neural_seats'])<=set(range(4)) and job['neural_seats'],'Invalid table roles')
                    require(not teacher_game or job['neural_seats']==list(range(4)),'Teacher phases must cover all four NN seats')
                    require(all(job['controllers'][s]==controller(ident,parent_ids[0],profiles[s],config,pool_id,decision_mode=decision_mode)['candidate_id']
                        for s in job['neural_seats']),'Claimed controller differs from executed profile/model/budget')
                    session.profiles=profiles;session.begin(job['seed'],job['sampling_seed'],teacher_game=teacher_game)
                    setup=digest(dict(rules='steam-brass-patch-1.1.0',setup_seed=job['seed']))
                    require(split_for(setup)==job['partition'] and job['partition'] in ('train','validation'),'Discovery/teacher setup split differs')
                    line=[('personality-search/'+job['controllers'][s] if s in job['neural_seats'] else 'personality-parent/'+parent_ids[job['parent_indices'][s]]) for s in range(4)]
                    g=dict(game_id=uuid.uuid4().hex,setup_group=setup,players=4,split=job['partition'],opponents=line,records=[],result=None)
                    c=host.reset(job['seed']);rngs=[random.Random(digest(dict(seed=job['sampling_seed'],seat=s))) for s in range(4)]
                    ledger=CostLedger(ready['reward']);extras=[];before=dict(session.telemetry);start=time.monotonic()
                    for step in range(execution['max_decisions']):
                        if c['terminal']:break
                        view=host.observe(c['actor']);choices=host.choices();seat=view['seat'];ledger.observe(view);tick=time.monotonic()
                        if seat in job['neural_seats']:
                            selected,target,info=session.select(view,choices,rngs[seat],ledger)
                            extra=dict(step=step,role='teacher' if teacher_game else 'discovery',model_id=ident,
                                candidate_id=job['controllers'][seat],teacher=target,**info)
                            kind='external';probability=1.
                        else:
                            session.finish_turn()
                            with service.cpu.acquire(1,session.timings):
                                policy=policies[parent_ids[job['parent_indices'][seat]]]
                                policy.utility_remaining[rngs[seat]]=tuple(ledger.remaining)
                                selected,probability,_=policy.select_batch([(view,choices,rngs[seat])])[0]
                            extra=dict(step=step,role='parent',model_id=parent_ids[job['parent_indices'][seat]],context=None,goal=None,teacher=None)
                            kind='model'
                        row=dict(step=step,seat=seat,view=view,choices=choices,chosen=selected,style=[0.]*4,intent=[0.]*8,
                            teacher=None,seconds=time.monotonic()-tick,behavior=dict(kind=kind,policy_id=line[seat],probability=probability,temperature=1.))
                        if extra['teacher'] is not None:validate_personality_teacher(extra['teacher'],row,ident,ready['reward'])
                        g['records'].append(row);extras.append(extra);ledger.chosen(view,choices[selected]);c=host.apply(c,choices[selected])
                    require(c['terminal'],'Incomplete personality game; no replay/quality credit')
                    g['result']=host.call('result');terminal_targets(g['result']);session.finish_turn()
                    name=f'game-{job["index"]:06d}';path=out/(name+'.jsonl.gz')
                    write_shard(path,[g],dict(purpose='training' if g['split']=='train' else 'evaluation',producer=TRACES,
                        kind='personality-teacher' if teacher_game else 'personality-discovery',spec=digest(spec),job=job))
                    phases=copy_phases(session.phases)
                    actual={p:0 for p in phases[0]} if teacher_game else {}
                    if teacher_game:
                        from brass_ai.recovery.teacher_schedule import phase_for
                        for row,extra in zip(g['records'],extras):
                            if extra['teacher'] is not None:actual[phase_for(row['view']['context'])]+=1
                        require(all(actual[p]==sum(seat[p]['qualified_samples'] for seat in phases) for p in actual),'Phase sample receipts disagree')
                    side=dict(format=TRACES,contract=identity(),spec_id=digest(spec),game_id=g['game_id'],job=job,records=extras,
                        reservations=session.rounds,phases=phases,state=session.snapshot(),
                        rng=[r.getstate() for r in rngs],ledger=ledger.snapshot(),telemetry={k:session.telemetry[k]-before[k] for k in before},
                        admission=session.timings.report(),seconds=time.monotonic()-start)
                    side_path=out/(name+'.context.json');write_json(side_path,side)
                    receipts.append(dict(path=path.name,sha256=sha(path),context=side_path.name,context_sha256=sha(side_path),game_id=g['game_id'],job=job['index']))
                    print(f'Personality {"teacher" if teacher_game else "discovery"}: completed job {job["index"]+1}/{len(jobs)}',flush=True)
        finally:session.close()
        return receipts
    with service:
        with ThreadPoolExecutor(max_workers=len(channels),thread_name_prefix='personality-table') as pool:
            records=[r for group in pool.map(run_table,range(len(channels))) for r in group]
    result=dict(format=TRACES,contract=identity(),spec=spec,spec_id=digest(spec),records=sorted(records,key=lambda r:r['job']),
        seconds=time.monotonic()-began,inference=service.report(),failed_games=0)
    result['id']=digest(result);write_json(out/'manifest.json',result)
    paths=[project_path(root,bundle)/('manifest.json' if _calibration is not None else 'ready.json')]+[out/x[k] for x in records for k in ('path','context')]
    write_json(out/'pins.json',dict(paths=[p.relative_to(root).as_posix() for p in paths]))
    return result


def copy_phases(phases):
    import copy
    return copy.deepcopy(phases)


def read(directory,*,teacher_game=None,lazy=False):
    directory=Path(directory);manifest=read_json(directory/'manifest.json')
    require(manifest['format']==TRACES and manifest['contract']==identity() and manifest['id']==digest({k:v for k,v in manifest.items() if k!='id'})
        and manifest['spec_id']==digest(manifest['spec']),'Personality collection changed')
    require(not manifest['failed_games'] and len(manifest['records'])==len(manifest['spec']['jobs']),'Incomplete collection')
    if teacher_game is not None:require(manifest['spec']['teacher_game']==teacher_game,'Wrong collection purpose')
    pairs=iter_pairs(directory,manifest)
    return manifest,pairs if lazy else list(pairs)


def iter_pairs(directory,manifest):
    jobs={j['index']:j for j in manifest['spec']['jobs']}
    require(len(jobs)==len(manifest['records']) and {i['job'] for i in manifest['records']}==set(jobs),'Duplicate/missing collection job')
    require(len({i['game_id'] for i in manifest['records']})==len(jobs),'Duplicate game identity')
    for item in manifest['records']:
        path=local_file(directory,item['path']);side=local_file(directory,item['context'])
        require(sha(path)==item['sha256'] and sha(side)==item['context_sha256'],'Replay/context changed')
        games=read_shards([path]);extra=read_json(side)
        require(len(games)==1 and extra['game_id']==games[0]['game_id']==item['game_id'] and extra['spec_id']==manifest['spec_id']
            and extra['job']==jobs[item['job']] and len(extra['records'])==len(games[0]['records']),'Context attribution differs')
        job=extra['job'];game=games[0]
        from brass_ai.search.common import SearchConfig
        spec=manifest['spec'];cfg=SearchConfig(**spec['search']).validate()
        require(all(job['controllers'][s]==controller(spec['model_id'],spec['parent_incumbent_id'],Profile.parse(job['profiles'][s]),cfg,spec['pool_id'],decision_mode=spec['decision_mode'])['candidate_id']
            for s in job['neural_seats']),'Recorded controller recipe differs')
        require(game['setup_group']==digest(dict(rules='steam-brass-patch-1.1.0',setup_seed=job['seed']))
            and game['split']==job['partition']==split_for(game['setup_group']),'Context setup/split differs')
        for row,ctx in zip(game['records'],extra['records']):
            require(row['step']==ctx['step'],'Context steps differ')
            if row['seat'] in job['neural_seats']:
                require(ctx['model_id']==manifest['spec']['model_id'] and ctx['candidate_id']==job['controllers'][row['seat']]
                    and ctx['context']['profile']==job['profiles'][row['seat']]['style'],'Controller attribution differs')
            else:require(ctx['role']=='parent' and ctx['teacher'] is None and ctx['context'] is None,'Parent actions cannot teach a personality')
            if ctx['teacher'] is not None:validate_personality_teacher(ctx['teacher'],row,manifest['spec']['model_id'],manifest['spec']['search']['reward'])
        yield games[0],extra
