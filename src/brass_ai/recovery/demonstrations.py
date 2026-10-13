"""Audited four-player demonstrations; BC labels never become PPO/search labels."""
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import gzip
import json
from pathlib import Path
import shutil
import os
import time

import torch
import numpy as np

from brass_ai.bank import admission, read, sha
from brass_ai.host import RulesHost
from brass_ai.learning.checkpoint import load_checkpoint, save_checkpoint, resume_optimizer
from brass_ai.learning.model import FourSeatNet,ModelConfig
from brass_ai.learning.contracts import compatibility, digest, project_path, require
from brass_ai.learning.paths import bound_reference
from brass_ai.learning.replay import split_for
from .replay import read_shards,write_cache,cache_receipt
from brass_ai.step5.league import write_json, seeds_for, implementation_id
from .config import execution_config
from .execution import ActorPool
from .inference import InferenceService
from .updates import Dataset, prepared, supervised_epoch
from .telemetry import environment
from .progress import Progress


def generate(root,output,games,config,*,seed=17100000,mono=None,partition=None,budget=None,work_budget=None):
    """A deterministic 80/10/10 setup partition, with balanced admitted profiles."""
    root=Path(root);out=project_path(root,output)
    require(type(games) is int and games>0,'Positive demonstration game count required')
    execution_config(config)
    pool=read(root/'training/opponents/pool.json');profiles=sorted(pool['training'])
    require(profiles,'No admitted teachers')
    for p in profiles:
        admission(root,p,'training')
        require('pottery-flex' not in p,'Held-out teacher selected')
    from .legacy_search import legacy_spec
    from .config import LEGACY_BUDGET,LEGACY_TIME_BUDGET,legacy_opponent_work_budget_config
    budget=budget or LEGACY_BUDGET
    legacy_opponent_work_budget_config(work_budget)
    work_budget=dict(LEGACY_TIME_BUDGET)
    teacher_specs={p:legacy_spec(root,p,budget,work_budget=work_budget) for p in profiles}
    spec=dict(format='v3-demonstration-collection/1.0.0',games=games,seed=seed,partition=partition,
        teachers=teacher_specs,execution=config,budget=budget,implementation=implementation_id(),compatibility=compatibility())
    if (out/'spec.json').exists(): require(read(out/'spec.json')==spec,'Demonstration spec changed')
    else:
        require(not out.exists() or not any(out.iterdir()),'Use an empty demonstration output')
        write_json(out/'spec.json',spec)
    jobs=[];cursor=seed
    for i in range(games):
        if partition: selected,cursor=seeds_for(cursor,1,partition);selected=selected[0]
        else: selected=cursor;cursor+=1
        policies=[teacher_specs[profiles[(i*4+s)%len(profiles)]] for s in range(4)]
        provenance=dict(producer='recovery/2.0.0',purpose='training',kind='v3-demonstration',
            spec=digest(spec),seed=selected,sampling_seed=seed*7+i,policies=[p['policy_id'] for p in policies])
        jobs.append(dict(index=i,seed=selected,rotation=i%4,sampling_seed=seed*7+i,policies=policies,
            path=str(out/f'game-{i:06d}.jsonl.gz'),provenance=provenance))
    with InferenceService(config,'cpu') as service,ActorPool(root,'artifacts/r1-host/guarded',service,config,mono=mono,heavy=True) as actors:
        records=actors.collect(jobs)
    result=dict(format='v3-demonstration-collection-report/1.0.0',spec=digest(spec),games=games,replays=records,
        eligible_for_bc=False,note='Run demo-audit to verify replay transitions and produce an admitted manifest')
    write_json(out/'report.json',result);return result


def source_seeds(header):
    p=header['provenance'];seeds=[]
    if type(p.get('seed')) is int: seeds.append(p['seed'])
    seeds+=p.get('seeds',[])
    seeds+=[j['seed'] for j in p.get('jobs',[])]
    return {digest(dict(rules='steam-brass-patch-1.1.0',setup_seed=s)):s for s in seeds}


def _audit_receipts(paths):
    """Reuse independently recorded generation receipts, never a cache's own claim."""
    receipts={}
    for directory in sorted({Path(p).parent for p in paths}):
        try:
            report=read(directory/'report.json')
            if not isinstance(report,dict) or report.get('format')!='v3-demonstration-collection-report/1.0.0':continue
            if report.get('spec')!=digest(read(directory/'spec.json')):continue
            records=report.get('replays')
            if not isinstance(records,list):continue
        except (OSError,ValueError):
            continue  # Older/unfinished collections still receive a full audit.
        by_name={}
        for record in records:
            if not isinstance(record,dict) or not isinstance(record.get('path'),str):continue
            receipt=record.get('features')
            if not isinstance(receipt,dict) or record.get('sha256')!=receipt.get('replay_sha256'):continue
            # Collections may have moved between Windows and the lab. The
            # receipt pins the complete replay bytes, not its old absolute path.
            name=Path(record['path'].replace('\\','/')).name
            by_name[name]=receipt if name not in by_name or by_name[name]==receipt else None
        receipts.update({str(directory/name):receipt for name,receipt in by_name.items() if receipt})
    return receipts


def audit(root,paths,output,*,mono=None,workers=1,budget=None,work_budget=None):
    root=Path(root);out=project_path(root,output)
    from .config import LEGACY_BUDGET
    budget=budget or LEGACY_BUDGET
    require(not out.exists(),'Refusing to overwrite a demonstration manifest')
    allowed=set(read(root/'training/opponents/pool.json')['training'])
    for identity in allowed: admission(root,identity,'training')
    calibration=None
    if budget:
        from .legacy_search import teachers
        mode=read(root/'training/opponents/pool.json').get('backend')
        allowed.update(teachers(root,budget));calibration=dict(format='direct-v3-s1-budget/1.0.0' if mode=='S1' else 'direct-masked-budget/1.0.0',budget=budget,
            parents=sorted(read(root/'training/opponents/pool.json')['training']))
        if mode=='masked_ismcts' and budget['role'] in ('lab','reference'):
            from .config import LEGACY_OPPONENT_WORK_BUDGET,LEGACY_TIME_BUDGET,current_work_budget
            from .legacy_search import legacy_spec
            work_budget=current_work_budget(work_budget,LEGACY_TIME_BUDGET)
            old={k:v for k,v in LEGACY_OPPONENT_WORK_BUDGET.items() if k!='watchdog_seconds'}
            old['contract']='legacy-opponent-turn-work/1.0.0'
            calibration['work_budgets']=list({digest(w):w for w in (work_budget,LEGACY_TIME_BUDGET,LEGACY_OPPONENT_WORK_BUDGET,old)}.values())
            allowed.update(legacy_spec(root,p,budget,work_budget=w)['policy_id'] for p in calibration['parents'] for w in calibration['work_budgets'])
    require(type(workers) is int and 1<=workers<=32,'Invalid audit CPU allowance')
    paths=sorted(set(str(project_path(root,p)) for p in paths))
    require(paths,'No demonstration replay paths supplied')
    receipts=_audit_receipts(paths)
    tasks=[(str(root),p,sorted(allowed),mono,receipts.get(p)) for p in paths]
    next_progress=20
    def report_progress(completed):
        nonlocal next_progress
        while next_progress<=100 and completed*100>=next_progress*len(tasks):
            print(f'demo-audit: {next_progress}% ({completed}/{len(tasks)} replay files checked)',flush=True)
            next_progress+=20
    if workers>1:
        with ProcessPoolExecutor(max_workers=min(workers,len(paths)),mp_context=multiprocessing.get_context('spawn')) as executor:
            pending={executor.submit(_audit_path,task):i for i,task in enumerate(tasks)}
            partial=[None]*len(tasks)
            for completed,future in enumerate(as_completed(pending),1):
                # Report completion immediately, retaining source order for the
                # existing manifest and duplicate-selection behavior.
                partial[pending[future]]=future.result()
                report_progress(completed)
    else:
        partial=map(_audit_path,tasks)
    # Commit a shard only after all its transitions pass. Serial and parallel
    # audits use the same eligibility rules and parent-owned deduplication.
    entries=[];excluded=[];seen=set();groups={}
    for completed,(accepted,rejected) in enumerate(partial,1):
        excluded.extend(rejected)
        for entry in accepted:
            if entry['semantic_sha256'] in seen:
                excluded.append(dict(path=entry['path'],game_id=entry['game_id'],reason='Duplicate semantic game'));continue
            seen.add(entry['semantic_sha256'])
            require(groups.setdefault(entry['setup_group'],entry['split'])==entry['split'],'Setup family crosses partitions')
            entries.append(entry)
        if workers==1:report_progress(completed)
    return _manifest(root,out,entries,excluded,allowed,calibration,workers)


def _manifest(root,out,entries,excluded,allowed,calibration,workers):
    require(entries,'No eligible demonstrations; generate fresh current-v4 training games. '+
            '; '.join(e['reason'] for e in excluded[:3]))
    manifest=dict(format='bc-dataset/1.0.0',compatibility=compatibility(),implementation=implementation_id(),
        label_kind='qualified-v3-choice/1.0.0',context='neutral-predecision/1.0.0',teachers=sorted(allowed),
        entries=entries,excluded=excluded,partitions=dict(Counter(e['split'] for e in entries)),
        calibration=calibration,audit_workers=workers)
    write_json(out,manifest);return manifest


def _audit_path(args):
    """A fresh private rules host per shard; deduplication commits in the parent."""
    root,name,allowed,mono,*receipts=args;root=Path(root);path=Path(name);entries=[];excluded=[]
    try:
        with gzip.open(path,'rt',encoding='utf-8') as stream: header=json.loads(next(stream))
        require(header['provenance'].get('purpose')=='training','Source purpose is not training')
        receipt=receipts[0] if receipts else None
        # Missing/stale derived data is rebuilt from the authoritative replay.
        # Only an exact independent receipt can skip repeated input validation.
        if receipt and receipt!=cache_receipt(path):receipt=None
        seeds=source_seeds(header)
        games=read_shards([path],heuristic_only=True,receipts={str(path):receipt})
        replay_sha256=sha(path)
        with RulesHost(root/'artifacts/r1-host/guarded',mono=mono) as host:
            for game in games:
                require(game['setup_group'] in seeds,'No reproducible setup seed')
                external={r['behavior']['policy_id'] for r in game['records'] if r['behavior']['kind']=='external'}
                require(external and external<=set(allowed),'Old/unadmitted external opponent in demonstration')
                context=host.reset(seeds[game['setup_group']])
                for row in game['records']:
                    require(not context['terminal'] and context['actor']==row['seat'],'Replay acting seat differs')
                    require(host.observe(row['seat'])==row['view'],'Permitted replay observation differs')
                    legal=host.choices();require(legal==row['choices'],'Complete legal choices differ')
                    context=host.apply(context,legal[row['chosen']])
                require(context['terminal'] and host.call('result')==game['result'],'Replay terminal result differs')
                signature=digest(dict(group=game['setup_group'],opponents=game['opponents'],
                    decisions=[(r['seat'],r['choices'][r['chosen']]['id']) for r in game['records']]))
                entries.append(dict(path=path.relative_to(root).as_posix(),sha256=replay_sha256,game_id=game['game_id'],
                    setup_group=game['setup_group'],split=game['split'],semantic_sha256=signature,
                    teachers=dict(Counter(r['behavior']['policy_id'] for r in game['records'])),
                    label_kind='qualified-v3-choice/1.0.0',context='neutral-predecision/1.0.0',transitions_verified=True))
        write_cache(path,games)
        features=cache_receipt(path)
        for entry in entries: entry['features']=features
    except (ValueError,KeyError,StopIteration,OSError,EOFError,UnicodeError) as error:
        entries=[]
        excluded.append(dict(path=path.relative_to(root).as_posix(),reason=str(error) or type(error).__name__))
    return entries,excluded


def _loader_init():
    # Spawned readers do CPU work only, without one tensor thread pool per core.
    torch.set_num_threads(1)


def _prepared_shard(task):
    path,entries,partition,allowed,compact=task;path=Path(path)
    require(all(e['sha256']==entries[0]['sha256'] for e in entries),'Conflicting demonstration hashes')
    require(sha(path)==entries[0]['sha256'],'Demonstration source changed')
    ids={e['game_id'] for e in entries}
    require(len(ids)==len(entries),'Missing/duplicate demonstration games')
    receipts=[e.get('features') for e in entries]
    require(all(r==receipts[0] for r in receipts),'Conflicting demonstration feature receipts')
    games=[g for g in read_shards([path],partition=partition,receipts={str(path):receipts[0]},heuristic_only=True)
           if g['game_id'] in ids]
    require({g['game_id'] for g in games}==ids,'Missing/duplicate demonstration games')
    require(all(r['behavior']['kind']=='external' and r['behavior']['policy_id'] in allowed for g in games for r in g['records']),
            'Fresh training prohibits imported neural replay; demonstrations must be entirely audited heuristics')
    data=prepared(games,neutral=True)
    if compact:
        # Features have crossed the full validation boundary. BC needs small
        # label/diagnostic metadata, not repeated board/history JSON over IPC.
        data=[(dict(seat=r['seat'],chosen=r['chosen'],behavior=r['behavior'],
                    view=dict(context={k:r['view']['context'][k] for k in ('era','phase')}),
                    choices=[{k:c[k] for k in ('id','kind','industry') if k in c} for c in r['choices']],
                    style=r['style'],intent=r['intent'],teacher=None),target,identity,encoded)
              for r,target,identity,encoded in data]
    return data


def load_dataset(root,manifest_path,partition,*,workers=1,progress=False,compact=False):
    require(type(workers) is int and 1<=workers<=64,'Invalid imitation loader CPU allowance')
    root=Path(root);manifest=read(project_path(root,manifest_path))
    require(manifest['format']=='bc-dataset/1.0.0','Incompatible BC manifest')
    from brass_ai.learning.contracts import assert_heuristic_data_compatible
    assert_heuristic_data_compatible(manifest['compatibility'])
    require(manifest['label_kind']=='qualified-v3-choice/1.0.0' and manifest['context']=='neutral-predecision/1.0.0','Invalid BC labels/context')
    allowed=set(manifest['teachers'])
    parents=set(read(root/'training/opponents/pool.json')['training']);admitted=set(parents)
    # Reuse previously audited HEURISTIC data without admitting old planners or
    # importing neural replay into the new population. Verify archived evidence.
    from brass_ai.bank import load_bank,validate_admission
    historical={}
    for filename in ('pre-masked-pool.json','pre-s1-training-pool.json','s1-training-smoke-pool.json','pre-era-pool.json','pre-era-r2-pool.json','pre-era-r3-pool.json','pre-industry-pool.json','pre-economy-pool.json'):
        archive=root/'training/opponents/archive'/filename
        if not archive.exists():continue
        for identity,record in read(archive)['training'].items():
            validate_admission(root,identity,record,'training')
            historical[identity]=record;admitted.add(identity)
    if manifest.get('calibration'):
        from .legacy_search import teachers
        c=manifest['calibration']
        if c.get('format')=='direct-v3-s1-budget/1.0.0':
            from .config import legacy_budget_config
            from .legacy_search import CONTRACT
            legacy_budget_config(c['budget'])
            require(set(c['parents'])<=admitted,'S1 dataset parent evidence missing')
            for parent in c['parents']:
                bank,name=parent.split('/');profile=load_bank(root,bank)['payload']['profiles'][name]
                require(profile['mode']=='S1' and profile['milliseconds']==1000,'S1 dataset source differs')
                admitted.add('v3-s1-forest/'+digest(dict(parent=parent,budget=c['budget'],contract=CONTRACT)))
        elif c.get('format')=='direct-masked-budget/1.0.0':
            admitted.update(teachers(root,c['budget']))
            from .legacy_search import legacy_spec
            from .config import legacy_opponent_work_budget_config
            for work in c.get('work_budgets',[]):
                legacy_opponent_work_budget_config(work)
                admitted.update(legacy_spec(root,p,c['budget'],work_budget=work)['policy_id'] for p in c['parents'])
            # Archived, audited heuristic choices remain reusable as BC only;
            # their former forest identities never become active opponents.
            for parent in set(historical)|parents:
                bank,name=parent.split('/');profile=load_bank(root,bank)['payload']['profiles'][name]
                if profile['mode']=='masked_ismcts':
                    admitted.add('masked-v3-forest/'+digest(dict(parent=parent,budget=c['budget'],version=2)))
                    admitted.add('masked-v3-forest/'+digest(dict(parent=parent,budget=c['budget'],search_free=False,version=3)))
        else:
            path=project_path(root,c['budget']['calibration']);require(sha(path)==c['sha256'],'Archived dataset evidence changed')
            report=read(path);require(report['identity']==c['identity'] and report['training_eligible'],'Archived dataset evidence invalid')
            for child in c['identity']['bank']['parents'].values():admitted.add('lab-v3/'+digest(dict(child=child,identity=c['identity'])))
    require(allowed<=admitted,'Manifest teacher no longer admitted')
    for p in allowed&parents: admission(root,p,'training')
    chosen=[e for e in manifest['entries'] if e['split']==partition]
    require(chosen,'No '+partition+' demonstration setup groups')
    by_path={}
    for e in chosen:
        require(e['transitions_verified'] and e['split']==split_for(e['setup_group']),'Unaudited/relabeled demonstration')
        by_path.setdefault(project_path(root,e['path']),[]).append(e)
    require(len({e['game_id'] for e in chosen})==len(chosen),'Missing/duplicate demonstration games')
    tasks=[(str(path),entries,partition,allowed,compact) for path,entries in by_path.items()]
    width=min(workers,len(tasks));parts=[None]*len(tasks)
    log=Progress(f'imitation load {partition} (CPU, {width} workers)',len(tasks),'shards') if progress else None
    if width>1:
        with ProcessPoolExecutor(max_workers=width,mp_context=multiprocessing.get_context('spawn'),initializer=_loader_init) as executor:
            # Bound queued results as well as live readers; retain manifest order.
            remaining=iter(enumerate(tasks));pending={}
            def submit():
                item=next(remaining,None)
                if item is not None:
                    index,task=item;pending[executor.submit(_prepared_shard,task)]=index
            for _ in range(min(len(tasks),2*width)):submit()
            completed=0
            while pending:
                future=next(as_completed(pending));index=pending.pop(future)
                parts[index]=future.result();completed+=1
                if log:log(completed)
                submit()
    else:
        for i,task in enumerate(tasks):
            parts[i]=_prepared_shard(task)
            if log:log(i+1)
    data=[row for part in parts for row in part]
    return data,manifest


def balance(data):
    counts=Counter((d[0]['behavior']['policy_id'],d[0]['view']['context']['era'],d[0]['choices'][d[0]['chosen']]['kind']) for d in data)
    weights=[]
    for d in data:
        key=(d[0]['behavior']['policy_id'],d[0]['view']['context']['era'],d[0]['choices'][d[0]['chosen']]['kind'])
        weights.append(min(3.0,len(data)/(len(counts)*counts[key])))
    # Normalize while retaining the declared threefold cap, even for rare bins.
    weights=np.asarray(weights,dtype=np.float64)
    low,high=0.,1.
    while np.minimum(3.,high*weights).mean()<1: high*=2
    for _ in range(40):
        middle=(low+high)/2
        if np.minimum(3.,middle*weights).mean()<1: low=middle
        else: high=middle
    return [(dict(d[0],_bc_weight=min(3.,high*w)),*d[1:]) for d,w in zip(data,weights)]


def checkpoint_alias(out,entry,name,*,root=None):
    source=project_path(root or out,entry['path'],base=out)
    require(source.is_relative_to(out),'BC checkpoint outside its run')
    require(sha(source)==entry['sha256'],'BC checkpoint changed')
    temporary=out/(name+'.tmp')
    temporary.unlink(missing_ok=True)
    try: os.link(source,temporary)
    except OSError: shutil.copyfile(source,temporary)
    temporary.replace(out/name)


def train_bc(root,run,manifest_path,initial,config,*,device='cpu',epochs=10,patience=2,min_epochs=3,lr=1e-4,retention=None,workers=1,from_imitation_run=None):
    from .retention import run_lock,owned
    _,out=owned(root,run)
    with run_lock(out):
        return _train_bc(root,run,manifest_path,initial,config,device=device,epochs=epochs,patience=patience,min_epochs=min_epochs,lr=lr,retention=retention,workers=workers,from_imitation_run=from_imitation_run)


def _train_bc(root,run,manifest_path,initial,config,*,device='cpu',epochs=10,patience=2,min_epochs=3,lr=1e-4,retention=None,workers=1,from_imitation_run=None):
    root=Path(root);out=project_path(root,run)
    require(epochs>=min_epochs>=1 and patience>0,'Invalid BC stopping settings')
    require(initial is None,'Imitation starts from random residual weights; omit --initial')
    from .config import update_config
    from .updates import autocast
    update_config(config);autocast(config,device)  # Fail unsupported precision before loading.
    parent_model=parent_origin=None
    if from_imitation_run is not None:
        require(project_path(root,from_imitation_run)!=out,'A changed dataset needs a new imitation run directory')
        parent_model,parent_origin=completed_imitation(root,from_imitation_run)
        require(parent_model.config==ModelConfig.preset('medium'),'Imitation continuation architecture differs')
    print(f'Imitation: device={device}, precision={config["precision"]}, loader_workers={workers}, '
          f'tensor_threads={torch.get_num_threads()}, effective_batch={config["effective_batch_size"]}, '
          f'microbatch={config["microbatch_size"]}; replay preparation uses CPU',flush=True)
    torch.manual_seed(812771)
    training,manifest=load_dataset(root,manifest_path,'train',workers=workers,progress=True,compact=True)
    validation,_=load_dataset(root,manifest_path,'validation',workers=workers,progress=True,compact=True)
    require({d[2][0] for d in training}.isdisjoint({d[2][0] for d in validation}),'BC game leakage')
    signature=dict(format='bc-training/2.2.0',manifest_sha256=sha(project_path(root,manifest_path)),
        manifest_path=project_path(root,manifest_path).relative_to(root).as_posix(),
        initialization='random',seed=812771,retention=retention,
        architecture=ModelConfig.preset('medium').to_dict(),update=config,lr=lr,epochs=epochs,
        patience=patience,min_epochs=min_epochs,implementation=implementation_id())
    if parent_origin:
        signature.update(initialization='fresh-imitation-continuation',parent_imitation=parent_origin)
    state_path=out/'state.json'
    resuming=state_path.exists()
    if resuming:
        state=read(state_path)
        if parent_origin:
            signature['parent_imitation']=bound_reference(root,parent_origin,state['signature'].get('parent_imitation'),('bc_run','path','manifest'))
        require({k:v for k,v in state['signature'].items() if k!='manifest_path'}==
                {k:v for k,v in signature.items() if k!='manifest_path'} and
                project_path(root,state['signature']['manifest_path'])==project_path(root,signature['manifest_path']),
                'BC run settings/source changed')
        signature=state['signature']
        entry=state['last'];path=project_path(root,entry['path'],base=out)
        require(path.is_relative_to(out) and sha(path)==entry['sha256'],'BC checkpoint changed')
        model,raw=load_checkpoint(path,device=device)
        require(raw['step']==state['epoch'],'BC epoch/checkpoint mismatch')
    else:
        require(not out.exists() or not any(p.name!='.active-run.json' for p in out.iterdir()),'Use a new BC run')
        model=FourSeatNet(ModelConfig.preset('medium')).to(device)
        if parent_model is not None:model.load_state_dict(parent_model.state_dict())
        state=dict(signature=signature,epoch=0,best_loss=None,bad_epochs=0,history=[],
            parent=parent_origin['model_id'] if parent_origin else None,stopped=False)
    optimizer=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=1e-4)
    if resuming: resume_optimizer(raw,optimizer)
    else:
        # The same-run starting anchor is random for round one, or the previous
        # selected BC model for an explicit new-dataset continuation.
        save_checkpoint(out/'initial.pt',model,optimizer,training=dict(
            mode='step5-bc-continuation-initial' if parent_origin else 'step5-bc-random-initial',
            initialization=signature['initialization'],fresh_run=out.relative_to(root).as_posix(),signature=digest(signature)))
        state['initial']=dict(path='initial.pt',sha256=sha(out/'initial.pt'))
        state['last']=dict(state['initial']);write_json(state_path,state)
    training=balance(training)
    balanced_validation=balance(validation)
    # Reuse encoded arrays and device metadata across all supervised epochs.
    training=Dataset(training,device,policy=False,progress='imitation prepare training tensors')
    validation=Dataset(validation,device,policy=False,progress='imitation prepare validation tensors')
    balanced_validation=Dataset(balanced_validation,device,policy=False,progress='imitation prepare balanced validation tensors')
    for epoch in (() if state['stopped'] else range(state['epoch'],epochs)):
        train_report=supervised_epoch(model,optimizer,training,config,progress=f'imitation epoch {epoch+1}/{epochs} train ({device})')
        val_report=supervised_epoch(model,None,validation,config,training=False,progress=f'imitation epoch {epoch+1}/{epochs} validation ({device})')
        balanced_val_report=supervised_epoch(model,None,balanced_validation,config,training=False,progress=f'imitation epoch {epoch+1}/{epochs} balanced validation ({device})')
        improved=state['best_loss'] is None or val_report['loss']<state['best_loss']-1e-4
        evidence=dict(epoch=epoch+1,train=train_report,validation=val_report,balanced_validation=balanced_val_report)
        state['history'].append(evidence);state['epoch']=epoch+1
        state['bad_epochs']=0 if improved else state['bad_epochs']+1
        metadata=dict(mode='step5-v3-bc',parent=state['parent'],signature=digest(signature),fresh_run=out.relative_to(root).as_posix(),initialization=signature['initialization'],
            label_kind='qualified-v3-choice/1.0.0',context='neutral-predecision/1.0.0',
            manifest_sha256=signature['manifest_sha256'],evidence=evidence,promoted=False)
        filename=f'epoch-{epoch+1:04d}.pt'
        save_checkpoint(out/filename,model,optimizer,step=epoch+1,training=metadata)
        state['last']=dict(path=filename,sha256=sha(out/filename))
        if improved: state.update(best_loss=val_report['loss'],best=dict(state['last']))
        state['stopped']=epoch+1>=min_epochs and state['bad_epochs']>=patience
        # Commit an immutable checkpoint pointer before materializing aliases.
        write_json(state_path,state)
        checkpoint_alias(out,state['best'],'best.pt',root=root);checkpoint_alias(out,state['last'],'last.pt',root=root)
        if retention and retention['enabled']:
            from .retention import prune
            prune(root,out,retention,apply=True)
        print(json.dumps(dict(epoch=epoch+1,train_loss=train_report['loss'],validation_loss=val_report['loss'],best=improved)),flush=True)
        if state['stopped']: break
    checkpoint_alias(out,state['best'],'best.pt',root=root);checkpoint_alias(out,state['last'],'last.pt',root=root)
    report=dict(format='bc-training-report/1.0.0',signature=signature,epochs=state['epoch'],
        history=state['history'],candidate=(out/'best.pt').relative_to(root).as_posix(),candidate_sha256=sha(out/'best.pt'),
        environment=environment(device),promoted=False,note='Select with fresh gameplay evaluation; imitation loss alone is not strength')
    write_json(out/'report.json',report);return report


def completed_imitation(root,run,*,_seen=None):
    """Explicit fresh BC -> PPO handoff; never a general checkpoint importer."""
    from .retention import owned,active
    root,run=owned(root,run)
    seen=set() if _seen is None else _seen
    require(run not in seen,'Cyclic imitation lineage');seen.add(run)
    require(not active(run),'Imitation is still active; finish it before self-play')
    require((run/'state.json').is_file() and (run/'report.json').is_file(),'Finish the imitation run before handing off its weights')
    state=read(run/'state.json');report=read(run/'report.json');signature=state['signature']
    require(report.get('format')=='bc-training-report/1.0.0' and report.get('signature')==signature,
            'Missing/mismatched completed imitation report')
    initialization=signature.get('initialization')
    # This training-only teacher revision does not invalidate completed BC.
    # Accept only the explicitly reviewed preceding source with exactly the same
    # learning contracts; all completion/hash/lineage checks below still apply.
    from brass_ai.learning.contracts import CONFIG_DIR
    reviewed=read(CONFIG_DIR/'imitation-handoff-sources.json')['compatible_sources']
    source_ok=signature.get('implementation')==implementation_id() or any(
        signature.get('implementation')==entry['implementation'] and digest(compatibility())==entry['learning_compatibility']
        for entry in reviewed)
    require(initialization in ('random','fresh-imitation-continuation') and source_ok,
            'Imitation must be fresh training from the current or reviewed compatible source')
    parent=None
    if initialization=='fresh-imitation-continuation':
        expected=signature['parent_imitation']
        _,parent=completed_imitation(root,expected['bc_run'],_seen=seen)
        parent=bound_reference(root,parent,expected,('bc_run','path','manifest'))
    require(report['epochs']==state['epoch'] and (state['stopped'] or state['epoch']==signature['epochs']),
            'Imitation has not completed')
    manifest=project_path(root,signature['manifest_path'])
    require(sha(manifest)==signature['manifest_sha256'],'Imitation dataset manifest changed')
    entry=state['best'];path=project_path(root,entry['path'],base=run)
    require(path.is_relative_to(run) and sha(path)==entry['sha256']==report['candidate_sha256'],
            'Selected imitation checkpoint changed')
    with torch.random.fork_rng(devices=[]):model,raw=load_checkpoint(path,device='cpu')
    metadata=raw['training']
    require(metadata.get('mode')=='step5-v3-bc' and metadata.get('initialization')==initialization
            and metadata.get('parent')==(parent['model_id'] if parent else None)
            and metadata.get('fresh_run') and project_path(root,metadata['fresh_run'])==run
            and metadata.get('signature')==digest(signature)
            and metadata.get('manifest_sha256')==signature['manifest_sha256'],
            'Checkpoint is not the completed fresh imitation candidate')
    require(model.config.to_dict()==signature['architecture'],'Imitation architecture differs')
    origin=dict(bc_run=run.relative_to(root).as_posix(),path=path.relative_to(root).as_posix(),
        sha256=entry['sha256'],model_id=raw['model_id'],manifest=manifest.relative_to(root).as_posix(),
        manifest_sha256=signature['manifest_sha256'],bc_signature=digest(signature))
    return model,origin


def fresh_imitation(root,model,settings,update,run,device,*,workers=1):
    """Optional heuristic BC inside a new random run; no checkpoint is imported."""
    if settings['manifest'] is None:return dict(enabled=False,initialization='random',note='No heuristic dataset supplied')
    import copy
    progress=run/'imitation-progress.json'
    identity=dict(run=run.relative_to(root).as_posix(),settings=settings,update=update,
        manifest_sha256=sha(project_path(root,settings['manifest'])),implementation=implementation_id(),model=model.config.to_dict())
    if progress.exists():
        previous=read(progress)
        require(previous.get('identity')==identity,'Incomplete imitation belongs to a different recipe/source/dataset')
        print('Restarting uncommitted imitation from this run\'s original random seed',flush=True)
    write_json(progress,dict(format='fresh-imitation-progress/1.0.0',identity=identity,history=[]))
    print(f'Fresh imitation: device={device}, loader_workers={workers}; replay preparation uses CPU',flush=True)
    train_rows,manifest=load_dataset(root,settings['manifest'],'train',workers=workers,progress=True,compact=True)
    val_rows,_=load_dataset(root,settings['manifest'],'validation',workers=workers,progress=True,compact=True)
    require({d[2][0] for d in train_rows}.isdisjoint({d[2][0] for d in val_rows}),'Imitation setup leakage')
    training=Dataset(balance(train_rows),device,policy=False,progress='imitation prepare training tensors')
    validation=Dataset(val_rows,device,policy=False,progress='imitation prepare validation tensors')
    optimizer=torch.optim.AdamW(model.parameters(),lr=settings['learning_rate'],weight_decay=1e-4)
    best_loss=float('inf');best=None;bad=0;history=[]
    for epoch in range(settings['epochs']):
        train_report=supervised_epoch(model,optimizer,training,update,progress=f'imitation epoch {epoch+1}/{settings["epochs"]} train ({device})')
        val_report=supervised_epoch(model,None,validation,update,training=False,progress=f'imitation epoch {epoch+1}/{settings["epochs"]} validation ({device})')
        improved=val_report['loss']<best_loss-1e-4
        if improved:best_loss=val_report['loss'];best=copy.deepcopy(model.state_dict());bad=0
        else:bad+=1
        history.append(dict(epoch=epoch+1,train=train_report,validation=val_report,improved=improved))
        write_json(progress,dict(format='fresh-imitation-progress/1.0.0',identity=identity,history=history))
        print(f'Fresh imitation {epoch+1}: train {train_report["loss"]:.4f}, validation {val_report["loss"]:.4f}',flush=True)
        if epoch+1>=settings['min_epochs'] and bad>=settings['patience']:break
    model.load_state_dict(best);model.validate_parameter_count()
    return dict(enabled=True,initialization='random',manifest=settings['manifest'],manifest_sha256=sha(project_path(root,settings['manifest'])),
                history=history,best_validation=best_loss,optimizer_reset_before_ppo=True)
