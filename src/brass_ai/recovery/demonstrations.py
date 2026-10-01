"""Audited four-player demonstrations; BC labels never become PPO/search labels."""
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
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
from brass_ai.learning.replay import split_for
from .replay import read_shards,write_cache,cache_receipt
from brass_ai.step5.league import write_json, seeds_for, implementation_id
from .config import execution_config
from .execution import ActorPool
from .inference import InferenceService
from .updates import Dataset, prepared, supervised_epoch
from .telemetry import environment


def generate(root,output,games,config,*,seed=17100000,mono=None,partition=None,budget=None):
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
    teacher_specs={p:legacy_spec(root,p,budget) for p in profiles}
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


def audit(root,paths,output,*,mono=None,workers=1,budget=None):
    root=Path(root);out=project_path(root,output)
    require(not out.exists(),'Refusing to overwrite a demonstration manifest')
    allowed=set(read(root/'training/opponents/pool.json')['training'])
    for identity in allowed: admission(root,identity,'training')
    calibration=None
    if budget:
        from .legacy_search import teachers
        allowed.update(teachers(root,budget));calibration=dict(format='direct-masked-budget/1.0.0',budget=budget)
    require(type(workers) is int and 1<=workers<=32,'Invalid audit CPU allowance')
    paths=sorted(set(str(project_path(root,p)) for p in paths))
    if workers>1:
        with ProcessPoolExecutor(max_workers=min(workers,len(paths)),mp_context=multiprocessing.get_context('spawn')) as executor:
            partial=list(executor.map(_audit_path,[(str(root),p,sorted(allowed),mono) for p in paths]))
        entries=[];excluded=[];seen=set();groups={}
        for accepted,rejected in partial:
            excluded.extend(rejected)
            for entry in accepted:
                if entry['semantic_sha256'] in seen:
                    excluded.append(dict(path=entry['path'],game_id=entry['game_id'],reason='Duplicate semantic game'));continue
                seen.add(entry['semantic_sha256'])
                require(groups.setdefault(entry['setup_group'],entry['split'])==entry['split'],'Setup family crosses partitions')
                entries.append(entry)
        return _manifest(root,out,entries,excluded,allowed,calibration,workers)
    entries=[];excluded=[];seen=set();groups={}
    with RulesHost(root/'artifacts/r1-host/guarded',mono=mono) as host:
        for name in sorted(set(str(project_path(root,p)) for p in paths)):
            path=Path(name)
            try:
                with gzip.open(path,'rt',encoding='utf-8') as stream: header=json.loads(next(stream))
                require(header['provenance'].get('purpose')=='training','Source purpose is not training')
                seeds=source_seeds(header);games=read_shards([path])
                for game in games:
                    require(game['setup_group'] in seeds,'No reproducible setup seed in source provenance')
                    teachers={r['behavior']['policy_id'] for r in game['records'] if r['behavior']['kind']=='external'}
                    eligible=teachers & allowed
                    require(eligible,'No current admitted v3 demonstration decisions')
                    # Old/held-out external policies cannot silently enter a demo game.
                    require(teachers<=allowed,'Old/unadmitted external opponent in demonstration')
                    signature=digest(dict(group=game['setup_group'],opponents=game['opponents'],
                        decisions=[(r['seat'],r['choices'][r['chosen']]['id']) for r in game['records']]))
                    if signature in seen:
                        excluded.append(dict(path=str(path.relative_to(root)),game_id=game['game_id'],reason='Duplicate semantic game'))
                        continue
                    context=host.reset(seeds[game['setup_group']])
                    for row in game['records']:
                        require(not context['terminal'] and context['actor']==row['seat'],'Replay acting seat differs')
                        require(host.observe(row['seat'])==row['view'],'Permitted replay observation differs')
                        legal=host.choices()
                        require(legal==row['choices'],'Complete legal choices differ')
                        context=host.apply(context,legal[row['chosen']])
                    require(context['terminal'] and host.call('result')==game['result'],'Replay terminal result differs')
                    seen.add(signature)
                    require(groups.setdefault(game['setup_group'],game['split'])==game['split'],'Setup family crosses partitions')
                    counts=Counter(r['behavior']['policy_id'] for r in game['records'] if r['behavior']['policy_id'] in eligible)
                    entries.append(dict(path=path.relative_to(root).as_posix(),sha256=sha(path),game_id=game['game_id'],
                        setup_group=game['setup_group'],split=game['split'],semantic_sha256=signature,
                        teachers=dict(counts),label_kind='qualified-v3-choice/1.0.0',
                        context='neutral-predecision/1.0.0',transitions_verified=True))
                if games:
                    write_cache(path,games)
                    for entry in entries:
                        if entry['path']==path.relative_to(root).as_posix(): entry['features']=cache_receipt(path)
            except (ValueError,KeyError,StopIteration) as error:
                excluded.append(dict(path=str(path.relative_to(root)),reason=str(error)))
    return _manifest(root,out,entries,excluded,allowed,calibration,workers)


def _manifest(root,out,entries,excluded,allowed,calibration,workers):
    require(entries,'No eligible demonstrations; generate fresh current-v3 training games')
    manifest=dict(format='bc-dataset/1.0.0',compatibility=compatibility(),implementation=implementation_id(),
        label_kind='qualified-v3-choice/1.0.0',context='neutral-predecision/1.0.0',teachers=sorted(allowed),
        entries=entries,excluded=excluded,partitions=dict(Counter(e['split'] for e in entries)),
        calibration=calibration,audit_workers=workers)
    write_json(out,manifest);return manifest


def _audit_path(args):
    """A fresh private rules host per shard; deduplication commits in the parent."""
    root,name,allowed,mono=args;root=Path(root);path=Path(name);entries=[];excluded=[]
    with gzip.open(path,'rt',encoding='utf-8') as stream: header=json.loads(next(stream))
    try:
        require(header['provenance'].get('purpose')=='training','Source purpose is not training')
        seeds=source_seeds(header);games=read_shards([path])
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
                entries.append(dict(path=path.relative_to(root).as_posix(),sha256=sha(path),game_id=game['game_id'],
                    setup_group=game['setup_group'],split=game['split'],semantic_sha256=signature,
                    teachers=dict(Counter(r['behavior']['policy_id'] for r in game['records'])),
                    label_kind='qualified-v3-choice/1.0.0',context='neutral-predecision/1.0.0',transitions_verified=True))
        write_cache(path,games)
        for entry in entries: entry['features']=cache_receipt(path)
    except (ValueError,KeyError,StopIteration) as error: excluded.append(dict(path=path.relative_to(root).as_posix(),reason=str(error)))
    return entries,excluded


def load_dataset(root,manifest_path,partition):
    root=Path(root);manifest=read(project_path(root,manifest_path))
    require(manifest['format']=='bc-dataset/1.0.0','Incompatible BC manifest')
    from brass_ai.learning.contracts import assert_heuristic_data_compatible
    assert_heuristic_data_compatible(manifest['compatibility'])
    require(manifest['label_kind']=='qualified-v3-choice/1.0.0' and manifest['context']=='neutral-predecision/1.0.0','Invalid BC labels/context')
    allowed=set(manifest['teachers'])
    parents=set(read(root/'training/opponents/pool.json')['training']);admitted=set(parents)
    # Reuse previously audited HEURISTIC data without admitting old planners or
    # importing neural replay into the new population. Verify archived evidence.
    archive=root/'training/opponents/archive/pre-masked-pool.json'
    if archive.exists():
        from brass_ai.bank import load_bank
        for identity,record in read(archive)['training'].items():
            evidence=project_path(root,record['qualification'])
            require(sha(evidence)==record['sha256'],'Historical heuristic evidence changed')
            report=read(evidence);name=record['profile'];bank=identity.split('/')[0]
            load_bank(root,bank)
            require(report['bank']==bank and report['profiles'][name]['training_eligible'],'Historical teacher evidence invalid')
            admitted.add(identity)
    if manifest.get('calibration'):
        from .legacy_search import teachers
        c=manifest['calibration']
        if c.get('format')=='direct-masked-budget/1.0.0':admitted.update(teachers(root,c['budget']))
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
        path=project_path(root,e['path']);require(sha(path)==e['sha256'],'Demonstration source changed')
        by_path.setdefault(path,set()).add(e['game_id'])
    games=[]
    for path,ids in by_path.items():
        receipt=next((e.get('features') for e in chosen if project_path(root,e['path'])==path),None)
        games.extend(g for g in read_shards([path],partition=partition,receipts={str(path):receipt},heuristic_only=True) if g['game_id'] in ids)
    require(len(games)==len(chosen),'Missing/duplicate demonstration games')
    require(all(r['behavior']['kind']=='external' and r['behavior']['policy_id'] in allowed for g in games for r in g['records']),
            'Fresh training prohibits imported neural replay; demonstrations must be entirely audited heuristics')
    data=prepared(games,lambda r:r['behavior']['kind']=='external' and r['behavior']['policy_id'] in allowed,neutral=True)
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


def checkpoint_alias(out,entry,name):
    source=out/entry['path']
    require(sha(source)==entry['sha256'],'BC checkpoint changed')
    temporary=out/(name+'.tmp')
    temporary.unlink(missing_ok=True)
    try: os.link(source,temporary)
    except OSError: shutil.copyfile(source,temporary)
    temporary.replace(out/name)


def train_bc(root,run,manifest_path,initial,config,*,device='cpu',epochs=10,patience=2,min_epochs=3,lr=1e-4,retention=None):
    from .retention import run_lock,owned
    _,out=owned(root,run)
    with run_lock(out):
        return _train_bc(root,run,manifest_path,initial,config,device=device,epochs=epochs,patience=patience,min_epochs=min_epochs,lr=lr,retention=retention)


def _train_bc(root,run,manifest_path,initial,config,*,device='cpu',epochs=10,patience=2,min_epochs=3,lr=1e-4,retention=None):
    root=Path(root);out=project_path(root,run)
    require(epochs>=min_epochs>=1 and patience>0,'Invalid BC stopping settings')
    torch.manual_seed(812771)
    training,manifest=load_dataset(root,manifest_path,'train')
    validation,_=load_dataset(root,manifest_path,'validation')
    require({d[2][0] for d in training}.isdisjoint({d[2][0] for d in validation}),'BC game leakage')
    require(initial is None,'Imitation starts from random residual weights; omit --initial')
    signature=dict(format='bc-training/2.2.0',manifest_sha256=sha(project_path(root,manifest_path)),
        manifest_path=project_path(root,manifest_path).relative_to(root).as_posix(),
        initialization='random',seed=812771,retention=retention,
        architecture=ModelConfig.preset('medium').to_dict(),update=config,lr=lr,epochs=epochs,
        patience=patience,min_epochs=min_epochs,implementation=implementation_id())
    state_path=out/'state.json'
    if state_path.exists():
        state=read(state_path);require(state['signature']==signature,'BC run settings/source changed')
        entry=state['last'];require(sha(out/entry['path'])==entry['sha256'],'BC checkpoint changed')
        model,raw=load_checkpoint(out/entry['path'],device=device)
        require(raw['step']==state['epoch'],'BC epoch/checkpoint mismatch')
    else:
        require(not out.exists() or not any(p.name!='.active-run.json' for p in out.iterdir()),'Use a new BC run')
        model=FourSeatNet(ModelConfig.preset('medium')).to(device)
        state=dict(signature=signature,epoch=0,best_loss=None,bad_epochs=0,history=[],parent=None,stopped=False)
    optimizer=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=1e-4)
    if state_path.exists(): resume_optimizer(raw,optimizer)
    training=balance(training)
    balanced_validation=balance(validation)
    # Reuse encoded arrays and device metadata across all supervised epochs.
    training=Dataset(training,device,policy=False)
    validation=Dataset(validation,device,policy=False)
    balanced_validation=Dataset(balanced_validation,device,policy=False)
    for epoch in (() if state['stopped'] else range(state['epoch'],epochs)):
        train_report=supervised_epoch(model,optimizer,training,config)
        val_report=supervised_epoch(model,None,validation,config,training=False)
        balanced_val_report=supervised_epoch(model,None,balanced_validation,config,training=False)
        improved=state['best_loss'] is None or val_report['loss']<state['best_loss']-1e-4
        evidence=dict(epoch=epoch+1,train=train_report,validation=val_report,balanced_validation=balanced_val_report)
        state['history'].append(evidence);state['epoch']=epoch+1
        state['bad_epochs']=0 if improved else state['bad_epochs']+1
        metadata=dict(mode='step5-v3-bc',parent=state['parent'],signature=digest(signature),fresh_run=out.relative_to(root).as_posix(),initialization='random',
            label_kind='qualified-v3-choice/1.0.0',context='neutral-predecision/1.0.0',
            manifest_sha256=signature['manifest_sha256'],evidence=evidence,promoted=False)
        filename=f'epoch-{epoch+1:04d}.pt'
        save_checkpoint(out/filename,model,optimizer,step=epoch+1,training=metadata)
        state['last']=dict(path=filename,sha256=sha(out/filename))
        if improved: state.update(best_loss=val_report['loss'],best=dict(state['last']))
        state['stopped']=epoch+1>=min_epochs and state['bad_epochs']>=patience
        # Commit an immutable checkpoint pointer before materializing aliases.
        write_json(state_path,state)
        checkpoint_alias(out,state['best'],'best.pt');checkpoint_alias(out,state['last'],'last.pt')
        if retention and retention['enabled']:
            from .retention import prune
            prune(root,out,retention,apply=True)
        print(json.dumps(dict(epoch=epoch+1,train_loss=train_report['loss'],validation_loss=val_report['loss'],best=improved)),flush=True)
        if state['stopped']: break
    checkpoint_alias(out,state['best'],'best.pt');checkpoint_alias(out,state['last'],'last.pt')
    report=dict(format='bc-training-report/1.0.0',signature=signature,epochs=state['epoch'],
        history=state['history'],candidate=str(out/'best.pt'),candidate_sha256=sha(out/'best.pt'),
        environment=environment(device),promoted=False,note='Select with fresh gameplay evaluation; imitation loss alone is not strength')
    write_json(out/'report.json',report);return report


def fresh_imitation(root,model,settings,update,run,device):
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
    train_rows,manifest=load_dataset(root,settings['manifest'],'train')
    val_rows,_=load_dataset(root,settings['manifest'],'validation')
    require({d[2][0] for d in train_rows}.isdisjoint({d[2][0] for d in val_rows}),'Imitation setup leakage')
    training=Dataset(balance(train_rows),device,policy=False);validation=Dataset(val_rows,device,policy=False)
    optimizer=torch.optim.AdamW(model.parameters(),lr=settings['learning_rate'],weight_decay=1e-4)
    best_loss=float('inf');best=None;bad=0;history=[]
    for epoch in range(settings['epochs']):
        train_report=supervised_epoch(model,optimizer,training,update)
        val_report=supervised_epoch(model,None,validation,update,training=False)
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
