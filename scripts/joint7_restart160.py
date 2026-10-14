"""Restore the audited joint-7 generation 160, preserving the entire later run.

Default is a read-only check. --apply stages verified files, archives the current
run by rename, and publishes generation 160 at its ORIGINAL path. Checkpoint
bytes (including Adam/RNG and fresh_run binding) are never rewritten. Exported
JSON does not substitute for a missing full checkpoint or historical opponent.
"""
import argparse
import copy
from datetime import datetime,timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys
import uuid

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))

from brass_ai.bank import read,sha
from brass_ai.learning.contracts import LAB_ROOT,digest,project_path,require
from brass_ai.recovery.config import champion_mix_config,time_budget_config
from brass_ai.recovery.population import retain
from brass_ai.recovery.retention import owned,safe,link,run_lock
from brass_ai.recovery.resume import check_signature
from brass_ai.step5.league import seeds_for,implementation_id,write_json

EXPORT='results/v4-selfplay-joint-g0179-20261014T200020968013Z'
TARGET_SHA='19c16fe4a7a7903d3ff338a59640bbd9defeeb735147fe09dfbde13910056b33'
COUNTERS=('games_done','next_seed','next_search_games','context_counter','ordinary_seconds','search_seconds')


def verified_export(folder):
    manifest=read(folder/'export-manifest.json')
    require(manifest['committed_generation']==179,'Expected reviewed generation-179 export')
    for e in manifest['files']:
        path=safe(folder,folder/e['path']);raw=path.read_bytes()
        require(e['sha256'] in (hashlib.sha256(raw).hexdigest(),hashlib.sha256(raw.replace(b'\r\n',b'\n')).hexdigest()),
                f'Export changed: {e["path"]}')
    return manifest


def reconstruct(folder):
    """Special-case audited counter replay, cross-checked against a full state.

    This does NOT recover weights/Adam/RNG from diagnostic reports. Only the
    deterministic schedule counters are replayed, and must exactly reproduce
    the independently saved generation-170 state (timings within 1 microsecond).
    """
    folder=Path(folder).resolve()
    manifest=verified_export(folder);latest=read(folder/'state.json')
    before=read(folder/'state-before-source-c69ec655374946af-0150.json')
    after=read(folder/'state-before-source-bcd676c01c5f82df-0170.json')
    require(latest['iteration']==179 and before['iteration']==150 and after['iteration']==170,'Wrong audited state interval')
    require(latest['history'][:171]==after['history'] and after['history'][:151]==before['history'],'Checkpoint lineage differs')
    require(latest['history'][160]['sha256']==TARGET_SHA,'Wrong generation-160 checkpoint')
    cfg=after['signature']['config']
    require((cfg['games_per_iteration'],cfg['search_games'],cfg['search_every_games'])==(320,5,160),'Audited collection recipe differs')
    retained=read(folder/'retention-history.json')
    rows={v['summary']['iteration']:v['summary'] for k,v in retained.items() if k.startswith('iteration-')}
    counters={k:before[k] for k in COUNTERS};target=None
    for i in range(150,170):
        r=rows[i];collections=r['search']['collection_summaries']
        require(r['incoming_model_id']==after['history'][i]['model_id'] and
                r['post_update_model_id']==after['history'][i+1]['model_id'] and
                r['games_done']==320*(i+1) and r['actor']['learner_trajectories']==320,
                'Counter replay lacks matching committed model/game evidence')
        require(len(collections)==r['search']['collection_count']==2,'Unexpected teacher collection history')
        counters['next_seed']=seeds_for(counters['next_seed'],320+5*len(collections))[1]
        counters['games_done']+=320;counters['context_counter']+=320
        counters['ordinary_seconds']+=sum(r['timings'][k] for k in ('collection_seconds','replay_load_seconds','ppo_seconds'))
        for batch in collections:
            require(batch['due_games']==counters['next_search_games'],'Teacher cursor differs')
            counters['next_search_games']+=160;counters['search_seconds']+=batch['elapsed_seconds']
        if i==159:target=copy.deepcopy(counters)
    for key in COUNTERS:
        require(math.isclose(counters[key],after[key],rel_tol=0,abs_tol=1e-6) if key.endswith('seconds') else counters[key]==after[key],
                f'Counter reconstruction failed generation-170 cross-check: {key}')
    require({e['model_id'] for e in retain(after['history'],cfg['population'])}=={e['model_id'] for e in after['population']},
            'Historical reservoir does not reproduce the known population')
    state=copy.deepcopy(after);state.update(target,iteration=160,history=after['history'][:161])
    known={}
    for s in (before,after,latest):
        for e in s['population']+[s['champion']]:known[e['model_id']]=e
    prefix=Path(manifest['run'])
    state['population']=[copy.deepcopy(known.get(e['model_id'],dict(e,
        path=(prefix/'population'/('weights-'+e['model_id']+'.pt')).as_posix(),sha256=None,weights_only=True)))
        for e in retain(state['history'],cfg['population'])]
    require(all(e['generation']<=160 for e in state['population']),'Future opponent in restored population')
    require(state['champion']['generation']==160 and state['champion_gate']['accepted_games']==51200 and
            all(e['games_done']<=51200 for e in state['evaluations']+state['champion_gate']['recent']),
            'Future evaluation/promotion in target state')
    state['resume']=[copy.deepcopy(state['history'][-1])]
    state['recent_training']=[] # Diagnostic cache only; originals remain archived.
    state.pop('teacher_work_cost',None) # Cost estimate cache; new reward has a new recipe identity.
    state.pop('champion_mix',None) # First exposure starts at the restored committed progress.
    history={k:v for k,v in retained.items() if v['summary'].get('games_done',10**20)<=51200}
    return state,history,dict(format='joint7-restart160/1.0.0',run=manifest['run'],
        source_state_digest=digest(latest),target_generation=160,checkpoint=state['history'][-1],
        counters=target,counter_crosscheck_generation=170,counter_max_error=max(abs(counters[k]-after[k]) for k in COUNTERS),
        population_generations=[e['generation'] for e in state['population']],
        stage_origin_games=51200,stage_games=1600,
        note='Plan only until all original model/Adam/RNG files are verified; later run is archived intact')


def locate(root,run,entry,*,backup_run=None,checkpoint_backup=None):
    relative=project_path(root,entry['path']).relative_to(run)
    candidates=[safe(run,run/relative)]
    if backup_run is not None:candidates.append(safe(backup_run,backup_run/relative))
    if checkpoint_backup is not None and entry.get('generation')==160 and not entry.get('weights_only'):
        require(not any(link(p) for p in [checkpoint_backup,*checkpoint_backup.parents]),'Linked checkpoint backup')
        candidates.append(checkpoint_backup.resolve())
    for path in (run/'.retention').glob('cleanup-*.json'):
        for item in read(path).get('entries',[]):
            if item.get('path')==relative.as_posix() and item.get('quarantine'):
                candidates.append(safe(run,run/item['quarantine']))
    for path in candidates:
        if path.is_file() and (entry.get('sha256') is None or sha(path)==entry['sha256']):return path
    return None


def resolve_models(root,run,state,*,backup_run=None,checkpoint_backup=None):
    """Fail closed on missing optimizer or population; never use future models."""
    import torch
    from brass_ai.learning.checkpoint import load_checkpoint,resume_optimizer
    entries=state['population']+[state['history'][-1],state['evaluation_anchor'],state['champion']]
    files={};missing=[];validated={}
    torch.set_num_threads(2)
    for e in entries:
        path=locate(root,run,e,backup_run=backup_run,checkpoint_backup=checkpoint_backup)
        if path is None:missing.append(e['path']);continue
        if str(path) not in validated:
            with torch.random.fork_rng(devices=[]):model,raw=load_checkpoint(path)
            require(raw['model_id']==e['model_id'] and raw['step']==e['generation'],'Recovery checkpoint identity differs')
            require(raw['training'].get('fresh_run')==run.relative_to(root).as_posix(),'Recovery checkpoint run lineage differs')
            validated[str(path)]=(sha(path),raw['model_id'],raw['step'])
            if e['path']==state['history'][-1]['path']:
                require(raw['optimizer'] is not None and raw['optimizer']['state'],'Generation 160 requires complete Adam state, not best.pt')
                require(raw['training'].get('fresh_run')==run.relative_to(root).as_posix() and
                        raw['training'].get('signature')==digest(state['signature']),'Generation 160 run/source binding differs')
                opt=torch.optim.AdamW(model.parameters());resume_optimizer(raw,opt,restore_rng=False)
                del opt
            del model,raw
        checksum,identity,generation=validated[str(path)]
        require(identity==e['model_id'] and generation==e['generation'],'Repeated checkpoint identity differs')
        e['sha256']=checksum;files[Path(e['path']).relative_to(run.relative_to(root)).as_posix()]=dict(source=str(path),sha256=checksum)
    return files,sorted(set(missing))


def publish(root,run,state,history,files,receipt):
    """Caller holds the original run lock. Both destinations are verified siblings."""
    require(digest(read(run/'state.json'))==receipt['source_state_digest'],'Live run advanced since the audited export')
    suffix=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+uuid.uuid4().hex[:8]
    _,staging=owned(root,run.parent/(run.name+'-staging160-'+suffix))
    _,archive=owned(root,run.parent/(run.name+'-before-restart160-'+suffix))
    require(not staging.exists() and not archive.exists(),'Restart destination exists')
    staging.mkdir()
    for relative,source in files.items():
        target=safe(staging,staging/relative);target.parent.mkdir(parents=True,exist_ok=True)
        require(sha(source['source'])==source['sha256'],'Recovery source changed')
        shutil.copyfile(source['source'],target);require(sha(target)==source['sha256'],'Recovery copy changed')
    # Retain available old reports/specs and source-transition backups. Future
    # iteration directories are never copied, so iteration 160 cannot reuse them.
    reports={e['report'] for e in state['evaluations']+state['champion_gate']['recent']}
    for name in sorted(reports):
        path=project_path(root,name)
        for p in (path,path.with_name('spec.json')):
            if p.is_file():
                relative=p.relative_to(run);target=safe(staging,staging/relative);target.parent.mkdir(parents=True,exist_ok=True)
                shutil.copyfile(safe(run,p),target)
    for transition in state.get('source_transitions',[]):
        path=safe(run,run/transition['backup'])
        if path.is_file():shutil.copyfile(path,safe(staging,staging/path.name))
    receipt=dict(receipt,applied=True,archive=archive.relative_to(root).as_posix(),checkpoint_bytes_unchanged=True)
    state['restart_origin']=dict(archive=receipt['archive'],source_state_digest=receipt['source_state_digest'],
        target_checkpoint=copy.deepcopy(state['history'][-1]),receipt=(run/'restart160.json').relative_to(root).as_posix())
    write_json(staging/'state.json',state);write_json(staging/'retention-history.json',history)
    write_json(staging/'restart160.json',receipt)
    # Absolute resolved sibling paths are checked before either directory move.
    require(staging.parent==archive.parent==run.parent and run.is_relative_to(root/'artifacts'),'Unsafe restart move')
    run.rename(archive)
    try:staging.rename(run)
    except BaseException:
        archive.rename(run)
        raise
    # Move our own marker back, so the normal run_lock exit removes it and the
    # archived run is not left falsely marked active. The OS guard stays held.
    marker=archive/'.active-run.json'
    if marker.exists():marker.rename(run/'.active-run.json')
    from brass_ai.recovery.champion import publish_best
    publish_best(root,run,state)
    return receipt


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=LAB_ROOT);p.add_argument('--evidence',default=EXPORT)
    p.add_argument('--run',default='artifacts/step5/v4-selfplay-joint')
    p.add_argument('--config',default='configs/step5/joint7-g150-gate72.json')
    p.add_argument('--backup-run',type=Path,help='Read-only earlier full-run backup containing retired opponent files')
    p.add_argument('--checkpoint-backup',type=Path,help='Original full generation-0160.pt, exact reviewed SHA required')
    p.add_argument('--plan-only',action='store_true',help='Audit exported counters without claiming checkpoints exist')
    p.add_argument('--apply',action='store_true');p.add_argument('--report',type=Path)
    a=p.parse_args();root=a.root.resolve();_,run=owned(root,a.run)
    state,history,receipt=reconstruct(project_path(root,a.evidence))
    require(receipt['run']==run.relative_to(root).as_posix(),'Use the original run path to preserve checkpoint bindings')
    config=champion_mix_config(time_budget_config(read(project_path(root,a.config))))
    signature=dict(state['signature'],config=config,implementation=implementation_id())
    check_signature(state,signature,reviewed=True)
    receipt.update(config_id=digest(config),ready=False,applied=False)
    require(not (a.plan_only and a.apply),'--plan-only cannot apply a restart')
    if not a.plan_only:
        require((run/'state.json').is_file(),'Actual lab run unavailable; report JSON alone cannot resume Adam/RNG')
        backup=a.backup_run.resolve() if a.backup_run else None
        if backup:require(not any(link(q) for q in [backup,*backup.parents]),'Linked backup run')
        with run_lock(run):
            require(digest(read(run/'state.json'))==receipt['source_state_digest'],'Live run differs from the generation-179 review')
            files,missing=resolve_models(root,run,state,backup_run=backup,checkpoint_backup=a.checkpoint_backup)
            receipt.update(ready=not missing,missing=missing,verified_model_files=len(files))
            if a.apply:
                require(not missing,'Cannot restore complete generation 160; missing files: '+', '.join(missing))
                receipt=publish(root,run,state,history,files,receipt)
    if a.report:write_json(a.report,receipt)
    print(json.dumps(receipt,indent=2))
    return 0 if a.plan_only or receipt['ready'] else 2


if __name__=='__main__':raise SystemExit(main())
