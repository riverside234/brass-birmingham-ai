"""Owned-run retention with dependency pins, quarantine, and exact restore.

Only known generated artifacts are eligible. Dataset sources, active populations,
unfinished jobs and formal evaluation evidence are dependencies, not garbage.
"""
from contextlib import contextmanager
import json
import os
from pathlib import Path, PureWindowsPath
import socket
import stat
import time
import uuid

import torch

from brass_ai.bank import read,sha
from brass_ai.learning.contracts import project_path,require
from brass_ai.step5.league import write_json
from .config import RETENTION,POPULATION


def link(path):
    path=Path(path)
    try:info=path.lstat()
    except FileNotFoundError:return False
    return stat.S_ISLNK(info.st_mode) or bool(getattr(info,'st_file_attributes',0)&0x400)


def owned(root,run):
    root=Path(root).resolve();base=root/'artifacts';original=Path(run) if Path(run).is_absolute() else root/run
    require(not any(link(p) for p in [original,*original.parents] if p.is_relative_to(base)),'Linked run root')
    run=project_path(root,run)
    require(run.is_relative_to(base) and run!=base,'Retention requires a specific owned artifacts run')
    require(not any(p.is_symlink() for p in [run,*run.parents] if p.is_relative_to(base)), 'Symlink run root')
    return root,run


def safe(run,path):
    original=Path(path)
    require(not link(original),'Linked retention target')
    resolved=original.resolve()
    require(resolved.is_relative_to(run) and resolved!=run,'Retention target escapes owned run')
    for parent in original.parents:
        if parent==run: break
        require(not link(parent),'Linked retention parent')
    return resolved


def active(run,allow_owner=False):
    file=Path(run)/'.active-run.json'
    if not file.exists(): return False
    data=read(file)
    if allow_owner and data['pid']==os.getpid() and data['host']==socket.gethostname(): return False
    if data['host']!=socket.gethostname(): return True
    if os.name=='nt':
        import ctypes
        kernel=ctypes.WinDLL('kernel32',use_last_error=True)
        kernel.OpenProcess.restype=ctypes.c_void_p
        handle=kernel.OpenProcess(0x1000,False,data['pid'])
        if not handle: return ctypes.get_last_error()==5
        try:
            status=ctypes.c_ulong();return not kernel.GetExitCodeProcess(ctypes.c_void_p(handle),ctypes.byref(status)) or status.value==259
        finally: kernel.CloseHandle(ctypes.c_void_p(handle))
    try: os.kill(data['pid'],0);return True
    except ProcessLookupError: return False
    except PermissionError: return True


@contextmanager
def _run_guard(run):
    """Serialize marker replacement, including the stale-marker recovery gap.

    Keep the guard inode stable outside the run; unlinking a locked guard would
    let another contender lock a different inode. The OS releases it on crash.
    """
    from brass_ai.learning.contracts import digest
    directory=run.parent/'.run-locks'
    require(not link(directory),'Linked run guard directory')
    directory.mkdir(exist_ok=True)
    path=directory/(digest(os.path.normcase(str(run)))+'.lock')
    require(not link(path),'Linked run guard')
    with path.open('a+b') as handle:
        handle.seek(0)
        try:
            if os.name=='nt':
                import msvcrt
                msvcrt.locking(handle.fileno(),msvcrt.LK_NBLCK,1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        except OSError as error:
            raise ValueError('Run is active in another process') from error
        try:yield
        finally:
            handle.seek(0)
            if os.name=='nt':msvcrt.locking(handle.fileno(),msvcrt.LK_UNLCK,1)
            else:fcntl.flock(handle.fileno(),fcntl.LOCK_UN)


@contextmanager
def run_lock(run):
    run=Path(run).resolve();run.mkdir(parents=True,exist_ok=True);path=run/'.active-run.json'
    with _run_guard(run):
        require(not active(run),'Run is active in another process')
        if path.exists(): path.unlink()
        token=uuid.uuid4().hex
        with path.open('x',encoding='utf-8') as f: json.dump(dict(pid=os.getpid(),host=socket.gethostname(),token=token),f)
        try: yield
        finally:
            if path.exists() and read(path).get('token')==token: path.unlink()


def strings(value):
    if isinstance(value,str): yield value
    elif isinstance(value,dict):
        for v in value.values(): yield from strings(v)
    elif isinstance(value,list):
        for v in value: yield from strings(v)


def controls(artifacts):
    # Dependency runtimes have tens of thousands of files and contain no run
    # controls. Avoid walking them at every training commit.
    result={k:[] for k in ('state.json','spec.json','pins.json')}
    excluded={'.retention-trash','patch-tools','node_modules','__pycache__','.git',
              'r2-pythondeps','r4-deps','r4-pythondeps','r0-pythonlibs','r0-validation-deps'}
    pending=[str(artifacts)]
    while pending:
        parent=pending.pop()
        # DirEntry reuses metadata from directory enumeration. Path.lstat for
        # every subdirectory added thousands of redundant Windows system calls.
        try:entries=os.scandir(parent)
        except FileNotFoundError:continue  # another owned run retired an empty directory
        with entries:
            for entry in entries:
                if entry.is_dir(follow_symlinks=False):
                    if entry.name in excluded:continue
                    if os.name=='nt' and getattr(entry.stat(follow_symlinks=False),'st_file_attributes',0)&0x400:continue
                    pending.append(entry.path)
                elif entry.name in result:result[entry.name].append(Path(entry.path))
    return result


def feature_sidecars(path):
    """All admitted generations accompany the same authoritative raw replay."""
    return tuple(Path(str(path)+'.features'+version+'.'+ext)
        for version in ('','-v2','-v3') for ext in ('npz','json'))


def references(root,run):
    # Other owned runs can retire their control directories after enumeration.
    # Rebuild the complete graph on that race instead of silently dropping an
    # unread dependency or aborting training after an otherwise valid commit.
    # Persistent missing inputs still fail closed; never delete from a partial
    # graph and never reuse an earlier pass's pins.
    for attempt in range(3):
        try:return _references(root,run)
        except FileNotFoundError:
            if attempt==2:raise


def _references(root,run):
    """Small control files only. Historical reports do not pin every old replay."""
    pins={};control=[];quarantined={}
    # Repeated manifests often name the same paths. Resolve once in this read
    # pass; never cache across cleanup passes or bypass safe() before deletion.
    resolved_paths={}
    def resolve(path,base=None):
        # Absolute (including relocated Windows) references ignore a local base.
        # Reuse their checked resolution for both schema interpretations.
        key=(path,None if Path(path).is_absolute() or PureWindowsPath(path).is_absolute() else base)
        if key not in resolved_paths:resolved_paths[key]=project_path(root,path,base=base)
        return resolved_paths[key]
    for receipt in (run/'.retention').glob('cleanup-*.json'):
        for entry in read(receipt)['entries']:
            if entry['path'].endswith('.json') and entry.get('quarantine') and not (entry.get('deleted') or entry.get('restored')):
                original=safe(run,run/entry['path']);stored=safe(run,run/entry['quarantine'])
                if not original.exists() and stored.is_file():
                    require(sha(stored)==entry['sha256'],'Quarantined manifest changed')
                    quarantined[original]=stored
    for directory in (root/'training/datasets',root/'training/models'):
        if directory.exists(): control.extend((p,read(p),'dataset or model manifest') for p in directory.rglob('*.json'))
    artifacts=root/'artifacts'
    inventory=controls(artifacts)
    for p in inventory['state.json']:
        if '.retention-trash' in p.parts: continue
        d=read(p)
        if d.get('format') in ('step5-recovery-league/2.1.0','step5-recovery-league/2.2.0'):
            selected={k:d.get(k) for k in ('resume','population','origin','evaluation_anchor','pins','champion','champion_mix')}
            selected['il_kl_probe']=d.get('il_kl_limit',{}).get('probe')
            selected['quality_members']=d.get('quality_pool',{}).get('members')
            selected['ratings']=d.get('ratings')
            selected['accepted_gate']=d.get('champion_gate',{}).get('accepted_gate')
            cfg=d.get('signature',{}).get('config',{})
            selected['datasets']=[cfg.get('rehearsal_manifest'),cfg.get('imitation',{}).get('manifest')]
        elif 'best' in d and 'last' in d: selected={k:d.get(k) for k in ('best','last','signature')}
        else: selected=d
        control.append((p,selected,'active state'))
    for p in inventory['spec.json']:
        if '.retention-trash' in p.parts: continue
        report=p.parent/'report.json';d=read(p)
        unfinished=not report.exists()
        cfg=d.get('config',{})
        formal=(cfg.get('panel')=='full' and cfg.get('setups',0)>=30) or (
            str(d.get('format','')).startswith('searchlab-paired') and cfg.get('workers')==1 and cfg.get('setups',0)>=30)
        if unfinished or formal: control.append((p,d,'unfinished or formal experiment'))
        if formal and report.exists(): control.append((report,read(report),'formal evidence'))
    for p in inventory['pins.json']:
        if '.retention-trash' not in p.parts: control.append((p,read(p),'explicit pin'))
    visited=set()
    for p,data,reason in control:
        visited.add(resolve(p))
        for value in dict.fromkeys(strings(data)):
            if not value.endswith(('.pt','.jsonl.gz','.jsonl','.npz','.json','.svg')): continue
            for resolved in dict.fromkeys((resolve(value),resolve(value,base=p.parent))):
                # Remember dependencies even while their original path is in
                # quarantine, so a new pin can rescue it before permanent purge.
                if resolved.is_relative_to(run): pins[resolved]=reason+' '+str(p.relative_to(root))
                if resolved.is_relative_to(root) and resolved.suffix=='.json' and resolved not in visited:
                    source=resolved if resolved.is_file() else quarantined.get(resolved)
                    if source is not None:
                        visited.add(resolved);control.append((resolved,read(source),'referenced manifest'))
    # Derived caches travel with their authoritative dataset/evidence source.
    for path,reason in list(pins.items()):
        if str(path).endswith('.jsonl.gz'):
            for derived in feature_sidecars(path):
                pins[resolve(derived)]=reason
    return pins


def population_entries(history,config=POPULATION):
    from .population import retain
    return retain(history,config)


def _same_payload(left,right):
    if type(left) is not type(right):return False
    if isinstance(left,torch.Tensor):
        return left.dtype==right.dtype and left.shape==right.shape and torch.equal(left,right)
    if isinstance(left,dict):
        return left.keys()==right.keys() and all(_same_payload(left[k],right[k]) for k in left)
    if isinstance(left,(list,tuple)):
        return len(left)==len(right) and all(_same_payload(a,b) for a,b in zip(left,right))
    return left==right


def slim_checkpoint(root,run,entry,existing=None):
    path=project_path(root,entry['path']);require(sha(path)==entry['sha256'],'Population checkpoint changed')
    destination=run/'population'/('weights-'+entry['model_id']+'.pt')
    if existing is not None:
        require(existing['model_id']==entry['model_id'] and project_path(root,existing['path'])==destination,
                'Population cache receipt changed')
        require(destination.is_file() and sha(destination)==existing['sha256'],'Population cache changed')
        cached_sha=existing['sha256']
    elif path==destination:
        cached_sha=entry['sha256']
    else:
        raw=torch.load(path,map_location='cpu',weights_only=True);raw['optimizer']=None
        # Opponents need model/compatibility/lineage, not copied per-game reports.
        raw['training']={k:v for k,v in raw.get('training',{}).items() if k not in ('evidence','imitation')}
        if destination.exists():
            # An interrupted publication has no committed checksum yet. Verify
            # the complete derived payload, including calibration and lineage.
            cached=torch.load(destination,map_location='cpu',weights_only=True)
            require(_same_payload(raw,cached),'Population cache content changed')
        else:
            destination.parent.mkdir(parents=True,exist_ok=True);temporary=destination.with_name(destination.name+'.tmp')
            torch.save(raw,temporary);temporary.replace(destination)
        cached_sha=sha(destination)
    return dict(entry,path=destination.relative_to(root).as_posix(),sha256=cached_sha,weights_only=True)


def update_population(root,run,state,config=POPULATION):
    if 'quality_pool' in state:
        from .quality_pool import reference
        members=[]
        for member in state['quality_pool']['members']:
            previous=member if member.get('weights_only') else None
            frozen=slim_checkpoint(root,run,reference(member),previous)
            members.append(dict(member,**frozen))
        state['quality_pool']['members']=members
        state['population']=[reference(m) for m in members]
        state['resume']=state['history'][-state['signature']['config'].get('retention',RETENTION)['resume_checkpoints']:]
        return state
    # Retired generations retain immutable metadata but are not candidates.
    committed={e['model_id']:e for e in state.get('population',[])}
    available=dict(committed)
    # A frozen champion may outlive the bounded population. Its cache receipt
    # remains authoritative if unchanged weights later re-enter the population.
    for champion in (state.get('champion'),state.get('champion_mix',{}).get('previous')):
        if champion and champion.get('weights_only'):committed.setdefault(champion['model_id'],champion)
    for e in state['history']:
        path=project_path(root,e['path'])
        if path.exists(): available[e['model_id']]=e
    # The active pool is ordered anchors + recent; reappearing full checkpoints
    # must not reorder it. Lineage order also survives a fresh recipe migration.
    order={e['model_id']:i for i,e in enumerate(state['history'])}
    selected=population_entries(sorted(available.values(),key=lambda e:order[e['model_id']]),config)
    state['population']=[slim_checkpoint(root,run,e,committed.get(e['model_id'])) for e in selected]
    state['resume']=state['history'][-state['signature']['config'].get('retention',RETENTION)['resume_checkpoints']:]
    return state


def plan(root,run,config=RETENTION):
    root,run=owned(root,run);require(run.exists(),'Run does not exist')
    require(not active(run,allow_owner=True),'Cannot prune an active run')
    pins=references(root,run);state_path=run/'state.json';state=read(state_path) if state_path.exists() else {}
    keep=set(pins);reasons=dict(pins);candidates={}
    def candidate(path,reason):
        path=safe(run,path)
        if path.is_file() and path not in keep: candidates[path]=reason
    if state.get('format') in ('step5-recovery-league/2.1.0','step5-recovery-league/2.2.0'):
        # Old full rating slots are retired only after their compact cumulative
        # outcomes/panels have been archived and committed in state.json.
        from .ratings import archived_history,ARCHIVE_FORMAT
        archived=archived_history(root,state.get('ratings',{}))
        for slot in archived['slots']:
            directory=safe(run,run/'ratings'/'slots'/f'generation-{slot["generation"]:06d}')
            for name in ('spec.json','report.json','panels.json','progress.svg','bootstrap.npz','games.jsonl','games-manifest.json'):
                candidate(directory/name,'retired strength-rating slot')
            for p in directory.glob('*.jsonl.gz*'):candidate(p,'retired strength-rating replay or features')
        # Only the active content-addressed archive stays pinned. Abandoned
        # pre-commit publications and superseded archives can be regenerated.
        for path in (run/'ratings').glob('history-*.json'):
            token=path.stem.removeprefix('history-')
            if len(token)==64 and all(c in '0123456789abcdef' for c in token) and path not in keep:
                require(read(safe(run,path)).get('format')==ARCHIVE_FORMAT,'Unknown rating archive')
                candidate(path,'superseded compact rating history')
        # Recent full actor replays are diagnostic samples, never future PPO
        # data or a fitting dependency.
        for slot in state.get('ratings',{}).get('slots',[]):
            directory=project_path(root,slot['report']['path']).parent
            for p in sorted(directory.glob('*.jsonl.gz'))[config['evaluation_samples']:]:
                candidate(p,'completed strength-rating replay')
                for derived in feature_sidecars(p):candidate(derived,'derived strength-rating features')
        for p in (run/'opponent-pool'/'probes').glob('*.npz'):
            candidate(p,'evicted or uncommitted quality-pool probabilities')
        for e in state['history'][-config['resume_checkpoints']:]:
            path=project_path(root,e['path'])
            if path.is_relative_to(run): keep.add(path);reasons[path]='recent resume checkpoint'
        for e in state['history'][:-config['resume_checkpoints']]:
            if project_path(root,e['path']).is_relative_to(run): candidate(project_path(root,e['path']),'retired full checkpoint')
        for path in (run/'population').glob('*.pt'): candidate(path,'retired population weights')
        rounds=sorted((p for p in run.glob('iteration-*') if p.is_dir() and p.name[10:].isdigit() and (p/'report.json').exists()),
                      key=lambda p:int(p.name[10:]))
        for directory in rounds[:-config['replay_rounds']]:
            for p in directory.glob('*.jsonl.gz*'): candidate(p,'expired ordinary or unpinned teacher replay')
            if int(directory.name.split('-')[-1])<state['iteration'] and str(read(directory/'report.json').get('format','')).startswith('step5-recovery-report/'):
                candidate(directory/'report.json','archived iteration report')
        # Keep two recent completed development screens per panel. Formal or
        # unfinished experiments and explicitly referenced files remain pinned.
        panels={}
        for directory in sorted((run/'evaluations').glob('*')):
            report=directory/'report.json'
            if not directory.is_dir() or not report.is_file():continue
            result=read(report)
            if result.get('format')=='four-player-champion/1.0.0':
                # Gate outcomes stay in the compact report. Completed raw games
                # never train PPO and are not kept for every promotion attempt.
                files=sorted(directory.glob('*.jsonl.gz'))
                for p in files[config['evaluation_samples']:]:
                    candidate(p,'completed champion selection game')
                    for derived in feature_sidecars(p):candidate(derived,'derived champion features')
                panels.setdefault('champion',[]).append(directory)
                continue
            if result.get('spec',{}).get('config',{}).get('panel')!='development':continue
            # A reviewed-source retry remains the same monitor/broader panel;
            # its immutable source suffix must not create an immortal group.
            panel_name=directory.name.split('-source-',1)[0].rsplit('-',1)[0]
            panels.setdefault(panel_name,[]).append(directory)
        for directories in panels.values():
            for directory in directories[:-2]:
                if any(p.is_relative_to(directory) for p in pins):continue
                for pattern in ('*.jsonl.gz*','spec.json','report.json','screen-report.json','confirmation-report.json'):
                    for p in directory.glob(pattern):candidate(p,'archived development screen')
                for p in (directory/'.retention').glob('cleanup-*.json'):
                    if settled(read(p)):candidate(p,'completed development cleanup receipt')
    elif 'best' in state and 'last' in state:
        for p in run.glob('epoch-*.pt'): candidate(p,'unused BC epoch')
    report_path=run/'report.json'
    if report_path.exists():
        report=read(report_path);spec=report.get('spec',{})
        formal=report.get('promotion_eligible',False) or spec.get('config',{}).get('panel')=='full' or (
            str(report.get('format','')).startswith('searchlab-paired') and spec.get('config',{}).get('workers')==1 and
            len(spec.get('seeds',[]))>=30)
        if not formal and any(k in str(report.get('format','')) for k in ('evaluation','searchlab-paired','searchlab-distill')):
            files=sorted(run.glob('*.jsonl.gz'))
            for p in files[config['evaluation_samples']:]:
                candidate(p,'completed development screen')
                for derived in feature_sidecars(p): candidate(derived,'derived development features')
    # Profiles are derived and not training data.
    profiles=sorted(p for p in run.rglob('*.pt.trace.json') if '.retention-trash' not in p.parts)
    for p in profiles[:-config['profile_traces']]: candidate(p,'old profiler trace')
    entries=[dict(path=p.relative_to(run).as_posix(),sha256=sha(p),bytes=p.stat().st_size,reason=reason)
             for p,reason in sorted(candidates.items()) if p not in keep]
    return dict(format='retention-plan/1.0.0',run=str(run),created=time.time(),
        generation=state.get('iteration',state.get('epoch',0)),entries=entries,
        reclaimable_bytes=sum(e['bytes'] for e in entries),
        protected=[dict(path=p.relative_to(run).as_posix(),reason=reasons.get(p,'dependency')) for p in sorted(keep) if p.exists()])


def prune(root,run,config=RETENTION,*,apply=False):
    root,run=owned(root,run);decision=plan(root,run,config)
    if not apply: return dict(decision,dry_run=True)
    from .history import compact_history
    archived=[e for e in decision['entries'] if e['path'].endswith('/report.json') and e['reason'].startswith('archived')]
    compact_history(run,[(e['path'],e['sha256'],read(safe(run,run/e['path']))) for e in archived],
                    config.get('history_reports',128))
    if not decision['entries']:
        return dict(decision,dry_run=False,**purge(root,run,config,automatic=True))
    # A single small history file retains learning/evaluation metrics, replacing
    # one large report directory per retired round. Archive before moving bytes.
    token=str(int(time.time()))+'-'+uuid.uuid4().hex[:8]
    manifest=run/'.retention'/('cleanup-'+token+'.json');trash=run/'.retention-trash'/token
    decision.update(dry_run=False,quarantine_hours=config['quarantine_hours'],status='moving')
    write_json(manifest,decision)
    for entry in decision['entries']:
        source=safe(run,run/entry['path']);destination=safe(run,trash/entry['path'])
        require(sha(source)==entry['sha256'],'Artifact changed during cleanup')
        entry['quarantine']=destination.relative_to(run).as_posix();entry['moving']=True
        write_json(manifest,decision)
        destination.parent.mkdir(parents=True,exist_ok=True);source.replace(destination)
        entry['moved']=True;entry['moving']=False
        write_json(manifest,decision)
    decision['status']='quarantined';write_json(manifest,decision)
    report_path=run/'report.json'
    if report_path.exists() and any('development' in e['reason'] for e in decision['entries']):
        report=read(report_path);report.update(raw_records_pruned=True,promotion_eligible=False)
        report['retention_manifest']=manifest.relative_to(root).as_posix();write_json(report_path,report)
    result=purge(root,run,config,automatic=True)
    return dict(decision,manifest=str(manifest),**result)


def settled(data):
    return data.get('status')=='quarantined' and all(e.get('restored') or e.get('deleted') for e in data['entries'])


def remove_empty_parents(run,path):
    parent=Path(path).parent
    while parent!=run:
        safe(run,parent)
        try:parent.rmdir()
        except OSError:break
        parent=parent.parent


def purge(root,run,config=RETENTION,*,automatic=False):
    root,run=owned(root,run);require(not active(run,allow_owner=True),'Cannot purge an active run')
    state=read(run/'state.json') if (run/'state.json').exists() else {};generation=state.get('iteration',state.get('epoch',0))
    count=0;rescued=0;deleted_bytes=0;pins=None
    for manifest in sorted((run/'.retention').glob('cleanup-*.json')):
        data=read(manifest)
        if time.time()-data['created']<data['quarantine_hours']*3600: continue
        # Zero-hour mode is safe immediately AFTER a committed state. Delayed
        # quarantine keeps its previous-generation guard.
        if automatic and data['quarantine_hours'] and generation<=data['generation']: continue
        if settled(data):continue
        if pins is None:pins=references(root,run)
        for entry in data['entries']:
            if entry.get('moving') and not entry.get('moved'):
                origin=safe(run,run/entry['path']);destination=safe(run,run/entry['quarantine'])
                require(not (origin.exists() and destination.exists()),'Ambiguous interrupted cleanup')
                if not origin.exists() and destination.exists():
                    require(sha(destination)==entry['sha256'],'Interrupted quarantine changed')
                    entry.update(moved=True,moving=False);write_json(manifest,data)
            if not entry.get('moved') or entry.get('restored') or entry.get('deleted'): continue
            path=safe(run,run/entry['quarantine']);origin=safe(run,run/entry['path'])
            if entry.get('restoring') and not path.exists() and origin.exists():
                require(sha(origin)==entry['sha256'],'Restored dependency changed')
                entry.update(restored=True,restoring=False);write_json(manifest,data)
                remove_empty_parents(run,path);continue
            if entry.get('deleting') and not path.exists():
                entry.update(deleted=True,deleting=False);write_json(manifest,data)
                remove_empty_parents(run,path);remove_empty_parents(run,origin);continue
            require(path.exists() and sha(path)==entry['sha256'],'Quarantine changed')
            if origin in pins:
                require(not origin.exists(),'Restoring dependency would overwrite a file')
                entry['restoring']=True;write_json(manifest,data)
                origin.parent.mkdir(parents=True,exist_ok=True);path.replace(origin)
                entry.update(restored=True,restoring=False,deleting=False);rescued+=1;write_json(manifest,data)
                remove_empty_parents(run,path);continue
            entry['deleting']=True;write_json(manifest,data)
            path.unlink();entry.update(deleted=True,deleting=False);count+=1;deleted_bytes+=entry['bytes'];write_json(manifest,data)
            remove_empty_parents(run,path);remove_empty_parents(run,origin)
    # Bound the cleanup journal too; unsettled receipts remain recoverable.
    # With only two completed receipts, no fresh dependency scan is necessary.
    if pins is None:
        if len(list((run/'.retention').glob('cleanup-*.json')))<=2:
            return dict(permanently_deleted_files=0,permanently_deleted_bytes=0,rescued_dependencies=0)
        pins=references(root,run)
    completed=sorted((p for p in (run/'.retention').glob('cleanup-*.json') if settled(read(p)) and p.resolve() not in pins),
                     key=lambda p:read(p)['created'])
    for path in completed[:-2]:safe(run,path).unlink()
    return dict(permanently_deleted_files=count,permanently_deleted_bytes=deleted_bytes,rescued_dependencies=rescued)


def restore(root,manifest):
    root=Path(root).resolve();manifest=project_path(root,manifest);data=read(manifest)
    _,run=owned(root,data['run']);safe(run,manifest);require(not active(run),'Cannot restore an active run')
    with run_lock(run):
        return _restore(run,manifest,read(manifest))


def _restore(run,manifest,data):
    count=0
    for entry in data['entries']:
        if not (entry.get('moved') or entry.get('moving')) or entry.get('restored') or entry.get('deleted'): continue
        source=safe(run,run/entry['quarantine']);destination=safe(run,run/entry['path'])
        # The rename and journal publication are separate durable operations.
        # Complete an interrupted restore only when its recorded intent and
        # exact destination bytes agree; never treat an arbitrary file as ours.
        if entry.get('restoring') and not source.exists() and destination.exists():
            require(sha(destination)==entry['sha256'],'Restored dependency changed')
            entry.update(restored=True,restoring=False)
            write_json(manifest,data);count+=1
            continue
        if entry.get('moving') and destination.exists() and not source.exists(): continue
        require(not destination.exists(),'Restore would overwrite a current artifact')
        require(sha(source)==entry['sha256'],'Quarantine changed')
        entry['restoring']=True;write_json(manifest,data)
        destination.parent.mkdir(parents=True,exist_ok=True);source.replace(destination)
        entry.update(restored=True,restoring=False)
        write_json(manifest,data);count+=1
    return dict(restored_files=count)
