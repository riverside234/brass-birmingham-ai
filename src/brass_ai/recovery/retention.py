"""Owned-run retention with dependency pins, quarantine, and exact restore.

Only known generated artifacts are eligible. Dataset sources, active populations,
unfinished jobs and formal evaluation evidence are dependencies, not garbage.
"""
from contextlib import contextmanager
import json
import os
from pathlib import Path
import socket
import time
import uuid

import torch

from brass_ai.bank import read,sha
from brass_ai.learning.contracts import project_path,require
from brass_ai.step5.league import write_json
from .config import RETENTION,POPULATION


def link(path):
    path=Path(path)
    return path.is_symlink() or (path.exists() and bool(getattr(path.lstat(),'st_file_attributes',0)&0x400))


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
def run_lock(run):
    run=Path(run);run.mkdir(parents=True,exist_ok=True);path=run/'.active-run.json'
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
    for parent,dirs,files in os.walk(artifacts,followlinks=False):
        dirs[:]=[d for d in dirs if d not in ('.retention-trash','patch-tools','node_modules','__pycache__','.git')
                 and d not in ('r2-pythondeps','r4-deps','r0-validation-deps') and not link(Path(parent)/d)]
        for name in result:
            if name in files: result[name].append(Path(parent)/name)
    return result


def references(root,run):
    """Small control files only. Historical reports do not pin every old replay."""
    pins={};control=[]
    for directory in (root/'training/datasets',root/'training/models'):
        if directory.exists(): control.extend((p,read(p),'dataset or model manifest') for p in directory.rglob('*.json'))
    artifacts=root/'artifacts'
    inventory=controls(artifacts)
    for p in inventory['state.json']:
        if '.retention-trash' in p.parts: continue
        d=read(p)
        if str(d.get('format','')).startswith('step5-recovery-league/2.1'):
            selected={k:d.get(k) for k in ('resume','population','origin','evaluation_anchor','pins')}
            cfg=d.get('signature',{}).get('config',{})
            selected['datasets']=cfg.get('rehearsal_manifest');selected['calibration']=cfg.get('legacy_budget',{}).get('calibration')
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
        visited.add(p.resolve())
        for value in strings(data):
            if not value.endswith(('.pt','.jsonl.gz','.npz','.json')): continue
            candidates=[root/value,p.parent/value] if not Path(value).is_absolute() else [Path(value)]
            for candidate in candidates:
                resolved=candidate.resolve()
                if resolved.is_relative_to(run) and resolved.exists(): pins[resolved]=reason+' '+str(p.relative_to(root))
                if resolved.is_relative_to(root) and resolved.suffix=='.json' and resolved.is_file() and resolved not in visited:
                    visited.add(resolved);control.append((resolved,read(resolved),'referenced manifest'))
    # Derived caches travel with their authoritative dataset/evidence source.
    for path,reason in list(pins.items()):
        if str(path).endswith('.jsonl.gz'):
            for suffix in ('.features.npz','.features.json'):
                derived=Path(str(path)+suffix)
                if derived.exists(): pins[derived.resolve()]=reason
    return pins


def population_entries(history,config=POPULATION):
    by_id={e['model_id']:e for e in history}
    entries=list(by_id.values());recent=entries[-config['recent']:]
    older=entries[:-config['recent']]
    anchors=[]
    if older:
        for i in range(min(config['anchors'],len(older))):
            index=round(i*(len(older)-1)/max(1,min(config['anchors'],len(older))-1))
            anchors.append(older[index])
    return list({e['model_id']:e for e in anchors+recent}.values())


def slim_checkpoint(root,run,entry):
    path=project_path(root,entry['path']);require(sha(path)==entry['sha256'],'Population checkpoint changed')
    destination=run/'population'/('weights-'+entry['model_id']+'.pt')
    if not destination.exists():
        raw=torch.load(path,map_location='cpu',weights_only=True);raw['optimizer']=None
        destination.parent.mkdir(parents=True,exist_ok=True);temporary=destination.with_name(destination.name+'.tmp')
        torch.save(raw,temporary);temporary.replace(destination)
    return dict(entry,path=destination.relative_to(root).as_posix(),sha256=sha(destination),weights_only=True)


def update_population(root,run,state,config=POPULATION):
    # Retired generations retain immutable metadata but are not candidates.
    available={e['model_id']:e for e in state.get('population',[])}
    for e in state['history']:
        path=project_path(root,e['path'])
        if path.exists(): available[e['model_id']]=e
    # The active pool is ordered anchors + recent; reappearing full checkpoints
    # must not reorder it. Lineage order also survives a fresh recipe migration.
    order={e['model_id']:i for i,e in enumerate(state['history'])}
    selected=population_entries(sorted(available.values(),key=lambda e:order[e['model_id']]),config)
    state['population']=[slim_checkpoint(root,run,e) for e in selected]
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
    if state.get('format')=='step5-recovery-league/2.1.0':
        for e in state['history'][-config['resume_checkpoints']:]:
            path=project_path(root,e['path'])
            if path.is_relative_to(run): keep.add(path);reasons[path]='recent resume checkpoint'
        for e in state['history'][:-config['resume_checkpoints']]:
            if project_path(root,e['path']).is_relative_to(run): candidate(project_path(root,e['path']),'retired full checkpoint')
        for path in (run/'population').glob('*.pt'): candidate(path,'retired population weights')
        rounds=sorted(p for p in run.glob('iteration-*') if p.is_dir() and (p/'report.json').exists())
        for directory in rounds[:-config['replay_rounds']]:
            for p in directory.glob('*.jsonl.gz*'): candidate(p,'expired ordinary or unpinned teacher replay')
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
                for derived in run.glob(p.name+'.features.*'): candidate(derived,'derived development features')
    # Profiles are derived and not training data.
    profiles=sorted(run.rglob('*.pt.trace.json'))
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
    purge(root,run,config,automatic=True)
    return dict(decision,manifest=str(manifest))


def purge(root,run,config=RETENTION,*,automatic=False):
    root,run=owned(root,run);require(not active(run,allow_owner=True),'Cannot purge an active run')
    state=read(run/'state.json') if (run/'state.json').exists() else {};generation=state.get('iteration',state.get('epoch',0))
    count=0
    for manifest in (run/'.retention').glob('cleanup-*.json'):
        data=read(manifest)
        if time.time()-data['created']<data['quarantine_hours']*3600: continue
        if automatic and generation<=data['generation']: continue
        for entry in data['entries']:
            if entry.get('moving') and not entry.get('moved'):
                origin=safe(run,run/entry['path']);destination=safe(run,run/entry['quarantine'])
                require(not (origin.exists() and destination.exists()),'Ambiguous interrupted cleanup')
                if not origin.exists() and destination.exists():
                    require(sha(destination)==entry['sha256'],'Interrupted quarantine changed')
                    entry.update(moved=True,moving=False);write_json(manifest,data)
            if not entry.get('moved') or entry.get('restored') or entry.get('deleted'): continue
            path=safe(run,run/entry['quarantine']);require(path.exists() and sha(path)==entry['sha256'],'Quarantine changed')
            path.unlink();entry['deleted']=True;count+=1;write_json(manifest,data)
    return dict(permanently_deleted_files=count)


def restore(root,manifest):
    root=Path(root).resolve();manifest=project_path(root,manifest);data=read(manifest)
    _,run=owned(root,data['run']);safe(run,manifest);require(not active(run),'Cannot restore an active run')
    count=0
    for entry in data['entries']:
        if not (entry.get('moved') or entry.get('moving')) or entry.get('restored') or entry.get('deleted'): continue
        source=safe(run,run/entry['quarantine']);destination=safe(run,run/entry['path'])
        if entry.get('moving') and destination.exists() and not source.exists(): continue
        require(not destination.exists(),'Restore would overwrite a current artifact')
        require(sha(source)==entry['sha256'],'Quarantine changed')
        destination.parent.mkdir(parents=True,exist_ok=True);source.replace(destination);entry['restored']=True
        write_json(manifest,data);count+=1
    return dict(restored_files=count)
