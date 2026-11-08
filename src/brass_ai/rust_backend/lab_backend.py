"""Explicit teacher-off B07 adoption, receipt checks and reference fallback.

Backend selection is operational state, recorded in each subsequent checkpoint;
it never changes the signed reward, optimizer, exploration or data recipe.
"""
import copy
from pathlib import Path

from brass_ai.bank import read,sha
from brass_ai.learning.contracts import digest,project_path,require
from brass_ai.step5.league import write_json
from .lab_replay_compat import accepts
from .lab_contract import pin,SOURCE

CONTRACT='brass-ordinary-backend/1.0.0'


def checked_pin(item):
    require(pin(item['path'])==item,'Backend runtime/evidence changed: '+item['path'])
    return read(item['path'])


def check_build(item):
    build=checked_pin(item)
    require(build['shared_core']==SOURCE and build['search_enabled'] is False and
            build['search_method']=='search_v1' and build['v205_enabled'] is False,'Unqualified ordinary binary')
    for entry in [build['collector'],build['bootstrap']['binary'],*build['bootstrap']['dependencies']]:
        require(pin(entry['path'])==entry,'Ordinary runtime changed')
    return dict(build,_receipt_sha256=item['sha256'])


def qualify_receipt(value,state):
    require(value.get('milestone')=='B07' and value.get('passed') is True and
            accepts(value.get('implementation')),'B07 qualification is absent or source differs')
    require(value['generation']==state['iteration'] and value['checkpoint_sha256']==state['history'][-1]['sha256']
            and value['config_sha256']==digest(state['signature']['config']),
            'B07 qualification is stale; qualify the actual committed checkpoint/config')
    require(value['ordinary']['matched']['equal'] and
            value['ordinary']['performance']['useful_accepted_work_gain'],
            'B07 adoption requires measured useful accepted-work improvement')
    require(value['canary']['passes']==2 and value['canary']['passed'] and
            value['continuation']['passed'],'B07 canary/restore scope incomplete')
    require(state['signature']['config'].get('teacher_phase',{}).get('mode','off')=='off',
            'Backend-only adoption requires teacher off')


def validate_state(state,raw=None,*,check_runtime=True):
    value=state.get('native_backend')
    if value is None:
        require(raw is None or raw['training'].get('native_backend') is None,'Native checkpoint lost backend state')
        return None
    require(value['contract']==CONTRACT and value['mode'] in ('rust','reference') and
            accepts(value['implementation']) and value['implementation']==state['signature']['implementation'] and
            value['config_sha256']==digest(state['signature']['config']), 'Backend source/config binding changed')
    config=state['signature']['config']
    if config.get('teacher_phase',{}).get('mode','off')!='off':
        from .lab_shadow_config import is_shadow,CONTRACT as SHADOW
        shadow=value.get('shadow',{})
        require(is_shadow(config) and shadow.get('contract')==SHADOW and
            shadow.get('tables')==dict(train=28,held_out=4),'Teacher on-mode remains a separate gate; invalid shadow adoption')
    if raw is not None and raw['training'].get('native_backend')!=value:
        transition=state.get('native_backend_switch')
        require(transition is not None and transition['checkpoint']==state['history'][-1] and
                transition['before']==raw['training'].get('native_backend') and transition['after']==value,
                'Backend/checkpoint mismatch without an explicit switch receipt')
    if raw is not None and 'native_backend_execution' in raw['training']:
        from .lab_replay_compat import accepts_execution
        require(accepts_execution(raw['training']['native_backend_execution']),
                'Native checkpoint execution source/review changed')
    if check_runtime and value['mode']=='rust':check_build(value['build'])
    receipt=checked_pin(value['qualification'])
    require(receipt['passed'] and accepts(receipt['implementation']),'Backend qualification changed')
    require(sha(value['rollback']['path'])==value['rollback']['sha256'],'Backend rollback manifest changed')
    return value


def rollback_fork(root,value):
    """Verify every retained full-checkpoint/dependency byte before rollback."""
    from .lab_continuation import DiagnosticFork
    checked_pin(value['rollback'])
    return DiagnosticFork(root,Path(value['rollback']['path']).parent,
        generation=value['adopted_generation'],checkpoint_sha256=value['adopted_checkpoint']['sha256'])


class RoutedPool:
    def __init__(self,root,host,service,config,state,*,mono=None):
        from brass_ai.recovery.execution import ActorPool
        self.reference=ActorPool(root,host,service,config['execution'],mono=mono)
        self.root,self.service,self.config,self.state,self.mono=root,service,config,state,mono
        self.host=host
        self.last_collection_report=None
        self.backend=validate_state(state)
        self._ordinary_build=None
    def collect(self,jobs,*,on_records=None):
        training=all(j['provenance'].get('purpose')=='training' for j in jobs)
        if not self.backend or self.backend['mode']=='reference':
            result=self.reference.collect(jobs,on_records=on_records)
            self.last_collection_report=self.reference.last_collection_report
            return result
        if not training:
            # Evaluation keeps its reference semantics without retaining an
            # idle second population of Python workers during native collection.
            from brass_ai.recovery.execution import ActorPool
            with ActorPool(self.root,self.host,self.service,self.config['execution'],mono=self.mono) as evaluation:
                result=evaluation.collect(jobs,on_records=on_records)
                self.last_collection_report=evaluation.last_collection_report
                return result
        from .lab_pool import NativePool
        from .lab_perf_runtime import resolve
        if self._ordinary_build is None:self._ordinary_build=resolve(self.root,check_build(self.backend['build']))
        build=self._ordinary_build  # Deployment/reload starts a new RoutedPool.
        with NativePool(self.root,self.service,self.config['execution'],build,
                        self.config,digest(self.state['signature']),mono=self.mono) as pool:
            result=pool.collect(jobs,on_records=on_records)
            self.last_collection_report=pool.last_collection_report
            return result
    def __enter__(self):self.reference.__enter__();return self
    def __exit__(self,*args):return self.reference.__exit__(*args)


def adopt(root,run,qualification,output):
    """Caller owns run lock. Restore all optimizers before any state publication."""
    from . import lab_continuation as continuation
    from brass_ai.recovery.resume import commit_transition
    root,run,output=Path(root),Path(run),Path(output)
    state=read(run/'state.json');qpin=pin(qualification);q=checked_pin(qpin);qualify_receipt(q,state)
    check_build(q['ordinary_build'])
    require('native_backend' not in state,'Backend already selected; use explicit switch/verification')
    source=continuation.preflight(root,run,generation=state['iteration'],checkpoint_sha256=q['checkpoint_sha256'])
    fork=continuation.create_fork(root,output,source)
    continuation.restore(fork,device='cpu',allow_cuda_topology_change=True)
    continuation.verify_dependencies(root,source)
    # The complete byte-for-byte fork retains all checkpoint/Adam/cache rollback
    # dependencies. Never replace this with a metadata-only backup.
    changed=state['signature']!=source['target_signature']
    if changed:state=commit_transition(run,state,source['target_signature'])
    value=dict(contract=CONTRACT,mode='rust',implementation=state['signature']['implementation'],
        config_sha256=digest(state['signature']['config']),qualification=qpin,build=q['ordinary_build'],
        adopted_generation=state['iteration'],adopted_checkpoint=copy.deepcopy(state['history'][-1]),
        rollback=pin(output/'diagnostic-fork.json'))
    check_build(value['build'])
    state['native_backend']=value
    state['native_backend_switch']=dict(checkpoint=copy.deepcopy(state['history'][-1]),before=None,after=copy.deepcopy(value))
    write_json(run/'state.json',state)
    return dict(adopted=True,teacher='off',training_advanced=False,rollback=value['rollback'],
                next_commit='Pending: next normally authorized G-to-G+1 iteration and restart')


def reference_switch(root,run):
    """Rollback backend at the current checkpoint; preserve all ordinary progress."""
    from .lab_snapshot import pending_paths
    from brass_ai.learning.checkpoint import load_checkpoint
    # A missing/broken native executable must not prevent returning to reference.
    state=read(Path(run)/'state.json');current=validate_state(state,check_runtime=False)
    require(current is not None and current['mode']=='rust','No active Rust backend')
    require(not pending_paths(Path(run),state['iteration']),'Finish/recover pending work before switching backend')
    entry=state['history'][-1];path=project_path(root,entry['path'])
    require(sha(path)==entry['sha256'],'Current rollback checkpoint changed')
    _,raw=load_checkpoint(path)
    validate_state(state,raw,check_runtime=False)
    rollback_fork(root,current)
    backup=Path(run)/('state-before-native-reference-'+entry['sha256'][:16]+'.json')
    require(not backup.exists() or read(backup)==state,'Existing reference-switch backup differs')
    write_json(backup,state)
    value=dict(current,mode='reference')
    state['native_backend_switch']=dict(checkpoint=copy.deepcopy(entry),before=raw['training'].get('native_backend'),after=value)
    state['native_backend']=value;write_json(Path(run)/'state.json',state)
    return dict(mode='reference',checkpoint=entry,ordinary_progress_preserved=True,source=state['signature']['implementation'])
