"""Production shadow barrier using the already qualified Rust teacher worker.

Teacher tables stay outside ordinary replay. Completed tables are sealed once
for interrupted-update recovery; disk reloads take full row/value admission.
"""
import gzip
import json
from pathlib import Path
import time

import torch

from brass_ai.bank import read,sha
from brass_ai.learning.contracts import CONFIG_DIR,digest,require,project_path
from brass_ai.recovery.retention import owned,safe
from brass_ai.step5.league import write_json
from .lab_contract import pin
from .lab_replay_compat import accepts
from .lab_teacher_overlay import source_fingerprint

LIMIT=128*1024**2
READINESS_SEED='3269395f284558e8d8a6c625130cff9d258abefbaa552c9b73255c919e654bcf'


def readiness_seed(root,state):
    """Bootstrap the plan's last-non-null rule from actual hash-bound evidence."""
    seed=read(CONFIG_DIR/'b07-shadow-readiness.json')
    require(digest(seed)==READINESS_SEED,'Shadow readiness seed differs from reviewed evidence')
    if state['iteration']!=seed['generation'] or state.get('teacher_phase',{}).get('readiness') is not None:return None
    require(state['history'][-1]==seed['checkpoint'] and seed['fit_checkpoint'] in state['history'],
        'Shadow readiness seed checkpoint lineage changed')
    receipt=seed['receipt']
    from brass_ai.recovery.utility import reward_spec
    require(receipt['ready'] is True and receipt['reward_id']==digest(reward_spec(state['signature']['config'])),
        'Shadow readiness seed reward differs')
    for name in ('source','exported_source'):
        path=project_path(root,seed[name])
        if path.is_file() and sha(path)==seed['sha256']:
            require(read(path)['value_phase']['calibration_receipt']==receipt,'Measured readiness receipt changed')
            return seed
    raise ValueError('Missing pinned g1254 readiness report; restore its original or Git-visible results copy')


def teacher_build(state,*,prepared=True):
    from .lab_backend import checked_pin
    q=checked_pin(state['native_backend']['qualification'])
    require(q['passed'] and q['canary']['passed'] and q['canary']['passes']==2,
        'Shadow requires the completed B07 teacher canaries')
    receipt=q['teacher_build'];build=checked_pin(receipt)
    require(build['contract']=='brass-lab-b07-teacher-build/1.0.0' and
        build['producer']==source_fingerprint() and accepts(build['implementation']) and
        build['wire_regressions_passed'] is True,'Shadow teacher build differs from qualification')
    for item in [build['binary'],build['tests'],build['build_log'],build['bootstrap']['binary'],*build['bootstrap']['dependencies']]:
        require(pin(item['path'])==item,'Shadow runtime changed: '+item['path'])
    original=dict(build,receipt=receipt)
    if not prepared:return original
    from .lab_teacher_runtime import resolve
    from brass_ai.learning.contracts import SOURCE_ROOT
    selected=resolve(SOURCE_ROOT,original)
    from brass_ai.learning.post_transfer_reward import is_revised
    if is_revised(state['signature']['config']):
        from .lab_reward_runtime import resolve as resolve_reward
        selected=resolve_reward(SOURCE_ROOT,selected)
    return selected


class TableCache:
    def __init__(self,run,directory,jobs):
        self.run=run;self.directory=safe(run,directory);self.directory.mkdir(parents=True,exist_ok=True)
        self.jobs={j['expected']['job_id']:j for j in jobs}

    def paths(self,job):
        stem=job['expected']['job_id']
        require(stem in self.jobs and job==self.jobs[stem],'Unknown shadow cache job')
        path=safe(self.run,self.directory/(stem+'.json.gz'))
        return path,safe(self.run,path.with_name(path.name+'.receipt.json'))

    def load(self,job):
        path,sidecar=self.paths(job)
        if not path.exists():
            require(not sidecar.exists(),'Shadow cache lost its sealed table');return None
        if not sidecar.exists():
            # A crash before receipt publication cannot confer validation trust.
            destination=safe(self.run,path.with_name(path.name+'.unsealed'))
            require(not destination.exists(),'Unsealed shadow cache already preserved')
            path.rename(destination);return None
        receipt=read(sidecar);require(receipt['job']==digest(job) and receipt['file']==pin(path),'Shadow cache changed')
        with gzip.open(path,'rb') as stream:raw=stream.read(LIMIT+1)
        require(len(raw)<=LIMIT,'Shadow cache exceeds table bound')
        payload=json.loads(raw)
        require(payload['job']==job and payload['result']['job_id']==job['expected']['job_id'],
            'Shadow cache model/profile/generation differs')
        result=payload['result']
        result['metrics']['shadow_cache_seconds']=receipt['post_seconds']
        return result

    def publish(self,result,job):
        began=time.monotonic();path,sidecar=self.paths(job)
        require(not path.exists() and not sidecar.exists(),'Shadow cache already published')
        raw=json.dumps(dict(job=job,result=result),allow_nan=False,separators=(',',':')).encode()
        require(len(raw)<=LIMIT,'Shadow cache exceeds table bound')
        data=gzip.compress(raw,compresslevel=1,mtime=0)
        partial=safe(self.run,path.with_name(path.name+'.partial'))
        with partial.open('wb') as stream:stream.write(data)
        partial.replace(path)
        spent=time.monotonic()-began
        receipt=dict(job=digest(job),file=pin(path),post_seconds=spent)
        write_json(sidecar,receipt)
        elapsed=time.monotonic()-began
        return elapsed


class ShadowCollector:
    def __init__(self,root,run,state,*,mono=None):
        from .lab_shadow_config import is_active
        self.root,self.run=owned(root,run);self.build=teacher_build(state);self.mono=mono or 'mono'
        safe(self.run,self.run/'search-values')
        require(is_active(state['signature']['config']),'Teacher collection requires its explicit reviewed recipe')

    def collect(self,phase,state,model,service,directory):
        from .lab_teacher import TeacherCollector,calibrate
        from .lab_backend import checked_pin
        require(phase.options['mode'] in ('shadow','on'),'Teacher barrier requires shadow/on mode')
        if phase.options['mode']=='on':
            from .lab_teacher_activation import authorize
            authorize(state,state['signature']['config'])
        from brass_ai.recovery import entropy_schedule as S
        if S.KEY in state['signature']['config']['update']:
            schedule=S.restore_run(state['signature']['config'],state)
            require(schedule['progress']==schedule['config']['ramp_iterations'] and not S.waiting(schedule),
                'Finish the committed entropy/reward schedule before live shadow qualification')
        began=time.monotonic();device=next(model.parameters()).device;before=dict(service.metrics)
        devices=[device.index or 0] if device.type=='cuda' else []
        with torch.random.fork_rng(devices=devices):
            seed=readiness_seed(self.root,state) if phase.receipt is None else None
            if seed is not None:phase.receipt=seed['receipt']
            jobs=phase.begin(state,model,export_hash=state['history'][-1]['sha256'],epoch=state['iteration'],
                native_source=source_fingerprint())
            if not jobs:
                phase.metrics=dict(deferred=True,reason='Current-reward utility readiness unavailable',tables={})
                return
            cache=TableCache(self.run,directory/'teacher-shadow',jobs)
            cached={j['expected']['job_id']:cache.load(j) for j in jobs}
            missing=[j for j in jobs if cached[j['expected']['job_id']] is None]
            collector=None;live=[];cpu_metrics=None
            if missing:
                build=self.build;bootstrap=[build['bootstrap']['binary']['path']]
                import os
                if os.name!='nt':bootstrap.insert(0,self.mono)
                # Use the same typical/wide qualified roots as the B07 canary.
                ordinary=checked_pin(state['native_backend']['build'])
                from .lab_paths import runtime_path
                fixtures=read(runtime_path(ordinary['b01']['path']).parent/'inputs/fixtures.json')
                roots=[r for r in fixtures['roots'] if r['view']['context']['phase']=='action' and len(r['expected']['choices'])>1]
                require(roots,'Missing qualified teacher calibration roots')
                origins=[dict(state=r['expected']['state'],rail_keys=list(range(64))) for r in
                    (roots[0],max(roots,key=lambda r:len(r['expected']['choices'])))]
                from brass_ai.recovery.collector_runtime import native_inference
                block=native_inference()
                if block is not None and block['teacher']=='cpu-ort':
                    from .lab_cpu_runtime import teacher_plan,core_allocation
                    from .lab_teacher import calibrate_cpu,count_calibration
                    lanes=min(len(missing),service.cpu.capacity,block['workers'])
                    plan=teacher_plan(self.root,build,service,missing,block)
                    cores=core_allocation(lanes) if block['pinning']=='one-lane-per-cpu' else []
                    calibration=(count_calibration(build['core_source'],missing[0]) if block['teacher_budget']['mode']=='count' else
                        calibrate_cpu(plan['binary'],build['core_source'],origins,missing[0],plan['inference'],cores))
                    write_json(directory/'teacher-shadow/calibration.json',calibration)
                    collector=TeacherCollector(plan['binary'],build['core_source'],bootstrap,service,calibration,
                        lanes=lanes,model=model,history=state['history'],inference=plan['inference'],cores=cores)
                    cpu_metrics=dict(receipt=plan['receipt'],budget=plan['inference']['budget'],
                        exported_bundles=plan['exported_bundles'],lanes=lanes,cores=len(cores),pinning=block['pinning'])
                else:
                    calibration=calibrate(build['binary']['path'],build['core_source'],origins,missing[0],service)
                    write_json(directory/'teacher-shadow/calibration.json',calibration)
                    lanes=min(len(missing),service.cpu.capacity,state['signature']['config']['execution']['processes']*
                        state['signature']['config']['execution']['hosts_per_process'])
                    collector=TeacherCollector(build['binary']['path'],build['core_source'],bootstrap,service,calibration,
                        lanes=lanes,model=model,history=state['history'])
                live=collector.collect(missing,on_result=cache.publish)
                cached.update({r['job_id']:r for r in live})
            results=[cached[j['expected']['job_id']] for j in jobs]
            # Cached JSON never inherits a process-local verification capability.
            # Rebuild that capability through normal complete row admission.
            if collector is None:
                phase.admit_collection(results,model)
            else:
                verifier=collector.verifier
                from .lab_admission import admit_teacher_rows
                class MixedVerifier:
                    def take(self,result,job,model):
                        if job['expected']['job_id'] in verifier.cache:return verifier.take(result,job,model)
                        return admit_teacher_rows(result['rows'],job=job,model=model,history=state['history'])
                phase.admit_collection(results,model,verifier=MixedVerifier())
            if cpu_metrics is not None:phase.metrics['cpu_inference']=cpu_metrics
            phase.metrics.update(shadow=phase.options['mode']=='shadow',mode=phase.options['mode'],
                cached_tables=len(jobs)-len(missing),new_tables=len(live),wall_seconds=time.monotonic()-began,
                gradient_qualification='Measured during ordinary PPO; never applied' if phase.options['mode']=='shadow' else
                    'Qualified fixed beta; teacher policy gradients use the existing guarded PPO step')
            phase.metrics['inference']={key:service.metrics.get(key,0)-before.get(key,0) for key in
                ('positions','batches','model_loads','shared_requests','shared_replies','inline_requests')}
            if phase.search_value is not None:
                phase.metrics['recovered_value_partials']=phase.search_value.phase.buffer.recovered_partials
            if seed is not None:phase.metrics['readiness_seed']=dict(source=seed['source'],sha256=seed['sha256'],fit_checkpoint=seed['fit_checkpoint'])
            write_json(directory/'teacher-shadow/collection.json',phase.metrics)
