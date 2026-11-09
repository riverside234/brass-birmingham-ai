"""B07 process-isolated ordinary collector; only bounded receipts reach the learner.

The existing ready-family scheduler still releases branches immediately. Native
parts cross the coordinator as file/hash handles, never full board histories.
Bounded isolated publishers perform complete admission once per family.
Durable replay and cold reader validation retain the B03 contracts unchanged.
"""
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import copy
import gzip
import hashlib
import io
import json
import multiprocessing
import os
from pathlib import Path
import queue
import threading
import time
import traceback
import uuid
import zipfile

from brass_ai.bank import sha
from brass_ai.learning.contracts import digest, require
from brass_ai.step5.league import write_json
from . import lab_admission as admission
from .lab_collection import collect_lane, frozen_id, route
from .lab_families import FORMAT, collect_families, _failure_scope
from .lab_transport import Pipe
from . import lab_shared, lab_perf


def part_write(directory, part):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    path=directory/(uuid.uuid4().hex+'.part.json.gz')
    began=time.perf_counter();timings={}
    value=json.dumps(part,separators=(',',':'),allow_nan=False).encode()
    timings['json_encoding_seconds']=time.perf_counter()-began;began=time.perf_counter()
    require(len(value)<=admission.LIMIT,'Native part exceeds bound')
    packed=gzip.compress(value,compresslevel=1,mtime=0)
    timings['compression_seconds']=time.perf_counter()-began;began=time.perf_counter()
    from brass_ai.recovery.replay import CheckedGame
    checked=isinstance(part['game'],CheckedGame)
    if checked:
        from . import lab_features
        path=path.with_name(path.name.replace('.json.gz','.npz'))
        features,meta=lab_features.pack([part['game']],hashlib.sha256(packed).hexdigest(),compressed=False)
        # One transient file/receipt, using the existing NPZ arrays. The actor
        # has validated its bindings; the publisher independently validates
        # behavior. No dense feature lists or board histories cross the learner.
        buffer=io.BytesIO(features)
        with zipfile.ZipFile(buffer,'a',compression=zipfile.ZIP_STORED) as archive:
            archive.writestr('record.json.gz',packed)
            archive.writestr('features.json',json.dumps(meta,separators=(',',':')))
        packed=buffer.getvalue()
        require(len(packed)<=admission.LIMIT,'Native part exceeds bound')
        timings['feature_handoff_seconds']=time.perf_counter()-began;began=time.perf_counter()
    with path.open('xb') as raw:raw.write(packed)
    fingerprint=hashlib.sha256(packed).hexdigest()
    timings['write_hash_seconds']=time.perf_counter()-began
    handle=dict(path=str(path),sha256=fingerprint,branch_expected=part['branch_expected'],transfer_timings=timings)
    if checked:
        handle['feature_archive']='derived-features/4.0.0'
        # This proof travels over the actor's private IPC, never in the part JSON.
        handle['actor_validated']=getattr(part['game'],'actor_validated',False)
    return handle


def part_read(directory, handle):
    if handle.get('format')==lab_shared.FORMAT:
        return lab_shared.decode(lab_shared.read(handle))
    raw=Path(handle['path']);path=raw.resolve();base=Path(directory).resolve()
    require(path.parent==base and not raw.is_symlink(), 'Foreign native part path')
    with path.open('rb') as stream:
        size=os.fstat(stream.fileno()).st_size
        require(0<size<=admission.LIMIT,'Native part exceeds bound')
        packed=stream.read(size+1)
    require(len(packed)==size,'Native part size changed')
    require(len(packed)<=admission.LIMIT and hashlib.sha256(packed).hexdigest()==handle['sha256'],'Native part changed')
    combined=handle.get('feature_archive');meta=None;record=packed
    if combined:
        require(combined=='derived-features/4.0.0','Unknown part feature archive')
        with zipfile.ZipFile(io.BytesIO(packed)) as archive:
            require(len(archive.infolist())==6 and sum(x.file_size for x in archive.infolist())<=admission.LIMIT,
                    'Part archive exceeds bound')
            record=archive.read('record.json.gz');meta=json.loads(archive.read('features.json'))
        require(meta['replay_sha256']==hashlib.sha256(record).hexdigest(),'Part feature binding changed')
    with gzip.GzipFile(fileobj=io.BytesIO(record),mode='rb') as stream:data=stream.read(admission.LIMIT+1)
    require(len(data)<=admission.LIMIT,'Native part exceeds bound')
    value=json.loads(data)
    require(value['branch_expected']==handle['branch_expected'],'Native part role changed')
    if combined:
        from . import lab_features
        arrays=lab_features.unpack(packed,[value['game']],meta,combined=True)
        value['game']=lab_features.bind(value['game'],arrays)
        value['game'].actor_validated=handle.get('actor_validated') is True
    return value


class Publisher:
    """Admission-bound durable receipts; unclaimed partial outputs cannot train."""
    def __init__(self, directory, producer, config, scope, selected_mode=None, previous_producer=None):
        self.directory=Path(directory);self.producer=producer;self.config=config;self.scope=scope
        self.selected_mode=selected_mode or 'strict'
        self.previous_producer=previous_producer

    def binding(self,job,producer=None):
        generation=job['provenance'].get('critic_table',{}).get('generation')
        if generation is None:generation=job['provenance']['exploration']['age']
        return admission.binding(job,producer or self.producer,generation=generation,
            model_id=job['provenance']['learner_behavior']['model_id'],config=self.config,scope=self.scope)

    def path(self,job):return self.directory/'replay'/(frozen_id(job)+'.jsonl.gz')

    def retire_unclaimed(self,job):
        path=self.path(job);record=Path(str(path)+'.record.json')
        if record.exists() or not path.exists():return
        # This current pending family never published its final commit marker.
        # Preserve exact old bytes, but do not let them block deterministic
        # recollection (elapsed seconds/build IDs need not match a failed try).
        from brass_ai.recovery.retention import safe
        from brass_ai.recovery.replay import _cache_paths
        base=self.directory.resolve();destination=base/'unclaimed'/uuid.uuid4().hex[:16]
        safe(base,path);safe(base,destination);destination.mkdir(parents=True,exist_ok=False)
        retained=[]
        for source,name in zip((path,*_cache_paths(path)),('replay.jsonl.gz','features.npz','features.json'),strict=True):
            if not source.exists():continue
            safe(base,source);target=destination/name;safe(base,target)
            require(source.resolve().is_relative_to(base) and target.resolve().is_relative_to(base)
                    and not record.exists(),'Unclaimed family ownership changed')
            source.rename(target)
            retained.append(dict(source=source.name,path=name))
        write_json(destination/'owner.json',dict(job_id=frozen_id(job),training_admitted=False,
            files=retained,reason='Replay/feature bytes without the final independent record marker; retained, never admitted'))

    def load(self,job):
        path=self.path(job);receipt=path.with_name(path.name+'.record.json')
        if not receipt.exists():return None
        from brass_ai.recovery.retention import safe
        for target in (path,receipt):safe(self.directory.resolve(),target)
        record=json.loads(receipt.read_text())
        allowed=[self.binding(job)]
        if self.previous_producer is not None:
            previous=self.previous_producer if isinstance(self.previous_producer,list) else [self.previous_producer]
            allowed.extend(self.binding(job,p) for p in previous)
        require(record['path']==str(path) and record['native_admission']['admission'] in allowed,
                'Cached collection belongs to a different job/producer/config')
        # A disk sidecar alone cannot confer validation trust after a crash.
        # Re-admit it against this frozen plan before returning it to the caller.
        game=admission.read(path,record['native_admission'],allow_diagnostic=self.scope=='diagnostic',reuse_validation=False,
                            fast_recovery=self.selected_mode!='strict')
        issued={k:v for k,v in record['native_admission'].items() if k!='validation'}
        expected=admission.record(path,game,issued['admission'],issued)
        require(all(record.get(k)==v for k,v in expected.items() if k!='native_admission'),
                'Cached collection record differs from validated family')
        # Cold disk metadata is never trusted. Only AFTER complete validation
        # issue a current-process receipt for readers, preserving feature bytes.
        # It travels over private IPC; do not rewrite the old disk sidecar.
        admission._attest(issued,coverage=record['native_admission'].get('validation',{}).get('coverage'))
        return dict(record,native_admission=issued)

    def publish(self,job,parent,child,*,validation=None):
        from brass_ai.recovery import replay_binding
        checks_before=replay_binding.report()
        began=time.perf_counter();parts=self.directory/'parts';timings={}
        self.retire_unclaimed(job)
        timings['unclaimed_retirement_seconds']=time.perf_counter()-began;began=time.perf_counter()
        handles=[h for h in (parent,child) if h]
        if validation is not None:lab_perf.check(validation,job)
        parent=part_read(parts,parent);child=part_read(parts,child) if child else None
        timings['part_read_seconds']=time.perf_counter()-began;began=time.perf_counter()
        value=dict(format=FORMAT,job_id=frozen_id(job),frozen_job={k:v for k,v in job.items() if k!='path'},
            producer=self.producer,parent=parent,child=child,training_admitted=False,sha256=None)
        timings['family_assembly_seconds']=time.perf_counter()-began
        timings['actor_part_transfer']=[h.get('transfer_timings',{}) for h in handles]
        # The sole full family validation is inside admission.publish/convert.
        # In diagnostic scope still enforce the unchanged natural branch draw.
        if self.scope=='diagnostic':admission.branch_trigger(value,job,False)
        began=time.perf_counter()
        record=admission.publish(self.path(job),value,self.binding(job),validation_receipt=True,timings=timings,
                                 coverage=lab_perf.coverage(validation,child=child is not None) if validation else None,_owned_family=True)
        record['publication_seconds']=time.perf_counter()-began
        record['publication_phases']=timings
        record['binding_checks']=replay_binding.since(checks_before)
        semantic=validation is None or validation['semantic']
        branch=child is not None and (validation is None or validation['branch'])
        record['validation_metrics']={name:dict(eligible=1,executed=int(executed),skipped=int(not executed),reused=0,failed=0)
            for name,executed in (('structural',True),('sampling_rng_probability',semantic),
                                  ('branch_deep',branch),('schema_deep',semantic))}
        if not job.get('branching'):record['validation_metrics']['branch_deep']=dict(eligible=0,executed=0,skipped=0,reused=0,failed=0)
        record['native_metrics']=[dict(role=role,timings=part['timings'],
            native_cost=part.get('native_receipt',{}).get('cost'))
            for role,part in (('parent',parent),('child',child)) if part]
        write_json(Path(record['path']+'.record.json'),record)
        for handle in handles:
            if handle.get('format')==lab_shared.FORMAT:continue
            path=Path(handle['path']).resolve()
            # part_read already checked the exact bytes. This private lease is
            # retired here; hashing again before deletion cannot protect data
            # already detached/independently admitted by the publisher.
            require(path.parent==parts.resolve() and not path.is_symlink(),'Retired native part changed')
            path.unlink()
        return record


def _writer(connection, directory, producer, config, scope, cpu, stop, selected_mode=None, previous_producer=None):
    import torch
    from brass_ai.recovery.telemetry import Timings
    from brass_ai.recovery.collector_runtime import from_config,install,recipe
    install(from_config(config));config=recipe(config)
    torch.set_num_threads(1);publisher=Publisher(directory,producer,config,scope,selected_mode,previous_producer)
    connection.send(dict(ready=True))
    while not stop.is_set():
        if not connection.poll(.1):continue
        command=connection.recv()
        if command is None:return
        try:
            with cpu.acquire(1,Timings()):
                result=getattr(publisher,command['op'])(*command['args'],**(
                    dict(validation=command['validation']) if command.get('validation') is not None else {}))
            connection.send(dict(result=result))
        except BaseException:
            connection.send(dict(error=traceback.format_exc()));return


def _actors(connections, root, build, execution, mono, lease, cpu, stop, directory, inference=None, core=None):
    import torch
    from brass_ai.recovery.telemetry import Timings
    from .lab_b02_smoke import reference_part
    from brass_ai.recovery.collector_runtime import from_config,install,recipe
    install(from_config(dict(execution=execution)));execution=recipe(dict(execution=execution))['execution']
    if core is not None:
        from .lab_cpu_runtime import pin_process
        pin_process(core)
    torch.set_num_threads(execution['worker_threads'])
    def lane(connection):
        endpoint=lease.acquire();endpoint.timeout=30.
        try:
            from .lab_session import OrdinarySession
            command=[build['bootstrap']['binary']['path']]
            if os.name!='nt':command.insert(0,mono)
            with Pipe(command,binary=False) as bootstrap, OrdinarySession(root,build['collector']['path'],endpoint,execution,mono,cpu=inference) as session:
                connection.send(dict(ready=True))
                while not stop.is_set():
                    if not connection.poll(.1):continue
                    command=connection.recv()
                    if command is None:return
                    job,continuation=command['job'],command['continuation']
                    try:
                        with cpu.acquire(1,Timings()):
                            if route(job)[0]=='rust':
                                part=collect_lane(root,build['collector']['path'],bootstrap,job,endpoint,execution,
                                    continuation=continuation,cancel=stop,
                                    submit_branch=lambda packet:connection.send(dict(branch=packet)),session=session,mono=mono,
                                    raw_export=True,validation=command.get('validation'))
                            else:
                                require(continuation is None,'Reference continuation unsupported')
                                part=reference_part(root,job,endpoint,execution,mono)
                        if isinstance(part,lab_shared.Export):
                            def allocate(size):
                                connection.send(dict(shared_allocate=dict(size=size)))
                                while not connection.poll(.05):require(not stop.is_set(),'Shared allocation canceled')
                                reply=connection.recv()
                                require('shared_lease' in reply,'Missing shared allocation reply')
                                return reply['shared_lease']
                            result=lab_shared.write(part,allocate,admit=lambda:cpu.acquire(1,Timings()),cancel=stop)
                        else:
                            with cpu.acquire(1,Timings()):result=part_write(directory,part)
                        del part  # Do not retain a complete export during the next game.
                        connection.send(dict(result=result))
                    except BaseException:
                        connection.send(dict(error=traceback.format_exc()));return
        finally:endpoint.close()
    with ThreadPoolExecutor(max_workers=len(connections)) as pool:
        list(pool.map(lane,connections))


def recover_cached(jobs,load,*,workers,cancel,on_error):
    """Bounded CPU work/IPC; abort readers before executor joins on failure."""
    require(type(workers) is int and 1<=workers<=24,'Invalid recovery worker count')
    remaining=iter(jobs);pending={};records={}
    with ThreadPoolExecutor(max_workers=workers) as pool,_failure_scope(cancel,on_error):
        def submit():
            require(not cancel.is_set(),'Native recovery canceled')
            job=next(remaining,None)
            if job is not None:pending[pool.submit(load,job)]=job['index']
        for _ in range(min(len(jobs),2*workers)):submit()
        while pending:
            require(not cancel.is_set(),'Native recovery canceled')
            done,_=wait(pending,timeout=.05,return_when=FIRST_COMPLETED)
            for future in done:
                index=pending.pop(future);record=future.result()
                if record is not None:
                    require(record['job']==index and index not in records,'Misrouted cached collection')
                    records[index]=record
                submit()
    return [records[j['index']] for j in jobs if j['index'] in records]


def publisher_workers(jobs,capacity):
    # Validation is CPU-bound Python work, like the reference's per-actor
    # writers. Shared admission still bounds actors + writers + replay readers.
    # Keep short batches cheap: Windows spawn/import cost dominated the eight-
    # family profile. Scale beyond two only when there is work to amortize it.
    desired=max(2,(len(jobs)+7)//8)
    return min(24,desired,len(jobs),capacity)


class NativePool:
    def __init__(self,root,service,execution,build,config,signature,*,mono='mono',scope='ordinary',cpu=None):
        require(scope in ('ordinary','diagnostic'),'Unknown native pool scope')
        self.cpu=cpu
        from .lab_teacher_activation import is_on
        require(config.get('teacher_phase',{}).get('mode','off') in ('off','shadow') or is_on(config),
                'Ordinary backend excludes unqualified teacher on-mode')
        require(config.get('diversity',{}).get('mode','off')=='off','DOMiNO is retired')
        self.root=Path(root);self.service=service;self.execution=execution;self.build=build
        self.config=config;self.signature=signature;self.mono=mono or 'mono';self.scope=scope
        from brass_ai.recovery.collector_runtime import DEFAULTS,current,validate
        validate(execution)
        self.runtime=dict(current(),**{k:execution[k] for k in DEFAULTS if k in execution})
        self.last_collection_report=None;self.processes=[];self.connections=[];self.channel_leases=[]
        self.mp=multiprocessing.get_context('spawn');self.stop=self.mp.Event()
        self.shared=None;self.health_lock=threading.Lock();self.health_due=0.
        self.producer=dict(build_sha256=build['_receipt_sha256'],b01_sha256=build['b01']['sha256'],
                           signature=signature,profile='search_v1',v205_enabled=False)
        previous=build.get('_previous_build_sha256s',
            (build['_previous_build_sha256'],) if '_previous_build_sha256' in build else ())
        producers=[dict(self.producer,build_sha256=p) for p in previous]
        self.previous_producer=producers if len(producers)>1 else (producers[0] if producers else None)

    def receive(self,connection,*,submit=None,timeout=120,shared_allocate=None):
        if not hasattr(self,'health_lock'):self.health_lock=threading.Lock();self.health_due=0.
        end=time.monotonic()+timeout
        while time.monotonic()<end:
            require(not self.stop.is_set(),'Native pool canceled')
            if connection.poll(.05):
                try:message=connection.recv()
                except EOFError as error:raise RuntimeError('Native worker exited without a result') from error
                require('error' not in message,message.get('error',''))
                if 'shared_allocate' in message:
                    require(shared_allocate is not None,'Unexpected shared allocation')
                    connection.send(dict(shared_lease=shared_allocate(message['shared_allocate']['size'])))
                    continue
                if 'branch' in message:
                    require(submit is not None,'Unexpected branch reply');submit(message['branch']);continue
                return message
            require(self.service.error is None,'Native inference service failed: '+str(self.service.error))
            now=time.monotonic()
            if now>=self.health_due:
                # Reference dispatch checks liveness on a one-second idle
                # timeout. Re-scanning every process in every lane every 50ms
                # adds millions of process polls at the 48-lane lab setting.
                with self.health_lock:
                    if now>=self.health_due:
                        require(all(p.is_alive() for p in self.processes),'Native worker exited without a result')
                        self.health_due=now+5.
        raise TimeoutError('Native worker timed out; partial family cannot train')

    def collect(self,jobs,*,on_records=None):
        require(jobs and not self.processes,'Use a fresh native pool collection')
        self.stop=self.mp.Event()
        self.shared=lab_shared.Registry(self.stop)
        jobs=copy.deepcopy(jobs)
        require(all(j['provenance']['spec']==self.signature for j in jobs),'Native collection signature differs')
        for job in jobs:job.setdefault('stable_game_id',digest(dict(job=frozen_id(job),seed=job['seed'])))
        selected_mode=self.runtime['collector_validation'] if self.scope=='ordinary' else 'strict'
        validations=lab_perf.plan(jobs,selected_mode=selected_mode)
        parents={Path(j['path']).parent.resolve() for j in jobs}
        require(len(parents)==1,'Native jobs must share an iteration directory')
        directory=next(iter(parents))/'native-ordinary';directory.mkdir(parents=True,exist_ok=True)
        config=self.execution;lanes_per=config['hosts_per_process']
        # As in the C# reference, inspect/recover caches before starting actors.
        # Presence is only a scheduling hint: every hit is independently read
        # and fully validated by a publisher before it is returned or reused.
        probe=Publisher(directory,self.producer,self.config,self.scope)
        candidates=[j for j in jobs if Path(str(probe.path(j))+'.record.json').exists()]
        available=queue.Queue();width=lanes=0
        began=time.perf_counter();first=[];failure_lock=threading.Lock()
        def fail(error):
            with failure_lock:
                if not first:first.append((error,error.__traceback__))
            self.stop.set();self.service.cpu.abort()
        try:
            publishers=queue.Queue();publisher_count=publisher_workers(jobs,self.service.cpu.capacity)
            for _ in range(publisher_count):
                writer,child=self.mp.Pipe();self.connections.append(writer)
                from brass_ai.recovery.collector_runtime import configured
                p=self.mp.Process(target=_writer,args=(child,str(directory),self.producer,configured(self.config,self.runtime),self.scope,self.service.cpu,self.stop,selected_mode,self.previous_producer))
                self.processes.append(p)
                try:p.start()
                finally:child.close()
                publishers.put(writer)
            for connection in self.connections:require(self.receive(connection).get('ready') is True,'Native startup failed')
            writer_startup=time.perf_counter()-began;callback_lock=threading.Lock();owner=self
            class Store:
                def call(self,op,*args,validation=None):
                    writer=publishers.get(timeout=120)
                    try:
                        writer.send(dict(op=op,args=args,validation=validation));return owner.receive(writer)['result']
                    except BaseException as error:fail(error);raise
                    finally:publishers.put(writer)
                def notify(self,record):
                    try:
                        # Preserve the coordinator's original serialized callback
                        # contract while independent families publish in parallel.
                        with callback_lock:
                            if record and on_records:on_records([record])
                    except BaseException as error:fail(error);raise
                def load(self,job):
                    record=self.call('load',job)
                    self.notify(record)
                    return record
                def publish(self,job,parent,child,*,cancel=None):
                    require(not owner.stop.is_set(),'Native publication canceled')
                    handles=[h for h in (parent,child) if h and h.get('format')==lab_shared.FORMAT]
                    for handle in handles:owner.shared.verify(handle)
                    try:record=self.call('publish',job,parent,child,validation=validations[job['index']])
                    finally:
                        for handle in handles:owner.shared.release(handle)
                    require(not owner.stop.is_set(),'Native publication canceled')
                    self.notify(record)
                    return record
            store=Store();recovery_began=time.perf_counter()
            cached=recover_cached(candidates,store.load,workers=publisher_count,cancel=self.stop,on_error=fail)
            recovery_seconds=time.perf_counter()-recovery_began
            recovered={r['job'] for r in cached};missing=[j for j in jobs if j['index'] not in recovered]
            actor_began=time.perf_counter()
            if missing:
                # Start only missing parents, with all configured lanes for
                # continuations. A cache-only restart needs no engine or NN IPC.
                width=min(config['processes'],len(missing));lanes=width*lanes_per
                lease=self.service.channels(lanes);self.channel_leases.append(lease);new_connections=[]
                cores=[]
                if self.cpu is not None and self.cpu['pinning']=='one-lane-per-cpu':
                    from .lab_cpu_runtime import core_allocation
                    cores=core_allocation(width)
                for number in range(width):
                    pairs=[]
                    try:
                        for _ in range(lanes_per):
                            a,b=self.mp.Pipe();pairs.append((a,b));self.connections.append(a)
                        extra=() if self.cpu is None else (self.cpu,cores[number] if cores else None)
                        p=self.mp.Process(target=_actors,args=([b for a,b in pairs],str(self.root),self.build,dict(config,**self.runtime),
                            self.mono,lease,self.service.cpu,self.stop,str(directory/'parts'),*extra))
                        self.processes.append(p);p.start()
                    finally:
                        for a,b in pairs:b.close()
                    for a,b in pairs:available.put(a);new_connections.append(a)
                for connection in new_connections:require(self.receive(connection).get('ready') is True,'Native startup failed')
            actor_startup=time.perf_counter()-actor_began if missing else 0.
            def collect(job,continuation,submit):
                connection=available.get(timeout=120)
                try:
                    connection.send(dict(job=job,continuation=continuation,validation=validations[job['index']]))
                    return self.receive(connection,submit=submit,timeout=360,shared_allocate=self.shared.allocate)['result']
                except BaseException as error:
                    fail(error);raise
                finally:available.put(connection)
            records,report=collect_families(jobs,store,collect,collect,lanes=max(1,lanes),cancel=self.stop,
                model_cohorts=config.get('model_cohorts',False),on_error=fail,
                publication_workers=min(publisher_count,max(1,lanes)),cached_records=cached)
            live_seconds=report['wall_seconds']
            report.update(startup_seconds=writer_startup+actor_startup,processes=width,publisher_processes=publisher_count,
                writer_startup_seconds=writer_startup,actor_startup_seconds=actor_startup,
                cache_recovery_seconds=recovery_seconds,cache_candidates=len(candidates),
                live_collection_seconds=live_seconds,wall_seconds=time.perf_counter()-began,
                lanes=lanes,limit=2*lanes,publication_workers=min(publisher_count,lanes),
                board_histories_cross_learner=False,structural_family_validation_passes_per_publication=1,
                native_families=sum(route(r['native_admission']['admission']['job'])[0]=='rust' for r in records))
            report['validation_policy']=dict(contract=lab_perf.CONTRACT,mode=selected_mode,
                eligible=len(records),semantic_audited=sum(r['native_admission'].get('validation',{}).get('coverage',{}).get('semantic_audited',True) for r in records),
                family_every=64,branch_every=16)
            checks={}
            for record in records:
                if record['job'] in recovered:continue
                for name,counts in record.get('validation_metrics',{}).items():
                    total=checks.setdefault(name,dict(eligible=0,executed=0,skipped=0,reused=0,failed=0))
                    for key,value in counts.items():total[key]+=value
            report['validation_policy'].update(publication_checks=checks,cold_recovery_families=len(cached),
                                               first_per_stratum_per_collection=True)
            names={k for r in records if r['job'] not in recovered for k in r.get('binding_checks',{})}
            report['binding_checks']={k:sum(r.get('binding_checks',{}).get(k,0) for r in records if r['job'] not in recovered)
                                      for k in names}
            report['shared_buffers']=self.shared.report()
            report['collector_runtime']=dict(build_sha256=self.build['_receipt_sha256'],
                performance_wire=self.build.get('performance_wire','legacy-compact-arrays-forced/1'),
                completion_export=self.build.get('completion_export','legacy-separate-pages'),
                immutable_bindings=self.runtime['replay_bindings'])
            if self.cpu is not None:
                report['collector_runtime']['cpu_inference']=dict(contract=self.cpu['contract'],pinning=self.cpu['pinning'],
                    model_cache=self.cpu['model_cache'],runtime_sha256=self.cpu['runtime']['sha256'],
                    models=len(self.cpu['bundles']),exported_bundles=self.cpu['exported_bundles'])
            self.last_collection_report=report
            return records
        except BaseException:
            self.stop.set();self.service.cpu.abort()
            if first:
                error,tb=first[0];raise error.with_traceback(tb)
            raise
        finally:self.close()

    def close(self):
        began=time.perf_counter()
        self.stop.set()
        for connection in self.connections:
            try:connection.send(None)
            except (OSError,EOFError):pass
        for process in self.processes:
            if process.pid is None:
                process.close();continue
            process.join(timeout=3)
            if process.is_alive():
                import psutil
                try:descendants=psutil.Process(process.pid).children(recursive=True)
                except psutil.NoSuchProcess:descendants=[]
                for child in reversed(descendants):
                    try:child.kill()
                    except psutil.NoSuchProcess:pass
                process.terminate();process.join(timeout=3)
            process.close()
        for connection in self.connections:connection.close()
        self.connections=[];self.processes=[]
        for lease in self.channel_leases:self.service.release_channels(lease)
        self.channel_leases=[]
        if getattr(self,'shared',None) is not None:self.shared.close()
        if self.last_collection_report is not None and self.last_collection_report.get('shutdown_seconds') is None:
            self.last_collection_report['shutdown_seconds']=time.perf_counter()-began
            self.last_collection_report['collection_total_seconds']=self.last_collection_report['wall_seconds']+self.last_collection_report['shutdown_seconds']

    def __enter__(self):return self
    def __exit__(self,*_):self.close()
