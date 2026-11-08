"""Matched frozen authored-data PI00–PI03 component probe, bounded CPU only.

No NN, checkpoint, simulator game or training update is run. Raw/decoded data
originate from the same fixture. Real publishers/IPC/files/readers are measured.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import copy
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import queue
import sys
import threading
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/p) for p in ('src','scripts','tests/R2','tests/Recovery','tests/FeatureEngineering','tests/RustBackendLab')]
import torch
from brass_ai.learning.contracts import digest,require
from brass_ai.recovery.resources import CPUAdmission
from brass_ai.rust_backend import lab_pool as P,lab_shared as S,lab_perf as V,lab_admission as A
from brass_ai.step5.league import write_json
from test_b03 import family_fixture
from test_collector_performance import exported


def probe_sources():
    return {name:hashlib.sha256((ROOT/name).read_bytes().replace(b'\r\n',b'\n')).hexdigest() for name in
        ('tests/RustBackendLab/profile_collector_performance.py','tests/RustBackendLab/test_collector_performance.py',
         'tests/RustBackendLab/test_b03.py','scripts/rust_lab_b06_ordinary.py','scripts/rust_lab_b06_metrics.py')}


def arm(out,frozen,raw,*,shared,fast,workers):
    from brass_ai.recovery import replay_binding
    out.mkdir();mp=multiprocessing.get_context('spawn');stop=mp.Event();cpu=CPUAdmission(mp,max(8,workers),0)
    registry=S.Registry(stop);available=queue.Queue();processes=[];connections=[]
    producer=frozen[0][0]['producer'];config=frozen[0][2];handles=[];report=dict(immutable_bindings=replay_binding.enabled())
    validations=V.plan([f[0]['frozen_job'] for f in frozen],selected_mode='production_fast' if fast else 'strict')
    began_total=time.perf_counter()
    try:
        began=time.perf_counter()
        for _ in range(workers):
            parent,child=mp.Pipe();p=mp.Process(target=P._writer,args=(child,str(out),producer,config,'ordinary',cpu,stop))
            p.start();child.close();processes.append(p);connections.append(parent);available.put(parent)
        for connection in connections:require(connection.poll(30) and connection.recv().get('ready'),'Publisher startup failed')
        report['startup_seconds']=time.perf_counter()-began;began=time.perf_counter()
        for value in raw:
            handles.append(S.write(value,registry.allocate) if shared else P.part_write(out/'parts',S.decode(value)))
        report['handoff_seconds']=time.perf_counter()-began
        report['temporary_payload_write_bytes']=sum(Path(h['path']).stat().st_size for h in handles if 'path' in h)
        def publish(item):
            (family,_,_),handle=item;connection=available.get(timeout=30)
            try:
                connection.send(dict(op='publish',args=(family['frozen_job'],handle,None),
                    validation=validations[family['frozen_job']['index']] if fast else None))
                require(connection.poll(30),'Publisher timed out');reply=connection.recv()
                require('error' not in reply,reply.get('error',''));return reply['result']
            finally:
                if shared:registry.release(handle)
                available.put(connection)
        began=time.perf_counter()
        with ThreadPoolExecutor(max_workers=workers) as pool:records=list(pool.map(publish,zip(frozen,handles)))
        report['publication_seconds']=time.perf_counter()-began;began=time.perf_counter();checks_before=replay_binding.report()
        games=[A.read(r['path'],r['native_admission']) for r in records]
        report['reader_seconds']=time.perf_counter()-began
        report['reader_binding_checks']=replay_binding.since(checks_before)
        began=time.perf_counter();checks_before=replay_binding.report()
        # Reuse the actual training preparation implementation.
        from rust_lab_b06_ordinary import prepared_rows
        prepared=[prepared_rows([g],config,f[0]['frozen_job']['provenance']['learner_behavior']['policy_id'])
                  for f,g in zip(frozen,games,strict=True)]
        report['preparation_seconds']=time.perf_counter()-began
        report['preparation_binding_checks']=replay_binding.since(checks_before)
        names={k for r in records for k in r.get('binding_checks',{})}
        report['publication_binding_checks']={k:sum(r.get('binding_checks',{}).get(k,0) for r in records) for k in names}
        identities=[]
        for g in games:
            hashed=hashlib.sha256()
            for row in g['records']:
                e=g.encoded(row,None)
                for value in (e.state,e.actions,e.stable_to_relative):hashed.update(value.tobytes())
            identities.append(dict(game=digest(g),features=hashed.hexdigest()))
        # Additional equality work is outside accepted-work and reported.
        report['semantic_outputs']=identities
        report['prepared_rows']=sum(len(p) for p in prepared)
        report['_prepared']=prepared
        report['coverage']=[r['native_admission'].get('validation',{}).get('coverage') for r in records]
        report['worker_phase_sums']={key:sum(r['publication_phases'].get(key,0.) for r in records)
                                    for key,value in records[0]['publication_phases'].items() if isinstance(value,(int,float))}
    finally:
        began=time.perf_counter();stop.set()
        for connection in connections:
            try:connection.send(None)
            except (OSError,EOFError):pass
        for p in processes:
            p.join(timeout=3)
            if p.is_alive():p.terminate();p.join(timeout=3)
            p.close()
        for connection in connections:connection.close()
        registry.close();report['shutdown_seconds']=time.perf_counter()-began
        report['resources']=registry.report();report['cpu']=cpu.report()
    report['accepted_work_seconds']=sum(report[k] for k in
        ('startup_seconds','handoff_seconds','publication_seconds','reader_seconds','preparation_seconds','shutdown_seconds'))
    report['probe_wall_seconds']=time.perf_counter()-began_total
    return report


def arguments():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',required=True,type=Path);p.add_argument('--families',type=int,default=4)
    p.add_argument('--decisions',type=int,default=64);p.add_argument('--workers',type=int,default=2)
    p.add_argument('--reverse',action='store_true',help='Reverse arm order for the second bounded probe')
    p.add_argument('--bindings-compare',action='store_true',help='Only compare identical fast shared paths with immutable bindings off/on')
    p.add_argument('--export-results',action='store_true',help='Export success/failure from an artifacts/rust-backend output to results/')
    return p.parse_args()


def run(args,out):
    torch.set_num_threads(1)
    require(2<=args.families<=24 and 8<=args.decisions<=128 and 1<=args.workers<=min(24,args.families)
            and args.families*args.decisions<=1024,'Keep component smoke bounded')
    frozen=[family_fixture(seed=i*100,decisions=args.decisions) for i in range(args.families)]
    raw=[exported(f[0]['parent']) for f in frozen]
    report=dict(scope='Frozen authored CPU components only; no NN/checkpoint/simulator/training; not whole-training throughput',
                checkpoint_sha256=None,frozen_jobs=digest([f[0]['frozen_job'] for f in frozen]),
                families=args.families,decisions=args.families*args.decisions,workers=args.workers,arms={})
    from brass_ai.step5.league import implementation_id
    report['implementation']=implementation_id()
    report['probe_sources']=probe_sources()
    from brass_ai.recovery.replay_binding import enabled
    report['immutable_bindings']=enabled()
    order=[('legacy_strict',False,False),('shared_strict',True,False),('shared_fast',True,True)]
    if args.bindings_compare:order=[('shared_fast_bindings_off',True,True),('shared_fast_bindings_on',True,True)]
    if args.reverse:order.reverse()
    for name,shared,fast in order:
        previous=os.environ.get('BRASS_REPLAY_BINDINGS')
        try:
            if args.bindings_compare:os.environ['BRASS_REPLAY_BINDINGS']='0' if name.endswith('_off') else '1'
            value=arm(out/name,frozen,raw,shared=shared,fast=fast,workers=args.workers);report['arms'][name]=value
            write_json(out/name/'report.json',{k:v for k,v in value.items() if k!='_prepared'})
        finally:
            if previous is None:os.environ.pop('BRASS_REPLAY_BINDINGS',None)
            else:os.environ['BRASS_REPLAY_BINDINGS']=previous
        print(name,json.dumps({k:v for k,v in value.items() if k.endswith('_seconds') or k=='temporary_payload_write_bytes'}),flush=True)
    expected=report['arms'][order[0][0]]
    from rust_lab_b06_metrics import compare_rows
    prepared_outputs=[report['arms'][key].pop('_prepared') for key,_,_ in order]
    for value in prepared_outputs[1:]:
        for left,right in zip(prepared_outputs[0],value,strict=True):compare_rows(left,right)
    require(all(v['semantic_outputs']==expected['semantic_outputs'] and v['prepared_rows']==expected['prepared_rows']
                for v in report['arms'].values()),'Training game/features/row coverage differs')
    report['equal_training_outputs']=True
    require(implementation_id()==report['implementation'] and probe_sources()==report['probe_sources'],
            'Source changed during component profile; do not use its timings')
    report['source_unchanged']=True
    return report


def main():
    import traceback
    args=arguments();out=args.out.resolve()
    require(out.is_relative_to(ROOT/'artifacts') and not out.exists(),'Use a new artifacts output')
    if args.export_results:require(out.is_relative_to(ROOT/'artifacts/rust-backend'),'Results export needs artifacts/rust-backend output')
    out.mkdir(parents=True);report=dict(passed=False,scope='Bounded authored CPU components; no NN/checkpoint/training')
    try:report.update(run(args,out),passed=True)
    except BaseException as error:
        report.update(error=str(error),traceback=traceback.format_exc(),
            arms={p.parent.name:json.loads(p.read_text()) for p in out.glob('*/report.json')});raise
    finally:
        write_json(out/'report.json',report);print('Profile: '+str(out/'report.json'),flush=True)
        if args.export_results:
            from rust_lab_results import export
            try:print('Git-visible results: '+str(export(out)),flush=True)
            except Exception as error:
                if report['passed']:raise
                print('Result export also failed: '+str(error),file=sys.stderr,flush=True)


if __name__=='__main__':main()
