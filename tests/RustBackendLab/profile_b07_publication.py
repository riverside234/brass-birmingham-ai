"""Short authored-data publication/recovery profile; no simulator, NN or training.

Both arms use identical frozen families/features. Startup is outside the timed
work phases. This is a CPU component comparison, not whole-training throughput.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import copy
import hashlib
import json
import multiprocessing
from pathlib import Path
import queue
import statistics
import sys
import time
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/p) for p in ('src','scripts','tests/R2','tests/Recovery','tests/FeatureEngineering','tests/RustBackendLab')]
import torch
from brass_ai.learning.contracts import digest,require
from brass_ai.recovery.replay import check_game,CheckedGame
from brass_ai.recovery.resources import CPUAdmission
from brass_ai.rust_backend import lab_pool as P,lab_admission as A,lab_features as F
from brass_ai.step5.league import write_json
from test_b03 import family_fixture


def profile(out,frozen,workers,*,serial_recovery):
    out.mkdir();mp=multiprocessing.get_context('spawn');stop=mp.Event();cpu=CPUAdmission(mp,8,0)
    available=queue.Queue();processes=[];connections=[];handles=[]
    producer=frozen[0][0]['producer'];config=frozen[0][2]
    for family,_,_ in frozen:
        part=copy.deepcopy(family['parent']);part['game']=check_game(part['game']);part['game'].actor_validated=True
        handles.append(P.part_write(out/'parts',part))
    def call(op,*args):
        connection=available.get(timeout=30)
        try:
            connection.send(dict(op=op,args=args));require(connection.poll(30),'Profile worker timed out')
            reply=connection.recv();require('error' not in reply,reply.get('error',''))
            return reply['result']
        finally:available.put(connection)
    try:
        began=time.perf_counter()
        for _ in range(workers):
            parent,child=mp.Pipe();p=mp.Process(target=P._writer,args=(child,str(out),producer,config,'ordinary',cpu,stop))
            p.start();child.close();processes.append(p);connections.append(parent);available.put(parent)
        for connection in connections:
            require(connection.poll(30) and connection.recv().get('ready'),'Profile startup failed')
        startup=time.perf_counter()-began;began=time.perf_counter()
        with ThreadPoolExecutor(max_workers=workers) as pool:
            records=list(pool.map(lambda item:call('publish',item[0][0]['frozen_job'],item[1],None),zip(frozen,handles)))
        publication=time.perf_counter()-began
        # Simulate old-source durable sidecars. They must pass a complete cold
        # validation, and only a newly issued in-memory receipt can be reused.
        for record in records:
            old=copy.deepcopy(record);old['native_admission']['validation']['validator']='prior-reviewed-source'
            write_json(Path(record['path']+'.record.json'),old)
        began=time.perf_counter()
        if serial_recovery:restored=[call('load',family['frozen_job']) for family,_,_ in frozen]
        else:
            restored=P.recover_cached([f['frozen_job'] for f,_,_ in frozen],lambda job:call('load',job),
                workers=workers,cancel=stop,on_error=lambda error:stop.set())
        recovery=time.perf_counter()-began;began=time.perf_counter();identities=[]
        for record in restored:
            receipt=(json.loads(Path(record['path']+'.record.json').read_text()) if serial_recovery else record)
            game=A.read(record['path'],receipt['native_admission'])
            arrays=hashlib.sha256()
            for row in game['records']:
                e=game.encoded(row,None)
                for value in (e.state,e.actions,e.stable_to_relative):arrays.update(value.tobytes())
            identities.append(dict(game=digest(game),features=arrays.hexdigest()))
        read_seconds=time.perf_counter()-began
        phases={k:sum(r['publication_phases'].get(k,0.) for r in records)
                for k,v in records[0]['publication_phases'].items() if isinstance(v,(int,float))}
        return dict(workers=workers,startup_seconds=startup,publication_seconds=publication,
            cold_recovery_seconds=recovery,reader_seconds=read_seconds,
            recovery_and_read_seconds=recovery+read_seconds,publication_worker_phase_sums=phases,
            cpu_peak=cpu.report()['peak'],semantic_outputs=identities)
    finally:
        stop.set()
        for connection in connections:
            try:connection.send(None)
            except (OSError,EOFError):pass
        for p in processes:
            p.join(timeout=3)
            if p.is_alive():p.terminate();p.join(timeout=3)
        for connection in connections:connection.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--families',type=int,default=8)
    parser.add_argument('--decisions',type=int,default=128)
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--read-existing',type=Path,help='Short ABBA legacy/current binding comparison on a previous profile')
    args=parser.parse_args();torch.set_num_threads(1)
    require(2<=args.families<=16 and 8<=args.decisions<=256 and 2<=args.workers<=8,'Keep the CPU profile bounded')
    out=args.out.resolve();require(out.is_relative_to(ROOT/'artifacts') and not out.exists(),'Use a new artifacts/ output')
    out.mkdir(parents=True)
    if args.read_existing:
        base=args.read_existing.resolve()
        require(base.is_relative_to(ROOT/'artifacts'),'Use an owned existing profile')
        saved=json.loads((base/'report.json').read_text());count=min(4,saved['families'])
        records=[]
        for path in sorted((base/'baseline/replay').glob('*.record.json'))[:count]:
            record=json.loads(path.read_text());proof=record['native_admission']['admission']
            _,_,config=family_fixture()
            records.append(P.Publisher(base/'baseline',proof['producer'],config,'ordinary').load(proof['job']))
        require(len(records)==count,'Incomplete profile cache')
        def legacy(game,arrays,*,start=0):return CheckedGame(game).bind(arrays[start:start+len(game['records'])])
        current=F.bind;read_features=F.read;times={'legacy':[],'current':[]};expected=None
        def deep_read(*args,**kwargs):return read_features(*args,**dict(kwargs,deep=True))
        for arm in ('legacy','current','current','legacy'):
            with patch.object(F,'bind',legacy if arm=='legacy' else current),\
                    patch.object(F,'read',deep_read if arm=='legacy' else read_features):
                began=time.perf_counter();games=[A.read(r['path'],r['native_admission']) for r in records]
                times[arm].append(time.perf_counter()-began)
            identities=[digest(g) for g in games]
            if expected is None:expected=identities
            require(identities==expected,'Binding changed training games')
        report=dict(scope='ABBA warm read: legacy deep checks/double input hashes vs attested reuse; same bytes/receipts; no NN/games/training',
            families=count,decisions=sum(len(g['records']) for g in games),timings=times,
            median_seconds={k:statistics.median(v) for k,v in times.items()},equal_training_games=True,
            implementation=A._validation_version())
        write_json(out/'report.json',report);print(json.dumps(report),flush=True);return
    frozen=[family_fixture(seed=i*100,decisions=args.decisions) for i in range(args.families)]
    baseline=profile(out/'baseline',frozen,2,serial_recovery=True)
    candidate=profile(out/'candidate',frozen,args.workers,serial_recovery=False)
    require(baseline['semantic_outputs']==candidate['semantic_outputs'],'Training game/features differ')
    report=dict(scope='Authored frozen CPU publication/cold recovery only; no NN, simulator or training; startup separate',
        families=args.families,decisions=args.families*args.decisions,source_results_commit='097327ef4fab7190f8d91ce4996dd1d4db4c5ce2',
        baseline=baseline,candidate=candidate,equivalent_training_game_and_features=True,
        implementation=A._validation_version())
    write_json(out/'report.json',report)
    for label,arm in (('baseline',baseline),('candidate',candidate)):
        print(label,json.dumps({k:v for k,v in arm.items() if k not in ('semantic_outputs','publication_worker_phase_sums')}),flush=True)
    print('Profile:',out/'report.json',flush=True)


if __name__=='__main__':main()
