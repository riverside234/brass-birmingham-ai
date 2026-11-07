"""Short matched R2/R3 CPU publication/read comparison; no games or GPU."""
import hashlib
import json
from pathlib import Path
import statistics
import sys
import tempfile
import time
import types
import zipfile

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/p) for p in ('src','scripts','tests/RustBackendLab','tests/R2','tests/Recovery','tests/FeatureEngineering')]


def baseline():
    from brass_ai.learning.contracts import require
    path=ROOT/'distribution/rust-lab-b07-g1250/rust-lab-b07-g1250-r2.zip'
    require(hashlib.sha256(path.read_bytes()).hexdigest()=='1cda06a47ee5c7b875e55ca011b722549d2d79191dca750b36227c3e4d17df92','Original R2 archive changed')
    modules=[]
    with zipfile.ZipFile(path) as archive:
        for name in ('lab_admission','lab_pool'):
            key='brass_ai.rust_backend._b07_r2_'+name
            module=types.ModuleType(key);module.__package__='brass_ai.rust_backend';sys.modules[key]=module
            exec(compile(archive.read('src/brass_ai/rust_backend/'+name+'.py'),str(path)+'/'+name+'.py','exec'),module.__dict__)
            modules.append(module)
    modules[1].admission=modules[0]
    return modules


def run(output):
    import torch
    from test_b03 import family_fixture
    from brass_ai.rust_backend import lab_admission as A,lab_pool as P
    from brass_ai.step5.league import write_json
    from rust_lab_b06_metrics import compare_games,compare_rows
    from rust_lab_b06_ordinary import prepared_rows
    torch.set_num_threads(2)
    old_a,old_p=baseline()
    fixtures=[family_fixture(kind=k) for k in ('selfplay','historical','legacy')]
    for a in (old_a,A):
        for f,b,_ in fixtures:a.convert(f,b)
        a._validation_version()
    samples=[]
    with tempfile.TemporaryDirectory(prefix='b07-r3-publication-',dir=ROOT/'artifacts') as tmp:
        for repeat in range(3):
            games={}
            for arm in (('r2','r3') if repeat%2==0 else ('r3','r2')):
                a,p=(old_a,old_p) if arm=='r2' else (A,P)
                loaded=[];records=[];phases={}
                began=time.perf_counter()
                for i,(family,_,config) in enumerate(fixtures):
                    out=Path(tmp)/f'{repeat}-{arm}'/str(i)
                    publisher=p.Publisher(out,family['producer'],config,'ordinary')
                    records.append(publisher.publish(family['frozen_job'],p.part_write(out/'parts',family['parent']),None))
                phases['publication_seconds']=time.perf_counter()-began
                for label in ('first_read_seconds','warm_read_seconds'):
                    began=time.perf_counter()
                    values=[a.read(r['path'],r['native_admission']) for r in records]
                    phases[label]=time.perf_counter()-began
                    if label=='first_read_seconds':loaded=values
                phases['publication_first_read_seconds']=phases['publication_seconds']+phases['first_read_seconds']
                samples.append(dict(arm=arm,repeat=repeat,**phases));games[arm]=loaded
            compare_games(games['r2'],games['r3'])
            for (family,_,config),a,b in zip(fixtures,games['r2'],games['r3'],strict=True):
                learner=family['frozen_job']['provenance']['learner_behavior']['policy_id']
                compare_rows(prepared_rows([a],config,learner),prepared_rows([b],config,learner))
    medians={arm:{key:statistics.median(s[key] for s in samples if s['arm']==arm) for key in phases} for arm in ('r2','r3')}
    report=dict(passed=True,samples=samples,medians=medians,identical_games_features_targets=True,
        scope='Three alternating batches; three eight-decision authored families each. Serial publication isolates reduced work; no game/GPU/parallel-publisher throughput claim.',lab_qualification='Pending')
    write_json(output,report);print(json.dumps(report,indent=2));return report


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True)
    run(p.parse_args().out)
