"""Short R3/R4 matched publication/read smoke with retained actor features."""
import copy
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
    path=ROOT/'distribution/rust-lab-b07-g1250/rust-lab-b07-g1250-r3.zip'
    require(hashlib.sha256(path.read_bytes()).hexdigest()=='fd28167f8ea8bbd1fd5087e37ae5ed70937c5e9da2b316bad867ce7494ee840b','Original R3 changed')
    modules={}
    with zipfile.ZipFile(path) as archive:
        for name in ('lab_collection','lab_families','lab_admission','lab_pool'):
            key='brass_ai.rust_backend._r3_'+name
            module=types.ModuleType(key);module.__package__='brass_ai.rust_backend';sys.modules[key]=module
            exec(compile(archive.read('src/brass_ai/rust_backend/'+name+'.py'),str(path)+'/'+name+'.py','exec'),module.__dict__)
            modules[name]=module
    old=modules['lab_admission'];pool=modules['lab_pool'];families=modules['lab_families']
    old.route=modules['lab_collection'].route;families.route=old.route
    old.validate_family=families.validate_family;pool.admission=old
    return old,pool


def run(out):
    import torch
    from test_b03 import family_fixture
    from brass_ai.recovery.replay import check_game
    from brass_ai.rust_backend import lab_admission as A, lab_pool as P
    from brass_ai.step5.league import write_json
    from rust_lab_b06_metrics import compare_games,compare_rows
    from rust_lab_b06_ordinary import prepared_rows
    torch.set_num_threads(2);old_a,old_p=baseline()
    fixtures=[family_fixture(kind=k) for k in ('selfplay','historical','legacy')]
    for family,binding,_ in fixtures:
        # Both actor paths already encoded these rows; R3 discards the metadata
        # at JSON handoff, R4 preserves it. Encoding is outside both timed arms.
        g=check_game(family['parent']['game']);g.actor_validated=True;family['parent']['game']=g
        for a in (old_a,A):a.convert(family,binding)
    samples=[]
    with tempfile.TemporaryDirectory(prefix='b07-r4-',dir=ROOT/'artifacts') as tmp:
        for repeat in range(2):
            games={}
            for arm in (('r3','r4') if repeat==0 else ('r4','r3')):
                a,p=(old_a,old_p) if arm=='r3' else (A,P)
                records=[];phases={};began=time.perf_counter()
                for i,(family,_,config) in enumerate(fixtures):
                    directory=Path(tmp)/f'{repeat}-{arm}'/str(i)
                    pub=p.Publisher(directory,family['producer'],config,'ordinary')
                    records.append(pub.publish(family['frozen_job'],p.part_write(directory/'parts',family['parent']),None))
                phases['publication_seconds']=time.perf_counter()-began
                for label in ('first_read_seconds','warm_read_seconds'):
                    began=time.perf_counter();values=[a.read(r['path'],r['native_admission']) for r in records]
                    phases[label]=time.perf_counter()-began
                    if label=='first_read_seconds':games[arm]=values
                phases['publication_first_read_seconds']=phases['publication_seconds']+phases['first_read_seconds']
                phases['validation_assembly_seconds']=sum(r['publication_phases']['full_validation_and_assembly_seconds'] for r in records)
                samples.append(dict(arm=arm,repeat=repeat,**phases))
            compare_games(games['r3'],games['r4'])
            for (family,_,config),x,y in zip(fixtures,games['r3'],games['r4'],strict=True):
                learner=family['frozen_job']['provenance']['learner_behavior']['policy_id']
                compare_rows(prepared_rows([x],config,learner),prepared_rows([y],config,learner))
    medians={a:{k:statistics.median(s[k] for s in samples if s['arm']==a) for k in phases} for a in ('r3','r4')}
    report=dict(passed=True,medians=medians,samples=samples,identical_games_features_targets=True,
        scope='Two alternating pairs; same three small authored families, including archived V4 fallback. Component timings, no g1250 GPU/accepted-work claim.')
    write_json(out,report);return report


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',required=True,type=Path)
    print(json.dumps(run(p.parse_args().out),indent=2))
