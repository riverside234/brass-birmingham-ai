"""Short CPU-only B06/B07 publication comparison on identical authored families."""
import json
from pathlib import Path
import statistics
import sys
import tempfile
import time
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/p) for p in ('src','scripts','tests/RustBackendLab','tests/R2','tests/Recovery','tests/FeatureEngineering')]


def run(output):
    import torch
    from test_b03 import family_fixture
    from brass_ai.rust_backend import lab_admission as A,lab_families as F,lab_pool as P
    from brass_ai.step5.league import write_json
    from rust_lab_b06_metrics import compare_games,compare_rows
    from rust_lab_b06_ordinary import prepared_rows
    torch.set_num_threads(2)
    fixtures=[family_fixture(kind=k) for k in ('selfplay','historical','legacy')]
    # Warm validation/schema caches equally; no engine, GPU or game is run.
    for f,a,_ in fixtures:A.convert(f,a)
    samples=[]
    with tempfile.TemporaryDirectory(prefix='b07-publication-',dir=ROOT/'artifacts') as temp:
        for repeat in range(3):
            games={}
            for arm in (('b06','b07') if repeat%2==0 else ('b07','b06')):
                directory=Path(temp)/f'{repeat}-{arm}';counts={'full_validation':0};records=[];loaded=[]
                original=A.validate_family
                def checked(*args,**kwargs):
                    counts['full_validation']+=1;return original(*args,**kwargs)
                with patch.object(A,'validate_family',side_effect=checked),patch.object(F,'validate_family',side_effect=checked):
                    began=time.perf_counter()
                    for index,(family,admitted,config) in enumerate(fixtures):
                        out=directory/str(index);job=family['frozen_job']
                        if arm=='b06':
                            sealed=F.FamilyStore(out/'families',family['producer']).publish(job,family['parent'],None)
                            record=A.publish(out/'replay.jsonl.gz',sealed,admitted)
                        else:
                            publisher=P.Publisher(out,family['producer'],config,'ordinary')
                            record=publisher.publish(job,P.part_write(out/'parts',family['parent']),None)
                        records.append(record);loaded.append(A.read(record['path'],record['native_admission']))
                    accepted=time.perf_counter()-began;first_passes=counts['full_validation']
                    began=time.perf_counter()
                    for record in records:A.read(record['path'],record['native_admission'])
                    warm=time.perf_counter()-began
                samples.append(dict(arm=arm,repeat=repeat,publish_and_read_seconds=accepted,warm_read_seconds=warm,
                    full_validations_to_first_read=first_passes,warm_full_validations=counts['full_validation']-first_passes))
                games[arm]=loaded
            compare_games(games['b06'],games['b07'])
            for (family,_,config),a,b in zip(fixtures,games['b06'],games['b07'],strict=True):
                learner=family['frozen_job']['provenance']['learner_behavior']['policy_id']
                compare_rows(prepared_rows([a],config,learner),prepared_rows([b],config,learner))
    medians={arm:{k:statistics.median(s[k] for s in samples if s['arm']==arm)
                 for k in ('publish_and_read_seconds','warm_read_seconds')} for arm in ('b06','b07')}
    report=dict(passed=True,samples=samples,medians=medians,
        scope='Three alternating CPU fixture batches, three eight-decision families per arm; same rows/features/targets. No games or GPU; not lab throughput.',
        validation_passes_per_family=dict(b06=3,b07=1,b06_warm=1,b07_warm=0),
        lab_qualification='Pending')
    write_json(output,report);print(json.dumps(report,indent=2))
    return report


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True)
    run(p.parse_args().out)
