"""Bounded CPU input/reader/bootstrap comparison. No training or live games."""
import argparse
import json
from pathlib import Path
import statistics
import tempfile
import time
import numpy as np
from brass_ai.bank import sha
from brass_ai.learning.batching import Request
from brass_ai.learning.contracts import SOURCE_ROOT,require
from brass_ai.recovery import inputs,plackett_luce as pl
from brass_ai.recovery.encoding import PreparedEncoder
from brass_ai.recovery.ingress import reader_pool
from brass_ai.recovery.rating_ingress import RatingIngress,validate_shard
from brass_ai.recovery.quality_config import RATING
from brass_ai.recovery.retention_digests import PlanDigests,METADATA_REUSE
from brass_ai.recovery.admitted_inputs import request as admitted_request
from test_collection_throughput import position_with_history,replay_fixture
from test_strength_rating import simulated,AGENTS


def input_trial(admitted,count=40):
    view,choices=position_with_history(256);encoder=PreparedEncoder()
    began=time.perf_counter();outputs=[]
    for step in range(count):
        v=dict(view,context=dict(view['context'],revision=step));c=choices
        if admitted:v,c=inputs.BOUNDARY.admit(v,c)
        else:inputs.BOUNDARY.check(v,c)
        request=(admitted_request if admitted else Request)(str(step),'f'*64,v,c,time.monotonic()+60)
        require(request.digest()==request.input_digest,'Input digest differs')
        prediction=encoder.encode(request.view,request.choices,request.style,request.intent)
        require(request.digest()==request.input_digest,'Response input digest differs')
        recorded=encoder.encode(v,c,request.style,request.intent,validated=True)
        require(np.array_equal(prediction.state,recorded.state),'Recorded encoding differs')
        outputs.append((request.input_digest,prediction.state.tobytes(),prediction.actions.tobytes()))
    return time.perf_counter()-began,outputs


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    out=a.output.resolve();require(out.is_relative_to(SOURCE_ROOT/'artifacts') and not out.exists(),'Use a new isolated artifacts directory')
    times={'before':[],'after':[]};input_trial(False,2);input_trial(True,2)
    for repeat in range(3):
        values={}
        for name in (('before','after') if repeat%2==0 else ('after','before')):
            elapsed,values[name]=input_trial(name=='after');times[name].append(elapsed)
        require(values['before']==values['after'],'Admission changed request identities or encoded tensors')
    result=dict(input_seconds=times,input_median_seconds={k:statistics.median(v) for k,v in times.items()},input_identity_and_encoding_equal=True)
    print('input',result['input_median_seconds'],flush=True)
    with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as temp:
        records,jobs,spec=replay_fixture(Path(temp),8)
        began=time.perf_counter();serial=[validate_shard(([r],jobs,spec))[0][0][1] for r in records]
        serial_seconds=time.perf_counter()-began;began=time.perf_counter()
        with reader_pool(4) as pool:
            ingress=RatingIngress(pool,jobs,spec)
            for record in reversed(records):ingress.submit([record])
            rows,metrics=ingress.finish(records)
        require(rows==serial,'Parallel replay changed outcomes/order')
        result['replay']=dict(shards=8,serial_seconds=serial_seconds,parallel4_with_startup_seconds=time.perf_counter()-began,
            metrics=metrics,rows_identical=True)
        path=Path(temp)/'cleanup-fixture';path.write_bytes(b'a'*(8*1024*1024));cache=PlanDigests()
        first=cache.digest(path);require(cache.digest(path)==first==sha(path),'Cleanup digest differs')
        result['cleanup']=cache.report()
    reports={};samples={};times={}
    games=simulated(setups=32)
    for workers in (4,16):
        began=time.perf_counter();reports[workers],samples[workers]=pl.estimate(games,AGENTS,RATING,compute=dict(pl.COMPUTE,workers=workers))
        times[workers]=time.perf_counter()-began;print('bootstrap',workers,times[workers],flush=True)
    require(np.array_equal(samples[4],samples[16]),'Bootstrap samples changed')
    require(reports[4]['ratings']==reports[16]['ratings'] and reports[4]['differences']==reports[16]['differences'],
            'Ratings/paired confidence intervals/decisions changed')
    result['bootstrap']=dict(games=len(games),agents=len(AGENTS),refits=RATING['bootstrap'],seconds=times,
        effective_workers={k:r['bootstrap_execution']['effective_workers'] for k,r in reports.items()},
        samples_and_ratings_and_paired_intervals_equal=True,max_sample_difference=float(np.max(np.abs(samples[4]-samples[16]))))
    result['scope']='CPU authored fixtures only; no collection/evaluation games or production checkpoints. Small reader/PL workloads include process startup and do not predict production speedups. Windows always rehashes cleanup files.'
    out.mkdir(parents=True);(out/'report.json').write_text(json.dumps(result,indent=2)+'\n')
    print(out/'report.json',flush=True)


if __name__=='__main__':main()
