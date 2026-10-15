"""Bounded hand-matching latency/allocation comparison on an immutable corpus."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time
import tracemalloc


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runtime',type=Path,required=True)
    p.add_argument('--inputs',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();sys.path.insert(0,str(args.runtime.resolve()/'src'))
    from brass_ai.recovery.encoding import PreparedEncoder
    from brass_ai.learning.contracts import compatibility
    rows=json.loads(args.inputs.read_text());result={}
    for mode in ('cold','persistent_miss','warm_hit'):
        enc=PreparedEncoder(cache_size=256 if mode=='warm_hit' else 0)
        def encode(row):
            current=PreparedEncoder(cache_size=0) if mode=='cold' else enc
            return current.encode(row['view'],row['choices'],deadline=time.monotonic()+60)
        for row in rows:encode(row)
        samples=[];large=[]
        for _ in range(7):
            for row in rows:
                start=time.perf_counter();e=encode(row);ms=(time.perf_counter()-start)*1000;samples.append(ms)
                if len(row['choices'])>=300:large.append(ms)
        tracemalloc.start();memory_enc=PreparedEncoder(cache_size=256 if mode=='warm_hit' else 0)
        for row in rows:
            current=PreparedEncoder(cache_size=0) if mode=='cold' else memory_enc
            current.encode(row['view'],row['choices'])
        _,peak=tracemalloc.get_traced_memory();tracemalloc.stop()
        def summary(values):
            ordered=sorted(values)
            return dict(calls=len(values),median_ms=statistics.median(values),p95_ms=ordered[int(.95*(len(ordered)-1))])
        result[mode]=dict(**summary(samples),large=summary(large),peak_python_bytes=peak)
    report=dict(compatibility=compatibility(),positions=len(rows),candidates=sum(len(r['choices']) for r in rows),
        max_candidates=max(len(r['choices']) for r in rows),inputs_sha256=hashlib.sha256(args.inputs.read_bytes()).hexdigest(),
        measurements=result,scope='CPU only, 7 passes, absolute request clock active. Cold includes construction; misses/hits reuse an encoder. Independent allocation pass, not total RSS/GPU memory. No search/model work or strength comparison.')
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
