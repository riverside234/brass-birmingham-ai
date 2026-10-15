"""Bounded same-feature encoder benchmark; no training or strength evaluation.

Micro timings retain the historical report's three encoding modes. The second
probe isolates coordinator -> eight worker roots -> replay on the same permitted
inputs, comparing former evaluator-local ownership with the active actor cache.
This is encoding latency, not an end-to-end search throughput measurement.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import platform
import statistics
import time
from unittest.mock import Mock

import numpy as np
import torch

from brass_ai.learning.contracts import SOURCE_ROOT,compatibility,require
from brass_ai.learning.features import Encoder
from brass_ai.learning.model import ModelConfig
from brass_ai.recovery.encoding import PreparedEncoder
from brass_ai.recovery.inference import Endpoint,RemoteEvaluator,share_encoder_cache
from brass_ai.step5.league import implementation_id,write_json
from fe_fixtures import benchmark_positions


def summary(values):
    values=sorted(values)
    return dict(n=len(values),median_ms=statistics.median(values),
        p95_ms=values[int(.95*(len(values)-1))],min_ms=min(values),max_ms=max(values),samples_ms=values)


def verify(got,want):
    require(got.choice_ids==want.choice_ids,'Changed candidate ordering')
    for key in ('state','actions','stable_to_relative'):
        np.testing.assert_array_equal(getattr(got,key),getattr(want,key))
        require(not getattr(got,key).flags.writeable,'Mutable cached input')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--inputs',type=Path)
    args=parser.parse_args();out=args.output.resolve()
    require(out.is_relative_to(SOURCE_ROOT/'artifacts') and not out.exists(),'Use a new isolated artifacts directory')
    began=time.perf_counter();source=implementation_id();torch.set_num_threads(2)
    rows=json.loads(args.inputs.read_text(encoding='utf-8')) if args.inputs else benchmark_positions()
    require(1<=len(rows)<=16,'Bounded corpus only')
    oracle=Encoder();expected=[oracle.encode(r['view'],r['choices'],[0.]*4,[0.]*8) for r in rows]
    fingerprint=hashlib.sha256(b''.join(e.state.tobytes()+e.actions.tobytes()+e.stable_to_relative.tobytes() for e in expected)).hexdigest()
    write_json(out/'inputs.json',rows)
    micro={name:[] for name in ('new_encoder','persistent_miss','warm_hit')}
    roots={name:[] for name in ('former_local','shared_actor')};construction={name:[] for name in roots}
    refreshed={name:[] for name in roots};counts={name:[] for name in roots}
    # Alternate case order across four blocks. Inputs, outputs and concurrency
    # are identical; the only comparison is ownership/lifetime of input caches.
    for block in range(4):
        names=list(micro) if block%2==0 else list(reversed(micro))
        for name in names:
            persistent=PreparedEncoder(cache_size=256 if name=='warm_hit' else 0)
            for row in rows:persistent.encode(row['view'],row['choices'],[0.]*4,[0.]*8)
            for _ in range(3):
                for row,want in zip(rows,expected):
                    start=time.perf_counter()
                    enc=PreparedEncoder(cache_size=0) if name=='new_encoder' else persistent
                    got=enc.encode(row['view'],row['choices'],[0.]*4,[0.]*8)
                    micro[name].append(1000*(time.perf_counter()-start));verify(got,want)
        for name in (list(roots) if block%2==0 else list(reversed(roots))):
            lanes=[Endpoint(Mock(),Mock(),i) for i in range(9)]
            shared=share_encoder_cache(lanes) if name=='shared_actor' else None
            # Emulate former constructor ownership using the same current
            # feature implementation, without changing production source.
            def clients(identity):
                result=[]
                for endpoint in lanes:
                    if name=='former_local':endpoint._encoder=None
                    result.append(RemoteEvaluator(endpoint,identity))
                return result
            start=time.perf_counter();owners=clients('a'*64);opponents=clients('b'*64)
            replay=PreparedEncoder() if name=='former_local' else owners[0].encoder
            construction[name].append(1000*(time.perf_counter()-start))
            with ThreadPoolExecutor(max_workers=8) as pool:
                # Warm thread startup separately from root timings.
                list(pool.map(lambda i:i,range(8)))
                for row,want in zip(rows,expected):
                    def encode(client):return client.encoder.encode(row['view'],row['choices'],[0.]*4,[0.]*8)
                    start=time.perf_counter();values=[encode(owners[0])]
                    values+=list(pool.map(encode,owners[1:]))
                    values.append(replay.encode(row['view'],row['choices'],[0.]*4,[0.]*8,validated=True))
                    roots[name].append(1000*(time.perf_counter()-start))
                    for got in values:verify(got,want)
            caches=([shared] if shared is not None else [c.encoder.cache for c in owners+opponents]+[replay.cache])
            counts[name].append({k:sum(c.report()[k] for c in caches) for k in ('warm_hit','miss','retained','capacity','evictions')})
            start=time.perf_counter();replacement=clients('c'*64);refreshed[name].append(1000*(time.perf_counter()-start))
            for old,new in zip(owners,replacement):
                require((old.encoder is new.encoder)==(name=='shared_actor'),'Encoder lifetime mismatch')
            # Cross-model features are reusable; model outputs are never measured
            # or cached by this probe. Actual prediction parity has its own test.
            for row,want in zip(rows,expected):verify(replacement[0].encoder.encode(row['view'],row['choices'],[0.]*4,[0.]*8),want)
            for endpoint in lanes:endpoint.close()
        print(json.dumps(dict(block=block+1,of=4)),flush=True)
    require(implementation_id()==source,'Production source changed during benchmark')
    report=dict(implementation=source,compatibility=compatibility(),model=ModelConfig.preset().to_dict(),
        platform=platform.platform(),python=platform.python_version(),torch=torch.__version__,cpu_threads=2,
        positions=len(rows),eras=sorted({r['view']['context']['era'] for r in rows}),
        candidates=[len(r['choices']) for r in rows],input_sha256=fingerprint,exact_parity=True,
        encoding={k:summary(v) for k,v in micro.items()},
        root_workers_replay={k:dict(latency=summary(roots[k]),construction=summary(construction[k]),
            model_replacement=summary(refreshed[k]),cache_blocks=counts[k]) for k in roots},
        elapsed_seconds=time.perf_counter()-began,
        scope='14-position permitted corpus; encoding only, four alternating blocks, eight worker threads. No model forward, native search, training, GPU throughput or playing-strength claim.')
    write_json(out/'report.json',report)
    print(json.dumps({k:v for k,v in report.items() if k in ('elapsed_seconds','positions','exact_parity')}),flush=True)


if __name__=='__main__':main()
