"""Bounded real-replay benchmark: serial full-shard vs selected-row/16 readers."""
import argparse
import copy
import gzip
import json
import time
from pathlib import Path
from unittest.mock import patch

import torch

from brass_ai.bank import read,sha
from brass_ai.learning.contracts import SOURCE_ROOT,project_path,require
from brass_ai.recovery import demonstrations as demos,rehearsal_window as windows
from brass_ai.recovery.ingress import reader_pool
from brass_ai.step5.league import write_json,implementation_id


def run(args):
    root=SOURCE_ROOT;out=project_path(root,args.output)
    require(out.is_relative_to(root/'artifacts') and not out.exists(),'Use a new isolated artifacts directory')
    torch.set_num_threads(2);out.mkdir(parents=True)
    manifest=read(project_path(root,args.manifest));entries=[e for e in manifest['entries'] if e['split']=='train']
    source=project_path(root,entries[0]['path']);before={str(source):sha(source)}
    with gzip.open(source,'rt',encoding='utf-8') as handle:
        header=json.loads(next(handle));games=[json.loads(line) for line in handle]
    original=next(g for g in games if g['game_id']==entries[0]['game_id'])
    # Replicate one real, audited game into separate diagnostic shards. Views,
    # labels and actions stay unchanged; only game IDs and file receipts vary.
    generated=[]
    for i in range(16):
        game=copy.deepcopy(original);game['game_id']+=f'-rehearsal-bench-{i}'
        path=out/f'shard-{i:02d}.jsonl.gz'
        with gzip.open(path,'wt',encoding='utf-8',compresslevel=1) as handle:
            handle.write(json.dumps(header)+'\n'+json.dumps(game)+'\n')
        generated.append(dict(entries[0],path=path.relative_to(root).as_posix(),sha256=sha(path),
            game_id=game['game_id'],features=None))
    data_manifest=out/'manifest.json';write_json(data_manifest,dict(manifest,entries=generated))
    index=windows.TrainIndex(root,data_manifest)
    config=dict(windows.DEFAULT,rows_per_iteration=min(2048,index.rows),shard_draws=64)
    iteration=next(i for i in range(100) if len(index.select(config,49,i))==16)
    original_prepare=demos._prepared_shard
    def previous(task,selection=None):
        full=original_prepare(task)
        return full if selection is None else [full[i] for i in selection]
    results=[];reference=None
    def measure(label,executor=None,workers=1):
        nonlocal reference
        rng=torch.get_rng_state().clone();began=time.perf_counter()
        with windows.window(root,out/label,index,config,seed=49,iteration=iteration,
                            device='cpu',workers=workers,executor=executor) as data:
            signatures=[(r['data_sha256'],r['index_sha256']) for r in data.data.receipts]
            exact=(signatures,data.identities,data.report['selection_sha256'])
            if reference is None:reference=exact
            require(exact==reference,'Parallel/selected loading changed training rows or order')
            report=data.report
        require(torch.equal(rng,torch.get_rng_state()),'Loading advanced training RNG')
        results.append(dict(mode=label,wall_seconds=time.perf_counter()-began,**report))
    with patch.object(demos,'_prepared_shard',previous):measure('previous-serial')
    measure('selected-serial')
    started=time.perf_counter()
    with reader_pool(args.workers) as pool:
        # A production pool was already started during ordinary collection.
        from brass_ai.recovery.ingress import _reader_ready
        for future in [pool.submit(_reader_ready) for _ in range(args.workers)]:require(future.result(),'Reader failed')
        startup=time.perf_counter()-started
        measure('selected-parallel',pool,args.workers)
    require(before=={p:sha(Path(p)) for p in before},'Benchmark changed source replay')
    write_json(out/'report.json',dict(passed=True,implementation=implementation_id(),results=results,
        source_rows=index.rows,shards=16,worker_startup_seconds=startup,
        speedup=results[0]['preparation_seconds']/results[-1]['preparation_seconds'],
        exact_rows_order_bytes_rng=True,source_unchanged=True,
        scope='One real diagnostic game replicated into 16 distinct shard files; no sidecar caches; CPU, two tensor threads. Single measured pass per mode; worker startup reported separately. Not full joint7 throughput or strength.'))
    print(out/'report.json',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--workers',type=int,default=16)
    run(parser.parse_args())
