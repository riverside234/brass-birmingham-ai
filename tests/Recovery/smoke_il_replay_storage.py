"""Small IL transport/ingress benchmark using existing games, no new searches.

Use --sources with one or two complete heuristic shards and a NEW --output.
Original replays/caches are never written. Timings are warm local-file timings,
not a production storage or GPU-training benchmark.
"""
import argparse
from collections import Counter
import copy
import gzip
import json
from pathlib import Path
import statistics
import time

import numpy as np
import torch

from brass_ai.bank import sha
from brass_ai.learning.contracts import SOURCE_ROOT, decode_json, require
from brass_ai.recovery import demonstrations as demos, il_replay_storage as storage
from brass_ai.recovery.prepared_store import write_prepared
from brass_ai.recovery.replay import cache_receipt, write_shard
from brass_ai.step5.league import write_json


def read_raw(path):
    with gzip.open(path, 'rt', encoding='utf-8') as stream:
        header = decode_json(next(stream))
        games = [decode_json(line) for line in stream]
    if header['format'] == storage.FORMAT:
        games = [storage.unpack(game) for game in games]
    else:
        require(header['format'] == storage.LEGACY_FORMAT, 'Only heuristic legacy/compact shards')
    return header, games


def timed(fn, repeats=1):
    times = []
    for _ in range(repeats):
        began = time.perf_counter(); value = fn(); times.append(time.perf_counter()-began)
    return value, statistics.median(times)


def run(sources, output, native_audit=False):
    require(not output.exists(), 'Use a new smoke output directory')
    output.mkdir(parents=True)
    torch.set_num_threads(2)
    results = []
    for number, source in enumerate(sources):
        before = sha(source)
        header, games = read_raw(source)
        require(games and len({g['split'] for g in games}) == 1, 'Use a single-partition shard')
        proof = copy.deepcopy(header['provenance'])
        # These are isolated transport copies, not a new qualification receipt.
        proof['purpose'] = 'training'
        proofs = [j['provenance'] for j in proof['jobs']] if 'jobs' in proof else [proof]
        for p in proofs:p.update(kind='v3-demonstration', purpose='training', replay_format=storage.FORMAT)
        row = dict(source=str(source), source_sha256=before, games=len(games),
                   decisions=sum(len(g['records']) for g in games), formats={})
        prepared = []
        for label in ('legacy', 'compact'):
            destination = output/f'{number}-{label}.jsonl.gz'
            provenance = copy.deepcopy(proof)
            if label == 'legacy':
                for p in [j['provenance'] for j in provenance['jobs']] if 'jobs' in provenance else [provenance]:
                    p.pop('replay_format', None)
            _, write_seconds = timed(lambda: write_shard(destination, games, provenance))
            (_, restored), read_seconds = timed(lambda: read_raw(destination), repeats=5)
            require(json.dumps(restored, separators=(',', ':'), allow_nan=False) ==
                    json.dumps(games, separators=(',', ':'), allow_nan=False), 'Replay roundtrip changed values')
            entries = [dict(game_id=g['game_id'], sha256=sha(destination), features=cache_receipt(destination),
                            teachers=dict(Counter(r['behavior']['policy_id'] for r in g['records']))) for g in games]
            task = (str(destination), entries, games[0]['split'], set(games[0]['opponents']), True)
            data, prepare_seconds = timed(lambda: demos._prepared_shard(task), repeats=2)
            prepared.append(data)
            receipt = write_prepared(data, output, f'dense-{number}-{label}')
            row['formats'][label] = dict(bytes=destination.stat().st_size, write_with_features_seconds=write_seconds,
                read_expand_median_seconds=read_seconds, prepare_median_seconds=prepare_seconds,
                dense_data_bytes=receipt['data_bytes'], dense_data_sha256=receipt['data_sha256'])
            if native_audit and label == 'compact':
                (accepted, rejected), seconds = timed(lambda: demos._audit_path(
                    (str(SOURCE_ROOT), str(destination), sorted({p for g in games for p in g['opponents']}),
                     None, cache_receipt(destination))))
                require(not rejected and len(accepted) == len(games), 'Native audit failed: '+str(rejected))
                row['native_audit'] = dict(games=len(accepted), seconds=seconds, transitions_verified=True)
        for a, b in zip(*prepared):
            require(a[:3] == b[:3], 'Prepared labels/metadata changed')
            for field in ('state', 'actions', 'stable_to_relative'):
                require(np.array_equal(getattr(a[3], field), getattr(b[3], field)), 'Prepared arrays changed')
        require(row['formats']['legacy']['dense_data_sha256'] == row['formats']['compact']['dense_data_sha256'],
                'Dense prepared training bytes changed')
        require(sha(source) == before, 'Original source changed')
        row['saved_fraction'] = 1-row['formats']['compact']['bytes']/row['formats']['legacy']['bytes']
        results.append(row)
        print(json.dumps(row), flush=True)
    report = dict(format='compact-il-storage-smoke/1.0.0', cases=results,
                  exact_replay_and_dense_training_parity=True,
                  scope='Existing local heuristic games; 5 warm raw reads, 2 prepare reads per format. No new search or training.')
    write_json(output/'report.json', report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--sources', nargs='+', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--native-audit', action='store_true')
    args = parser.parse_args()
    run([p.resolve() for p in args.sources], args.output.resolve(), args.native_audit)
