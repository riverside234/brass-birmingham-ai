"""Differential L08 corpus/NN check against the executed, frozen L01 oracle."""
import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def rows(folder, filename):
    with gzip.open(ROOT/'artifacts/rust-backend'/folder/filename, 'rt', encoding='utf-8') as stream:
        for line in stream:
            yield json.loads(line)


def cases():
    legal = {r['id']: r['expected']['choices'] for r in rows('20261025-l05-capture-v2', 'roots.jsonl.gz')}
    for r in rows('20261025-l06-capture', 'roots.jsonl.gz'):
        ctx = r['state']['context']
        if ctx['actor'] >= 0 and not ctx['terminal']:
            yield 'root/'+r['id'], r['partition'], r['views'][ctx['actor']], legal[r['id']], r['state']
    for folder, filename in [('20261025-l06-capture', 'samples.jsonl.gz'), ('20261025-l06-edges', 'edges.jsonl.gz')]:
        for r in rows(folder, filename):
            if not r['expected']['ok']:
                continue
            name = r.get('id', r.get('root'))
            for i, e in enumerate([r['expected']] + [s['expected'] for s in r['suffix']]):
                ctx = e['state']['context']
                if ctx['actor'] >= 0 and not ctx['terminal'] and e['choices']:
                    yield 'future/'+name+'/'+str(i), r.get('partition', 'authored'), e['views'][ctx['actor']], e['choices'], e['state']


def context(index=0):
    return dict(model_id='1145fecba3289511df1c518bb71fbfb5b2f6df351777151a5eb999b89cba0761',
                reward_id='g890-frozen-utility', profile_id='parity-'+str(index % 3),
                style=[-.5, 0., .25, 1.] if index % 3 else [0.]*4,
                intent=[.125, -.25, .5, 0., 1., -1., 0., .25] if index % 3 else [0.]*8,
                remaining=[0., .015, .05, .075], max_cost=.075, stable_descriptors=None)


def verify(args):
    import numpy as np
    from brass_ai.learning.features import Encoder
    from brass_ai.native.protocol import NativeWorker
    from brass_ai.native.features import NativeFeatureEncoder, GROUP, ENTITY
    from brass_ai.learning.state_layout import model_layout
    from brass_ai.learning.local_entities import entity_layout
    from brass_ai.learning.seat_descriptors import unknown_numpy, relative, descriptor
    assert model_layout().identity == GROUP and entity_layout()['identity'] == ENTITY
    encoder = Encoder()
    counts, stages, seats, partitions, kinds = (Counter() for _ in range(5))
    output_stages, output_seats, output_matrix = Counter(), Counter(), Counter()
    oracle_ns = native_ns = candidates = hits = misses = 0
    max_candidates, checked = 0, 0
    seen = set()
    nn = None
    if args.nn:
        import torch
        torch.set_num_threads(1); torch.set_num_interop_threads(1)
        from brass_ai.learning.checkpoint import load_checkpoint
        checkpoint = next((ROOT/'test_nn').glob('*/checkpoint.pt'))
        checkpoint_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        assert checkpoint_sha == 'ecad3b932a8dd9d3b5d0cc49b5e9bb94a9e98b4536ffdc86a44179d003995cb8'
        nn, raw = load_checkpoint(checkpoint)
        nn.eval()
        assert nn.utility_calibration is None and raw['model_id'] == context()['model_id']
        assert nn.config.utility['insolvency']['max_cost'] == .075
        assert nn.state_encoder.permutation.tolist() == list(model_layout().permutation)
    output_count = 0
    output_max_error = 0.
    with NativeWorker(args.worker, timeout=60) as worker:
        native = [NativeFeatureEncoder(worker, s) for s in (0, 1)]
        for name, partition, view, choices, state in cases():
            # Repeated suffix roots are the same mathematical input; retain one receipt.
            identity = hashlib.sha256(json.dumps([view, choices], sort_keys=True, separators=(',', ':')).encode()).hexdigest()
            if identity in seen:
                continue
            seen.add(identity)
            if args.limit and checked >= args.limit:
                break
            ctx = context(checked)
            try:
                start = time.perf_counter_ns()
                py = encoder.encode(view, choices, ctx['style'], ctx['intent'])
                oracle_ns += time.perf_counter_ns()-start
                start = time.perf_counter_ns()
                rs = native[checked % 2].encode(view, choices, ctx)
                native_ns += time.perf_counter_ns()-start
                for field in ('state', 'actions', 'stable_to_relative'):
                    x, y = getattr(py, field), getattr(rs, field)
                    if not np.array_equal(x, y):
                        ix = np.argwhere(x != y)
                        raise AssertionError(dict(field=field, count=len(ix), max_error=float(np.max(np.abs(x-y))),
                            examples=[dict(index=i.tolist(), python=float(x[tuple(i)]), rust=float(y[tuple(i)])) for i in ix[:12]]))
                assert py.choice_ids == rs.choice_ids and rs.legal_mask.all()
                assert np.array_equal(rs.descriptors, unknown_numpy())
                # Use the typed simulator -> observation -> encoder path too, without reimporting JSON features.
                if checked % 17 == 0:
                    s = checked % 2
                    imported = worker.simulation(s, 'import', state=state)
                    handle = imported['handle']
                    first = worker.call('features', session=s, command=dict(kind='from_state', handle=handle,
                                        revision=view['context']['revision'], context=ctx))['data']['tensors']
                    assert np.array_equal(np.asarray(first['state'], dtype=np.float32), py.state)
                    assert np.array_equal(np.asarray(first['actions'], dtype=np.float32).reshape(-1, 327), py.actions[:64])
                    worker.simulation(s, 'release', handle=handle)
                    counts['direct_state'] += 1
                matrix_key = '/'.join((name.split('/')[0], view['context']['era'], view['context']['phase'], str(view['seat'])))
                if nn is not None and (name.startswith('root/') or not output_matrix[matrix_key] or checked % 101 == 0):
                    def infer(e, d):
                        with torch.inference_mode():
                            return nn(torch.from_numpy(e.state[None]), torch.from_numpy(e.actions[None]),
                                      torch.from_numpy(getattr(e, 'legal_mask', np.ones(len(choices), dtype=bool))[None]),
                                      torch.from_numpy(e.stable_to_relative[None]),
                                      remaining=torch.from_numpy(getattr(e, 'remaining', np.asarray(ctx['remaining'], dtype=np.float32))[None]),
                                      descriptors=torch.from_numpy(d[None]))
                    a, b = infer(py, unknown_numpy()), infer(rs, rs.descriptors)
                    assert len(a) == len(b) == 5
                    for x, y in zip(a, b):
                        assert torch.isfinite(x).all() and torch.isfinite(y).all()
                        output_max_error = max(output_max_error, float((x-y).abs().max()))
                        torch.testing.assert_close(x, y, rtol=1e-6, atol=1e-6)
                    stable = [descriptor(role, s*43, [((i+s) % 9)/8 for i in range(28)])
                              for s, role in enumerate(('learner', 'snapshot', 'il', 'rule_bot'))]
                    explicit_ctx = dict(ctx, stable_descriptors=stable, profile_id='known-critic')
                    explicit = native[checked % 2].encode(view, choices, explicit_ctx)
                    d = relative(stable, py.stable_to_relative)
                    assert np.array_equal(explicit.descriptors, d)
                    c, e = infer(py, d), infer(explicit, explicit.descriptors)
                    for x, y in zip(c, e):
                        output_max_error = max(output_max_error, float((x-y).abs().max()))
                        torch.testing.assert_close(x, y, rtol=1e-6, atol=1e-6)
                    assert torch.equal(a[0], c[0]), 'critic descriptors affected policy'
                    output_count += 2; output_matrix[matrix_key] += 1
                    output_stages[view['context']['phase']] += 1; output_seats[view['seat']] += 1
                checked += 1; candidates += len(choices); max_candidates = max(max_candidates, len(choices))
                hits += rs.stats['effect_hits']; misses += rs.stats['effect_misses']
                counts[name.split('/')[0]] += 1; stages[view['context']['phase']] += 1; seats[view['seat']] += 1
                partitions[partition] += 1; kinds.update(c['kind'] for c in choices)
                if checked % 100 == 0:
                    print(json.dumps(dict(checked=checked, candidates=candidates, last=name)), flush=True)
            except Exception:
                print('FIRST DIFFERENCE: '+name+' '+str(view['context']), flush=True)
                raise
    result = dict(status='passed', exact_float32=True, cases=checked, candidates=candidates, maximum_candidates=max_candidates,
        counts=counts, phases=stages, actors=seats, partitions=partitions, kinds=kinds, effect_hits=hits, effect_misses=misses,
        python_total_ns=oracle_ns, diagnostic_native_total_ns=native_ns, diagnostic_includes_json=True,
        nn_output_pairs=output_count, nn_phases=output_stages, nn_actors=output_seats, nn_matrix=output_matrix, nn_max_absolute_error=output_max_error,
        mechanical_only=True, calibration_ready=False, live_strength_training_admitted=False)
    if nn is not None:
        assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == checkpoint_sha
        result.update(checkpoint_sha256=checkpoint_sha, model_id=raw['model_id'])
    args.report.write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--reference', type=Path, required=True)
    p.add_argument('--worker', type=Path, required=True)
    p.add_argument('--report', type=Path, required=True)
    p.add_argument('--limit', type=int, default=0)
    p.add_argument('--nn', action='store_true')
    args = p.parse_args()
    sys.path[:0] = [str(args.reference.resolve()/'src')] + [str(ROOT/d) for d in
        ('artifacts/r2-pythondeps', 'artifacts/r4-pythondeps', 'artifacts/r0-pythonlibs', 'artifacts/r0-validation-deps')]
    verify(args)
