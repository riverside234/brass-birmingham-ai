"""Reproduce the generation-150 joint-7 audit using retained JSON only."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import statistics

from review_joint7 import audit as earlier_audit
from evidence_storage import atomic_write


EXPORT = 'results/v4-selfplay-joint-g0150-20261013T173011108713Z'
EXPANDED_FORMAT = 'joint7-generation150-audit/1.0.0'
COMPACT_FORMAT = 'joint7-generation150-audit/2.0.0'


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def read_metrics(path):
    """Load either report version, verifying v2's complete iteration sidecar."""
    path = Path(path)
    summary = read(path)
    if not isinstance(summary, dict):
        raise ValueError('Expected a generation-150 audit object')
    if summary.get('format') == EXPANDED_FORMAT:
        if not isinstance(summary.get('iterations'), list) or 'iterations_ref' in summary:
            raise ValueError('Invalid expanded audit iterations')
        return summary
    if summary.get('format') != COMPACT_FORMAT:
        raise ValueError('Unsupported generation-150 audit format')
    reference = summary.get('iterations_ref')
    if not isinstance(reference, dict) or 'iterations' in summary:
        raise ValueError('Invalid audit iteration reference')
    digest = reference.get('sha256')
    count = reference.get('count')
    if (not isinstance(digest, str) or len(digest) != 64 or
            any(c not in '0123456789abcdef' for c in digest) or
            type(count) is not int or count < 0 or
            reference.get('path') != f'iterations-{digest}.json.gz'):
        raise ValueError('Invalid audit iteration reference')
    detail = path.parent / reference['path']
    if detail.resolve().parent != path.parent.resolve():
        raise ValueError('Audit iteration sidecar must be in the report directory')
    payload = detail.read_bytes()
    if hashlib.sha256(payload).hexdigest() != digest:
        raise ValueError('Audit iteration sidecar checksum mismatch')
    iterations = json.loads(gzip.decompress(payload).decode('utf-8'))
    if not isinstance(iterations, list) or len(iterations) != count:
        raise ValueError('Audit iteration sidecar count mismatch')
    result = {key: value for key, value in summary.items() if key != 'iterations_ref'}
    result.update(format=EXPANDED_FORMAT, iterations=iterations)
    return result


def write_metrics(path, data):
    """Publish compact evidence without losing details or replacing live sidecars."""
    path = Path(path)
    if (not isinstance(data, dict) or data.get('format') != EXPANDED_FORMAT or
            not isinstance(data.get('iterations'), list) or 'iterations_ref' in data):
        raise ValueError('Expected expanded generation-150 audit data')
    raw = json.dumps(data['iterations'], separators=(',', ':'), allow_nan=False).encode('utf-8')
    # Remove timestamps and platform-specific gzip header bytes.
    compressed = gzip.compress(raw, compresslevel=9, mtime=0)
    compressed = compressed[:9] + b'\xff' + compressed[10:]
    digest = hashlib.sha256(compressed).hexdigest()
    reference = dict(path=f'iterations-{digest}.json.gz', sha256=digest, count=len(data['iterations']))
    summary = {key: value for key, value in data.items() if key != 'iterations'}
    summary.update(format=COMPACT_FORMAT, iterations_ref=reference)
    summary_bytes = (json.dumps(summary, indent=2, allow_nan=False) + '\n').encode('utf-8')
    path.parent.mkdir(parents=True, exist_ok=True)
    detail = path.parent / reference['path']
    # Publish the immutable, content-addressed sidecar before pointing to it.
    # A failed summary write leaves the previous summary/sidecar pair readable.
    for target, payload in ((detail, compressed), (path, summary_bytes)):
        if target.exists() and target.read_bytes() == payload:
            continue
        atomic_write(target, payload)
    return summary


def stats(values):
    values = [x for x in values if x is not None]
    return dict(n=len(values), mean=statistics.mean(values), std=statistics.pstdev(values),
                min=min(values), max=max(values)) if values else None


def get(value, path):
    for key in path.split('.'):
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def normalize(row):
    row = dict(row)
    actor = row['actor'] = dict(row['actor'])
    if actor.get('epochs'):
        actor['last_epoch'] = actor['epochs'][-1]
    if 'reward' in actor:
        actor['reward_components'] = actor['reward']['components']
        actor['reward_distribution'] = actor['reward'].get('distribution')
    search = row['search'] = dict(row['search'])
    if 'teaching' in search:
        search['phases'] = search['teaching']['phases']
    for key in ('collection', 'updates', 'collections'):
        search.pop(key, None)
    row.pop('execution', None)
    row.pop('inference', None)
    if 'by_composition' in row:
        row['by_composition'] = {k: v.get('matches', {}).get('policies', v)
                                 for k, v in row['by_composition'].items()}
    return row


def reward_moments(rows):
    records = [r['actor']['reward_distribution'] for r in rows if r['actor'].get('reward_distribution')]
    if not records:
        return None
    names = ('win', 'placement', 'vp', 'cost')
    count = sum(r['utility']['count'] for r in records)
    means = {k: sum(r['weighted'][k]['mean'] * r['utility']['count'] for r in records) / count for k in names}
    cov = {a: {b: sum(r['utility']['count'] * (r['covariance'][a][b] +
              (r['weighted'][a]['mean'] - means[a]) * (r['weighted'][b]['mean'] - means[b]))
              for r in records) / count for b in names} for a in names}
    return dict(iterations=len(records), trajectories=count, weighted_means=means, covariance=cov,
        weighted_std={k: cov[k][k]**.5 for k in names},
        utility_mean=sum(means.values()), utility_std=sum(sum(v.values()) for v in cov.values())**.5,
        cap_count=sum(r['capped_trajectories'] for r in records),
        scope='Pooled trajectory population moments, including between-iteration variation; cost has negative sign')


def window(rows, lo, hi):
    selected = [rows[i] for i in range(lo, hi + 1)]
    compositions = {}
    for kind in ('selfplay', 'historical', 'legacy'):
        entries = [(r['iteration'], r['by_composition'][kind][r['incoming_policy_id']])
                   for r in selected if r.get('by_composition')]
        count = sum(v['seat_games'] for _, v in entries)
        compositions[kind] = dict(iterations=[i for i, _ in entries], games=count,
            **{k: dict(pooled=sum(v[k]*v['seat_games'] for _, v in entries)/count,
                       **stats(v[k] for _, v in entries)) for k in ('win_share', 'placement', 'vp')})
    fields = ['actor.optimizer_steps', 'actor.last_epoch.actor', 'actor.last_epoch.value',
        'actor.last_epoch.post_update_kl', 'actor.last_epoch.clip_fraction', 'actor.last_epoch.entropy',
        'actor.last_epoch.utility_prediction.mse', 'actor.baseline_explained_variance',
        'actor.categorical_probe.categorical_kl', 'actor.categorical_probe.entropy_over_log_legal',
        'actor.gradient_norms.mean', 'actor.advantage.std', 'actor.return_distribution.std',
        'actor.legal_actions.mean', 'search.post_teacher_sampled_kl', 'search.teacher_rows',
        'search.optimizer_steps', 'search.fit_before.meaningful_top1', 'search.fit_before.teacher_kl',
        'search.fit_after.teacher_kl', 'search.fit_before.teacher_entropy']
    fields += ['timings.' + k for k in ('collection_seconds', 'replay_load_seconds', 'ppo_seconds',
               'search_and_distill_seconds', 'evaluation_seconds', 'cleanup_seconds', 'iteration_seconds')]
    metrics = {key: stats(get(r, key) for r in selected) for key in fields}
    debt = [r['actor']['insolvency'] for r in selected]
    metrics['insolvency'] = {k: sum(v[k] for v in debt) for k in ('events', 'affected_trajectories', 'learner_trajectories')}
    metrics['kl_stops'] = sum(r['actor']['kl_early_stop'] for r in selected)
    observed = [get(r, 'search.post_teacher_sampled_kl') for r in selected]
    metrics['post_teacher_observed'] = sum(v is not None for v in observed)
    metrics['post_teacher_over_01'] = sum(v > .01 for v in observed if v is not None)
    metrics['reward_iteration_means'] = {k: stats(r['actor']['reward_components'][k] for r in selected)
                                         for k in ('win', 'placement', 'vp', 'cost', 'utility')}
    return dict(iterations=[lo, hi], compositions=compositions, metrics=metrics, reward=reward_moments(selected))


def audit(root):
    folder = root / EXPORT
    state, manifest, retained = [read(folder / p) for p in ('state.json', 'export-manifest.json', 'retention-history.json')]
    earlier = earlier_audit(root)
    rows = {r['iteration']: normalize(r) for r in earlier['iterations']}
    for key, value in retained.items():
        if key.startswith('iteration-'):
            row = normalize(value['summary'])
            rows[row['iteration']] = row
    for path in sorted(folder.glob('iteration-*/report.json')):
        row = normalize(read(path))
        rows[row['iteration']] = row
    assert set(rows) == set(range(150))
    checks = []
    for entry in manifest['files']:
        path = folder / entry['path']
        raw = hashlib.sha256(path.read_bytes()).hexdigest()
        lf = hashlib.sha256(path.read_text(encoding='utf-8').encode()).hexdigest()
        assert entry['sha256'] in (raw, lf), entry['path']
        checks.append(dict(path=entry['path'], raw_sha256=raw, lf_sha256=lf,
                           matches='raw' if raw == entry['sha256'] else 'LF-normalized'))
    assert read(folder / 'recorded-config.json') == state['signature']['config']
    old = read(root / 'results/selfplay-joint/state.json')
    assert state['history'][:len(old['history'])] == old['history']
    assert rows[60]['incoming_model_id'] == old['history'][-1]['model_id']
    assert all(rows[i]['post_update_model_id'] == state['history'][i+1]['model_id'] for i in range(60, 150))
    assert all(rows[i+1]['incoming_model_id'] == rows[i]['post_update_model_id'] for i in range(60, 149))
    windows = [(0,19),(20,39),(40,59),(60,89),(90,119),(120,149),(60,61),(62,62),(63,92),(93,93),(94,149),(60,149)]
    windows += [(i,i+9) for i in range(60,150,10)]
    phases = {}; coverage = {}
    for row in [rows[i] for i in range(60,150)]:
        for phase, counts in row['search'].get('phases', {}).items():
            target = phases.setdefault(phase, {})
            fields = coverage.setdefault(phase, {})
            for key, value in counts.items():
                if type(value) in (int,float):
                    target[key] = target.get(key,0) + value
                    fields[key] = fields.get(key,0) + 1
    evaluations = {}
    for path in sorted((folder/'evaluations').glob('*/report.json')):
        report = read(path); spec = read(path.with_name('spec.json'))
        evaluations[path.parent.name] = dict(spec=spec, result=report.get('result'),
            results=report.get('results'), outcomes=report.get('outcomes'), failed_games=report['failed_games'])
    return dict(format='joint7-generation150-audit/1.0.0', source_folder=EXPORT,
        manifest_checks=checks, manifest_missing=manifest['missing'], manifest_warnings=manifest['warnings'],
        state={k:state.get(k) for k in ('iteration','games_done','next_seed','context_counter','next_search_games',
                'ordinary_seconds','search_seconds','origin','champion','signature','teacher_work_cost','source_transitions')},
        latest=state['history'][-1], lineage_continuity_verified=True,
        windows=[window(rows,lo,hi) for lo,hi in windows], phases=phases, phase_coverage=coverage,
        champion_gates=state['champion_gate']['recent'], evaluations=evaluations,
        iterations=[rows[i] for i in range(150)],
        missing=dict(full_iterations=[i for i in range(150) if i not in (59,148,149)],
                     composition_iterations=[i for i,r in rows.items() if not r.get('by_composition')],
                     checkpoints='Actual lab checkpoint/Adam bytes unavailable; receipts only'))


def tables(data):
    lines = ['# Retained generation-150 measurements', '',
        'Rates are pooled over learner games. SD is across iteration means, not a per-game confidence interval.', '',
        '| Iterations | Matchup | Games | Win share | Iteration SD | Rank | VP |',
        '|---|---|---:|---:|---:|---:|---:|']
    for window in data['windows']:
        for kind, values in window['compositions'].items():
            lines.append(f'| {window["iterations"][0]}–{window["iterations"][1]} | {kind} | {values["games"]} | '
                f'{100*values["win_share"]["pooled"]:.2f}% | {100*values["win_share"]["std"]:.2f} pp | '
                f'{values["placement"]["pooled"]:.3f} | {values["vp"]["pooled"]:.2f} |')
    return '\n'.join(lines)+'\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    data = audit(args.root)
    output = args.root/'docs/joint7-150-evidence'
    output.mkdir(parents=True, exist_ok=True)
    write_metrics(output/'metrics.json', data)
    (output/'tables.md').write_text(tables(data),encoding='utf-8')
    print(output)


if __name__ == '__main__':
    main()
