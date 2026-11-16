"""Compare matched frozen evaluation panels with a paired setup bootstrap.

This reads official per-game outcomes, never trains or promotes a checkpoint.
Both reports must have identical source, configuration, setups, lineups, frozen
opponents and search budgets. Confidence intervals condition on these models.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import statistics


def require(condition, message):
    if not condition:
        raise ValueError(message)


def identities(entries):
    return [{k: entry[k] for k in ('sha256', 'model_id')} for entry in entries]


def interval(values, draws, rng):
    n = len(values)
    samples = sorted(statistics.mean(values[rng.randrange(n)] for _ in range(n)) for _ in range(draws))
    # Interpolate empirical quantiles; clusters, not seats, are resampled.
    def quantile(p):
        x = (draws - 1) * p
        lo, hi = math.floor(x), math.ceil(x)
        return samples[lo] + (samples[hi] - samples[lo]) * (x - lo)
    return [quantile(.025), quantile(.975)]


def panel_rows(panel, seeds):
    expected = {(seed, seat) for seed in seeds for seat in range(4)}
    rows = panel.get('outcomes', [])
    indexed = {(r['seed'], r['seat']): r for r in rows}
    require(len(rows) == len(indexed) and set(indexed) == expected,
            'Missing, duplicate or mismatched setup/seat outcomes')
    require(panel['games'] == len(rows) and panel['setups'] == len(seeds), 'Panel counts differ from outcomes')
    for row in rows:
        require(all(type(row[k]) in (int, float) and math.isfinite(row[k]) for k in ('win', 'rank', 'vp')),
                'Non-finite official outcome')
        require(0 <= row['win'] <= 1 and 1 <= row['rank'] <= 4 and row['vp'] >= 0, 'Invalid official outcome')
    for key, summary in [('win', 'win_share'), ('rank', 'mean_rank'), ('vp', 'mean_vp')]:
        require(math.isclose(panel[summary], statistics.mean(r[key] for r in rows), abs_tol=1e-9),
                'Panel summary differs from outcomes')
    return indexed


def compare(control, candidate, *, draws=10000, seed=731509):
    require(type(draws) is int and draws >= 1000, 'Use at least 1000 bootstrap draws')
    for report in (control, candidate):
        require(report['format'] == 'step5-evaluation-report/1.0.0', 'Expected a complete evaluation report')
        require(report.get('failed_games') == 0, 'Failed or unaccounted evaluation games')
    left, right = control['spec'], candidate['spec']
    require({k: v for k, v in left.items() if k != 'entries'} ==
            {k: v for k, v in right.items() if k != 'entries'},
            'Evaluation source, configuration, seeds, lineups or legacy opponents differ')
    require(len(left['entries']) >= 2 and identities(left['entries'][1:]) == identities(right['entries'][1:]),
            'Frozen reference identities or order differ')
    seeds, lineups = left['seeds'], left['lineups']
    require(len(seeds) >= 2 and len(set(seeds)) == len(seeds), 'Use at least two distinct setups')
    require(len(seeds) == left['config']['setups'], 'Specification setup count differs')
    require(set(control['results']) == set(candidate['results']) == set(lineups), 'Incomplete evaluation lineups')
    rng, panels = random.Random(seed), {}
    for name in sorted(lineups):
        a = panel_rows(control['results'][name], seeds)
        b = panel_rows(candidate['results'][name], seeds)
        metrics = {}
        for key in ('win', 'rank', 'vp'):
            delta = [statistics.mean(b[(s, seat)][key] - a[(s, seat)][key] for seat in range(4)) for s in seeds]
            metrics[key] = dict(control=statistics.mean(r[key] for r in a.values()),
                candidate=statistics.mean(r[key] for r in b.values()), delta=statistics.mean(delta),
                paired_setup_bootstrap_95=interval(delta, draws, rng),
                setup_deltas=[dict(seed=s, delta=d) for s, d in zip(seeds, delta)])
        panels[name] = dict(games_per_model=len(a), setups=len(seeds), metrics=metrics,
            by_seat={str(seat): {key: dict(control=statistics.mean(a[(s, seat)][key] for s in seeds),
                candidate=statistics.mean(b[(s, seat)][key] for s in seeds)) for key in ('win', 'rank', 'vp')}
                for seat in range(4)})
    return dict(format='matched-training-panels/1.0.0', matched=True, promotion_eligible=False,
        implementation=left['implementation'], config=left['config'], seeds=seeds,
        candidates=dict(control=identities(left['entries'][:1])[0], candidate=identities(right['entries'][:1])[0]),
        references=identities(left['entries'][1:]), bootstrap=dict(draws=draws, seed=seed, unit='setup'),
        panels=panels,
        scope='Paired candidate-minus-control deltas; higher win/VP and lower rank are better. '
              'Intervals condition on frozen models and opponents; timed search remains variable. '
              'No automatic promotion, multiplicity correction or training-seed uncertainty estimate.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--control', required=True, type=Path)
    parser.add_argument('--candidate', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path, help='New comparison.json; existing files are never replaced')
    args = parser.parse_args()
    inputs = {name: path.read_bytes() for name, path in [('control', args.control), ('candidate', args.candidate)]}
    report = compare(**{k: json.loads(v) for k, v in inputs.items()})
    report['inputs'] = {k: dict(path=str(getattr(args, k).resolve()), sha256=hashlib.sha256(v).hexdigest()) for k, v in inputs.items()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as out:
        out.write(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(args.output.resolve())


if __name__ == '__main__':
    main()
