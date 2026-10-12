"""Reproduce the joint-7 iteration-60 audit from retained JSON, without training."""
import argparse
import hashlib
import json
import statistics
from pathlib import Path


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def stats(values):
    values = list(values)
    if not values:
        return None
    return dict(n=len(values), mean=statistics.mean(values),
                std=statistics.pstdev(values), min=min(values), max=max(values))


def audit(root):
    folder = root / 'results/selfplay-joint'
    state, retained, latest = [read(folder / name) for name in
                               ('state.json', 'retention-history.json', 'iter59.json')]
    rows = {v['summary']['iteration']: v['summary'] for k, v in retained.items()
            if k.startswith('iteration-')}
    for report in state['recent_training']:
        actor = report['actor']
        rows[report['iteration']] = dict(report, actor=dict(actor,
            last_epoch=actor['epochs'][-1], reward_components=actor['reward']['components']))
    rows[59].update(by_composition={k: {latest['incoming_policy_id']:
        v['matches']['policies'][latest['incoming_policy_id']]}
        for k, v in latest['by_composition'].items()},
        incoming_policy_id=latest['incoming_policy_id'], timings=latest['timings'])
    windows = []
    for lo, hi in ((0, 19), (20, 39), (40, 59), *[(i, i + 9) for i in range(0, 60, 10)]):
        selected = [rows[i] for i in range(lo, hi + 1)]
        window = dict(iterations=[lo, hi], compositions={}, actor={}, reward={}, timings={})
        for kind in ('selfplay', 'historical', 'legacy'):
            measured = [(r['iteration'], r['by_composition'][kind][r['incoming_policy_id']])
                        for r in selected if 'by_composition' in r]
            window['compositions'][kind] = dict(iterations=[i for i, _ in measured],
                seat_games=sum(v['seat_games'] for _, v in measured),
                **{key: stats(v[key] for _, v in measured) for key in ('win_share', 'placement', 'vp')})
        for key in ('value', 'entropy', 'post_update_kl', 'clip_fraction'):
            window['actor'][key] = stats(r['actor']['last_epoch'][key] for r in selected)
        window['actor'].update(early_stops=sum(r['actor']['kl_early_stop'] for r in selected),
            optimizer_steps=stats(r['actor']['optimizer_steps'] for r in selected),
            utility_mse=stats(r['actor'].get('utility_prediction',
                r['actor']['last_epoch'].get('utility_prediction', {}))['mse'] for r in selected))
        for key in ('win', 'placement', 'vp', 'cost', 'utility'):
            window['reward'][key] = stats(r['actor']['reward_components'][key] for r in selected)
        debt = [r['actor']['insolvency'] for r in selected]
        window['insolvency'] = dict(events=sum(r['events'] for r in debt),
            affected=sum(r['affected_trajectories'] for r in debt),
            trajectories=sum(r['learner_trajectories'] for r in debt))
        for key in ('collection_seconds', 'replay_load_seconds', 'ppo_seconds',
                    'search_and_distill_seconds', 'evaluation_seconds', 'iteration_seconds'):
            window['timings'][key] = stats(r['timings'][key] for r in selected if 'timings' in r)
        window['teachers'] = {key: sum(r['search'].get(key, 0) for r in selected)
                              for key in ('collection_count', 'teacher_rows', 'optimizer_steps')}
        windows.append(window)
    sources = [*folder.glob('*.json'), *(root / 'results/joint-v7').glob('*.json')]
    phases = {}; phase_coverage = {}
    for row in rows.values():
        for phase, counts in row['search'].get('phases', row['search'].get('teaching', {}).get('phases', {})).items():
            dest = phases.setdefault(phase, {})
            coverage = phase_coverage.setdefault(phase, {})
            for key, value in counts.items():
                if type(value) in (int, float):
                    dest[key] = dest.get(key, 0) + value
                    coverage[key] = coverage.get(key, 0) + 1
    return dict(format='joint7-retained-audit/1.0.0',
        sources={p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        source_lf_sha256={p.relative_to(root).as_posix(): hashlib.sha256(p.read_text(encoding='utf-8').encode()).hexdigest() for p in sources},
        semantics='Iteration means and population SD across iterations; not per-game SD or confidence intervals. All outcome windows have equal per-iteration seat counts.',
        missing=dict(composition_iterations=[58], full_iteration_reports=list(range(59)),
                     timing_iterations=[58], lab_checkpoint_present=(root / state['history'][-1]['path']).exists()),
        windows=windows, phases=phases, phase_field_iterations=phase_coverage,
        champion=state['champion_gate']['recent'],
        latest_composition=rows[59]['by_composition'], latest_actor=latest['actor'],
        latest_teaching=latest['search']['teaching'],
        latest_teacher_fit={k:latest['search'][k] for k in ('fit_before','fit_after','teacher_rows','optimizer_steps')},
        latest_utilization=latest['utilization'],
        counters={k:state[k] for k in ('iteration','games_done','next_seed','context_counter','next_search_games',
                                      'search_seconds','ordinary_seconds')},
        iterations=[rows[i] for i in sorted(rows)])


def tables(data):
    lines = ['# Reproduced retained measurements', '', data['semantics'], '',
        'Iteration 58 has PPO/search summaries but no composition results or timings in this export.', '',
        '| Iterations | Matchup | Measured iterations / learner games | Win share ± SD | Rank | VP ± SD |',
        '|---|---|---:|---:|---:|---:|']
    for window in data['windows']:
        lo, hi = window['iterations']
        for kind, v in window['compositions'].items():
            w, p = v['win_share'], v['vp']
            lines.append(f'| {lo}–{hi} | {kind} | {len(v["iterations"])} / {v["seat_games"]} | '
                         f'{100*w["mean"]:.2f}% ± {100*w["std"]:.2f} pp | {v["placement"]["mean"]:.3f} | '
                         f'{p["mean"]:.2f} ± {p["std"]:.2f} |')
    lines += ['', '| Iterations | PPO steps mean | KL | Clip fraction | Entropy | Early stops | Utility MSE | Insolvent trajectories |',
              '|---|---:|---:|---:|---:|---:|---:|---:|']
    for w in data['windows'][:3]:
        a, debt = w['actor'], w['insolvency']
        lines.append(f'| {w["iterations"][0]}–{w["iterations"][1]} | {a["optimizer_steps"]["mean"]:.2f} | '
                     f'{a["post_update_kl"]["mean"]:.5f} | {a["clip_fraction"]["mean"]:.3f} | '
                     f'{a["entropy"]["mean"]:.3f} | {a["early_stops"]}/20 | {a["utility_mse"]["mean"]:.4f} | '
                     f'{debt["affected"]}/{debt["trajectories"]} ({100*debt["affected"]/debt["trajectories"]:.1f}%) |')
    lines += ['', '| Iterations | W | 0.1 P | 0.001 VP | −cost | Utility |', '|---|---:|---:|---:|---:|---:|']
    for w in data['windows'][:3]:
        r = w['reward']
        fields = [(key, factor) for key, factor in (('win',1),('placement',.1),('vp',.001),('cost',-1),('utility',1))]
        lines.append(f'| {w["iterations"][0]}–{w["iterations"][1]} | ' + ' | '.join(
            f'{factor*r[k]["mean"]:.5f} ± {abs(factor)*r[k]["std"]:.5f}' for k, factor in fields) + ' |')
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output', type=Path, default=Path('docs/joint7-60-evidence'))
    args = parser.parse_args()
    data = audit(args.root.resolve())
    output = args.output if args.output.is_absolute() else args.root / args.output
    output.mkdir(parents=True, exist_ok=True)
    (output / 'metrics.json').write_text(json.dumps(data, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    (output / 'tables.md').write_text(tables(data), encoding='utf-8')
    print(output)


if __name__ == '__main__':
    main()
