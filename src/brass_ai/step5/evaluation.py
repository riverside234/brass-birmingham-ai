"""Seat-rotated, setup-clustered neural/legacy comparisons kept out of training."""
from __future__ import annotations

from contextlib import ExitStack
from collections import Counter
from pathlib import Path
import statistics

from brass_ai.bank import read, sha, require
from brass_ai.learning.checkpoint import load_checkpoint
from brass_ai.learning.contracts import digest, project_path
from brass_ai.learning.legacy_policy import LegacyPolicy
from brass_ai.learning.replay import terminal_targets
from brass_ai.qualification import grouped_interval
from .league import collect_cached, pottery_metrics, seeds_for, write_json
from .policy import NeuralPolicy
from .search import SearchPolicy, SearchConfig


def outcome(game, candidate_id, seed):
    require(game['opponents'].count(candidate_id) == 1, 'Evaluation requires one candidate seat')
    seat = game['opponents'].index(candidate_id)
    targets = terminal_targets(game['result'])
    rivals = [v for s, v in enumerate(targets['vp']) if s != seat]
    return dict(seed=seed, seat=seat, win=targets['win_share'][seat],
                rank=targets['placement'][seat], vp=targets['vp'][seat],
                margin=targets['vp'][seat] - statistics.mean(rivals),
                margin_to_best=targets['vp'][seat] - max(rivals))


def summarize_outcomes(rows):
    require(rows, 'No complete evaluation games')
    return dict(games=len(rows), setups=len({r['seed'] for r in rows}),
                win_share=statistics.mean(r['win'] for r in rows),
                win_share_setup_bootstrap_95=grouped_interval(rows, 'win'),
                mean_rank=statistics.mean(r['rank'] for r in rows),
                mean_vp=statistics.mean(r['vp'] for r in rows),
                mean_margin=statistics.mean(r['margin'] for r in rows),
                mean_margin_to_best=statistics.mean(r['margin_to_best'] for r in rows),
                seat_counts=dict(Counter(str(r['seat']) for r in rows)))


def evaluate(root, output, candidate_path, references, config, *, setups=5, seed=910000,
             partition='validation', include_legacy=False, device='cpu', mono=None,
             search=False, focus='neutral'):
    root = Path(root).resolve(); directory = project_path(root, output)
    require(setups > 0 and partition in ('validation', 'test') and references,
            'Held-out partition and frozen neural references required')
    candidate_file = project_path(root, candidate_path)
    model, candidate_raw = load_checkpoint(candidate_file, device=device)
    seeds, _ = seeds_for(seed, setups, partition)
    host = project_path(root, config['host'])
    reference_models = [load_checkpoint(project_path(root, p), device=device) for p in references]
    neural_ids = [raw['model_id'] for _, raw in reference_models]
    require(candidate_raw['model_id'] not in neural_ids, 'Candidate cannot be its own frozen reference')
    lineups = [('recent-neural', ['neural:0'] * 3),
               ('mixed-neural', [f'neural:{i % len(references)}' for i in range(3)])]
    if len(set(neural_ids)) == 1:
        lineups = lineups[:1]
    pool = read(root / 'training/opponents/pool.json')
    if include_legacy:
        training_ids, held_ids = sorted(pool['training']), sorted(pool['evaluation'])
        require(training_ids and held_ids, 'Qualified legacy pools missing')
        lineups += [('mixed-v3', ['legacy-training:' + training_ids[i % len(training_ids)] for i in range(3)]),
                    ('heldout-pottery-v3', ['legacy-evaluation:' + held_ids[0]] * 3)]
    spec = dict(format='step5-evaluation/1.0.0', purpose='evaluation', partition=partition,
                candidate=str(Path(candidate_path).as_posix()), candidate_sha256=sha(candidate_file),
                candidate_id=candidate_raw['model_id'], references=[dict(path=p, sha256=sha(project_path(root, p)),
                    model_id=r['model_id'], training_mode=r['training'].get('mode')) for p, (_, r) in zip(references, reference_models)],
                seeds=seeds, lineups=[dict(name=n, opponents=o) for n, o in lineups],
                rotations=4, search=search, search_config=config['search'] if search else None, focus=focus)
    if (directory / 'spec.json').exists():
        require(read(directory / 'spec.json') == spec, 'Evaluation spec changed')
    else:
        write_json(directory / 'spec.json', spec)
    results, raw_evidence = {}, {}
    with ExitStack() as stack:
        if search:
            candidate = SearchPolicy(model, host, device, config=SearchConfig(**config['search']), mono=mono, focus=focus)
            stack.callback(candidate.close)
        else:
            candidate = NeuralPolicy(model, device, exploration=0, focus=focus)
        for name, identities in lineups:
            opponents = []
            with ExitStack() as opponent_stack:
                for identity in identities:
                    if identity.startswith('neural:'):
                        opponent = NeuralPolicy(reference_models[int(identity.split(':')[1])][0], device,
                                                exploration=0, focus='neutral')
                    else:
                        kind, policy = identity.split(':', 1)
                        opponent = LegacyPolicy(root, policy, purpose='evaluation' if kind == 'legacy-evaluation' else 'training', mono=mono)
                        opponent_stack.callback(opponent.close)
                    opponents.append(opponent)
                rows, games = [], []
                for setup_seed in seeds:
                    for seat in range(4):
                        base = [candidate] + opponents
                        table = [base[(s-seat) % 4] for s in range(4)]
                        path = directory / f'{name}-{setup_seed}-seat-{seat}.jsonl.gz'
                        provenance = dict(producer='step5/1.0.0', purpose='evaluation', seed=setup_seed,
                            rotation=seat, policies=[p.policy_id for p in table], sampling_seed=setup_seed * 7 + seat)
                        game = collect_cached(root, path, host, [setup_seed], table, config, provenance, mono=mono)[0]
                        require(game['split'] == partition, 'Evaluation setup crossed partition')
                        games.append(game); rows.append(outcome(game, candidate.policy_id, setup_seed))
                        raw_evidence[path.name] = sha(path)
                        print(f'evaluation {name}: setup {setup_seed}, seat {seat}, rank {rows[-1]["rank"]}', flush=True)
            results[name] = dict(**summarize_outcomes(rows), outcomes=rows,
                                 diversity=pottery_metrics(games, {candidate.policy_id}))
    neural_results = [v for k, v in results.items() if k.endswith('neural')]
    trained_reference_count = len({r['model_id'] for r in spec['references']
        if r['training_mode'] in ('step5-population-ppo', 'step5-reptile')})
    eligible = (partition == 'validation' and setups >= 30 and trained_reference_count >= 2 and
                all(v['win_share_setup_bootstrap_95'][0] > .25 for v in neural_results) and
                not any(v['diversity']['pottery_warning'] for v in results.values()))
    report = dict(format='step5-evaluation-report/1.0.0', spec=spec,
                  candidate_sha256=spec['candidate_sha256'], results=results,
                  promotion_eligible=eligible, failed_games=0, raw_record_sha256=raw_evidence,
                  limitation='Frozen opponents, five-setup smoke comparisons and legacy-only wins do not establish broad strength. Search comparisons must use equal whole-turn budgets. No human-play claim.')
    write_json(directory / 'report.json', report)
    return report


def promote(root, checkpoint, report_path, output):
    root = Path(root).resolve(); report = read(project_path(root, report_path))
    require(report['format'] == 'step5-evaluation-report/1.0.0' and report['promotion_eligible'],
            'Candidate has not passed neural-population promotion gates')
    require(sha(project_path(root, checkpoint)) == report['candidate_sha256'], 'Evaluation was for other weights')
    record = dict(format='step5-promoted/1.0.0', checkpoint=checkpoint,
                  checkpoint_sha256=report['candidate_sha256'], report=report_path,
                  report_sha256=sha(project_path(root, report_path)),
                  scope='Neural population validation only; laptop release and human playtests remain separate')
    write_json(project_path(root, output), record)
    return record
