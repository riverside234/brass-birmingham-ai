"""Joint BC rehearsal over audited batch4--7 data, using the existing trainer.

This is full-data supervised experience replay, inspired by CLEAR's rehearsal
principle, not an implementation of CLEAR's off-policy RL algorithm. No source
dataset, parent checkpoint, PPO setting or installed game file is modified.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys

SOURCE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE_ROOT / 'src'))
from brass_ai.learning.contracts import (LAB_ROOT, assert_heuristic_data_compatible,
    digest, project_path, read_json, require)

RECIPE = 'joint-il-rehearsal/1.0.0'
REFERENCE = 'https://proceedings.neurips.cc/paper_files/paper/2019/hash/fa7cdfad1a5aaf8370ebeda47a1ff1c3-Abstract.html'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save(path, value, *, immutable=False):
    path = Path(path)
    if immutable and path.exists():
        require(read_json(path) == value, f'Existing replay recipe differs: {path}; use a new --experiment')
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def dataset_source(root, run):
    """Resolve the actual manifest from a completed run, not a guessed filename."""
    run = project_path(root, run)
    state, report = read_json(run / 'state.json'), read_json(run / 'report.json')
    signature = state['signature']
    require(report.get('format') == 'bc-training-report/1.0.0' and report.get('signature') == signature,
            f'Incomplete or mismatched imitation run: {run}')
    require(report['epochs'] == state['epoch'] and
            (state['stopped'] or state['epoch'] == signature['epochs']), f'Imitation still incomplete: {run}')
    manifest = project_path(root, signature['manifest_path'])
    require(sha(manifest) == signature['manifest_sha256'], f'Source manifest changed: {manifest}')
    return dict(name=run.name, run=run.relative_to(root).as_posix(),
                manifest=manifest.relative_to(root).as_posix(), manifest_sha256=sha(manifest)), read_json(manifest)


def merge_manifests(root, runs):
    """Full union, preserving partitions and original audited replay receipts."""
    from brass_ai.learning.replay import split_for
    from brass_ai.step5.league import implementation_id
    root = Path(root).resolve()
    require(len(runs) >= 2, 'Supply a latest run and at least one historical run')
    require(len({project_path(root, p) for p in runs}) == len(runs), 'Repeated source run')
    sources = [dataset_source(root, p) for p in runs]
    require(len({s['name'] for s, _ in sources}) == len(sources), 'Source run names must be distinct')
    first = sources[0][1]
    entries, teachers, receipt_sources = [], set(), []
    seen_semantics, seen_ids, path_hashes, group_splits = {}, {}, {}, {}
    for source, manifest in sources:
        require(manifest.get('format') == 'bc-dataset/1.0.0', 'Only audited BC manifests are accepted')
        assert_heuristic_data_compatible(manifest['compatibility'])
        for field in ('label_kind', 'context', 'calibration'):
            require(manifest.get(field) == first.get(field),
                    f'{source["name"]}: differing {field}; do not merge incompatible teacher receipts')
        require(manifest['label_kind'] == 'qualified-v3-choice/1.0.0' and
                manifest['context'] == 'neutral-predecision/1.0.0', 'Unsupported labels or context')
        allowed = set(manifest['teachers'])
        require(allowed, 'Manifest has no admitted teachers')
        teachers.update(allowed)
        selected, duplicates = [], 0
        for entry in manifest['entries']:
            require(entry.get('transitions_verified') is True, 'Unaudited demonstration')
            group, partition = entry['setup_group'], entry['split']
            require(partition == split_for(group), 'Relabeled setup partition')
            require(group_splits.setdefault(group, partition) == partition, 'Setup crosses partitions')
            require(entry.get('label_kind') == manifest['label_kind'] and
                    entry.get('context') == manifest['context'], 'Entry label/context differs')
            require(entry.get('teachers') and set(entry['teachers']) <= allowed, 'Unlisted demonstration teacher')
            semantic = entry['semantic_sha256']
            require(isinstance(semantic, str) and len(semantic) == 64 and
                    all(c in '0123456789abcdef' for c in semantic), 'Invalid semantic game digest')
            identity = (semantic, group, partition)
            require(seen_ids.setdefault(entry['game_id'], identity) == identity, 'Conflicting game ID')
            path = project_path(root, entry['path'])
            require(path.is_file(), f'Missing replay: {path}')
            require(path_hashes.setdefault(str(path), entry['sha256']) == entry['sha256'], 'Conflicting shard hashes')
            if semantic in seen_semantics:
                require(seen_semantics[semantic] == (group, partition), 'Conflicting semantic game')
                duplicates += 1
                continue
            seen_semantics[semantic] = (group, partition)
            entries.append(entry)  # Original fields, hashes and feature receipts stay intact.
            selected.append(entry['game_id'])
        receipt_sources.append(dict(source, input_games=len(manifest['entries']),
                                    duplicate_games=duplicates, selected_game_ids=selected))
    partitions = dict(Counter(e['split'] for e in entries))
    require(partitions.get('train', 0) and partitions.get('validation', 0), 'Train and validation games are required')
    return dict(format='bc-dataset/1.0.0', compatibility=first['compatibility'],
                implementation=implementation_id(), label_kind=first['label_kind'], context=first['context'],
                calibration=first.get('calibration'), teachers=sorted(teachers), entries=entries,
                excluded=[], partitions=partitions,
                rehearsal=dict(format=RECIPE, method='full-union-global-shuffle', reference=REFERENCE,
                    latest_fraction=None, notes='Every unique training game is included. Existing teacher/era/action balancing remains. '
                    'Validation and test retain their audited setup partitions. Duplicate ownership prefers the latest source.',
                    script_sha256=hashlib.sha256(Path(__file__).read_text(encoding='utf-8').replace('\r\n', '\n').encode()).hexdigest(),
                    sources=receipt_sources))


def prepare(root, latest, history, experiment, run):
    root = Path(root).resolve()
    sources = [project_path(root, p) for p in [latest, *history]]
    destination, experiment = project_path(root, run), project_path(root, experiment)
    for output in (destination, experiment):
        require(output.is_relative_to(root / 'artifacts') and output != root / 'artifacts',
                'Outputs must be specific directories under artifacts/')
        require(all(not output.is_relative_to(p) and not p.is_relative_to(output) for p in sources),
                'Output overlaps a source run')
    require(not destination.is_relative_to(experiment) and not experiment.is_relative_to(destination),
            'Keep experiment metadata separate from the imitation run')
    manifest = merge_manifests(root, sources)
    path = experiment / 'manifest.json'
    save(path, manifest, immutable=True)
    by_id = {e['game_id']: e for e in manifest['entries']}
    summary = dict(format=RECIPE, manifest=path.relative_to(root).as_posix(), manifest_sha256=sha(path),
        initialization=project_path(root, latest).relative_to(root).as_posix(),
        candidate_run=destination.relative_to(root).as_posix(), partitions=manifest['partitions'],
        sources=[dict(name=s['name'], input_games=s['input_games'], duplicates=s['duplicate_games'],
                      retained=dict(Counter(by_id[g]['split'] for g in s['selected_game_ids'])))
                 for s in manifest['rehearsal']['sources']])
    save(experiment / 'preparation.json', summary, immutable=True)
    print(json.dumps(summary, indent=2), flush=True)
    return path, manifest


def training_command(args, manifest):
    return [sys.executable, str(SOURCE_ROOT / 'scripts/step5.py'), '--root', str(args.root),
        '--device', args.device, '--config', args.config, 'imitate', '--manifest', str(manifest),
        '--from-imitation-run', args.latest_run, '--run', args.run, '--epochs', str(args.epochs),
        '--min-epochs', str(args.min_epochs), '--patience', str(args.patience),
        '--learning-rate', str(args.learning_rate)] + ([] if args.workers is None else ['--workers', str(args.workers)])


def launch(command, root):
    print('Running: ' + shlex.join(command), flush=True)
    subprocess.run(command, cwd=root, check=True)  # argv, not a shell-built command.


def aggregate(reports, config):
    """Recombine natural BC validation exactly by the existing loss denominators."""
    count = sum(r['decisions'] for r in reports)
    meaningful = sum(r['meaningful_decisions'] for r in reports)
    denominator = lambda r: max(1, r['meaningful_decisions']) if config.get('actor_nonforced') else r['decisions']
    total_denominator = max(1, meaningful) if config.get('actor_nonforced') else count
    policy = sum(r['policy_loss'] * denominator(r) for r in reports) / total_denominator
    value = sum(r['value_loss'] * r['decisions'] for r in reports) / count
    return dict(loss=policy + config['value_weight'] * value, policy_loss=policy, value_loss=value,
        meaningful_top1=sum(r['meaningful_top1'] * r['meaningful_decisions'] for r in reports) / max(1, meaningful),
        decisions=count, meaningful_decisions=meaningful)


def validation_choice(rows, candidate):
    """Original baselines remain selectable; this is NOT a gameplay promotion."""
    require(all(math.isfinite(r['overall']['loss']) for r in rows), 'Non-finite validation loss')
    originals = [r for r in rows if r['run'] != candidate]
    require(originals, 'Missing preserved original model')
    best = min(originals, key=lambda r: r['overall']['loss'])
    challenger = next(r for r in rows if r['run'] == candidate)
    if challenger['overall']['loss'] < best['overall']['loss'] - 1e-4:
        best = challenger
    return dict(selected_imitation_run=best['run'], validation_loss=best['overall']['loss'],
                gameplay_promoted=False, note='Validation-only selection; compare win share and placement before choosing the RL start.')


def validate(args, manifest_path, manifest):
    import torch
    from brass_ai.recovery import demonstrations as demos
    from brass_ai.recovery.updates import Dataset, supervised_epoch
    from brass_ai.step5.league import implementation_id
    config = read_json(project_path(args.root, args.config))
    torch.set_num_threads(config['threads'])
    require(args.device != 'cuda' or torch.cuda.is_available(), 'CUDA requested but unavailable')
    update = dict(config['update'], value_weight=.5)
    workers = args.workers if args.workers is not None else config.get('audit_workers', 1)
    data, _ = demos.load_dataset(args.root, manifest_path, 'validation', workers=workers, progress=True, compact=True)
    owners = {g: s['name'] for s in manifest['rehearsal']['sources'] for g in s['selected_game_ids']}
    grouped = defaultdict(list)
    for row in data:
        grouped[owners[row[2][0]]].append(row)
    runs = list(dict.fromkeys(project_path(args.root, p).relative_to(args.root).as_posix()
                             for p in [args.latest_run, *args.baseline_runs, args.run]))
    rows = []
    for run in runs:
        model, origin = demos.completed_imitation(args.root, run)
        model.to(args.device)
        per_source = {}
        for name, batch in sorted(grouped.items()):
            dataset = Dataset(batch, args.device, policy=False, progress=f'validation {name} prepare')
            per_source[name] = supervised_epoch(model, None, dataset, update, training=False,
                progress=f'validation {Path(run).name} on {name}')
            del dataset
        rows.append(dict(run=project_path(args.root, run).relative_to(args.root).as_posix(),
            checkpoint=origin, per_source=per_source, overall=aggregate(list(per_source.values()), update)))
        del model
    candidate = project_path(args.root, args.run).relative_to(args.root).as_posix()
    result = dict(format=RECIPE, manifest_sha256=sha(manifest_path), implementation=implementation_id(),
        models=rows, **validation_choice(rows, candidate))
    save(project_path(args.root, args.experiment) / 'validation-comparison.json', result)
    print(json.dumps({k: v for k, v in result.items() if k != 'models'}, indent=2), flush=True)
    return result


def evaluate(args):
    """Same seeds/seats/v4 bank for each model; candidate also faces three batch7s."""
    output = project_path(args.root, args.experiment) / 'evaluations'
    runs = list(dict.fromkeys(project_path(args.root, p).relative_to(args.root).as_posix()
                             for p in [args.latest_run, *args.baseline_runs, args.run]))
    summaries = []
    for index, run in enumerate(runs):
        location = project_path(args.root, run)
        destination = output / f'{index:02d}-{location.name}'
        lineups = ['mixed-v3']
        if location == project_path(args.root, args.run):
            lineups.insert(0, 'recent-neural')
        command = [sys.executable, str(SOURCE_ROOT / 'scripts/step5.py'), '--root', str(args.root),
            '--device', args.device, '--config', args.evaluation_config, 'evaluate',
            '--candidate', str(location / 'best.pt'), '--references', str(location / 'initial.pt'),
            '--output', str(destination), '--setups', str(args.setups), '--seed', str(args.seed),
            '--partition', 'validation', '--no-search', '--lineups', *lineups]
        launch(command, args.root)
        report = read_json(destination / 'report.json')
        summaries.append(dict(run=location.relative_to(args.root).as_posix(),
            report=destination.relative_to(args.root).as_posix() + '/report.json',
            results={name: {k: panel[k] for k in ('games', 'win_share', 'win_share_setup_bootstrap_95', 'mean_rank', 'mean_vp')}
                     for name, panel in report['results'].items()}))
    result = dict(format=RECIPE, evaluations=summaries, gameplay_promoted=False,
                  note='Win share/placement first, VP diagnostic. No automatic replacement or RL launch.')
    save(output / 'comparison.json', result)
    print(json.dumps(result, indent=2), flush=True)
    return result


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['prepare', 'train', 'validate', 'evaluate'])
    p.add_argument('--root', type=Path, default=LAB_ROOT)
    p.add_argument('--latest-run', default='artifacts/step5/v4-batch7')
    p.add_argument('--history-runs', nargs='+', default=[f'artifacts/step5/v4-batch{i}' for i in range(6, 3, -1)])
    p.add_argument('--baseline-runs', nargs='*', default=['artifacts/step5/v4-batch6'])
    p.add_argument('--experiment', default='artifacts/il-replay/v4-batch7-joint')
    p.add_argument('--run', default='artifacts/step5/v4-batch7-joint')
    p.add_argument('--config', default='configs/step5/lab.json')
    p.add_argument('--evaluation-config', default='configs/step5/evaluation.json')
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    p.add_argument('--workers', type=int)
    p.add_argument('--epochs', type=int, default=6)
    p.add_argument('--min-epochs', type=int, default=2)
    p.add_argument('--patience', type=int, default=2)
    p.add_argument('--learning-rate', type=float, default=3e-5)
    p.add_argument('--setups', type=int, default=40)
    p.add_argument('--seed', type=int, default=1947000)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    require(args.root.is_absolute(), 'Use an absolute project --root')
    args.root = args.root.resolve()
    require(args.epochs >= args.min_epochs >= 1 and args.patience > 0, 'Invalid stopping settings')
    require(math.isfinite(args.learning_rate) and 0 < args.learning_rate <= 1, 'Invalid learning rate')
    require(args.workers is None or 1 <= args.workers <= 64, 'Invalid CPU loader allowance')
    require(args.setups > 0 and 0 <= args.seed < 2**31, 'Invalid evaluation setup settings')
    os.chdir(SOURCE_ROOT)  # Stable multiprocessing launch directory.
    path, manifest = prepare(args.root, args.latest_run, args.history_runs, args.experiment, args.run)
    if args.action == 'train':
        launch(training_command(args, path), args.root)
        validate(args, path, manifest)
    elif args.action == 'validate':
        validate(args, path, manifest)
    elif args.action == 'evaluate':
        evaluate(args)
    else:
        print('Next: ' + shlex.join(training_command(args, path)), flush=True)


if __name__ == '__main__':
    main()
