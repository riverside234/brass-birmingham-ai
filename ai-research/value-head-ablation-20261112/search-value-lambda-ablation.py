#!/usr/bin/env python3
"""Compare five search-value lambdas using fresh width-32 residual heads."""

import argparse
from collections import defaultdict
import gzip
import hashlib
from itertools import islice
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))

import numpy as np
import torch

from brass_ai.learning.checkpoint import load_checkpoint
from brass_ai.learning.contracts import digest, project_path, read_json, require
from brass_ai.recovery.search_value_training import Companion


RUN = 'artifacts/step5/joint7-v11-2-g1190-restart-3'
LAMBDAS = (0.95, 0.90, 0.85, 0.80, 0.75)
REFERENCE_LAMBDA = 0.80
POSITIONS = 64
SPLIT_SEED = 4504
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 4507


def buffer_games(directory, receipts, generation, reward_id):
    directory = directory.resolve()
    seen = set()
    for receipt in sorted(receipts, key=lambda r: (r['generation'], r['file'])):
        age, split = receipt['generation'], receipt['split']
        require(split in ('train', 'heldout'), f'Unexpected split in {receipt["file"]}')
        if (receipt['reward_id'] != reward_id or age >= generation
                or (split == 'train' and age < generation - 8)):
            continue

        path = (directory / receipt['file']).resolve()
        require(path.parent == directory, f'Invalid value shard path: {path}')
        data = path.read_bytes()
        require(hashlib.sha256(data).hexdigest() == receipt['sha256'], f'Changed value shard: {path}')
        rows = json.loads(gzip.decompress(data))
        require(len(rows) == receipt['rows'], f'Wrong row count in {path}')
        games = defaultdict(list)
        for row in rows:
            require((row['generation'], row['split'], row['reward_id']) == (age, split, reward_id)
                    and row['game_id'] == receipt.get('game_id', row['game_id']),
                    f'Wrong value row identity in {path}')
            games[row['game_id']].append(row)

        for game_id, game in sorted(games.items()):
            game.sort(key=lambda row: row['index'])
            require(game_id not in seen and [r['index'] for r in game] == list(range(1, POSITIONS + 1)),
                    f'Duplicate or incomplete value game: {game_id}')
            seen.add(game_id)
            yield game


def pack_games(companion, games):
    # Encode eight games at a time and keep only the inputs needed by the head.
    chunks, game_ids, splits, ages = [], [], [], []
    utility, roots, terminals = [], [], []
    while block := list(islice(games, 8)):
        rows = [row for game in block for row in game]
        chunks.append(companion._head_inputs(rows)[:3])
        for game in block:
            game_ids.append(game[0]['game_id'])
            splits.append(game[0]['split'])
            ages.append(game[0]['generation'])
            utility.append([row['u'] for row in game])
            roots.append([row['r'] for row in game])
            terminals.append(game[0]['terminal_utility'])
    require(chunks, 'No active search-value games in the checkpoint buffer')
    packed = tuple(torch.cat(parts) for parts in zip(*chunks, strict=True))
    labels = tuple(np.asarray(values, dtype=np.float64) for values in (utility, roots, terminals))
    require(all(np.isfinite(values).all() for values in labels), 'Nonfinite value labels')
    return packed, np.asarray(splits), sorted(set(ages)), labels, game_ids


def lambda_returns(utility, roots, terminal, lam):
    """The SearchValuePhase recurrence, batched across complete games."""
    next_utility = np.concatenate((utility[:, 1:], terminal[:, None]), axis=1)
    bootstrap = np.concatenate((roots[:, 1:], terminal[:, None]), axis=1)
    costs = next_utility - utility
    following = terminal
    targets = np.empty_like(utility)
    for position in reversed(range(utility.shape[1])):
        following = -costs[:, position] + (1 - lam) * bootstrap[:, position] + lam * following
        targets[:, position] = following
    return targets


def partition(splits):
    buffered = np.flatnonzero(splits == 'train')
    require(len(buffered) >= 2, 'Need at least two complete training-buffer games')
    validation_count = max(1, round(0.20 * len(buffered)))
    validation = np.sort(np.random.default_rng(SPLIT_SEED).permutation(buffered)[:validation_count])
    groups = {'fit': np.setdiff1d(buffered, validation), 'buffer-val': validation}
    official = np.flatnonzero(splits == 'heldout')
    if len(official):
        groups['official'] = official
    return groups


def score(prediction, utility, groups):
    errors = utility - prediction.detach().cpu().numpy().reshape(utility.shape)
    require(np.isfinite(errors).all(), 'Nonfinite value predictions')
    result = {}
    for name, games in groups.items():
        residuals = errors[games]
        squared_errors = residuals ** 2
        variance = float(np.var(utility[games]))
        mse = float(np.mean(squared_errors))
        ev = float(1 - np.var(residuals) / variance) if variance > 1e-12 else None
        result[name] = dict(mse=mse, ev=ev)
        if name == 'official':
            result[name]['game_mse'] = squared_errors.mean(axis=(1, 2))
    return result


def fit(model, inputs, labels, groups, generation, steps, seeds):
    features, baseline, mapping = inputs
    utility, roots, terminals = labels
    fit_games = groups['fit']
    fit_rows = (fit_games[:, None] * POSITIONS + np.arange(POSITIONS)).ravel()
    fit_indices = torch.as_tensor(fit_rows, device=features.device)
    fit_inputs = tuple(tensor[fit_indices] for tensor in inputs)
    targets = {
        lam: features.new_tensor(lambda_returns(
            utility[fit_games], roots[fit_games], terminals[fit_games], lam).reshape(-1, 4))
        for lam in LAMBDAS
    }

    results = defaultdict(list)
    row_count = len(fit_rows)
    for seed in seeds:
        planner = Companion(model, seed=seed, hidden=32)
        # A cap of one step per row lets the production sampler use the entire epoch.
        epoch_batches = planner._head_batches(row_count, generation, row_count, one_epoch=True)
        step_batches = planner._head_batches(row_count, generation, steps[-1])
        fixed_steps = [(end, step_batches[start:end])
                       for start, end in zip((0, *steps[:-1]), steps, strict=True)]

        # Start a fresh head for each schedule so the fixed-step scores stay comparable.
        for schedule in ((('1 epoch', epoch_batches),), fixed_steps):
            for lam in LAMBDAS:
                del model.search_value_head
                companion = Companion(model, seed=seed, hidden=32)
                cached = (*fit_inputs, targets[lam])
                for budget, batches in schedule:
                    companion._head_steps(cached, batches)
                    with torch.no_grad():
                        prediction = baseline + model.search_value_head(features).gather(1, mapping)
                    results[lam, budget].append(score(prediction, utility, groups))
        print(f'Completed seed {seed}', file=sys.stderr, flush=True)
    return results


def print_row(label, observations, groups):
    cells = []
    for name in groups:
        mse = [item[name]['mse'] for item in observations]
        ev = [item[name]['ev'] for item in observations]
        mean_ev = 'n/a' if ev[0] is None else f'{np.mean(ev):.6f}'
        cells.append(f'{np.mean(mse):10.6f} {mean_ev:>10}')
    print(f'{label:>15}  ' + '  '.join(cells))


def print_paired_mse(results, budgets, game_ids):
    candidates = [lam for lam in LAMBDAS if lam != REFERENCE_LAMBDA]
    count = len(game_ids)
    print(f'\nOfficial paired MSE differences vs lambda={REFERENCE_LAMBDA:.2f} ({count} games)')
    print('Averaged over seeds within each game. Negative favors the candidate.')
    draws = None
    if count >= 2:
        draws = np.random.default_rng(BOOTSTRAP_SEED).integers(count, size=(BOOTSTRAP_RESAMPLES, count))
        print(f'95% percentile intervals from {BOOTSTRAP_RESAMPLES:,} paired game resamples.')
    else:
        print('95% interval unavailable with one game.')

    width = max(len('95% CI lower'), *(len(game_id) for game_id in game_ids))
    columns = '  '.join(f'{lam:11.2f}' for lam in candidates)
    for budget in budgets:
        per_seed_mse = {
            lam: np.stack([item['official']['game_mse'] for item in results[lam, budget]])
            for lam in LAMBDAS
        }
        reference = per_seed_mse[REFERENCE_LAMBDA]
        differences = np.column_stack([(per_seed_mse[lam] - reference).mean(axis=0) for lam in candidates])
        rows = list(zip(game_ids, differences, strict=True))
        rows.append(('Mean', differences.mean(axis=0)))
        if draws is not None:
            means = differences[draws].mean(axis=1)
            lower, upper = np.quantile(means, (0.025, 0.975), axis=0)
            rows.extend([('95% CI lower', lower), ('95% CI upper', upper)])
        print(f'\nBudget {budget}')
        print(f'{"game_id":<{width}}  {columns}')
        for label, values in rows:
            print(f'{label:<{width}}  ' + '  '.join(f'{value:+11.8f}' for value in values))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', default=RUN, help='Lab training run with state.json and search-values/')
    parser.add_argument('--generation', type=int, help='Committed checkpoint (default: latest in state.json)')
    parser.add_argument('--device', default='cpu', help='Torch device, e.g. cpu or cuda:0')
    parser.add_argument('--steps', type=int, nargs='+', default=[256, 512],
                        help='Fixed Adam step counts to report alongside the full-epoch fit')
    parser.add_argument('--seeds', type=int, nargs='+', default=[4504, 4505, 4506], help='Paired initialization and batch seeds')
    args = parser.parse_args()
    if sorted(set(args.steps)) != args.steps or args.steps[0] < 1:
        parser.error('--steps must be strictly increasing positive integers')
    if len(set(args.seeds)) != len(args.seeds) or min(args.seeds) < 0:
        parser.error('--seeds must be distinct nonnegative integers')

    try:
        run = project_path(ROOT, args.run)
        state = read_json(run / 'state.json')
        generation = state['iteration'] if args.generation is None else args.generation
        entries = [entry for entry in state['history'] if entry['generation'] == generation]
        require(len(entries) == 1 and generation <= state['iteration'], f'No committed g{generation} in {run}')
        entry = entries[0]
        checkpoint = project_path(ROOT, entry['path'])
        torch.set_num_threads(2)
        model, raw = load_checkpoint(checkpoint, device=args.device)
        require(raw['step'] == generation and raw['model_id'] == entry['model_id'], 'Checkpoint identity differs')
        saved = raw['training'].get('search_value')
        reward_id = digest(model.config.utility)
        require(saved and saved['generation'] == generation and saved['reward_id'] == reward_id,
                f'g{generation} has no matching search-value buffer')
        receipts = saved['buffer']['receipts']
        del raw, saved
        model.eval().requires_grad_(False)
        if hasattr(model, 'search_value_head'):
            del model.search_value_head
        print(f'g{generation} | width=32 | frozen encoder and Utility head | device={args.device}', flush=True)
        games = buffer_games(run / 'search-values', receipts, generation, reward_id)
        inputs, splits, ages, labels, game_ids = pack_games(Companion(model, hidden=32), games)
        groups = partition(splits)
        counts = ', '.join(f'{name}={len(indices)}' for name, indices in groups.items())
        print(f'Games: {counts} | collection source generations: {ages} | split seed: {SPLIT_SEED}', flush=True)
        print(f'Seeds: {args.seeds} | scoring target: realized utility-to-go u', flush=True)
        baseline = score(inputs[1], labels[0], groups)
        results = fit(model, inputs, labels, groups, generation, args.steps, args.seeds)
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(1, f'Error: {exc}\n')

    print(f'{"lambda @ budget":>15}  ' + '  '.join(f'{name + " MSE/EV":^21}' for name in groups))
    print_row('Utility (w=0)', [baseline], groups)
    budgets = ('1 epoch', *args.steps)
    for budget in budgets:
        for lam in LAMBDAS:
            print_row(f'{lam:.2f} @ {budget}', results[lam, budget], groups)
    if 'official' in groups:
        print_paired_mse(results, budgets, [game_ids[index] for index in groups['official']])


if __name__ == '__main__':
    main()
