"""Lab-root command line entry points; no implicit game installation changes."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import uuid

import torch

from .checkpoint import load_checkpoint, resume_optimizer, save_checkpoint
from .contracts import LAB_ROOT, digest, load_config, project_path, require
from .export import export_onnx
from .features import Encoder
from .model import FourSeatNet, ModelConfig
from .replay import read_shards, split_for, write_shard
from .runner import ModelPolicy, RandomPolicy, collect_games, match_report
from .training import evaluate_values, on_policy_update, prepared, supervised_updates


def parser():
    p = argparse.ArgumentParser(description='R2 four-player training scaffold (fresh weights; no strength claim)')
    p.add_argument('--project-root', default=str(LAB_ROOT), help='All data/model/host paths resolve below this root')
    p.add_argument('--config', help='Configuration file below project root')
    p.add_argument('--device', choices=['cpu', 'cuda', 'auto'])
    p.add_argument('--size', choices=['small', 'medium', 'large'])
    p.add_argument('--seed', type=int)
    sub = p.add_subparsers(dest='command', required=True)
    init = sub.add_parser('init')
    init.add_argument('--checkpoint', default='artifacts/r2/fresh.pt')
    collect = sub.add_parser('collect')
    collect.add_argument('--checkpoint', help='Omit for diagnostic legal-random collection')
    collect.add_argument('--output', required=True, help='New gzip JSONL shard path')
    collect.add_argument('--games', type=int, default=4)
    collect.add_argument('--workers', type=int, default=4)
    collect.add_argument('--mono')
    train = sub.add_parser('train')
    train.add_argument('--shards', nargs='+', required=True)
    train.add_argument('--checkpoint', required=True, help='Output checkpoint')
    train.add_argument('--from-checkpoint')
    train.add_argument('--resume', action='store_true')
    train.add_argument('--mode', choices=['value', 'teacher'], default='value')
    train.add_argument('--steps', type=int, default=100)
    iterate = sub.add_parser('iterate', help='Experimental terminal-reward self-play, before R3 search')
    iterate.add_argument('--from-checkpoint')
    iterate.add_argument('--iterations', type=int, default=2)
    iterate.add_argument('--games', type=int, default=4)
    iterate.add_argument('--workers', type=int, default=4)
    iterate.add_argument('--mono')
    export = sub.add_parser('export')
    export.add_argument('--checkpoint', required=True)
    export.add_argument('--shards', nargs='+', required=True, help='Permitted-view examples for parity')
    export.add_argument('--output', default='artifacts/r2/model.onnx')
    evaluate = sub.add_parser('evaluate')
    evaluate.add_argument('--opponents', nargs=4, required=True, help='Four checkpoint paths or random')
    evaluate.add_argument('--games', type=int, default=4)
    evaluate.add_argument('--workers', type=int, default=4)
    evaluate.add_argument('--mono')
    evaluate.add_argument('--output', default='artifacts/r2/evaluation.json')
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    require(Path(args.project_root).is_absolute(), 'Use --project-root with an absolute path on this platform')
    root = Path(args.project_root).resolve()
    config = load_config(project_path(root, args.config) if args.config else None)
    config['project_root'] = str(root)
    device = args.device or config['device']
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    require(device != 'cuda' or torch.cuda.is_available(), 'CUDA requested but unavailable')
    torch.set_num_threads(config['threads'])
    seed = config['seed'] if args.seed is None else args.seed
    require(seed >= 0, 'Seed must be non-negative')
    torch.manual_seed(seed)
    resolve = lambda value: project_path(root, value)
    output = resolve(config['output'])
    host = resolve(config['host'])
    batch_size = config['batch_size']

    def fresh():
        return FourSeatNet(ModelConfig.preset(args.size or config['model_size'])).to(device)

    def get_policy(path):
        if path == 'random':
            return RandomPolicy()
        model, _ = load_checkpoint(resolve(path), device=device)
        return ModelPolicy(model, device, batch_size)

    def collect(seeds, policies, sampling_seed):
        return collect_games(host, seeds, policies, workers=args.workers,
            max_decisions=config['max_decisions'], mono=args.mono, sampling_seed=sampling_seed)

    if args.command == 'init':
        require(not resolve(args.checkpoint).exists(), 'Fresh checkpoint already exists')
        model = fresh()
        result = {'checkpoint': str(resolve(args.checkpoint)),
            'model_id': save_checkpoint(resolve(args.checkpoint), model, training={'mode': 'fresh', 'seed': seed}),
            'parameters': model.parameter_count, 'device': device}
    elif args.command == 'collect':
        require(args.games > 0, 'Games must be positive')
        policy = get_policy(args.checkpoint or 'random')
        games = collect(list(range(seed, seed + args.games)), [policy] * 4, seed + 1000003)
        write_shard(resolve(args.output), games, {'producer': 'R2/1.0.0', 'mode': 'diagnostic-collection'})
        result = {**match_report(games), 'shard': str(resolve(args.output))}
    elif args.command == 'train':
        require(not args.resume or args.from_checkpoint, '--resume requires --from-checkpoint')
        games = read_shards([resolve(p) for p in args.shards], partition='train')
        model, raw = load_checkpoint(resolve(args.from_checkpoint), device=device) if args.from_checkpoint else (fresh(), None)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'])
        training = {'mode': args.mode, 'batch_size': batch_size, 'learning_rate': config['learning_rate'],
                    'game_ids': sorted(g['game_id'] for g in games)}
        step = 0
        if args.resume:
            require(raw['training'] == training, 'Exact resume requires the same dataset and training configuration')
            resume_optimizer(raw, optimizer)
            step = raw['step']
        metrics = supervised_updates(model, optimizer, prepared(games), args.steps, batch_size, mode=args.mode)
        identity = save_checkpoint(resolve(args.checkpoint), model, optimizer, step=step + args.steps, training=training)
        result = {'model_id': identity, 'step': step + args.steps, 'last_metrics': metrics[-1], 'train_games': len(games)}
        # Validation positions never contribute a gradient. A missing partition
        # is reported instead of silently reusing training examples.
        all_games = read_shards([resolve(p) for p in args.shards])
        held = [g for g in all_games if g['split'] == 'validation']
        result['validation'] = evaluate_values(model, held, batch_size) if held else 'No held-out validation games in these shards'
    elif args.command == 'iterate':
        require(args.iterations > 0 and args.games > 0, 'Positive iterations/games required')
        model = load_checkpoint(resolve(args.from_checkpoint), device=device)[0] if args.from_checkpoint else fresh()
        optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'])
        run = output / ('selfplay-' + uuid.uuid4().hex[:12])
        metrics, next_seed = [], seed
        for iteration in range(args.iterations):
            seeds = []
            while len(seeds) < args.games:
                group = digest({'rules': 'steam-brass-patch-1.1.0', 'setup_seed': next_seed})
                if split_for(group) == 'train':
                    seeds.append(next_seed)
                next_seed += 1
            policy = ModelPolicy(model, device, batch_size)
            games = collect(seeds, [policy] * 4, seed + 1000003 + iteration)
            write_shard(run / f'iteration-{iteration:04d}.jsonl.gz', games,
                        {'producer': 'R2/1.0.0', 'mode': 'terminal-on-policy-experiment'})
            metric = on_policy_update(model, optimizer, games, batch_size)
            metrics.append(metric)
            save_checkpoint(run / f'iteration-{iteration:04d}.pt', model, optimizer,
                step=iteration + 1, training={'mode': 'terminal-on-policy-experiment', 'setup_seeds': seeds,
                                            'next_setup_seed': next_seed, 'iteration': iteration})
            print(json.dumps({'iteration': iteration, **metric}), flush=True)
        result = {'run': str(run), 'iterations': metrics, 'strength': 'unmeasured'}
    elif args.command == 'export':
        model, _ = load_checkpoint(resolve(args.checkpoint))
        games = read_shards([resolve(p) for p in args.shards])
        rows = [r for g in games for r in g['records']]
        encoder = Encoder()
        first = encoder.encode(rows[0]['view'], rows[0]['choices'], rows[0]['style'], rows[0]['intent'])
        second = next(r for r in rows if len(r['choices']) != len(first.choice_ids))
        second = encoder.encode(second['view'], second['choices'], second['style'], second['intent'])
        result = export_onnx(model, resolve(args.output), [[first], [second, first], [second]])
    else:
        require(args.games > 0 and args.games % 4 == 0, 'Evaluation requires complete four-seat rotations')
        pool = {path: get_policy(path) for path in set(args.opponents)}
        games = collect(list(range(seed, seed + args.games)), [pool[p] for p in args.opponents], seed + 1000003)
        result = match_report(games)
        target = resolve(args.output)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, indent=2), flush=True)
    return result
