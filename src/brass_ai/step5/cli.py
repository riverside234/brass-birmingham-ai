from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from brass_ai.learning.contracts import LAB_ROOT, project_path, read_json, require
from .assets import export_assets, import_runtime, preflight, require_preflight
from .league import train, validate_config
from .evaluation import evaluate, promote, validate_evaluation_config
from .meta import meta_train, meta_evaluate


def parser():
    p = argparse.ArgumentParser(description='Four-player self-play-first neural training and optional Reptile meta-RL')
    p.add_argument('--root', type=Path, default=LAB_ROOT)
    p.add_argument('--config', help='Defaults to configs/step5/evaluation.json for evaluate, otherwise configs/step5/lab.json')
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    p.add_argument('--mono')
    sub = p.add_subparsers(dest='command', required=True)
    assets = sub.add_parser('assets'); assets.add_argument('--bundle', default='artifacts/step5-transfer/private-rules-runtime.zip')
    transfer = sub.add_parser('import-runtime'); transfer.add_argument('--bundle', required=True)
    check = sub.add_parser('preflight'); check.add_argument('--record-parity', action='store_true')
    run = sub.add_parser('train')
    run.add_argument('--run', default='artifacts/step5/selfplay')
    initialization = run.add_mutually_exclusive_group()
    initialization.add_argument('--initial')
    initialization.add_argument('--from-run', help='Fork last committed weights, optimizer, history and setup cursor into a NEW run')
    run.add_argument('--iterations', type=int, default=1)
    evaluation = sub.add_parser('evaluate')
    evaluation.add_argument('--candidate', required=True)
    evaluation.add_argument('--references', nargs='+', required=True)
    evaluation.add_argument('--output', required=True)
    evaluation.add_argument('--setups', type=int)
    evaluation.add_argument('--seed', type=int)
    evaluation.add_argument('--partition', choices=['validation', 'test'])
    evaluation.add_argument('--include-legacy', action=argparse.BooleanOptionalAction, default=None)
    evaluation.add_argument('--search', action=argparse.BooleanOptionalAction, default=None)
    evaluation.add_argument('--focus')
    meta = sub.add_parser('meta-train')
    meta.add_argument('--initial', required=True); meta.add_argument('--references', nargs='+', required=True)
    meta.add_argument('--run', default='artifacts/step5/meta')
    meta.add_argument('--iterations', type=int, default=1); meta.add_argument('--tasks', type=int, default=3)
    meta.add_argument('--fraction', type=float, default=.1)
    metatest = sub.add_parser('meta-evaluate')
    metatest.add_argument('--candidate', required=True); metatest.add_argument('--references', nargs=3, required=True)
    metatest.add_argument('--output', required=True); metatest.add_argument('--seed', type=int, default=6510000)
    for q in (meta, metatest):
        q.add_argument('--inner-steps', type=int, default=2)
        q.add_argument('--support-games', type=int, default=4); q.add_argument('--query-games', type=int, default=4)
    promotion = sub.add_parser('promote')
    promotion.add_argument('--checkpoint', required=True); promotion.add_argument('--report', required=True)
    promotion.add_argument('--output', default='training/models/promoted.json')
    return p


def main(argv=None):
    a = parser().parse_args(argv)
    require(a.root.is_absolute(), 'Pass an absolute --root on this platform')
    root = a.root.resolve()
    require(a.device != 'cuda' or torch.cuda.is_available(), 'CUDA requested but unavailable')
    default_config = 'configs/step5/evaluation.json' if a.command == 'evaluate' else 'configs/step5/lab.json'
    config = read_json(project_path(root, a.config or default_config))
    if a.command == 'evaluate' and config.get('format') == 'step5-evaluation-config/1.0.0':
        validate_evaluation_config(config)
    else:
        validate_config(config)
    torch.set_num_threads(config['threads'])
    if a.command == 'assets':
        result = export_assets(root, bundle=a.bundle)
    elif a.command == 'import-runtime':
        result = import_runtime(root, a.bundle)
    elif a.command == 'preflight':
        result = preflight(root, device=a.device, mono=a.mono, record_parity=a.record_parity)
    elif a.command == 'promote':
        result = promote(root, a.checkpoint, a.report, a.output)
    else:
        require_preflight(root)
        if a.command == 'train':
            result = train(root, a.run, config, iterations=a.iterations, device=a.device, initial=a.initial,
                           mono=a.mono, from_run=a.from_run)
            result.pop('summaries', None)
        elif a.command == 'evaluate':
            result = evaluate(root, a.output, a.candidate, a.references, config, setups=a.setups, seed=a.seed,
                partition=a.partition, include_legacy=a.include_legacy, device=a.device, mono=a.mono,
                search=a.search, focus=a.focus)
            result = {k: result[k] for k in ('promotion_eligible', 'failed_games', 'candidate_sha256')}
        elif a.command == 'meta-train':
            result = meta_train(root, a.run, a.initial, a.references, config, iterations=a.iterations, tasks=a.tasks,
                inner_steps=a.inner_steps, support_games=a.support_games, query_games=a.query_games,
                fraction=a.fraction, device=a.device, mono=a.mono)
        else:
            result = meta_evaluate(root, a.output, a.candidate, a.references, config, seed=a.seed,
                inner_steps=a.inner_steps, support_games=a.support_games, query_games=a.query_games,
                device=a.device, mono=a.mono)
    print(json.dumps(result, indent=2, allow_nan=False))
