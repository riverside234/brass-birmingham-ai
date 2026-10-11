"""Evaluate one completed joint-IL policy against three original batch6 policies.

An explicit cross-run, evaluation-only entry point. Reuses Step 5's actors,
inference service, replay validation, official scoring and setup bootstrap.
Neither checkpoint metadata nor the training population is rewritten.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

SOURCE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE_ROOT / 'src'))

from brass_ai.bank import read, sha
from brass_ai.learning.contracts import LAB_ROOT, digest, project_path, require
from brass_ai.step5.evaluation import (check_evaluation_batch, evaluation_config,
    evaluation_jobs, outcome, summarize_outcomes)
from brass_ai.step5.league import implementation_id, seeds_for, write_json
from brass_ai.recovery.demonstrations import completed_imitation
from brass_ai.recovery.execution import ActorPool, neural_spec
from brass_ai.recovery.inference import InferenceService
from brass_ai.recovery.replay import read_shards
from brass_ai.recovery.retention import run_lock

FORMAT = 'joint-il-head-to-head/1.0.0'


def frozen_origins(root, candidate_run, reference_run):
    origins, architectures = [], []
    for run in (candidate_run, reference_run):
        model, origin = completed_imitation(root, run)
        require(model.config.architecture == 'brass-four-seat-mlp-resnet/1.0.0',
                'Comparison requires completed residual IL models')
        architectures.append(model.config.to_dict())
        origins.append(origin)
        del model
    require(architectures[0] == architectures[1], 'Comparison architectures differ')
    require(origins[0]['model_id'] != origins[1]['model_id'], 'Candidate and reference are identical')
    return origins


def checked_output(root, output, candidate_run, reference_run):
    directory = project_path(root, output)
    artifacts = project_path(root, 'artifacts')
    require(directory != artifacts and directory.is_relative_to(artifacts),
            'Evaluation output must be a dedicated directory under artifacts')
    for run in (candidate_run, reference_run):
        source = project_path(root, run)
        require(not (directory.is_relative_to(source) or source.is_relative_to(directory)),
                'Evaluation output overlaps an input IL run')
    return directory


def build_jobs(directory, spec):
    candidate, reference = [neural_spec(o['model_id'], 0, 'neutral') for o in spec['origins']]
    jobs = []
    for index, job in enumerate(evaluation_jobs(spec['seeds'])):
        provenance = dict(producer=FORMAT, purpose='evaluation', kind='one-vs-three',
            seed=job['seed'], sampling_seed=job['sampling_seed'], rotation=job['seat'], spec=digest(spec))
        jobs.append(dict(index=index, seed=job['seed'], rotation=job['seat'],
            sampling_seed=job['sampling_seed'], policies=[candidate, reference, reference, reference],
            path=str(directory / f'one-vs-three-{index:06d}.jsonl.gz'), provenance=provenance))
    return jobs


def checked_outcomes(games, records, jobs, spec):
    scheduled = {j['index']: j for j in jobs}
    require(len(records) == len(jobs) and {r['job'] for r in records} == set(scheduled),
            'Missing, duplicate or foreign evaluation receipts')
    rotations = {r['game_id']: scheduled[r['job']]['rotation'] for r in records}
    require(len(rotations) == len(records) and len(games) == len(records)
            and {g['game_id'] for g in games} == set(rotations), 'Evaluation replay identities differ')
    candidate_id, reference_id = [neural_spec(o['model_id'], 0, 'neutral')['policy_id'] for o in spec['origins']]
    for game in games:
        expected = [reference_id] * 4
        expected[rotations[game['game_id']]] = candidate_id
        require(game['opponents'] == expected, 'Expected exactly one candidate and three original references')
    check_evaluation_batch(games, evaluation_jobs(spec['seeds']), candidate_id, spec['config']['partition'])
    by_group = {digest(dict(rules='steam-brass-patch-1.1.0', setup_seed=s)): s for s in spec['seeds']}
    return sorted((outcome(g, candidate_id, by_group[g['setup_group']]) for g in games),
                  key=lambda r: (r['seed'], r['seat']))


def collect(root, directory, spec, *, device, mono=None):
    config = spec['config']; jobs = build_jobs(directory, spec); began = time.monotonic()
    with InferenceService(config['execution'], device) as service:
        for origin in spec['origins']:
            path = project_path(root, origin['path'])
            require(sha(path) == origin['sha256'], 'Frozen IL checkpoint changed')
            require(service.register_checkpoint(path) == origin['model_id'], 'Frozen model identity changed')
        with ActorPool(root, config['host'], service, config['execution'], mono=mono) as actor:
            records = actor.collect(jobs)
        games = read_shards([r['path'] for r in records], receipts={r['path']: r.get('features') for r in records})
        rows = checked_outcomes(games, records, jobs, spec)
        telemetry = service.report()
    for origin in spec['origins']:
        require(sha(project_path(root, origin['path'])) == origin['sha256'], 'Frozen IL checkpoint changed')
    return dict(format=FORMAT, spec=spec, results=dict(**summarize_outcomes(rows), outcomes=rows),
        failed_games=0, cached_games=sum(r['cached'] for r in records),
        raw_record_sha256={Path(r['path']).name: r['sha256'] for r in records},
        inference=telemetry, elapsed_seconds=time.monotonic()-began,
        gameplay_promoted=False, purpose='evaluation',
        note='One candidate versus three identical references, neutral policies without search. '
             'Official win shares and tiebreakers; 25% is the symmetric reference. No automatic promotion.')


def evaluate(args):
    import torch
    from brass_ai.step5.assets import require_preflight
    root = args.root.resolve()
    directory = checked_output(root, args.output, args.candidate_run, args.reference_run)
    require(args.device != 'cuda' or torch.cuda.is_available(), 'CUDA requested but unavailable')
    require_preflight(root)
    config = evaluation_config(read(project_path(root, args.config)), setups=args.setups,
        seed=args.seed, partition='validation', include_legacy=False, use_search=False,
        focus='neutral', lineups=['recent-neural'])
    torch.set_num_threads(config['threads'])
    origins = frozen_origins(root, args.candidate_run, args.reference_run)
    seeds, _ = seeds_for(config['seed'], config['setups'], config['partition'])
    spec = dict(format=FORMAT, purpose='evaluation', origins=origins, config=config, seeds=seeds,
        implementation=implementation_id(),
        script_sha256=hashlib.sha256(Path(__file__).read_text(encoding='utf-8').replace('\r\n', '\n').encode()).hexdigest())
    with run_lock(directory):
        spec_path = directory / 'spec.json'
        if spec_path.exists():
            require(read(spec_path) == spec, 'Evaluation specification changed; use a new --output')
        else:
            require(not any(p.name != '.active-run.json' for p in directory.iterdir()),
                    'Use a fresh evaluation directory')
            write_json(spec_path, spec)
        if (directory / 'report.json').exists():
            report = read(directory / 'report.json')
            require(report['spec'] == spec and report['failed_games'] == 0
                    and report['results']['games'] == 4 * len(seeds), 'Cached evaluation report differs')
        else:
            report = collect(root, directory, spec, device=args.device, mono=args.mono)
            write_json(directory / 'report.json', report)
        result = report['results']
        summary = dict(format=FORMAT, candidate=origins[0], reference=origins[1],
            reference_seats=3, search=False, symmetric_win_share=.25,
            results={k: v for k, v in result.items() if k != 'outcomes'},
            lower_95_above_symmetric=result['win_share_setup_bootstrap_95'][0] > .25,
            gameplay_promoted=False, report=str(directory / 'report.json'))
        write_json(directory / 'summary.json', summary)
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=LAB_ROOT)
    p.add_argument('--candidate-run', default='artifacts/step5/v4-batch7-joint')
    p.add_argument('--reference-run', default='artifacts/step5/v4-batch6')
    p.add_argument('--output', default='artifacts/il-replay/v4-batch7-joint/evaluations/joint-vs-batch6')
    p.add_argument('--config', default='configs/step5/evaluation.json')
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    p.add_argument('--setups', type=int, default=40, help='Each setup is played at all four seats')
    p.add_argument('--seed', type=int, default=1947000)
    p.add_argument('--mono')
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    require(args.root.is_absolute(), 'Use an absolute project --root')
    require(args.setups > 0 and 0 <= args.seed < 2**31, 'Invalid evaluation setup settings')
    os.chdir(SOURCE_ROOT)  # Stable launch directory for spawned simulation workers.
    return evaluate(args)


if __name__ == '__main__':
    main()
