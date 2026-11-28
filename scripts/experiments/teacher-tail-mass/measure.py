#!/usr/bin/env python3
"""Measure how long-search teacher targets change unvisited action mass."""

import argparse
from dataclasses import dataclass
import gzip
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'src'))

import numpy as np
import torch

from brass_ai.bank import sha
from brass_ai.learning.checkpoint import load_checkpoint
from brass_ai.learning.contracts import project_path, read_json, require
from brass_ai.recovery.encoding import PreparedEncoder
from brass_ai.recovery.teacher_labels import LabelRow, label_profile, validate_improved_policy
from brass_ai.recovery.teacher_term import policy_logits, search_bonus


RUN = 'artifacts/step5/joint7-v11-2-g4970-rail025-soft8-beta0.02'
SPLITS = ('train', 'heldout')
ROUNDING_TOLERANCE = 1e-12


@dataclass(frozen=True)
class TailMass:
    student: float
    teacher: float
    delta: float | None
    visited: int


def tail_mass(logits, label, scale):
    # The main policy supplies both the student and the detached target prior.
    logits = np.asarray(logits, dtype=np.float64)
    visited = np.asarray(label['counts']) > 0
    count = int(visited.sum())
    require(count > 0, 'Long teacher label has no visits')
    if visited.all():
        return TailMass(0., 0., None, count)

    # Match the training bonus tensor's precision, then sum the mass in float64.
    boosted = logits + search_bonus(label, scale).astype(np.float32)
    student_visited = np.logaddexp.reduce(logits[visited])
    student_tail = np.logaddexp.reduce(logits[~visited])
    teacher_visited = np.logaddexp.reduce(boosted[visited])
    teacher_tail = np.logaddexp.reduce(boosted[~visited])
    delta = (teacher_visited - student_visited) - (teacher_tail - student_tail)
    student = float(np.exp(student_tail - np.logaddexp(student_visited, student_tail)))
    teacher = float(np.exp(teacher_tail - np.logaddexp(teacher_visited, teacher_tail)))
    return TailMass(student, teacher, float(delta), count)


def checked_sources(run, state, expected, generations):
    require(state['iteration'] == expected,
            f'Expected committed g{expected}, found g{state["iteration"]}. Use --state for a saved snapshot.')
    entry = state['history'][-1]
    require(entry['generation'] == expected, 'Run history does not end at the expected checkpoint')
    checkpoint = project_path(ROOT, entry['path'])
    require(checkpoint.is_relative_to(run), 'Checkpoint is outside the requested lab run')
    buffer = state.get('teacher_phase', {}).get('buffer') or {}
    receipts = [receipt for receipt in buffer.get('receipts', ())
                if receipt['generation'] < expected and receipt['profile_digest'] == buffer['identity']]
    require(receipts, f'No committed teacher shards in g{expected} state')
    if generations:
        recent = set(sorted({receipt['generation'] for receipt in receipts})[-generations:])
        receipts = [receipt for receipt in receipts if receipt['generation'] in recent]

    sources = []
    missing = [] if checkpoint.is_file() else [checkpoint]
    for receipt in sorted(receipts, key=lambda row: (row['generation'], row['split'])):
        generation, split = receipt['generation'], receipt['split']
        name = f'teacher-labels/generation-{generation:06d}-{split}.jsonl.gz'
        require(split in SPLITS and receipt['path'] == name, 'Unexpected teacher shard path')
        path = project_path(ROOT, name, base=run)
        sidecar = path.with_name(path.name + '.receipt.json')
        missing.extend(file for file in (path, sidecar) if not file.is_file())
        sources.append((path, sidecar, receipt))
    if missing:
        raise FileNotFoundError(f'Missing {len(missing)} lab inputs, including {missing[0]}')
    for _, sidecar, receipt in sources:
        require(read_json(sidecar) == receipt, f'Teacher receipt differs from the committed state: {sidecar}')
    return checkpoint, entry, sources


def measure_shard(model, encoder, path, receipt, scale, batch_rows):
    scores = []
    batch = []

    def score_batch():
        scores.extend(tail_mass(logits, row.label, scale)
                      for row, logits in zip(batch, policy_logits(model, batch), strict=True))
        batch.clear()

    rows_seen = short_rows = 0
    with gzip.open(path, 'rt', encoding='utf-8') as stream:
        for line in stream:
            row = json.loads(line)
            label = row['teacher']
            rows_seen += 1
            require(all(label[key] == receipt[key] for key in ('generation', 'model_id', 'reward_id')),
                    f'Teacher row lineage differs from its receipt: {path}')
            if label_profile(label) == 'fast':
                short_rows += 1
                continue
            require(validate_improved_policy(label, row['choices'], view=row['view']) is not None,
                    f'Unadmitted long-search label in committed shard: {path}')
            encoded = encoder.encode(row['view'], row['choices'], [0.] * 4, [0.] * 8)
            batch.append(LabelRow(dict(teacher=label), encoded))
            if len(batch) == batch_rows:
                score_batch()
    if batch:
        score_batch()
    require(rows_seen == receipt['rows'], f'Teacher shard row count differs: {path}')
    return scores, short_rows


def report(split, scores, short_rows):
    count = len(scores)
    print(f'\n{split}: {count:,} long-search rows, {short_rows:,} short-search rows skipped')
    if not count:
        return None
    student = np.array([row.student for row in scores])
    teacher = np.array([row.teacher for row in scores])
    difference = 100 * (teacher - student)
    has_tail = sum(row.delta is not None for row in scores)
    leak = np.array([row.delta is not None and row.delta < -ROUNDING_TOLERANCE for row in scores])
    outward = float(np.maximum(difference, 0.).mean())
    inward = float(np.maximum(-difference, 0.).mean())
    print(f'  Mean unvisited mass: student {student.mean():.3%}, teacher {teacher.mean():.3%}, '
          f'teacher minus student {difference.mean():+.3f} pp')
    tail_rate = f'{leak.sum() / has_tail:.2%} of rows with an unvisited action' if has_tail else 'no unvisited actions'
    print(f'  Delta < 0: {leak.sum():,}/{has_tail:,} ({tail_rate}, {leak.mean():.2%} of all long rows)')
    print(f'  Rows moving at least 1 pp to unvisited actions: {np.mean(difference >= 1.):.2%}')
    if leak.any():
        extra = difference[leak]
        print(f'  Extra unvisited mass on Delta < 0 rows: mean {extra.mean():.3f} pp, '
              f'median {np.median(extra):.3f} pp, P90 {np.quantile(extra, .9):.3f} pp, '
              f'max {extra.max():.3f} pp')
    print(f'  Outward / inward mass moved per long row: {outward:.3f} / {inward:.3f} pp')
    print(f'  Mean teacher unvisited mass with one-way / two-way centering: '
          f'{np.minimum(student, teacher).mean():.3%} / {student.mean():.3%}')
    print(f'  Rows with only one visited action: {sum(row.visited == 1 for row in scores):,}/{count:,} '
          '(no within-visited ranking gradient under a mask)')
    return outward, inward


def recommend(training, minimum):
    print(f'\nRecommendation (training rows, diagnostic threshold {minimum:g} pp mean movement per row)')
    if training is None:
        print('  Collect training evidence before choosing a centering or mask method.')
        return
    outward, inward = training
    if outward < minimum:
        print('  Keep the current main teacher target. Observed outward mass is below the diagnostic threshold.')
    elif inward >= minimum:
        print('  A/B test one-way centering first. It removes the measured outward mass and retains '
              'the separate tail-suppression signal.')
    else:
        print('  A/B test two-way centering first. Outward mass is material and the retained '
              'tail-suppression signal would be small. A double-sided mask is an alternative if you '
              'want an unweighted within-visited KL instead of weighting it by the student visited mass.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=Path(RUN), help='Lab run holding the checkpoint and labels')
    parser.add_argument('--state', type=Path, help='Saved state.json snapshot. Defaults to the lab run state.')
    parser.add_argument('--expected-generation', type=int, required=True)
    parser.add_argument('--label-generations', type=int, default=0,
                        help='Use the newest N committed label generations. The default 0 uses all retained ones.')
    parser.add_argument('--device', default='cpu', help='PyTorch device, e.g. cpu or cuda:0')
    parser.add_argument('--threads', type=int, default=2, help='PyTorch CPU threads')
    parser.add_argument('--batch-rows', type=int, default=64, help='Maximum labels per policy forward')
    parser.add_argument('--min-shift-pp', type=float, default=.25,
                        help='Meaningful mean outward/inward mass per row for the printed recommendation')
    args = parser.parse_args()
    if (args.expected_generation < 1 or args.label_generations < 0 or args.threads < 1
            or args.batch_rows < 1 or not 0 < args.min_shift_pp <= 100):
        parser.error('Require a positive generation, thread and batch sizes, a non-negative label window, '
                     'and 0 < --min-shift-pp <= 100')

    try:
        run = project_path(ROOT, args.run)
        state = read_json(project_path(ROOT, args.state) if args.state else run / 'state.json')
        settings = state['signature']['config'].get('teacher_revision', {}).get('targets', {})
        require(settings.get('kind') == 'rebuilt-from-current-policy' and settings.get('labels') == 'long-only',
                'Expected the main teacher recipe that rebuilds targets from current logits on long labels')
        scale = settings['label_scale']
        checkpoint, entry, sources = checked_sources(run, state, args.expected_generation, args.label_generations)
        require(sha(checkpoint) == entry['sha256'], f'Checkpoint hash differs: {checkpoint}')
        torch.set_num_threads(args.threads)
        model, raw = load_checkpoint(checkpoint, device=args.device)
        require(raw['step'] == entry['generation'] and raw['model_id'] == entry['model_id'],
                'Loaded checkpoint differs from the committed g' + str(args.expected_generation))
        del raw
        model.eval()

        generations = sorted({receipt['generation'] for _, _, receipt in sources})
        print(f'Checkpoint g{entry["generation"]} ({entry["model_id"][:12]}); main policy T=1, label scale {scale:g}',
              flush=True)
        print(f'Scoring long-search rows in {len(sources)} committed shards from '
              f'g{generations[0]} to g{generations[-1]}. Rows have equal weight and are shown by collection split.',
              flush=True)
        per_split = {split: [] for split in SPLITS}
        skipped = dict.fromkeys(SPLITS, 0)
        encoder = PreparedEncoder()
        for path, _, receipt in sources:
            values, short_rows = measure_shard(model, encoder, path, receipt, scale, args.batch_rows)
            per_split[receipt['split']].extend(values)
            skipped[receipt['split']] += short_rows
        require(any(per_split.values()), 'No usable long-search labels in the selected shards')
    except (OSError, ValueError) as exc:
        parser.exit(2, f'Experiment failed: {exc}\n')

    results = {split: report(split, per_split[split], skipped[split]) for split in SPLITS}
    recommend(results['train'], args.min_shift_pp)


if __name__ == '__main__':
    main()
