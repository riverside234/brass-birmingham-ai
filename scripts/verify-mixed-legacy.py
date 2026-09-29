"""Check a complete mixed neural/legacy shard without treating it as strength evidence."""
import argparse
from collections import Counter
import gzip
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import torch
from brass_ai.bank import LAB_ROOT, below, require, sha
from brass_ai.learning.batching import model_id
from brass_ai.learning.checkpoint import load_checkpoint
from brass_ai.learning.replay import read_shards
from brass_ai.learning.training import on_policy_update
from brass_ai.qualification import write_new


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=LAB_ROOT)
    parser.add_argument('--shard', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    shard = below(root, args.shard)
    games = read_shards([shard])  # Includes legality, compatibility and terminal targets.
    with gzip.open(shard, 'rt', encoding='utf-8') as source:
        header = json.loads(next(source))
    torch.set_num_threads(2)
    model, _ = load_checkpoint(below(root, args.checkpoint))
    identity = model_id(model)
    declared = header['provenance']['policy_ids']
    require(len(declared) == 4 and len(set(declared)) == 4 and identity in declared,
            'Header must identify the exact model and three distinct bank profiles')
    require(all(p == identity or p.startswith('bank-') for p in declared), 'Unexpected policy')
    counts = Counter()
    for game in games:
        require(set(game['opponents']) == set(declared), 'Header and game identities disagree')
        for row in game['records']:
            behavior = row['behavior']
            counts[behavior['kind']] += 1
            if behavior['policy_id'] == identity:
                require(behavior['kind'] == 'model', 'Neural behavior mislabeled')
                require(not any(row['style']) and not any(row['intent']), 'Unexpected model context')
            else:
                require(behavior['kind'] == 'external', 'Bank behavior mislabeled')
                require(sum(row['style']) == 1, 'Missing legacy style context')
            require(row['teacher'] is None, 'Mixed collection invented teacher labels')
    require(counts['model'] > 0 and counts['external'] > 0, 'Both behavior kinds required')
    before = model_id(model)
    rejected = False
    try:
        on_policy_update(model, torch.optim.AdamW(model.parameters()), games)
    except ValueError as error:
        require('On-policy update rejects' in str(error), 'Unexpected learner rejection')
        rejected = True
    require(rejected and before == model_id(model), 'Mixed replay reached the on-policy optimizer')
    report = dict(passed=True, purpose='integration only; fresh untrained neural model',
                  games=len(games), decisions=sum(counts.values()), behavior_counts=dict(counts),
                  policy_ids=declared, shard_sha256=sha(shard),
                  collector_sha256=sha(root / 'scripts/collect-legacy-bank.py'),
                  verifier_sha256=sha(Path(__file__)), exact_model_identity=True,
                  all_four_seat_trajectories=True, mixed_on_policy_rejected=True)
    write_new(below(root, args.output), report)
    print(json.dumps(report))


if __name__ == '__main__':
    main()
