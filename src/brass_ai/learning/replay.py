"""Whole-game replay shards, terminal targets and coverage-aware teacher labels."""
from __future__ import annotations

import gzip
import json
import math
from pathlib import Path

from brass_ai.runtime_contracts import validate
from .contracts import (assert_compatible, compatibility, decode_json, digest, finite_vector,
                        keys, require)
from .features import Encoder


def terminal_targets(result):
    validate('result', result)
    scores = [(s['vp'], s['income'], s['cash']) for s in result['seats']]
    best = max(scores)
    winners = scores.count(best)
    ranks = [1 + sum(other > score for other in scores) for score in scores]
    shares = [1 / winners if score == best else 0.0 for score in scores]
    require(ranks == [s['rank'] for s in result['seats']] and
            all(abs(s['utility'] - shares[i]) < 1e-8 for i, s in enumerate(result['seats'])),
            'Terminal ranks/win shares disagree with official tie breakers')
    return {'win_share': shares, 'placement': ranks, 'vp': [s['vp'] for s in result['seats']]}


def validate_teacher(teacher, count):
    if teacher is None:
        return
    keys(teacher, ('source', 'probabilities', 'coverage', 'confidence', 'samples', 'horizon'))
    require(teacher['source'] == 'fresh-finalist/1.0.0', 'Unsupported teacher target; visits are not probabilities')
    finite_vector(teacher['probabilities'], count, 0, 1)
    coverage = teacher['coverage']
    require(isinstance(coverage, list) and len(coverage) == count and
            all(type(x) is bool for x in coverage) and any(coverage), 'Invalid teacher coverage')
    require(abs(sum(teacher['probabilities']) - 1) < 1e-6, 'Teacher probabilities must sum to one')
    require(all(c or p == 0 for c, p in zip(coverage, teacher['probabilities'])), 'Unexamined choices cannot receive targets')
    finite_vector([teacher['confidence']], 1, 0, 1)
    require(type(teacher['samples']) is int and teacher['samples'] > 0, 'Missing confirmation samples')
    require(isinstance(teacher['horizon'], str) and bool(teacher['horizon']), 'Missing teacher horizon')


def split_for(group_id):
    """Rotations/repeats of a setup share a partition, independent of shard order."""
    bucket = int(digest({'setup_group': group_id})[:8], 16) % 10
    return 'train' if bucket < 8 else ('validation' if bucket == 8 else 'test')


def validate_game(game):
    keys(game, ('game_id', 'setup_group', 'players', 'split', 'opponents', 'result', 'records'))
    require(type(game['players']) is int and game['players'] == 4, 'Replay must have four seats')
    require(all(isinstance(game[k], str) and game[k] for k in ('game_id', 'setup_group')), 'Missing game identity')
    require(game['split'] == split_for(game['setup_group']), 'Setup split mismatch')
    require(isinstance(game['opponents'], list) and len(game['opponents']) == 4 and
            all(isinstance(x, str) and x for x in game['opponents']), 'Four frozen opponent versions required')
    terminal_targets(game['result'])
    require(isinstance(game['records'], list) and bool(game['records']), 'Empty game')
    encoder = Encoder()
    previous = -1
    for step, row in enumerate(game['records']):
        keys(row, ('step', 'seat', 'view', 'choices', 'chosen', 'behavior', 'style', 'intent', 'teacher', 'seconds'))
        require(type(row['step']) is int and row['step'] == step, 'Non-contiguous decision sequence')
        require(type(row['seat']) is int and 0 <= row['seat'] < 4 and row['seat'] == row['view']['seat'], 'Seat attribution mismatch')
        encoder.encode(row['view'], row['choices'], row['style'], row['intent'])
        require(type(row['chosen']) is int and 0 <= row['chosen'] < len(row['choices']), 'Chosen action is outside legal candidates')
        revision = row['view']['context']['revision']
        require(revision > previous, 'Repeated or reversed decision revision')
        previous = revision
        keys(row['behavior'], ('kind', 'policy_id', 'probability', 'temperature'))
        b = row['behavior']
        require(b['kind'] in ('model', 'random', 'external'), 'Unknown behavior source')
        require(b['policy_id'] == game['opponents'][row['seat']], 'Opponent changed within game')
        finite_vector([b['probability']], 1, 0, 1)
        require(b['probability'] > 0, 'Selected action had zero behavior probability')
        finite_vector([b['temperature']], 1, 0)
        finite_vector([row['seconds']], 1, 0)
        validate_teacher(row['teacher'], len(row['choices']))
    require({r['seat'] for r in game['records']} == set(range(4)), 'Missing seat trajectories')
    return game


def write_shard(path, games, provenance):
    """A shard contains only COMPLETE games, never a truncated terminal target."""
    path = Path(path)
    require(not path.exists(), 'Refusing to overwrite an existing replay shard')
    require(bool(games), 'Cannot publish an empty shard')
    require(len({g['game_id'] for g in games}) == len(games), 'Duplicate game IDs')
    for game in games:
        validate_game(game)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    try:
        with gzip.open(temporary, 'wt', encoding='utf-8') as output:
            output.write(json.dumps({'format': 'four-seat-replay/1.0.0',
                'compatibility': compatibility(), 'provenance': provenance}, allow_nan=False) + '\n')
            for game in games:
                output.write(json.dumps(game, separators=(',', ':'), allow_nan=False) + '\n')
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def read_shards(paths, partition=None):
    require(partition in (None, 'train', 'validation', 'test'), 'Unknown partition')
    games, identities = [], set()
    for path in paths:
        with gzip.open(path, 'rt', encoding='utf-8') as source:
            header = decode_json(next(source))
            keys(header, ('format', 'compatibility', 'provenance'))
            require(header['format'] == 'four-seat-replay/1.0.0', 'Unsupported replay format')
            assert_compatible(header['compatibility'])
            require(isinstance(header['provenance'], dict), 'Invalid replay provenance')
            require(partition != 'train' or header['provenance'].get('purpose') != 'evaluation',
                    'Evaluation-only opponent replay cannot be used for training')
            for line in source:
                game = validate_game(decode_json(line))
                require(game['game_id'] not in identities, 'Duplicate game across shards')
                identities.add(game['game_id'])
                if partition is None or game['split'] == partition:
                    games.append(game)
    require(bool(games), 'No complete games in the requested partition')
    return games


def examples(games):
    """Keep attribution; four alternating seats never become one trajectory."""
    return [(row, terminal_targets(game['result']), (game['game_id'], row['seat']))
            for game in games for row in game['records']]
