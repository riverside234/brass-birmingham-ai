"""Versioned learning contracts, separate from immutable R0/R1 contracts."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path, PurePosixPath
from .paths import resolve_project_path

LAB_ROOT = Path('/data/not_backed_up/yxu209/ai-training/')
SOURCE_ROOT = Path(__file__).resolve().parents[3]
CONFIG_DIR = SOURCE_ROOT / 'configs/learning'


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def decode_json(text):
    def object_pairs(pairs):
        # Most replay objects have unique keys. Build them in C, retaining the
        # strict duplicate-key error (including the first duplicate) on failure.
        result = dict(pairs)
        if len(result) != len(pairs):
            seen = set()
            for key, _ in pairs:
                require(key not in seen, f'Duplicate JSON key: {key}')
                seen.add(key)
        return result
    def constant(value):
        raise ValueError(f'Non-finite JSON constant: {value}')
    return json.loads(text, object_pairs_hook=object_pairs, parse_constant=constant)


def read_json(path):
    return decode_json(Path(path).read_text(encoding='utf-8-sig'))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def keys(value, required, optional=()):
    require(isinstance(value, dict), 'Expected object')
    require(set(required) <= set(value) <= set(required) | set(optional),
            f'Unexpected/missing fields: {set(value) ^ set(required)}')


def finite_vector(value, length, low=None, high=None):
    require(isinstance(value, (list, tuple)) and len(value) == length, 'Invalid vector shape')
    require(all(type(x) in (int, float) and math.isfinite(x) and
                (low is None or x >= low) and (high is None or x <= high) for x in value),
            'Invalid vector values')


def project_path(root, relative, *, base=None):
    """All generated paths are under the declared project root; no cwd default."""
    return resolve_project_path(root, relative, base=base)


def compatibility():
    runtime = read_json(SOURCE_ROOT / 'configs/contracts/v0.2.0/manifest.json')
    wire = SOURCE_ROOT / 'configs/contracts/v0.2.0/wire.schema.json'
    require(hashlib.sha256(wire.read_bytes()).hexdigest() == runtime['contract_sha256'],
            'R1 contract fingerprint mismatch')
    schema = read_json(CONFIG_DIR / 'features-1.0.0.json')
    return {
        'learning_contract': 'four-seat-learning/1.0.0', 'players': 4,
        'rules': runtime['rules_profile_id'], 'wire_sha256': runtime['contract_sha256'],
        'observation': runtime['observation_schema'], 'choices': runtime['action_schema'],
        'objective': 'win-share/1.0.0', 'placement': 'competition-rank/1.0.0',
        'seat_order': 'stable-0-1-2-3', 'vp_scale': 100.0,
        'features_sha256': digest(schema),
        'producer_profile_sha256': digest(read_json(CONFIG_DIR / 'host-1.0.0.json')),
        'encoder_sha256': hashlib.sha256((Path(__file__).parent / 'features.py').read_text().replace('\r\n', '\n').encode()).hexdigest(),
        'architecture_sha256': hashlib.sha256((Path(__file__).parent / 'model.py').read_text().replace('\r\n', '\n').encode()).hexdigest(),
        'loss': 'win-ce+0.25-rank-huber+0.1-vp-huber/1.0.0',
        'reference_lock_sha256': digest(read_json(SOURCE_ROOT / 'configs/references/upstream.lock.json')),
        'heads': ['policy_logits', 'win_share_4', 'placement_4', 'vp_4'],
    }


def assert_compatible(actual):
    require(actual == compatibility(), 'Incompatible rules/features/four-seat targets/reference pins')


def assert_heuristic_data_compatible(actual):
    """A model-source change does not alter audited heuristic state/action data.

    Every rules, feature, encoder, target and reference field must still match.
    Callers must separately reject neural behavior and search teacher labels.
    Checkpoints/NN replay continue using exact assert_compatible.
    """
    expected=compatibility();field='architecture_sha256'
    require(type(actual) is dict and set(actual)==set(expected),'Invalid heuristic data compatibility fields')
    token=actual[field]
    require(type(token) is str and len(token)==64 and all(c in '0123456789abcdef' for c in token),'Invalid historical model-source digest')
    require({k:v for k,v in actual.items() if k!=field}=={k:v for k,v in expected.items() if k!=field},
            'Incompatible heuristic rules/features/encoder/four-seat targets')


def load_config(path=None):
    config = read_json(path or CONFIG_DIR / 'lab.json')
    keys(config, ('players', 'project_root', 'host', 'output', 'model_size', 'device',
                  'batch_size', 'learning_rate', 'seed', 'max_decisions', 'threads'))
    require(type(config['players']) is int and config['players'] == 4, 'Only four-player runs are supported')
    require(config['model_size'] in ('small', 'medium', 'large'), 'Unknown model size')
    require(config['device'] in ('cpu', 'cuda', 'auto'), 'Unsupported training device')
    for key in ('batch_size', 'max_decisions', 'threads'):
        require(type(config[key]) is int and config[key] > 0, f'Invalid {key}')
    require(type(config['seed']) is int and config['seed'] >= 0, 'Invalid seed')
    require(type(config['learning_rate']) in (float, int) and
            0 < config['learning_rate'] <= 1, 'Invalid learning rate')
    require(Path(config['project_root']).is_absolute() or PurePosixPath(config['project_root']).is_absolute(), 'Project root must be absolute')
    return config
