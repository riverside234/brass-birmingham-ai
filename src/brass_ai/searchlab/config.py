"""Execution presets for the independent search comparison/teacher experiments."""
import math

from brass_ai.learning.contracts import require
from brass_ai.step5.search import SearchConfig
from .s3 import TreeConfig


BASE_FIELDS = {'format', 'players', 'default_variants', 'seconds', 'validation_reserve_fraction',
    'simulations', 'future_turns', 'proposals', 'finalists', 'pilot_samples', 'confirmation_samples',
    'promotion_min_setups', 'default_partition', 'threads'}
SMALL_EXECUTION = dict(setups=1, workers=1, distill_games=2, distill_workers=1,
    batch_size=32, minibatch_size=32, max_candidate_cells=8192,
    distill_steps=4, distill_weight=.25, learning_rate=1e-4)


def validate_config(config):
    config = dict(config)
    version = config.get('format')
    require(version in ('searchlab-experiment/1.0.0', 'searchlab-experiment/1.1.0','searchlab-experiment/1.2.0'),
            'Unsupported search experiment config')
    expected = BASE_FIELDS if version.endswith('/1.0.0') else BASE_FIELDS | set(SMALL_EXECUTION)
    if version.endswith('/1.2.0'):
        expected |= {'execution','update'}
        from brass_ai.recovery.config import execution_config,update_config
        execution_config(config['execution']);update_config(config['update'])
        require(config['batch_size']==config['execution']['batch_size'] and
            config['max_candidate_cells']==config['execution']['max_candidate_cells'],
            'Inference aliases differ from the central execution config')
        require(config['minibatch_size']==config['update']['microbatch_size'] and
            config['max_candidate_cells']==config['update']['max_candidate_cells'],
            'Distillation memory aliases differ from the update config')
    require(set(config) == expected, 'Unknown or missing search experiment config field')
    if version.endswith('/1.0.0'):
        config.update(SMALL_EXECUTION, format='searchlab-experiment/1.1.0')
    require(type(config['players']) is int and config['players'] == 4, 'Four-player experiment required')
    for field in ('threads', 'setups', 'workers', 'distill_games', 'distill_workers', 'batch_size',
                  'minibatch_size', 'max_candidate_cells', 'distill_steps', 'promotion_min_setups',
                  'simulations', 'future_turns', 'proposals', 'finalists', 'pilot_samples', 'confirmation_samples'):
        require(type(config[field]) is int and config[field] > 0, 'Invalid search experiment ' + field)
    for field in ('seconds', 'validation_reserve_fraction', 'distill_weight', 'learning_rate'):
        require(type(config[field]) in (int, float) and math.isfinite(config[field]) and config[field] > 0,
                'Invalid search experiment ' + field)
    require(config['validation_reserve_fraction'] < .5 and config['distill_weight'] <= 1,
            'Invalid time reserve or distillation weight')
    require(config['promotion_min_setups'] >= 30, 'Weakened promotion threshold')
    require(config['default_partition'] in ('validation', 'test'), 'Comparison must use held-out setups')
    variants = config['default_variants']
    require(isinstance(variants, list) and 's2' in variants and len(variants) == len(set(variants))
            and set(variants) <= {'s2', 'r4', 's3'}, 'Include unique variants and the S2 control')
    TreeConfig(simulations=config['simulations'], future_turns=config['future_turns']).validate()
    SearchConfig(proposals=config['proposals'], finalists=config['finalists'], pilot_samples=config['pilot_samples'],
        confirmation_samples=config['confirmation_samples'], seconds=config['seconds']).validate()
    return config
