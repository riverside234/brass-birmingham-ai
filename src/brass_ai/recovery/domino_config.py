"""Versioned DOMiNO-based episodic PPO; no changes to inference features."""
import copy
import math
from brass_ai.learning.contracts import digest, require
from brass_ai.step5.policy import INDUSTRIES, context_for

CONTRACT = 'domino-episodic-ppo/1.0.0'
CONTEXT = 'ordinary-strategy-uniform/1.0.0'
ENCODING = 'industry-context-4-style-8-intent/1.0.0'
MIGRATION = 'domino-adoption/1.0.0'
NAMES = ('neutral',) + INDUSTRIES


def mapping():
    return [dict(id=i, name=n, style=context_for(n)[0], intent=context_for(n)[1]) for i,n in enumerate(NAMES)]


def defaults(mode='apply'):
    from .domino_features import manifest
    return dict(contract=CONTRACT, mode=mode, context_contract=CONTEXT, context_encoding=ENCODING,
        contexts=mapping(), strategy_count=7, reference_strategy=0,
        sampling='independent-uniform-per-neural-seat-game', historical_contexts='same-seven-contexts',
        descriptor_contract=manifest()['contract'], descriptor_manifest_sha256=digest(manifest()),
        objective='nearest-repulsive', quality_ratio=.95, statistics_ema_decay=.9,
        minimum_seat_games_per_strategy=32, minimum_distinct_games_per_strategy=16,
        dual_learning_rate=.01, dual_initial_logit=0., intrinsic_gamma=1., intrinsic_lambda=1.,
        intrinsic_value_weight=1., intrinsic_preservation_weight=1.,
        intrinsic_fit_scope='current-ordinary-learner-training-rows', branch_objective='existing-task-only',
        adoption=None)


def enabled(config): return config.get('diversity',{}).get('mode','off') != 'off'


def validate(c, recipe=None):
    require(type(c) is dict,'DOMiNO configuration must be an object')
    expected=defaults(c.get('mode'))
    require(type(c) is dict and set(c)==set(expected), 'Unknown/missing DOMiNO configuration field')
    require(c['mode'] in ('off','shadow','apply'), 'Unknown DOMiNO mode')
    for k,v in expected.items():
        if k in ('mode','adoption'): continue
        require(type(c[k]) is type(v) and c[k]==v, 'Unsupported DOMiNO contract setting: '+k)
        if isinstance(v,float): require(math.isfinite(c[k]), 'Nonfinite DOMiNO setting')
    require(digest(c['contexts'])==digest(mapping()), 'DOMiNO context mapping types/values differ')
    from .domino_features import validate_manifest
    validate_manifest()
    a=c['adoption']
    if a is not None:
        require(type(a) is dict and set(a)=={'generation','checkpoint_sha256','model_id'} and
            type(a['generation']) is int and a['generation']>0 and
            all(type(a[k]) is str and len(a[k])==64 and all(x in '0123456789abcdef' for x in a[k])
                for k in ('checkpoint_sha256','model_id')), 'Invalid DOMiNO adoption binding')
    if recipe is not None and enabled({'diversity':c}):
        from .config import quality_selfplay_recipe
        require(quality_selfplay_recipe(recipe) and all(k in recipe for k in ('critic_data','il_kl_limit','advantage','value_phase')),
                'DOMiNO requires explicit ordinary roles, PFSP and the fixed fingerprint probe')
        if c['mode']=='apply':
            from . import ppg_g210,dcpg_config
            require(recipe['value_phase'].get('enabled') is True and ppg_g210.active(recipe['value_phase']) and
                dcpg_config.enabled(recipe['update']), 'DOMiNO apply requires supported enabled PPG/DCPG')
    return c


def configure(config, state=None, mode='apply'):
    result=copy.deepcopy(config)
    if enabled(result):
        validate(result['diversity'],result)
        require(result['diversity']['mode']==mode, 'DOMiNO mode change needs a separate reviewed transition')
        return result
    result['diversity']=defaults(mode)
    if state is not None:
        e=state['history'][-1]
        result['diversity']['adoption']=dict(generation=state['iteration'],checkpoint_sha256=e['sha256'],model_id=e['model_id'])
    validate(result['diversity'],result)
    return result
