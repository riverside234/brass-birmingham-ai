"""Report-only, matched DOMiNO panel. Absence is deliberately opt-out."""
import copy
import math
from brass_ai.learning.contracts import require

CONTRACT = 'domino-context-evaluation/1.0.0'
MIGRATION = 'domino-context-evaluation-adoption/1.0.0'


def defaults(mode='report'):
    return dict(contract=CONTRACT, mode=mode, every_iterations=50, setups=16,
        seat_rotations=4, reuse_main_neutral=True, bootstrap_samples=2000,
        permutation_samples=1000, overhead_review_fraction=.05, raw_replay_panels=2)


def enabled(config):
    return config.get('context_evaluation', {}).get('mode', 'off') == 'report'


def validate(c, recipe):
    require(type(c) is dict and set(c)==set(defaults()), 'Invalid context_evaluation fields')
    require(c['mode'] in ('off','report') and type(c['mode']) is str, 'Invalid context evaluation mode')
    for key, value in defaults(c['mode']).items():
        require(type(c[key]) is type(value), 'Invalid context evaluation type: '+key)
        if isinstance(value,float):require(math.isfinite(c[key]), 'Nonfinite context evaluation setting')
        if key=='every_iterations':
            require(c[key] in (50,100), 'Context cadence must be 50 or explicitly reviewed 100')
        else:require(c[key]==value, 'Unsupported context evaluation contract setting: '+key)
    if c['mode']=='report':
        from .domino_config import validate as validate_diversity, enabled as diverse
        require(diverse(recipe), 'Context evaluation requires the seven-context DOMiNO interface')
        validate_diversity(recipe['diversity'])
        r=recipe.get('rating',{})
        require(r.get('enabled') is True and type(r.get('every_iterations')) is int and
            r['every_iterations']>0 and c['every_iterations']%r['every_iterations']==0 and
            r.get('setups',0)>=16 and r.get('seat_rotations')==4,
            'Context evaluation requires sufficient main setups and compatible main cadence')
        require(recipe.get('v4_search_enabled') is False, 'Context evaluation requires search-free v4')
    return c


def configure(config, mode='report', every_iterations=50):
    result=copy.deepcopy(config)
    result['context_evaluation']=dict(defaults(mode),every_iterations=every_iterations)
    validate(result['context_evaluation'],result)
    return result


def compatible(before, after):
    """No recipe edits can be smuggled into an evaluation-only transition."""
    without=lambda c:{k:v for k,v in c.items() if k!='context_evaluation'}
    if without(before)!=without(after):return False
    if 'context_evaluation' in before:validate(before['context_evaluation'],before)
    if 'context_evaluation' in after:validate(after['context_evaluation'],after)
    return True
