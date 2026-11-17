"""Strength-rating cadence after a committed generation (user request, g3170).

The rating block itself (config['rating'], copied into the rating state and every slot
spec) is unchanged: its 10-iteration grid still names every committed slot. This key only
skips due slots from `from_generation` on, so a PL rating runs every `every_iterations`
(20) committed iterations, counted from that generation: g3190, g3210, ... after g3170.
Slots up to and including `from_generation` keep the 10-iteration grid. The reference
setups keep rotating by one position per rating slot (ordinal), as before.

Only rating cost changes. Collection, training, the gate's slope units (PL points per
committed iteration) and every committed slot are untouched. Weights of generations that
are no longer rated are not kept as rating snapshots.
"""
import copy

from brass_ai.learning.contracts import keys, require

KEY = 'rating_cadence'
CONTRACT = 'rating-cadence/1.0.0'
CHOICES = (20,)


def validate(block, rating):
    keys(block, ('contract', 'from_generation', 'every_iterations'))
    base = rating['every_iterations']
    require(block['contract'] == CONTRACT and type(block['from_generation']) is int and block['from_generation'] > 0
            and block['from_generation'] % base == 0 and block['every_iterations'] in CHOICES
            and block['every_iterations'] % base == 0 and block['every_iterations'] > base,
            'Invalid rating cadence')
    return block


def applied(config): return KEY in config


def configure(config, every, generation):
    """`config` with ratings every `every` committed iterations after committed generation `generation`."""
    require(every in CHOICES, 'Use --rating-every ' + '/'.join(map(str, CHOICES)))
    result = copy.deepcopy(config)
    if KEY in result:
        require(result[KEY]['every_iterations'] == every, 'The rating cadence is already set; it does not change again')
        return result
    base = result['rating']['every_iterations']
    require(generation % base == 0, f'Change the rating cadence at a rated generation (a multiple of {base})')
    result[KEY] = validate(dict(contract=CONTRACT, from_generation=generation, every_iterations=every), result['rating'])
    return result


def due(config, generation):
    """Whether committed generation `generation` gets a rating slot."""
    base = config['rating']['every_iterations']
    if generation % base: return False
    block = config.get(KEY)
    if block is None or generation <= block['from_generation']: return True
    return (generation - block['from_generation']) % block['every_iterations'] == 0


def ordinal(config, generation):
    """The rating slot's position in the sequence of rated generations (reference-setup rotation)."""
    base = config['rating']['every_iterations']
    block = config.get(KEY)
    if block is None or generation <= block['from_generation']: return generation // base
    return block['from_generation'] // base + (generation - block['from_generation']) // block['every_iterations']


def previous(config):
    """The recipe without the cadence key (recipe recognition)."""
    if KEY not in config: return config
    result = copy.deepcopy(config); result.pop(KEY)
    return result


def project(old, new):
    """`old` with `new`'s cadence when it is adopted or kept; None when removed or changed."""
    a, b = old.get(KEY), new.get(KEY)
    if a == b: return old
    if a is not None: return None
    try:
        validate(b, new['rating'])
    except (ValueError, KeyError, TypeError):
        return None
    result = copy.deepcopy(old); result[KEY] = copy.deepcopy(b)
    return result


def summary(config):
    return copy.deepcopy(config.get(KEY))
