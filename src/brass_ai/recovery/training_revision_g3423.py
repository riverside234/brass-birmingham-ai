"""g3423 soft policy revision: step S1 of ai-research/SOFT_POLICY_TARGET_PLAN.md (doc:
ai-research/G3423_SOFT_POLICY_S1.md). Adopted once with --soft-policy at a committed boundary,
then kept.

A trainer-owned Linear(192 -> 1) without bias on the policy scorer's GELU layer predicts the
teacher term's rebuilt label target softened at T = 4, softmax((current logits + search bonus)
/ 4), on the same 128 long-label rows of each PPO step (no PPO rows: factor 0 until S4). Its
weight is 8 x the label term's own (8 beta). It starts as policy_scorer.2.weight / 4, is not
exported and never chooses a move. Its own gradient call and Adam (1e-4, clip 1.0); warm-up
2 head-only iterations, then the trunk multiplier .2, .4, .6, .8, 1.

The teacher recipe, labels, buffers, beta and its qualification are unchanged. The head and
its Adam live in the teacher phase's payload beside the term's beta state.

g3471: --soft-policy-weight W sets the weight to W x beta at a committed boundary (the user's
change; 8 -> 6 at g3471, 6 -> 12 at g3528). The head, its Adam and warm-up carry over. The block
records the latest change as weight_change = {generation, before}, with the change before it nested
as weight_change.previous, so the settings before any change can be rebuilt exactly. The head's
committed state, stamped under the earlier settings, is re-stamped at the change generation.
"""
import copy

from brass_ai.learning.contracts import keys, require

KEY = 'soft_policy'
CONTRACT = 'soft-policy-g3423/1.0.0'
SETTINGS = dict(
    contract=CONTRACT,
    step='S1',
    temperature=4.,
    weight=8.,
    weight_basis='label-term-beta',
    rows='teacher-long-labels',
    ppo_row_factor=0.,
    head=dict(input='policy_scorer.1', init='policy_scorer.2.weight/temperature', bias=False, exported=False),
    optimizer=dict(kind='adam', learning_rate=1e-4, weight_decay=0., gradient_clip=1.),
    warmup=dict(head_only_iterations=2, ramp_iterations=5),
)


WEIGHTS = (2., 4., 6., 8., 10., 12., 16.)  # x beta; 8 at adoption


def validate(block):
    keys(block, (*SETTINGS, 'from_generation'), ('weight_change',))
    plain = {k: v for k, v in block.items() if k not in ('from_generation', 'weight_change')}
    require(plain == dict(SETTINGS, weight=block['weight']) and block['weight'] in WEIGHTS
            and type(block['from_generation']) is int and block['from_generation'] >= 0, 'Invalid g3423 soft policy revision')
    # The change history, latest first: each change moves to a different reviewed weight, later than the one
    # before it, and the earliest starts from the adopted weight 8.
    weight, change, later = block['weight'], block.get('weight_change'), None
    while change is not None:
        keys(change, ('generation', 'before'), ('previous',))
        require(change['before'] in WEIGHTS and change['before'] != weight and type(change['generation']) is int
                and change['generation'] >= block['from_generation'] and (later is None or change['generation'] < later),
                'Invalid g3423 soft policy weight change')
        weight, later, change = change['before'], change['generation'], change.get('previous')
    require(weight == SETTINGS['weight'], 'Invalid g3423 soft policy weight change')
    return block


def previous_settings(block):
    """The block before its latest weight change (whose digest stamps the head's earlier committed state)."""
    change = block['weight_change']
    result = {k: copy.deepcopy(v) for k, v in block.items() if k != 'weight_change'}
    result['weight'] = change['before']
    if 'previous' in change: result['weight_change'] = copy.deepcopy(change['previous'])
    return result


def set_weight(config, weight, generation):
    """`config` with the soft head's weight at `weight` from committed generation `generation`."""
    require(applied(config), 'The soft policy weight changes an adopted soft head (--soft-policy first)')
    require(float(weight) in WEIGHTS and type(generation) is int,
            'The reviewed soft policy weights are ' + ', '.join(f'{w:g}' for w in WEIGHTS))
    block = validate(config[KEY])
    if block['weight'] == float(weight): return copy.deepcopy(config)
    earlier = block.get('weight_change')
    require(generation >= block['from_generation'] and (earlier is None or generation > earlier['generation']),
            'A soft policy weight change follows the adoption and the previous change')
    change = dict(generation=generation, before=block['weight'])
    if earlier is not None: change['previous'] = copy.deepcopy(earlier)
    result = copy.deepcopy(config)
    result[KEY] = validate(dict({k: copy.deepcopy(v) for k, v in block.items() if k != 'weight_change'},
                                weight=float(weight), weight_change=change))
    from .config import validate_config
    validate_config(result)
    return result


def applied(config): return KEY in (config or {})


def configure(config, generation):
    """`config` with the soft head adopted at committed generation `generation`; kept unchanged afterwards."""
    if applied(config):
        validate(config[KEY])
        return copy.deepcopy(config)
    from . import training_revision_g3270 as g3270
    require(g3270.applied(config) and config['teacher_phase'].get('mode') == 'on',
            'The g3423 soft policy head trains on the g3270 teacher revision\'s long-label rows, with teaching on')
    require(type(generation) is int and generation >= 0, 'Invalid adoption generation')
    result = copy.deepcopy(config)
    result[KEY] = validate(dict(copy.deepcopy(SETTINGS), from_generation=generation))
    from .config import validate_config
    validate_config(result)
    return result


def previous(config):
    """The recipe without the revision (recipe recognition)."""
    if not applied(config): return config
    result = copy.deepcopy(config); result.pop(KEY)
    return result


def project(old, new):
    """`old` with `new`'s revision when it is adopted or kept; None when it is removed or changed."""
    a, b = (old or {}).get(KEY), (new or {}).get(KEY)
    if a == b: return old
    if a is not None:
        # The one reviewed change of an adopted head: its weight, recorded as b's latest change from exactly a.
        try:
            validate(a); validate(b)
        except (ValueError, KeyError, TypeError):
            return None
        if 'weight_change' not in b or previous_settings(b) != a: return None
        result = copy.deepcopy(old); result[KEY] = copy.deepcopy(b)
        return result
    try:
        validate(b)
    except (ValueError, KeyError, TypeError):
        return None
    result = copy.deepcopy(old); result[KEY] = copy.deepcopy(b)
    return result


def settings(config):
    """What the trainer runs (soft_policy.SoftPolicy), or None before adoption."""
    return copy.deepcopy(validate(config[KEY])) if applied(config) else None


def tag(config):
    return f"soft-t4-w{config[KEY]['weight']:g}-s1" if applied(config) else 'none'


def summary(config):
    return copy.deepcopy(config.get(KEY))
