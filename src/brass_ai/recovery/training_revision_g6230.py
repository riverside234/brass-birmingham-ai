"""g6230 soft-head revision (the user's request of 2026-11-26; doc ai-research/G6230_SOFT_HEAD_REVISION.md).
Adopted once with --soft-head-revision at a committed boundary, on top of the g5685 clip revision, then kept.

Soft target  each teacher label row's soft target is fixed: softmax((old logits + search bonus) / T) over its legal
             actions with visits > 0 only, renormalized there (zero elsewhere), where the old logits are the policy
             logits stored at collection (label['logits']) and the bonus is the teacher term's own
             (teacher_term.search_bonus at the g3270 label scale). It no longer follows the current policy.
Soft output  the soft head's output is masked the same way: its log-softmax runs over the visited legal actions only.
Weight 0     a row with at most one visited legal action (a forced move, a one-candidate search) has soft loss
             weight 0 and its loss is not computed. The step's soft loss stays the sum of the other rows' KL over
             the step's rows, so the other rows keep their weight; a step without any is skipped by the head.
Held-out     the soft head's held-out KL is measured once per iteration, after PPO and before PPG, against targets
             built the same way for each validation-table long label (rows with at most one visited action left
             out). The checks before PPO and after PPG are dropped: over the 128 iterations in the g6230 export's
             history PPG moved it by a median 0.00016 (0 in 49), against 0.0013 for PPO.
Soft rate    the soft head's own Adam steps at a fixed 1e-4 for the first 100 iterations from adoption, then at the
             PPO step rate again (g5402's link).
PPG clip     the PPG value phase clips at 2.5 (was 2.4 from g5685), with its Adam moments rescaled once on the
             adoption generation's checkpoint, from g5466's normal fit of the last 30 committed iterations' logged
             PPG mean and P98 (receipt PPG_GROUP_KEY in the PPG Adam group).
Unchanged    the soft head's temperature (4), weight (8 x beta), clip (1.0), warm-up and its committed head and Adam
             state; the main policy's teacher targets (rebuilt from the current policy); PPO's clip (6.5) and rates.
"""
import copy

from brass_ai.learning.contracts import keys, require
from . import training_revision_g5466 as g5466
from . import training_revision_g5685 as g5685

KEY = 'soft_head_revision'
CONTRACT = 'soft-head-revision-g6230/1.0.0'
PPG_GRAD_CLIP = 2.5
BASE_PPG_GRAD_CLIP = 2.4  # g5685
SOFT_RATE = 1e-4
SOFT_RATE_ITERATIONS = 100
MIN_VISITED = 2  # visited legal actions a row needs for a soft loss
WINDOW = 30
SEARCH = 40
CLIP_RATE_TOLERANCE = 10.
SETTINGS = dict(
    contract=CONTRACT,
    soft_target=dict(kind='fixed', logits='collection-time (label logits)', bonus='teacher-term search bonus',
                     support='legal actions with visits > 0', normalized='over the support'),
    soft_output='log-softmax over the same support',
    soft_min_visited=MIN_VISITED,
    soft_heldout='after-ppo-only, fixed targets',
    soft_learning_rate=dict(rate=SOFT_RATE, iterations=SOFT_RATE_ITERATIONS, then='ppo-step-rate'),
    ppg_grad_clip=PPG_GRAD_CLIP,
    window_iterations=WINDOW,
)
VARIABLE = ('from_generation', 'ppg_adam_rescale')
PPG_GROUP_KEY = 'brass_ppg_adam_rescale_g6230'


def committed_ppg_norms(run, state, window=WINDOW):
    """The PPG pre-clip norm summaries (mean and P98) of the latest `window` committed iterations with PPG steps,
    among the latest SEARCH (g5685.committed_rows); every one must have clipped at 2.4."""
    found = {}
    for iteration, row in g5685.committed_rows(run, state, SEARCH).items():
        record = g5466._record(iteration, (row.get('value_phase') or {}).get('gradient_clipping'))
        if record is not None: found[iteration] = record
    latest = sorted(found, reverse=True)[:window]
    require(len(latest) == window, f'The g6230 PPG Adam rescale needs {window} committed iterations of PPG norm '
            f'statistics among the latest {SEARCH}; found {len(latest)}')
    require(all(found[i][1] == BASE_PPG_GRAD_CLIP for i in latest),
            'The g6230 PPG Adam rescale reads iterations that clipped PPG at 2.4')
    return [found[i][0] for i in sorted(latest)]


def ppg_adam_rescale(norms):
    """g5466's PPG fit (mean and P98 per iteration), from the 2.4 clip to 2.5."""
    return g5466.adam_rescale(norms, old_clip=BASE_PPG_GRAD_CLIP, new_clip=PPG_GRAD_CLIP)


def validate(block):
    keys(block, (*SETTINGS, *VARIABLE))
    require({k: v for k, v in block.items() if k not in VARIABLE} == SETTINGS
            and type(block['from_generation']) is int and block['from_generation'] >= 0, 'Invalid g6230 soft-head revision')
    rescale = block['ppg_adam_rescale']
    require(type(rescale) is dict and type(rescale.get('norms')) is list and len(rescale['norms']) == WINDOW
            and rescale == ppg_adam_rescale(rescale['norms']) and rescale['first_moment'] > 1 and rescale['second_moment'] > 1
            and abs(rescale['predicted_clipped_percent'] - rescale['logged_clipped_percent']) <= CLIP_RATE_TOLERANCE,
            'Invalid g6230 PPG Adam rescale')
    return block


def applied(config): return KEY in (config or {})


def configure(config, generation, *, gradient_norms):
    """`config` with the revision adopted at committed generation `generation`, where the latest committed PPG phases
    logged `gradient_norms` (committed_ppg_norms); kept unchanged afterwards."""
    if applied(config):
        validate(config[KEY])
        return copy.deepcopy(config)
    from . import training_revision_g3423 as g3423
    from .training_revision_g5402 import soft_follows_ppo
    require(g5685.applied(config) and g5685.ppg_grad_clip(config) == BASE_PPG_GRAD_CLIP and g3423.applied(config)
            and soft_follows_ppo(config),
            'The g6230 soft-head revision follows the g5685 clip revision (PPG clip 2.4) with the soft policy head on '
            'the PPO rate (g3423, g5402)')
    require(type(generation) is int and generation >= 0, 'Invalid adoption point')
    require(all(config[g5685.KEY]['from_generation'] <= r['iteration'] < generation for r in gradient_norms),
            'The g6230 PPG Adam rescale reads committed iterations trained at PPG clip 2.4')
    block = dict(copy.deepcopy(SETTINGS), from_generation=generation, ppg_adam_rescale=ppg_adam_rescale(gradient_norms))
    result = copy.deepcopy(config); result[KEY] = validate(block)
    from .config import validate_config
    validate_config(result)
    return result


def previous(config):
    """The recipe without the revision (recipe recognition); the block holds all of it."""
    if not applied(config): return config
    result = copy.deepcopy(config); result.pop(KEY)
    return result


def project(old, new):
    """`old` with `new`'s revision when it is adopted or kept; None when it is removed or changed."""
    a, b = (old or {}).get(KEY), (new or {}).get(KEY)
    if a == b: return old
    if a is not None: return None
    try:
        validate(b)
        if g5685.ppg_grad_clip(old) != BASE_PPG_GRAD_CLIP: return None
    except (ValueError, KeyError, TypeError):
        return None
    result = copy.deepcopy(old); result[KEY] = copy.deepcopy(b)
    return result


# ---- settings read by the training code --------------------------------------------------------

def ppg_grad_clip(config):
    """The PPG value phase's own clip: 2.5 with the revision, else g5685's (2.4, ...)."""
    if applied(config): return config[KEY]['ppg_grad_clip']
    return g5685.ppg_grad_clip(config)


def fixed_soft_targets(config):
    """True when the soft head trains and is measured on the fixed, visited-only targets."""
    return applied(config)


def soft_rate(config, generation):
    """The soft head's own fixed rate at `generation`: 1e-4 for SOFT_RATE_ITERATIONS iterations from adoption, else
    None (it follows the PPO step rate, or its settings' rate without the g5402 link)."""
    if not applied(config): return None
    start = config[KEY]['from_generation']
    return SOFT_RATE if start <= generation < start + SOFT_RATE_ITERATIONS else None


def soft_rate_until(config):
    """The first generation back on the PPO step rate."""
    return config[KEY]['from_generation'] + SOFT_RATE_ITERATIONS


def rescale_ppg(optimizer, config, generation):
    """Once, on the committed checkpoint of the adoption generation: every PPG Adam exp_avg and exp_avg_sq multiplied
    by the recorded factors (2.4 -> 2.5), the receipt kept in the Adam group. Returns the receipt, or None."""
    if not applied(config): return None
    return g5685._rescale(optimizer, config[KEY], generation, name='PPG', group_key=PPG_GROUP_KEY,
                          rescale_key='ppg_adam_rescale', contract=CONTRACT, revision='g6230')


def tag(config):
    return 'softfixed-lr1e4x100-ppgclip25' if applied(config) else 'none'


def summary(config):
    return copy.deepcopy(config.get(KEY))
