"""g5402 optimizer revision (the user's request of 2026-11-24; doc ai-research/G5402_OPTIMIZER_REVISION.md).
Adopted once with --optimizer-revision at a committed boundary, then kept.

grad clip     PPO clips its total gradient norm at 5.83 (update.grad_clip; it was 1.0, which every PPO step hit:
              pre-clip norms ran 4.5-5 with a P95 near 5.9). PPG has a clip of its own, kept at 1.0
              (ppg_grad_clip; value_phase.fit).
PPO LR        at adoption the PPO rate in force (base x controller multiplier) is cut by 20%, and that rate becomes
              the base, multiplier 1.0: 6e-6 -> 4.8e-6. The cut is one controller cut (1/1.25), so as after any
              cut the controller's three-iteration KL window restarts and its three-iteration cooldown runs. The
              g1307 floor stays 4e-6 (min multiplier 4e-6 / 4.8e-6). PPG's peak follows PPO's applied rate (x .65)
              as before, so it is 20% lower too.
Adam rescale  the PPO Adam's moments were built on gradients clipped to norm 1.0 at every step; unclipped they are
              about 4.7x larger, and the second moment (beta2 .999) would take ~1000 steps to catch up, so the first
              iterations would step ~2.7x today's size. Once, with the rebase, exp_avg is multiplied by
              E[min(n, 5.83)] and exp_avg_sq by E[min(n, 5.83)^2] (n: the pre-clip norm), from a normal fit of the
              committed iterations' norm statistics (adam_rescale). Each old step was scaled to norm 1 and would now
              be scaled to min(n, 5.83), so the moments are what they would have been under the new clip, and the
              step is 0.8x today's from the first one. Steps, the warm-up counter, the VP health window and the data
              signature are kept (resume.apply_optimizer_config -> rebase_ppo).
soft head     the soft policy head steps at the PPO LR of the same PPO step (learning_rate.upcoming), instead of its
              own 1e-4; its clip, Adam moments, warm-up and settings are unchanged.
era heads     the era heads (Canal, Rail and Rail-end ownership + VP) step at the PPG LR of the same PPG step: PPG's
              peak x its linear warm-up / cosine decay factor (was 1e-4 x that factor), the head-only steps after
              the trunk stops included. Their PPO-phase preservation steps take the PPO LR of the same PPO step.

The PPG value phase also reports the pre-clip gradient norms of all its attempted optimizer steps
(value_phase.clipping_report); that needs no adoption.
"""
import copy
import math

from brass_ai.learning.contracts import keys, require

KEY = 'optimizer_revision'
CONTRACT = 'optimizer-revision-g5402/1.0.0'
PPO_GRAD_CLIP = 5.83
PPG_GRAD_CLIP = 1.
BASE_GRAD_CLIP = 1.
LR_CUT = .8
SETTINGS = dict(
    contract=CONTRACT,
    ppo_grad_clip=PPO_GRAD_CLIP,
    ppg_grad_clip=PPG_GRAD_CLIP,
    ppo_learning_rate=dict(cut=LR_CUT, rebase='cut-rate-is-the-base-at-multiplier-1',
                           controller='kl-window-restarts-cooldown-runs', floor='g1307-4e-6-kept'),
    soft_policy_learning_rate='ppo-step-rate',
    era_heads_learning_rate=dict(ppg='ppg-step-rate', ppo='ppo-step-rate'),
)
BEFORE = ('learning_rate', 'multiplier', 'grad_clip', 'min_multiplier')
AFTER = ('learning_rate', 'min_multiplier')
NORMS = ('iteration', 'count', 'mean', 'std')
RESCALE_MODEL = 'normal-fit-of-committed-pre-clip-norms/1.0.0'
VARIABLE = ('from_generation', 'before', 'after', 'adam_rescale')


def _positive(value):
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def _digits(value): return float(f'{value:.12g}')


def cut_rate(learning_rate, multiplier):
    """80% of the rate in force, to 12 significant digits (6e-6 -> 4.8e-6, not 4.800000000000001e-6)."""
    return _digits(learning_rate * multiplier * LR_CUT)


def committed_gradient_norms(state):
    """The PPO pre-clip norm statistics of the committed iterations kept in state.recent_training. The rescale
    assumes each of their steps was clipped to 1.0, which holds when every step's norm was above it."""
    rows = state.get('recent_training') or []
    require(rows and rows[-1]['iteration'] + 1 == state['iteration'], 'The g5402 Adam rescale needs the committed '
            "iterations' training summaries (state.recent_training)")
    result = []
    for row in rows:
        actor = row['actor']; norms = actor['gradient_norms']
        require(norms['count'] == actor['optimizer_steps'] == actor['gradient_clip_steps'] > 0
                and norms['min'] > BASE_GRAD_CLIP, 'The g5402 Adam rescale needs PPO updates whose every step clipped at 1.0')
        result.append(dict(iteration=row['iteration'], count=norms['count'], mean=norms['mean'], std=norms['std']))
    return result


def adam_rescale(norms, old_clip=BASE_GRAD_CLIP, new_clip=PPO_GRAD_CLIP):
    """The factors that move the PPO Adam moments from gradients clipped at `old_clip` (every step) to the clip
    `new_clip`: E[min(n, c)] / old_clip for exp_avg and E[min(n, c)^2] / old_clip^2 for exp_avg_sq, n ~ a normal fit
    (pooled mean and variance) of the pre-clip norms."""
    require(norms and all(set(r) == set(NORMS) and type(r['count']) is int and r['count'] > 0
                          and _positive(r['mean']) and type(r['std']) in (int, float) and r['std'] >= 0 for r in norms),
            'Invalid committed PPO gradient norms')
    steps = sum(r['count'] for r in norms)
    mean = sum(r['count'] * r['mean'] for r in norms) / steps
    second = sum(r['count'] * (r['std'] ** 2 + r['mean'] ** 2) for r in norms) / steps
    std = math.sqrt(max(second - mean ** 2, 0.)); c = new_clip
    if std > 0:
        z = (c - mean) / std
        below = .5 * (1 + math.erf(z / math.sqrt(2))); density = math.exp(-z * z / 2) / math.sqrt(2 * math.pi)
        first = mean * below - std * density + c * (1 - below)  # E[min(X, c)]
        squared = (mean ** 2 + std ** 2) * below - std * (mean + c) * density + c * c * (1 - below)  # E[min(X, c)^2]
    else:
        first = min(mean, c); squared = first * first
    return dict(model=RESCALE_MODEL, old_clip=old_clip, new_clip=new_clip, norms=copy.deepcopy(norms), steps=steps,
                norm_mean=_digits(mean), norm_std=_digits(std),
                first_moment=_digits(first / old_clip), second_moment=_digits(squared / old_clip ** 2))


def validate(block):
    from .training_revision_g1307 import LR_FLOOR
    keys(block, (*SETTINGS, *VARIABLE))
    require({k: v for k, v in block.items() if k not in VARIABLE} == SETTINGS
            and type(block['from_generation']) is int and block['from_generation'] >= 0,
            'Invalid g5402 optimizer revision')
    rescale = block['adam_rescale']
    require(type(rescale) is dict and type(rescale.get('norms')) is list
            and rescale == adam_rescale(rescale['norms']) and rescale['first_moment'] > 1 and rescale['second_moment'] > 1,
            'Invalid g5402 Adam rescale')
    before, after = block['before'], block['after']
    keys(before, BEFORE); keys(after, AFTER)
    require(all(_positive(before[k]) for k in BEFORE) and all(_positive(after[k]) for k in AFTER), 'Invalid g5402 rates')
    require(before['grad_clip'] == BASE_GRAD_CLIP
            and after['learning_rate'] == cut_rate(before['learning_rate'], before['multiplier'])
            and after['learning_rate'] >= LR_FLOOR and after['min_multiplier'] == LR_FLOOR / after['learning_rate'],
            'The g5402 PPO rate is 80% of the committed rate, above the 4e-6 floor')
    return block


def applied(config): return KEY in (config or {})


def committed_multiplier(state):
    """The PPO LR controller multiplier in force at the committed generation (its last update's decision)."""
    rows = state.get('recent_training') or []
    require(rows and rows[-1]['iteration'] + 1 == state['iteration'], 'The g5402 revision needs the committed '
            "iteration's training summary (state.recent_training) to read the PPO LR in force")
    row = rows[-1]; controller = row['actor']['lr_controller']; schedule = row['actor']['learning_rate_schedule']
    require(controller['mode'] == 'apply' and controller['warmup_done'] and schedule['completed_steps'] >= schedule['steps']
            and schedule['base_lr'] == state['signature']['config']['learning_rate'] and _positive(controller['multiplier_after']),
            'The g5402 revision is adopted after the PPO warm-up, with the controller applied')
    return float(controller['multiplier_after'])


def configure(config, generation, *, multiplier, gradient_norms):
    """`config` with the revision adopted at committed generation `generation`, where the PPO controller holds
    `multiplier` (committed_multiplier) and the committed updates had `gradient_norms` (committed_gradient_norms);
    kept unchanged afterwards."""
    if applied(config):
        validate(config[KEY])
        return copy.deepcopy(config)
    from . import training_revision_g1307 as lr_floor
    from .lr_controller import validate_config as validate_controller
    from .ppg_g210 import active as ppg_g210
    controller = config.get('ppo_lr_controller')
    require(controller is not None and controller['mode'] == 'apply' and 'learning_rate_warmup' in config
            and lr_floor.applied(config) and config['update']['grad_clip'] == BASE_GRAD_CLIP
            and ppg_g210(config.get('value_phase', {})),
            'The g5402 optimizer revision follows the applied PPO LR controller with the 4e-6 floor, PPO clip 1.0 '
            'and the g210 PPG value phase (whose clip it keeps at 1.0)')
    require(type(generation) is int and generation >= 0 and _positive(multiplier)
            and controller['min_multiplier'] <= multiplier <= controller['max_multiplier'], 'Invalid adoption point')
    rate = cut_rate(config['learning_rate'], multiplier)
    block = dict(copy.deepcopy(SETTINGS), from_generation=generation,
                 before=dict(learning_rate=config['learning_rate'], multiplier=multiplier,
                             grad_clip=config['update']['grad_clip'], min_multiplier=controller['min_multiplier']),
                 after=dict(learning_rate=rate, min_multiplier=lr_floor.LR_FLOOR / rate),
                 adam_rescale=adam_rescale(gradient_norms))
    result = _adopt(config, validate(block))
    validate_controller(result['ppo_lr_controller'], result['update'])
    from .config import validate_config
    validate_config(result)
    return result


def _adopt(config, block):
    result = copy.deepcopy(config)
    result['update']['grad_clip'] = block['ppo_grad_clip']
    result['learning_rate'] = block['after']['learning_rate']
    result['ppo_lr_controller']['min_multiplier'] = block['after']['min_multiplier']
    result[KEY] = copy.deepcopy(block)
    return result


def previous(config):
    """The recipe without the revision (recipe recognition): the committed clip, base LR and floor multiplier."""
    if not applied(config): return config
    result = copy.deepcopy(config); block = result.pop(KEY); before = block['before']
    result['update']['grad_clip'] = before['grad_clip']
    result['learning_rate'] = before['learning_rate']
    result['ppo_lr_controller']['min_multiplier'] = before['min_multiplier']
    return result


def project(old, new):
    """`old` with `new`'s revision when it is adopted or kept; None when it is removed or changed."""
    a, b = (old or {}).get(KEY), (new or {}).get(KEY)
    if a == b: return old
    if a is not None: return None
    try:
        validate(b)
        before = b['before']
        if (old['learning_rate'], old['update']['grad_clip'], old['ppo_lr_controller']['min_multiplier']) != (
                before['learning_rate'], before['grad_clip'], before['min_multiplier']):
            return None
    except (ValueError, KeyError, TypeError):
        return None
    return _adopt(old, b)


# ---- settings read by the training code --------------------------------------------------------

def ppg_grad_clip(config):
    """The PPG value phase's own clip, or None (it clips at update.grad_clip, as PPO does)."""
    return config[KEY]['ppg_grad_clip'] if applied(config) else None


def soft_follows_ppo(config):
    """True when the soft policy head steps at the PPO LR of the same PPO step."""
    return applied(config)


def era_heads_follow(config):
    """True when the era heads step at the main network's LR of the same step (PPG in PPG, PPO in PPO)."""
    return applied(config)


def rebase_ppo(optimizer, config, generation):
    """Once, on the committed checkpoint of the adoption generation: the PPO Adam group's warm-up base becomes the
    cut rate and its controller multiplier 1.0, with the KL window restarted and the cooldown running, and every
    parameter's exp_avg and exp_avg_sq are multiplied by the recorded adam_rescale factors. Steps, the warm-up
    counter, the VP health window and the data signature are kept. A crashed attempt at that generation reloads the
    same checkpoint and rebases it the same way; a later checkpoint is already rebased. Returns the receipt, or None."""
    if not applied(config): return None
    from .learning_rate import KEY as WARMUP_KEY
    from .lr_controller import KEY as CONTROLLER_KEY
    block = config[KEY]; before, after = block['before'], block['after']
    require(len(optimizer.param_groups) == 1, 'The g5402 rebase expects one PPO Adam group')
    group = optimizer.param_groups[0]; warmup = group[WARMUP_KEY]; controller = group[CONTROLLER_KEY]
    if warmup['base_lr'] == after['learning_rate']: return None
    require(generation == block['from_generation'] and warmup['base_lr'] == before['learning_rate']
            and controller['multiplier'] == before['multiplier'] and warmup['completed_steps'] >= warmup['steps'],
            'The PPO Adam rebase runs once, on the checkpoint of the g5402 adoption generation')
    receipt = dict(contract=CONTRACT, generation=generation,
                   before=dict(base_lr=warmup['base_lr'], multiplier=controller['multiplier'],
                               rate=warmup['base_lr'] * controller['multiplier'], cooldown=controller['cooldown'],
                               window=copy.deepcopy(controller['window'])),
                   after=dict(base_lr=after['learning_rate'], multiplier=1., rate=after['learning_rate'],
                              cooldown=controller['config']['cooldown_iterations'], window=[]),
                   kept='Adam steps, the warm-up counter, the VP health window and the data signature')
    rescale = block['adam_rescale']; first, second = rescale['first_moment'], rescale['second_moment']
    states = [optimizer.state[p] for p in group['params'] if p in optimizer.state]
    require(states and all('exp_avg' in s and 'exp_avg_sq' in s and 'max_exp_avg_sq' not in s for s in states),
            'The g5402 Adam rescale needs the trained PPO Adam moments')
    for s in states:
        s['exp_avg'].mul_(first); s['exp_avg_sq'].mul_(second)
    receipt['adam_rescale'] = dict(first_moment=first, second_moment=second, parameters=len(states),
                                   norm_mean=rescale['norm_mean'], norm_std=rescale['norm_std'], steps=rescale['steps'])
    warmup['base_lr'] = after['learning_rate']
    controller.update(multiplier=1., window=[], cooldown=controller['config']['cooldown_iterations'])
    return receipt


def tag(config):
    return 'clip583-lr080-linked-heads' if applied(config) else 'none'


def summary(config):
    return copy.deepcopy(config.get(KEY))
