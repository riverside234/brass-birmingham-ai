"""g5466 auxiliary revision (the user's request of 2026-11-24; doc ai-research/G5466_AUXILIARY_REVISION.md).
Adopted once with --auxiliary-revision at a committed boundary, on top of the g5402 optimizer revision, then kept.

Rail-end in PPO  the restored Rail-end outputs (rail_end_slots, rail_end_links) are frozen before PPO and preserved
                 during it like the Canal and Rail outputs (value_preservation.FrozenValues: owner KL plus VP Huber,
                 at the head's loss weight), so PPO's preservation steps reach every era head alike. PPG already
                 trained them in the same batches, steps and ownership stop, and their own warm-up has ended
                 (multiplier 1.0). Their loss weight stays the run's .25.
PPG clip         the PPG value phase clips its total gradient norm at 2.2 (it was 1.0, which 81% of the last 30
                 iterations' attempted steps hit: pre-clip mean about 1.45, P95 about 2.2). PPO keeps 5.83; the
                 era heads' and the soft head's own clips stay 1.0 (their norms hit 1 in at most one step in 30
                 iterations, so the user left them).
PPG Adam         the PPG Adam moments were built on gradients clipped at 1.0. Once, on the adoption generation's
                 checkpoint (rescale_ppg), exp_avg is multiplied by E[min(n, 2.2)] / E[min(n, 1)] and exp_avg_sq by
                 E[min(n, 2.2)^2] / E[min(n, 1)^2] (n: the pre-clip norm), from a normal fit of each of the last 30
                 committed iterations' logged mean and P95 (P98 from g5466 on; value_phase.gradient_clipping), pooled
                 by steps. The fit
                 must reproduce the clip rate the logs show at 1.0. Without it the first PPG steps would run about
                 1.5x their size under the new clip until the second moment caught up (about 1,000 steps).
head rates       the era heads step at the PPO rate in PPG too: the rate PPO's last step of the iteration applied
                 (the rate PPG's own peak is .65 of), fixed for the whole phase, head-only steps included: no PPG
                 warm-up or cosine decay. PPG's own rate keeps both. In PPO the era heads and the soft head keep
                 the PPO step rate (g5402).
"""
import copy
import json
import math
from pathlib import Path

from brass_ai.learning.contracts import keys, require

KEY = 'auxiliary_revision'
CONTRACT = 'auxiliary-revision-g5466/1.0.0'
PPG_GRAD_CLIP = 2.2
BASE_PPG_GRAD_CLIP = 1.
WINDOW = 30  # committed iterations of PPG norm statistics: about 1,000 attempted steps, Adam's beta2 horizon
SEARCH = 40  # ... found among the latest this many committed iterations (an iteration without PPG steps is skipped)
CLIP_RATE_TOLERANCE = 10.  # percentage points between the fit's and the logged clip rate at the old clip
Z95 = 1.6448536269514722
# The logged upper percentile: P95 up to g5466, P98 after (value_phase.clipping_report); its standard normal quantile.
UPPER = dict(p95=Z95, p98=2.0537489106318225)
SETTINGS = dict(
    contract=CONTRACT,
    ppg_grad_clip=PPG_GRAD_CLIP,
    rail_end_ppo='preserved-like-canal-and-rail',
    era_heads_learning_rate=dict(ppg='ppo-rate-fixed', ppo='ppo-step-rate'),
    window_iterations=WINDOW,
)
NORMS = ('iteration', 'steps', 'clipped_steps', 'mean')  # plus one upper percentile, p95 or p98 (UPPER)
RESCALE_MODEL = 'normal-fit-of-logged-ppg-norm-mean-and-upper-percentile/1.0.0'
VARIABLE = ('from_generation', 'adam_rescale')
GROUP_KEY = 'brass_ppg_adam_rescale'  # the PPG Adam group's receipt: present once the moments are rescaled


def _digits(value): return float(f'{value:.12g}')


def _finite(value): return type(value) in (int, float) and math.isfinite(value)


def _moments(mean, std, c):
    """E[min(X, c)], E[min(X, c)^2] and P(X > c) for X ~ N(mean, std^2) (a point mass when std is 0)."""
    if std <= 0:
        x = min(mean, c)
        return x, x * x, float(mean > c)
    z = (c - mean) / std
    below = .5 * (1 + math.erf(z / math.sqrt(2))); density = math.exp(-z * z / 2) / math.sqrt(2 * math.pi)
    first = mean * below - std * density + c * (1 - below)
    squared = (mean ** 2 + std ** 2) * below - std * (mean + c) * density + c * c * (1 - below)
    return first, squared, 1 - below


def _record(iteration, clipping):
    """One committed iteration's PPG norm summary, or None without attempted steps."""
    if not isinstance(clipping, dict) or not clipping.get('steps'): return None
    norm = clipping['pre_clip_norm']; upper = next((k for k in UPPER if norm.get(k) is not None), None)
    require(upper is not None, 'A logged PPG norm summary without its upper percentile')
    return dict(iteration=iteration, steps=clipping['steps'], clipped_steps=clipping['clipped_steps'],
                mean=norm['mean'], **{upper: norm[upper]}), clipping['max_norm']


def committed_ppg_norms(run, state, window=WINDOW):
    """The PPG pre-clip norm summaries (value_phase.gradient_clipping, logged from g5402) of the latest `window`
    committed iterations with PPG steps: from state.recent_training, the run's uncompacted iteration reports and
    its compact history (retention-history.json). Every one must have clipped at the old 1.0."""
    found = {}
    def add(iteration, value_phase):
        if type(iteration) is not int or iteration >= state['iteration'] or iteration in found: return
        record = _record(iteration, (value_phase or {}).get('gradient_clipping'))
        if record is not None: found[iteration] = record
    for row in state.get('recent_training') or []: add(row.get('iteration'), row.get('value_phase'))
    run = Path(run)
    for path in sorted(run.glob('iteration-*/report.json')):
        try: number = int(path.parent.name.split('-', 1)[1])
        except ValueError: continue
        if number in found or number >= state['iteration'] or number < state['iteration'] - SEARCH: continue
        report = json.loads(path.read_text())
        add(report.get('iteration'), report.get('value_phase'))
    history = run / 'retention-history.json'
    if history.is_file():
        for value in json.loads(history.read_text()).values():
            summary = (value or {}).get('summary') or {}
            add(summary.get('iteration'), summary.get('value_phase'))
    latest = sorted((i for i in found if i >= state['iteration'] - SEARCH), reverse=True)[:window]
    require(len(latest) == window, f'The g5466 PPG Adam rescale needs {window} committed iterations of PPG norm '
            f'statistics (value_phase.gradient_clipping) among the latest {SEARCH}; found {len(latest)}')
    require(all(found[i][1] == BASE_PPG_GRAD_CLIP for i in latest),
            'The g5466 PPG Adam rescale reads iterations that clipped PPG at 1.0')
    return [found[i][0] for i in sorted(latest)]


def adam_rescale(norms, old_clip=BASE_PPG_GRAD_CLIP, new_clip=PPG_GRAD_CLIP):
    """The factors that move the PPG Adam moments from gradients clipped at `old_clip` to `new_clip`:
    E[min(n, new)] / E[min(n, old)] for exp_avg and E[min(n, new)^2] / E[min(n, old)^2] for exp_avg_sq, where each
    iteration's pre-clip norm n ~ N(mean, ((P - mean) / z)^2) for its logged upper percentile P (P95, z 1.645, or P98,
    z 2.054), pooled by its steps. Also the clip rate the fit predicts at `old_clip` beside the logged one, which
    validate() requires to agree."""
    def upper(r):
        keys = [k for k in UPPER if k in r]
        return keys[0] if len(keys) == 1 else None
    require(norms and all(upper(r) and set(r) == {*NORMS, upper(r)} and type(r['steps']) is int and r['steps'] > 0
                          and type(r['clipped_steps']) is int and 0 <= r['clipped_steps'] <= r['steps']
                          and _finite(r['mean']) and r['mean'] > 0 and _finite(r[upper(r)]) and r[upper(r)] >= r['mean']
                          for r in norms), 'Invalid committed PPG gradient norms')
    steps = sum(r['steps'] for r in norms)
    old = [0., 0., 0.]; new = [0., 0.]
    for r in norms:
        std = (r[upper(r)] - r['mean']) / UPPER[upper(r)]
        a1, a2, above = _moments(r['mean'], std, old_clip); b1, b2, _ = _moments(r['mean'], std, new_clip)
        w = r['steps'] / steps
        old[0] += w * a1; old[1] += w * a2; old[2] += w * above; new[0] += w * b1; new[1] += w * b2
    return dict(model=RESCALE_MODEL, old_clip=old_clip, new_clip=new_clip, norms=copy.deepcopy(norms), steps=steps,
                norm_mean=_digits(sum(r['steps'] * r['mean'] for r in norms) / steps),
                predicted_clipped_percent=_digits(100. * old[2]),
                logged_clipped_percent=_digits(100. * sum(r['clipped_steps'] for r in norms) / steps),
                first_moment=_digits(new[0] / old[0]), second_moment=_digits(new[1] / old[1]))


def validate(block):
    keys(block, (*SETTINGS, *VARIABLE))
    require({k: v for k, v in block.items() if k not in VARIABLE} == SETTINGS
            and type(block['from_generation']) is int and block['from_generation'] >= 0, 'Invalid g5466 auxiliary revision')
    rescale = block['adam_rescale']
    require(type(rescale) is dict and type(rescale.get('norms')) is list and len(rescale['norms']) == WINDOW
            and rescale == adam_rescale(rescale['norms']) and rescale['first_moment'] > 1 and rescale['second_moment'] > 1
            and abs(rescale['predicted_clipped_percent'] - rescale['logged_clipped_percent']) <= CLIP_RATE_TOLERANCE,
            'Invalid g5466 PPG Adam rescale')
    return block


def applied(config): return KEY in (config or {})


def configure(config, generation, *, gradient_norms):
    """`config` with the revision adopted at committed generation `generation`, where the latest committed PPG phases
    logged `gradient_norms` (committed_ppg_norms); kept unchanged afterwards."""
    if applied(config):
        validate(config[KEY])
        return copy.deepcopy(config)
    from . import training_revision_g5402 as g5402
    from .local_scoring_v3 import rail_end_restored
    from .ppg_g210 import active as ppg_g210
    require(g5402.applied(config) and g5402.ppg_grad_clip(config) == BASE_PPG_GRAD_CLIP
            and rail_end_restored(config.get('local_scoring_auxiliary')) and ppg_g210(config.get('value_phase', {})),
            'The g5466 auxiliary revision follows the g5402 optimizer revision (PPG clip 1.0, linked head rates) with '
            'the restored Rail-end heads and the g210 PPG value phase')
    require(type(generation) is int and generation >= 0, 'Invalid adoption point')
    block = dict(copy.deepcopy(SETTINGS), from_generation=generation, adam_rescale=adam_rescale(gradient_norms))
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
    try: validate(b)
    except (ValueError, KeyError, TypeError): return None
    result = copy.deepcopy(old); result[KEY] = copy.deepcopy(b)
    return result


# ---- settings read by the training code --------------------------------------------------------

def ppg_grad_clip(config):
    """The PPG value phase's own clip: 2.2 with the revision, else g5402's (1.0, or None: update.grad_clip)."""
    if applied(config): return config[KEY]['ppg_grad_clip']
    from .training_revision_g5402 import ppg_grad_clip as optimizer_revision_clip
    return optimizer_revision_clip(config)


def rail_end_in_ppo(config):
    """True when PPO freezes and preserves the restored Rail-end outputs like the Canal and Rail ones."""
    return applied(config)


def fixed_head_rate(config):
    """True when the era heads step at the PPO rate in PPG, fixed for the phase (no warm-up or decay)."""
    return applied(config)


def rescale_ppg(optimizer, config, generation):
    """Once, on the committed checkpoint of the adoption generation: every PPG Adam exp_avg and exp_avg_sq multiplied
    by the recorded factors, and the receipt kept in the Adam group (GROUP_KEY), so a later checkpoint is not
    rescaled again. A crashed attempt at that generation reloads the same checkpoint and rescales it the same way.
    Returns the receipt, or None."""
    if not applied(config): return None
    require(len(optimizer.param_groups) == 1, 'The g5466 PPG Adam rescale expects one PPG Adam group')
    group = optimizer.param_groups[0]; block = config[KEY]
    if GROUP_KEY in group:
        require(group[GROUP_KEY]['generation'] == block['from_generation'], 'The PPG Adam was rescaled by another adoption')
        return None
    require(generation == block['from_generation'],
            'The g5466 PPG Adam rescale runs once, on the checkpoint of its adoption generation')
    rescale = block['adam_rescale']; first, second = rescale['first_moment'], rescale['second_moment']
    states = [optimizer.state[p] for p in group['params'] if p in optimizer.state]
    require(states and all('exp_avg' in s and 'exp_avg_sq' in s and 'max_exp_avg_sq' not in s for s in states),
            'The g5466 PPG Adam rescale needs the trained PPG Adam moments')
    for s in states:
        s['exp_avg'].mul_(first); s['exp_avg_sq'].mul_(second)
    receipt = dict(contract=CONTRACT, generation=generation, old_clip=rescale['old_clip'], new_clip=rescale['new_clip'],
                   first_moment=first, second_moment=second, parameters=len(states), steps=rescale['steps'],
                   norm_mean=rescale['norm_mean'], predicted_clipped_percent=rescale['predicted_clipped_percent'],
                   logged_clipped_percent=rescale['logged_clipped_percent'],
                   kept='Adam steps, the PPG beta and trips, the replay cache and the PPG schedule')
    group[GROUP_KEY] = copy.deepcopy(receipt)
    return receipt


def tag(config):
    return 'ppgclip22-railendppo-headrate' if applied(config) else 'none'


def summary(config):
    return copy.deepcopy(config.get(KEY))
