"""g5685 clip revision (the user's request of 2026-11-25; doc ai-research/G5685_CLIP_REVISION.md).
Adopted once with --clip-revision at a committed boundary, on top of the g5402 and g5466 revisions, then kept.

PPO clip   PPO clips its total gradient norm at 6.5 (update.grad_clip; it was 5.83 from g5402). In the last 30
           committed iterations 9.0% of PPO steps were above 5.83 (pre-clip mean 4.78, P98 about 6.6).
PPG clip   the PPG value phase clips at 2.4 (it was 2.2 from g5466): 5.9% of its attempted steps were above 2.2
           (pre-clip mean 1.34, P98 about 2.4).
Adam       both Adams' moments were built on gradients clipped at the old values. Once, on the adoption
           generation's checkpoint, exp_avg is multiplied by E[min(n, new)] / E[min(n, old)] and exp_avg_sq by
           E[min(n, new)^2] / E[min(n, old)^2] (n: the pre-clip norm), from a normal fit of each of the last 30
           committed iterations' logged norms, pooled by steps: PPO's mean and standard deviation
           (actor.gradient_norms), PPG's mean and P98 (value_phase.gradient_clipping, the g5466 fit). Each fit must
           reproduce the clip rate the logs show at the old clip. PPO's receipt is kept in its Adam group
           (PPO_GROUP_KEY, resume.apply_optimizer_config), PPG's in its own (PPG_GROUP_KEY, ppg_g210.initialize), so
           a later checkpoint is not rescaled again.
Unchanged  the PPO and PPG rates and schedules, the era heads' and the soft head's own clips (1.0), the Rail-end
           preservation and the era heads' fixed PPO rate in PPG (g5466).
"""
import copy
import json
import math
from pathlib import Path

from brass_ai.learning.contracts import keys, require
from . import training_revision_g5466 as g5466

KEY = 'clip_revision'
CONTRACT = 'clip-revision-g5685/1.0.0'
PPO_GRAD_CLIP = 6.5
BASE_PPO_GRAD_CLIP = 5.83  # g5402
PPG_GRAD_CLIP = 2.4
BASE_PPG_GRAD_CLIP = 2.2  # g5466
WINDOW = 30  # committed iterations of norm statistics: about 1,350 PPO and 870 PPG steps
SEARCH = 40  # ... found among the latest this many committed iterations (an iteration without PPG steps is skipped)
CLIP_RATE_TOLERANCE = 10.  # percentage points between a fit's and the logged clip rate at the old clip
SETTINGS = dict(contract=CONTRACT, ppo_grad_clip=PPO_GRAD_CLIP, ppg_grad_clip=PPG_GRAD_CLIP, window_iterations=WINDOW)
VARIABLE = ('from_generation', 'ppo_adam_rescale', 'ppg_adam_rescale')
PPO_NORMS = ('iteration', 'steps', 'clipped_steps', 'mean', 'std')
PPO_MODEL = 'normal-fit-of-logged-ppo-norm-mean-and-std/1.0.0'
PPO_GROUP_KEY = 'brass_ppo_adam_rescale_g5685'  # the PPO Adam group's receipt: present once its moments are rescaled
PPG_GROUP_KEY = 'brass_ppg_adam_rescale_g5685'  # the same for the PPG Adam (g5466's receipt stays beside it)


def _finite(value): return type(value) in (int, float) and math.isfinite(value)


def _ppo_record(iteration, actor):
    """One committed iteration's PPO norm summary (updates.py: every optimizer step's pre-clip norm), or None."""
    norms = (actor or {}).get('gradient_norms')
    if not isinstance(norms, dict) or not norms.get('count'): return None
    return dict(iteration=iteration, steps=norms['count'], clipped_steps=actor['gradient_clip_steps'],
                mean=norms['mean'], std=norms['std'])


def committed_rows(run, state, search=SEARCH):
    """{iteration: training summary} of the latest `search` committed iterations, from state.recent_training, the
    run's uncompacted iteration reports and its compact history (retention-history.json), in that order of
    precedence. g6230 reads its PPG norms from the same rows."""
    rows = {}
    def add(iteration, row):
        if type(iteration) is not int or iteration >= state['iteration'] or iteration < state['iteration'] - search:
            return
        if iteration not in rows: rows[iteration] = row
    for row in state.get('recent_training') or []: add(row.get('iteration'), row)
    run = Path(run)
    for path in sorted(run.glob('iteration-*/report.json')):
        try: number = int(path.parent.name.split('-', 1)[1])
        except ValueError: continue
        if number in rows or number >= state['iteration'] or number < state['iteration'] - search: continue
        report = json.loads(path.read_text())
        add(report.get('iteration'), report)
    history = run / 'retention-history.json'
    if history.is_file():
        for value in json.loads(history.read_text()).values():
            summary = (value or {}).get('summary') or {}
            add(summary.get('iteration'), summary)
    return rows


def committed_norms(run, state, window=WINDOW):
    """The PPO and PPG pre-clip norm summaries of the latest `window` committed iterations with steps (each among the
    latest SEARCH; committed_rows). Every PPG one must have clipped at 2.2; configure() checks that the PPO ones
    trained after the g5402 adoption (5.83)."""
    ppo, ppg = {}, {}
    for iteration, row in committed_rows(run, state).items():
        record = _ppo_record(iteration, row.get('actor'))
        if record is not None: ppo[iteration] = record
        found = g5466._record(iteration, (row.get('value_phase') or {}).get('gradient_clipping'))
        if found is not None: ppg[iteration] = found
    result = {}
    for name, found in (('ppo', ppo), ('ppg', ppg)):
        latest = sorted(found, reverse=True)[:window]
        require(len(latest) == window, f'The g5685 {name.upper()} Adam rescale needs {window} committed iterations of '
                f'{name.upper()} norm statistics among the latest {SEARCH}; found {len(latest)}')
        result[name] = [found[i] for i in sorted(latest)]
    require(all(clip == BASE_PPG_GRAD_CLIP for _, clip in result['ppg']),
            'The g5685 PPG Adam rescale reads iterations that clipped PPG at 2.2')
    result['ppg'] = [record for record, _ in result['ppg']]
    return result


def ppo_adam_rescale(norms, old_clip=BASE_PPO_GRAD_CLIP, new_clip=PPO_GRAD_CLIP):
    """The factors that move the PPO Adam moments from gradients clipped at `old_clip` to `new_clip`:
    E[min(n, new)] / E[min(n, old)] for exp_avg and the same with squares for exp_avg_sq, where each iteration's
    pre-clip norm n ~ N(mean, std^2) of its logged steps, pooled by steps; with the clip rate the fit predicts at
    `old_clip` beside the logged one."""
    require(norms and all(set(r) == set(PPO_NORMS) and type(r['steps']) is int and r['steps'] > 0
                          and type(r['clipped_steps']) is int and 0 <= r['clipped_steps'] <= r['steps']
                          and _finite(r['mean']) and r['mean'] > 0 and _finite(r['std']) and r['std'] >= 0
                          for r in norms), 'Invalid committed PPO gradient norms')
    steps = sum(r['steps'] for r in norms)
    old = [0., 0., 0.]; new = [0., 0.]
    for r in norms:
        a1, a2, above = g5466._moments(r['mean'], r['std'], old_clip); b1, b2, _ = g5466._moments(r['mean'], r['std'], new_clip)
        w = r['steps'] / steps
        old[0] += w * a1; old[1] += w * a2; old[2] += w * above; new[0] += w * b1; new[1] += w * b2
    digits = g5466._digits
    return dict(model=PPO_MODEL, old_clip=old_clip, new_clip=new_clip, norms=copy.deepcopy(norms), steps=steps,
                norm_mean=digits(sum(r['steps'] * r['mean'] for r in norms) / steps),
                predicted_clipped_percent=digits(100. * old[2]),
                logged_clipped_percent=digits(100. * sum(r['clipped_steps'] for r in norms) / steps),
                first_moment=digits(new[0] / old[0]), second_moment=digits(new[1] / old[1]))


def ppg_adam_rescale(norms):
    """g5466's PPG fit (mean and P98 per iteration), from the 2.2 clip to 2.4."""
    return g5466.adam_rescale(norms, old_clip=BASE_PPG_GRAD_CLIP, new_clip=PPG_GRAD_CLIP)


def _fitted(rescale, fit):
    return (type(rescale) is dict and type(rescale.get('norms')) is list and len(rescale['norms']) == WINDOW
            and rescale == fit(rescale['norms']) and rescale['first_moment'] > 1 and rescale['second_moment'] > 1
            and abs(rescale['predicted_clipped_percent'] - rescale['logged_clipped_percent']) <= CLIP_RATE_TOLERANCE)


def validate(block):
    keys(block, (*SETTINGS, *VARIABLE))
    require({k: v for k, v in block.items() if k not in VARIABLE} == SETTINGS
            and type(block['from_generation']) is int and block['from_generation'] >= 0, 'Invalid g5685 clip revision')
    require(_fitted(block['ppo_adam_rescale'], ppo_adam_rescale), 'Invalid g5685 PPO Adam rescale')
    require(_fitted(block['ppg_adam_rescale'], ppg_adam_rescale), 'Invalid g5685 PPG Adam rescale')
    return block


def applied(config): return KEY in (config or {})


def configure(config, generation, *, gradient_norms):
    """`config` with the revision adopted at committed generation `generation`, where the latest committed phases
    logged `gradient_norms` (committed_norms: dict(ppo=..., ppg=...)); kept unchanged afterwards."""
    if applied(config):
        validate(config[KEY])
        return copy.deepcopy(config)
    from . import training_revision_g5402 as g5402
    require(g5466.applied(config) and config['update']['grad_clip'] == BASE_PPO_GRAD_CLIP
            and g5466.ppg_grad_clip(config) == BASE_PPG_GRAD_CLIP,
            'The g5685 clip revision follows the g5402 (PPO clip 5.83) and g5466 (PPG clip 2.2) revisions')
    require(type(generation) is int and generation >= 0, 'Invalid adoption point')
    keys(gradient_norms, ('ppo', 'ppg'))
    # The fits read only iterations trained at the old clips.
    require(all(config[g5402.KEY]['from_generation'] <= r['iteration'] < generation for r in gradient_norms['ppo'])
            and all(config[g5466.KEY]['from_generation'] <= r['iteration'] < generation for r in gradient_norms['ppg']),
            'The g5685 Adam rescales read committed iterations trained at PPO clip 5.83 and PPG clip 2.2')
    block = dict(copy.deepcopy(SETTINGS), from_generation=generation,
                 ppo_adam_rescale=ppo_adam_rescale(gradient_norms['ppo']),
                 ppg_adam_rescale=ppg_adam_rescale(gradient_norms['ppg']))
    result = _adopt(config, validate(block))
    from .config import validate_config
    validate_config(result)
    return result


def _adopt(config, block):
    result = copy.deepcopy(config)
    result['update']['grad_clip'] = block['ppo_grad_clip']
    result[KEY] = copy.deepcopy(block)
    return result


def previous(config):
    """The recipe without the revision (recipe recognition): the g5402 PPO clip; the block holds the rest."""
    if not applied(config): return config
    result = copy.deepcopy(config); result.pop(KEY)
    result['update']['grad_clip'] = BASE_PPO_GRAD_CLIP
    return result


def project(old, new):
    """`old` with `new`'s revision when it is adopted or kept; None when it is removed or changed."""
    a, b = (old or {}).get(KEY), (new or {}).get(KEY)
    if a == b: return old
    if a is not None: return None
    try:
        validate(b)
        if old['update']['grad_clip'] != BASE_PPO_GRAD_CLIP or g5466.ppg_grad_clip(old) != BASE_PPG_GRAD_CLIP:
            return None
    except (ValueError, KeyError, TypeError):
        return None
    return _adopt(old, b)


# ---- settings read by the training code --------------------------------------------------------

def ppg_grad_clip(config):
    """The PPG value phase's own clip: 2.4 with the revision, else g5466's (2.2, 1.0 or None)."""
    if applied(config): return config[KEY]['ppg_grad_clip']
    return g5466.ppg_grad_clip(config)


def _rescale(optimizer, block, generation, *, name, group_key, rescale_key, contract=CONTRACT, revision='g5685'):
    """Multiply one Adam's moments once by `block[rescale_key]`'s factors, at `block['from_generation']`, with the
    receipt in the group under `group_key`; g6230 reuses it for its own PPG clip."""
    require(len(optimizer.param_groups) == 1, f'The {revision} {name} Adam rescale expects one {name} Adam group')
    group = optimizer.param_groups[0]
    if group_key in group:
        require(group[group_key]['generation'] == block['from_generation'],
                f'The {name} Adam was rescaled by another adoption')
        return None
    require(generation == block['from_generation'],
            f'The {revision} {name} Adam rescale runs once, on the checkpoint of its adoption generation')
    rescale = block[rescale_key]; first, second = rescale['first_moment'], rescale['second_moment']
    states = [optimizer.state[p] for p in group['params'] if p in optimizer.state]
    require(states and all('exp_avg' in s and 'exp_avg_sq' in s and 'max_exp_avg_sq' not in s for s in states),
            f'The {revision} {name} Adam rescale needs the trained {name} Adam moments')
    for s in states:
        s['exp_avg'].mul_(first); s['exp_avg_sq'].mul_(second)
    receipt = dict(contract=contract, phase=name.lower(), generation=generation, old_clip=rescale['old_clip'],
                   new_clip=rescale['new_clip'], first_moment=first, second_moment=second, parameters=len(states),
                   steps=rescale['steps'], norm_mean=rescale['norm_mean'],
                   predicted_clipped_percent=rescale['predicted_clipped_percent'],
                   logged_clipped_percent=rescale['logged_clipped_percent'],
                   kept='Adam steps, rates, schedules and the other group options')
    group[group_key] = copy.deepcopy(receipt)
    return receipt


def rescale_ppo(optimizer, config, generation):
    """Once, on the committed checkpoint of the adoption generation: every PPO Adam exp_avg and exp_avg_sq multiplied
    by the recorded factors, the receipt kept in the Adam group. A crashed attempt at that generation reloads the same
    checkpoint and rescales it the same way. Returns the receipt, or None."""
    if not applied(config): return None
    return _rescale(optimizer, config[KEY], generation, name='PPO', group_key=PPO_GROUP_KEY, rescale_key='ppo_adam_rescale')


def rescale_ppg(optimizer, config, generation):
    """rescale_ppo for the PPG value phase's Adam."""
    if not applied(config): return None
    return _rescale(optimizer, config[KEY], generation, name='PPG', group_key=PPG_GROUP_KEY, rescale_key='ppg_adam_rescale')


def tag(config):
    return 'clip65-ppgclip24' if applied(config) else 'none'


def summary(config):
    return copy.deepcopy(config.get(KEY))
