"""g6670 revision (the user's request of 2026-11-28; doc ai-research/G6670_SEARCH_HEAD_CUT_AND_PL_WINDOW.md).
Adopted once with --g6670-revision at a committed boundary, on top of the g6230 soft-head revision, then kept.

Search head  the search-value head (paused since g4475) is cut. On the adoption iteration the head is still loaded
             from the committed checkpoint, but nothing reads it: the teacher tables already bind w = 0
             (teacher_search_head_weight 0, value_source 'main'), and the CPU bundles are head-less at w = 0. At that
             iteration's PPG boundary (SearchValueIteration.finish_ppg) the head module and its own Adam (the
             companion) are removed, so the committed checkpoint of the next generation has neither. Its stage record
             stays as committed (C, frozen since g4474) as history. The search tables still publish their value
             shards, the shards still retire on schedule, and the era heads keep training on the search tables
             (search-aux 2.0.0), as under the g4475 pause.
PL rating    the strength rating keeps and uses only the latest 64 rated history checkpoints (state.ratings.agents;
             the references generation-0, v4 and the trained-start baseline stay). Older ones move to
             ratings.retired_agents (name -> generation and model id, no weights path), so retention no longer pins
             their weights files; this run's scheduled cleanup then deletes them. The first trim runs at the
             trainer's startup rating pass. Retired checkpoints stored in another run (the parent g1190-restart-3,
             which branched this run at g4970 and stopped at g5177) are removed there once, after the first
             committed iteration (rating_parent_cleanup): retired from that run's own rating registry, with a
             backup of its state.json, then deleted where no control file of any run still pins them. Opponents are chosen among the kept checkpoints only. The PL fit reads
             only the latest 40,960 rated games (320 slots of 128); an agent that appears in those games is still a
             fit parameter after its checkpoint is retired. Game records and the compact archive are kept as they are.
PPG floor    value_phase.beta_clone_min .125 -> .25 (g3658 had lowered it from .25). The restored PPG beta is raised
             to .25 once if it is below (ppg_g210.initialize); KL trips, Adams, RNG, replay and cadence are kept.
             The search-head schedule's Stage C floor (.125) no longer binds: the effective floor is the larger one.
PPG clip     the PPG value phase clips at 3.0 (was 2.5 from g6230), with its Adam moments rescaled once on the adoption
             generation's checkpoint, from g5466's normal fit of the last 30 committed PPG phases' logged mean and P98
             (receipt PPG_GROUP_KEY in the PPG Adam group).
Unchanged    PPO's clip (6.5, the user kept it) and every rate; the teacher recipe, its labels, buffers and beta; the
             soft head (weight 4); the era heads; the rating protocol (32 setups x 4 seats, cadence 20).
"""
import copy

from brass_ai.learning.contracts import keys, require
from . import training_revision_g5466 as g5466
from . import training_revision_g5685 as g5685
from . import training_revision_g6230 as g6230

KEY = 'g6670_revision'
CONTRACT = 'training-revision-g6670/1.0.0'
PPG_GRAD_CLIP = 3.
BASE_PPG_GRAD_CLIP = 2.5  # g6230
PPG_BETA_FLOOR = .25
BASE_PPG_BETA_FLOOR = .125  # g3658
RATING_CHECKPOINTS = 64
RATING_GAMES = 40960  # 320 rating slots of 128 main games
WINDOW = 30
SEARCH = 40
CLIP_RATE_TOLERANCE = 10.
SETTINGS = dict(
    contract=CONTRACT,
    search_value_head='cut-at-adoption-ppg-boundary',
    rating=dict(checkpoints=RATING_CHECKPOINTS, games=RATING_GAMES, references='kept',
                retired='fit-parameters-while-their-games-are-in-the-window',
                other_runs='removed-once-after-the-first-commit-where-unpinned'),
    ppg_beta_floor=PPG_BETA_FLOOR,
    ppg_grad_clip=PPG_GRAD_CLIP,
    window_iterations=WINDOW,
)
VARIABLE = ('from_generation', 'ppg_adam_rescale')
PPG_GROUP_KEY = 'brass_ppg_adam_rescale_g6670'
CUT_CONTRACT = 'search-value-head-cut-g6670/1.0.0'
FLOOR_MIGRATION = 'ppg-beta-floor-quarter-g6670/1.0.0'


def committed_ppg_norms(run, state, window=WINDOW):
    """The PPG pre-clip norm summaries (mean and P98) of the latest `window` committed iterations with PPG steps,
    among the latest SEARCH (g5685.committed_rows); every one must have clipped at 2.5."""
    found = {}
    for iteration, row in g5685.committed_rows(run, state, SEARCH).items():
        record = g5466._record(iteration, (row.get('value_phase') or {}).get('gradient_clipping'))
        if record is not None: found[iteration] = record
    latest = sorted(found, reverse=True)[:window]
    require(len(latest) == window, f'The g6670 PPG Adam rescale needs {window} committed iterations of PPG norm '
            f'statistics among the latest {SEARCH}; found {len(latest)}')
    require(all(found[i][1] == BASE_PPG_GRAD_CLIP for i in latest),
            'The g6670 PPG Adam rescale reads iterations that clipped PPG at 2.5')
    return [found[i][0] for i in sorted(latest)]


def ppg_adam_rescale(norms):
    """g5466's PPG fit (mean and P98 per iteration), from the 2.5 clip to 3.0."""
    return g5466.adam_rescale(norms, old_clip=BASE_PPG_GRAD_CLIP, new_clip=PPG_GRAD_CLIP)


def validate(block):
    keys(block, (*SETTINGS, *VARIABLE))
    require({k: v for k, v in block.items() if k not in VARIABLE} == SETTINGS
            and type(block['from_generation']) is int and block['from_generation'] >= 0, 'Invalid g6670 revision')
    rescale = block['ppg_adam_rescale']
    require(type(rescale) is dict and type(rescale.get('norms')) is list and len(rescale['norms']) == WINDOW
            and rescale == ppg_adam_rescale(rescale['norms']) and rescale['first_moment'] > 1 and rescale['second_moment'] > 1
            and abs(rescale['predicted_clipped_percent'] - rescale['logged_clipped_percent']) <= CLIP_RATE_TOLERANCE,
            'Invalid g6670 PPG Adam rescale')
    return block


def applied(config): return KEY in (config or {})


def configure(config, generation, *, gradient_norms):
    """`config` with the revision adopted at committed generation `generation`, where the latest committed PPG phases
    logged `gradient_norms` (committed_ppg_norms); kept unchanged afterwards."""
    if applied(config):
        validate(config[KEY])
        return copy.deepcopy(config)
    from .training_revision_g4475 import search_value_paused
    from .training_revision_g1491 import teacher_search_head_weight
    require(g6230.applied(config) and g6230.ppg_grad_clip(config) == BASE_PPG_GRAD_CLIP
            and config['value_phase']['beta_clone_min'] == BASE_PPG_BETA_FLOOR,
            'The g6670 revision follows the g6230 soft-head revision (PPG clip 2.5) with the .125 PPG floor (g3658)')
    require(search_value_paused(config) and teacher_search_head_weight(config) == 0,
            'The search-value head is cut only while paused (g4475) and unread by the teacher tables (w = 0)')
    require(config.get('rating', {}).get('enabled') and config['rating']['setups'] * config['rating']['seat_rotations'] == 128,
            'The g6670 PL window reads 128 main games per rating slot')
    require(type(generation) is int and generation >= 0, 'Invalid adoption point')
    require(all(config[g6230.KEY]['from_generation'] <= r['iteration'] < generation for r in gradient_norms),
            'The g6670 PPG Adam rescale reads committed iterations trained at PPG clip 2.5')
    block = dict(copy.deepcopy(SETTINGS), from_generation=generation, ppg_adam_rescale=ppg_adam_rescale(gradient_norms))
    result = _adopt(config, validate(block))
    from .config import validate_config
    validate_config(result)
    return result


def _adopt(config, block):
    result = copy.deepcopy(config)
    result['value_phase']['beta_clone_min'] = block['ppg_beta_floor']
    result[KEY] = copy.deepcopy(block)
    return result


def previous(config):
    """The recipe without the revision (recipe recognition): the .125 PPG floor; the block holds the rest."""
    if not applied(config): return config
    result = copy.deepcopy(config); result.pop(KEY)
    result['value_phase']['beta_clone_min'] = BASE_PPG_BETA_FLOOR
    return result


def project(old, new):
    """`old` with `new`'s revision when it is adopted or kept; None when it is removed or changed."""
    a, b = (old or {}).get(KEY), (new or {}).get(KEY)
    if a == b: return old
    if a is not None: return None
    try:
        validate(b)
        if g6230.ppg_grad_clip(old) != BASE_PPG_GRAD_CLIP or old['value_phase']['beta_clone_min'] != BASE_PPG_BETA_FLOOR:
            return None
    except (ValueError, KeyError, TypeError):
        return None
    return _adopt(old, b)


# ---- settings read by the training code --------------------------------------------------------

def ppg_grad_clip(config):
    """The PPG value phase's own clip: 3.0 with the revision, else g6230's (2.5, ...)."""
    if applied(config): return config[KEY]['ppg_grad_clip']
    return g6230.ppg_grad_clip(config)


def raises_ppg_floor(config):
    """True when the PPG floor is back at .25 after g3658 lowered it (g3658.applied stays true for the lineage)."""
    return applied(config) and (config.get('value_phase') or {}).get('beta_clone_min') == PPG_BETA_FLOOR


def search_head_cut(config):
    """True when the search-value head is cut (removed at the adoption iteration's PPG boundary)."""
    return applied(config)


def removes_parent_checkpoints(config):
    """True when retired rating checkpoints stored in another run are removed there (rating_parent_cleanup)."""
    return applied(config)


def rating_window(config):
    """dict(checkpoints, games) of the PL rating with the revision, else None (all checkpoints and games)."""
    if not applied(config): return None
    return dict(checkpoints=config[KEY]['rating']['checkpoints'], games=config[KEY]['rating']['games'])


def rescale_ppg(optimizer, config, generation):
    """Once, on the committed checkpoint of the adoption generation: every PPG Adam exp_avg and exp_avg_sq multiplied
    by the recorded factors (2.5 -> 3.0), the receipt kept in the Adam group. Returns the receipt, or None."""
    if not applied(config): return None
    return g5685._rescale(optimizer, config[KEY], generation, name='PPG', group_key=PPG_GROUP_KEY,
                          rescale_key='ppg_adam_rescale', contract=CONTRACT, revision='g6670')


def tag(config):
    return 'svcut-pl64x40960-ppgfloor025-ppgclip30' if applied(config) else 'none'


def summary(config):
    return copy.deepcopy(config.get(KEY))
