"""g4770 era-head revision (the user's request of 2026-11-22; doc ai-research/G4770_ERA_HEAD_REVISION.md).
Adopted once with --era-head-revision at a committed boundary, then kept; --rail-end-weight 0.25 later raises the
restored Rail-end weight.

rail-end      the Rail-end owner and VP outputs (removed untrained at g3528) come back, fresh (zero, as the heads'
              own initialisation), at weight .05 (local_scoring_auxiliary.rail_end_restored; ladder .05 -> .25).
              They train only in PPG, in the same batches, forward and steps as the Canal and Rail outputs (one
              shared hidden layer per entity kind). PPO neither snapshots, transfers nor preserves them, and no
              lambda-return outputs are cached for them. PPO-game Canal rows train them on the actual Rail-era
              (game-end) map; search-table Canal rows on lambda .95 returns once the fresh head has trained four
              iterations on the actual map (training_revision_g3140.RAIL_END_WARMUP). The ownership stop reports
              their CE; Canal and Rail decide.
ordinary      every era head (Canal, Rail, Rail-end) trains PPO-game rows on the actual map (lambda 1): PPO keeps no
              frozen era-head outputs for returns, PPG computes no corrections and stages no local-bootstrap
              sidecar, and an older age's sidecar is not read. Search-table rows keep their lambda .95 returns.
search pool   each iteration the era heads' search tables are a uniform sample, without replacement, of the
              search-value training tables admitted in the last 8 iterations (train tables collected at G-7..G and
              the held-out tables that left the latest 32 in those iterations). The sample is as large as the
              tables the last 4 iterations admitted (the 2.0.0 pool), drawn with its own generator (generation).

Kept: the Canal/Rail weights, the PPO preservation of the Canal/Rail outputs, the ownership stop, the search
rows' lambda .95 Canal/Rail targets, the held-out sets and every collection setting.
"""
import copy

from brass_ai.learning.contracts import keys, require

KEY = 'era_head_revision'
CONTRACT = 'era-head-revision-g4770/1.0.0'
RAIL_END_WEIGHT = .05
RAIL_END_WEIGHTS = (.05, .25)
SETTINGS = dict(
    contract=CONTRACT,
    rail_end=dict(start_weight=RAIL_END_WEIGHT, weights=list(RAIL_END_WEIGHTS), rows='canal-rows-to-the-game-end-map',
                  training='ppg-only-same-batches-as-canal-rail', ppo='no-snapshot-transfer-or-preservation',
                  ordinary_targets='actual-map', search_targets='lambda-return-after-warm-up', stop='reported-not-decisive'),
    ordinary_targets='actual-map',
    search_targets=dict(lam=.95),
    search_pool=dict(window_iterations=8, sampled_iterations=4, draw='uniform-tables-without-replacement'),
)


def validate(block):
    keys(block, (*SETTINGS, 'from_generation'))
    require({k: v for k, v in block.items() if k != 'from_generation'} == SETTINGS
            and type(block['from_generation']) is int and block['from_generation'] >= 0,
            'Invalid g4770 era-head revision')
    return block


def applied(config): return KEY in (config or {})


def configure(config, generation):
    """`config` with the revision adopted at committed generation `generation`; kept unchanged afterwards."""
    from . import local_scoring_v3 as v3, training_revision_g3140 as g3140
    from .local_scoring_auxiliary import KEY as LOCAL
    if applied(config):
        validate(config[KEY])
        return copy.deepcopy(config)
    local = config.get(LOCAL); revision = g3140.block(config) if g3140.applied(config) else None
    require(v3.two_era(local) and local['mode'] == 'apply' and revision is not None
            and 'search_auxiliary' in revision and 'ownership_stop' in revision
            and g3140.search_rows(config) == g3140.SEARCH_ROWS,
            'The g4770 era-head revision follows drop-rail-end with the applied era heads, the ownership stop, '
            'search-aux and search-rows (Rail-end on the actual map)')
    require(type(generation) is int and generation > local['rail_end_removed'], 'Invalid adoption generation')
    result = copy.deepcopy(config)
    restored = result[LOCAL]
    restored['rail_end_restored'] = generation
    restored['weights'] = dict(restored['weights'], rail_end=RAIL_END_WEIGHT)
    # The search rows' Rail-end targets: lambda returns after the fresh head's warm-up on the actual map.
    result[g3140.KEY]['search_rows'] = g3140.rail_end_returns(generation + g3140.RAIL_END_WARMUP)
    result[KEY] = validate(dict(copy.deepcopy(SETTINGS), from_generation=generation))
    from .config import validate_config
    validate_config(result)
    return result


def set_rail_end_weight(config, weight):
    """`config` with the restored Rail-end head at `weight` (.05 -> .25 only); unchanged when already there."""
    from .local_scoring_auxiliary import KEY as LOCAL
    require(applied(config), 'The Rail-end weight follows the g4770 era-head revision (--era-head-revision)')
    require(weight in RAIL_END_WEIGHTS, 'Rail-end weight must be one of ' + ', '.join(f'{w:g}' for w in RAIL_END_WEIGHTS))
    current = config[LOCAL]['weights']['rail_end']
    require(weight >= current, 'The Rail-end weight only moves up')
    if weight == current: return copy.deepcopy(config)
    result = copy.deepcopy(config)
    result[LOCAL]['weights'] = dict(result[LOCAL]['weights'], rail_end=weight)
    from .config import validate_config
    validate_config(result)
    return result


def previous(config):
    """The recipe without the block (recipe recognition; g3140.previous restores the replaced companion)."""
    if not applied(config): return config
    result = copy.deepcopy(config); result.pop(KEY)
    return result


def project(old, new):
    """`old` with `new`'s block when it is adopted or kept; None when it is removed or changed."""
    a, b = (old or {}).get(KEY), (new or {}).get(KEY)
    if a == b: return old
    if a is not None: return None
    try:
        validate(b)
    except (ValueError, KeyError, TypeError):
        return None
    result = copy.deepcopy(old); result[KEY] = copy.deepcopy(b)
    return result


# ---- settings read by the training code --------------------------------------------------------

def ordinary_actual(config):
    """True when PPO-game rows train every era head on the actual map (no frozen outputs, corrections or sidecar)."""
    return applied(config)


def search_pool(config):
    """The search-table pool sampling, or None (the 2.0.0 pool: the tables of the last four iterations)."""
    return copy.deepcopy(SETTINGS['search_pool']) if applied(config) else None


def tag(config):
    """Static: the Rail-end weight is already in the g3140 tag (railend005, railend025)."""
    return 'ppoactual-pool4of8' if applied(config) else 'none'


def summary(config):
    from .local_scoring_auxiliary import KEY as LOCAL
    if not applied(config): return None
    return dict(copy.deepcopy(config[KEY]), local_weights=copy.deepcopy(config[LOCAL]['weights']),
                rail_end_restored=config[LOCAL]['rail_end_restored'])
