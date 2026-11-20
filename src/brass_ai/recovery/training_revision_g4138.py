"""g4138 teacher search revision (the user's request of 2026-11-20; doc ai-research/G4138_TEACHER_SEARCH.md).
Adopted once with --teacher-search-revision at a committed boundary, then kept.

tables        48 teacher tables every iteration: 44 train and 4 validation (the g3270 revision ran 36 + 4).
short search  128 simulations (was 96) at C_scale .6 (was .7), K 8. The long search (640 simulations,
              K 8, C_scale .2) and its policy labels do not change.
search value  48 of a value game's 64 roots are short searches, and every return bootstraps from the
              roots' search values. The search-value head therefore trains only on games collected from
              adoption on: it waits until eight such ages exist (the reviewed rebuild length), and the older
              games leave as they age out of the 16-iteration buffer. Held-out games leaving the latest 32
              train it only if they were collected from adoption on. The dedicated held-out check (realized
              utility-to-go) keeps every game, and the era heads keep the search tables of every age.

Kept: the policy-label buffer and its identity (long labels only), the teacher beta and its adaptation,
the soft head, the search head and its stage. The qualified recipe (teacher_phase, its g1491 marker and
count_search) is not edited; this top-level block overlays what the trainer and the CPU teacher run.
"""
import copy

from brass_ai.learning.contracts import keys, require

KEY = 'teacher_search_revision'
CONTRACT = 'teacher-search-g4138/1.0.0'
BASE_TABLES = dict(train=36, held_out=4)  # the g3270 revision
BASE_SHORT = dict(simulations=96, k=8, c_scale=.7)
SETTINGS = dict(
    contract=CONTRACT,
    tables=dict(train=44, held_out=4),
    short_search=dict(simulations=128, k=8, c_scale=.6),
    search_value=dict(train_games='collected-from-adoption', required_ages=8),
)


def validate(block):
    keys(block, (*SETTINGS, 'from_generation'))
    require({k: v for k, v in block.items() if k != 'from_generation'} == SETTINGS
            and type(block['from_generation']) is int and block['from_generation'] >= 0,
            'Invalid g4138 teacher search revision')
    return block


def applied(config): return KEY in (config or {})


def configure(config, generation):
    """`config` with the revision adopted at committed generation `generation`; kept unchanged afterwards."""
    if applied(config):
        validate(config[KEY])
        return copy.deepcopy(config)
    from . import training_revision_g1491 as g1491, training_revision_g3270 as g3270
    require(g3270.applied(config) and config['teacher_phase'].get('mode') == 'on' and
            g1491.count_search(config) == g1491.G2600_SEARCH and not g1491.settings(config)['short_labels'],
            'The g4138 teacher search revision follows the g3270 teacher revision with teaching on, long-only labels '
            'and the g2600 search profile (long C_scale .2, short C_scale .7, K 8)')
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
    if a is not None: return None
    try:
        validate(b)
    except (ValueError, KeyError, TypeError):
        return None
    result = copy.deepcopy(old); result[KEY] = copy.deepcopy(b)
    return result


# ---- settings read by the training code --------------------------------------------------------

def tables(config):
    """The adopted table counts, or None (the g3270 revision's 36 + 4 apply)."""
    return dict(SETTINGS['tables']) if applied(config) else None


def short_search(profile):
    """A committed count-search profile with this revision's short search (CPU budget keys)."""
    s = SETTINGS['short_search']
    return dict(profile, fast=s['simulations'], fast_k=s['k'], fast_c_scale=s['c_scale'])


def search_profile(config):
    """The count search the CPU teacher runs: the committed profile, with the short search when adopted.

    Its keys are CPU budget keys, so a budget matches it only with 128 short simulations at K 8 and C_scale .6.
    """
    from .training_revision_g1491 import count_search
    profile = count_search(config)
    return short_search(profile) if applied(config) and profile is not None else profile


def rebuild(config):
    """The search-value training rebuild: games collected from adoption only, after eight such ages."""
    if not applied(config): return None
    return dict(origin_generation=config[KEY]['from_generation'],
                required_ages=SETTINGS['search_value']['required_ages'])


def tag(config):
    return 'teacher48-short128-c060' if applied(config) else 'none'


def summary(config):
    return copy.deepcopy(config.get(KEY))
