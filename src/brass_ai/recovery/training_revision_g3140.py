"""g3140 value-phase revision (ai-research/TRAINING_IMPROVEMENT_PLAN_G2910.md, build steps 1, 5-8).

Each item is adopted at a committed boundary with --value-revision and persists.
Items only move forward; the value phase's cap, beta floor, schedule and strict
held-out early stop are unchanged, and its check always scores the four main
terms on real outcomes with the original weights.

time        step 1: full-buffer train-metric scans every tenth iteration (held-out
            evidence decides the utility receipt in between). The search-value head
            keeps training (user decision, g3140): it is not paused.
validation  step 5: the early stop validates on a quarter of the new games, which
            train in the next iteration; the permanent 5% hold-out goes.
heads       step 6: era-separated local-scoring heads with a per-entity trunk slice
            (local_scoring_v3), fresh at this generation; weight .05.
targets     step 7: win and utility heads train on half the outcome and half the
            lambda .95 bootstrapped return from the critic frozen before PPO.
weight=X    step 8: local-scoring weight .15, then .5 (both eras, owner and VP).
vp          step 8: total-VP loss weight .1 -> .5 in training (the check keeps .1).
rail-end    step 8: Canal rows also predict the Rail-end map, at half the Canal weight.
ownership-stop  (g3318, after heads) the era heads get their own early stop: every 16
            steps the held-out owner cross-entropy on the value phase's validation
            games decides whether the heads keep their steps (an increase restores
            heads and Adam and ends their fitting). It replaces the utility-loss stop
            for the heads: once that stop ends the trunk's phase, the heads keep
            fitting on the remaining batches with the trunk frozen (recovery/ownership_stop.py).
search-aux  (g3423, after ownership-stop; needs the g3270 teacher revision) completed search-
            teacher tables train the era heads: each table sends its era-end scoring maps once
            with its finish message. The played roots of the search-value training tables of
            the last four iterations (36 train + 4 leaving the held-out set, per iteration) are
            mixed into every PPG batch the heads train on, each root once per epoch, with the
            era heads' ownership and local-VP losses only. The search-value validation set
            (latest 32 held-out tables, those with maps) is pooled row by row with the ordinary
            validation rows in the ownership stop's criterion (recovery/search_auxiliary.py).

GPU-staged teacher rows and the background cleanup purge (also step 1) are
code changes with no setting: they change neither numbers nor recipe.
"""
import copy

from brass_ai.learning.contracts import keys, require

KEY = 'value_revision'
CONTRACT = 'value-revision-g3140/1.0.0'
OFF = dict(contract=CONTRACT, train_metrics_every=1, validation=None, bootstrap=None, value_weights=dict(rank=.25, vp=.1))
TIME = dict(train_metrics_every=10)
VALIDATION = dict(contract='new-game-validation/1.0.0', fraction=.25)
BOOTSTRAP = dict(contract='blended-critic-targets/1.0.0', lam=.95, share=.5)
VP_WEIGHT = .5
OWNERSHIP_STOP = dict(contract='ownership-ce-early-stop/1.0.0', interval_steps=16,
                      criterion='validation-owner-ce', after_value_stop='heads-only')
SEARCH_AUX = dict(contract='search-table-auxiliary/2.0.0', rows='played-roots',
                  train_tables='search-value-training-tables-admitted-in-the-last-iterations', train_iterations=4,
                  heldout='search-value-validation-tables-with-maps', sampling='mixed-into-every-ppg-batch-once-per-epoch',
                  losses='era-heads-ownership-and-local-vp', row_weight='ordinary-row', criterion='pooled-owner-ce')
ITEMS = ('time', 'validation', 'heads', 'targets', 'weight=0.15', 'weight=0.5', 'vp', 'rail-end', 'ownership-stop',
         'search-aux')


def validate(block):
    keys(block, tuple(OFF), ('ownership_stop', 'search_auxiliary'))
    require(block['contract'] == CONTRACT and block['train_metrics_every'] in (1, TIME['train_metrics_every'])
            and block['validation'] in (None, VALIDATION)
            and block['bootstrap'] in (None, BOOTSTRAP)
            and block['value_weights'] in (OFF['value_weights'], dict(OFF['value_weights'], vp=VP_WEIGHT))
            and block.get('ownership_stop', OWNERSHIP_STOP) == OWNERSHIP_STOP
            and block.get('search_auxiliary', SEARCH_AUX) == SEARCH_AUX
            and ('search_auxiliary' not in block or 'ownership_stop' in block),
            'Invalid g3140 value revision')
    return block


def block(config):
    return validate(config[KEY]) if KEY in config else OFF


def applied(config): return KEY in config


def parse(text):
    items = tuple(x.strip() for x in text.split(',') if x.strip())
    require(items and all(x in ITEMS for x in items) and len(set(items)) == len(items),
            'Use --value-revision with items from: ' + ', '.join(ITEMS))
    return items


def configure(config, items, generation):
    """The recipe with the requested items adopted at committed generation `generation`."""
    from . import local_scoring_v3 as v3
    from .local_scoring_auxiliary import KEY as LOCAL, accepted_warmup
    result = copy.deepcopy(config)
    revision = copy.deepcopy(block(result))
    for item in items:
        if item == 'time':
            revision.update(TIME)
        elif item == 'validation':
            revision['validation'] = copy.deepcopy(VALIDATION)
        elif item == 'targets':
            from .dcpg_config import enabled as preserves_outputs
            require(preserves_outputs(result['update']), 'Blended targets read the critic PPO freezes before its update')
            revision['bootstrap'] = copy.deepcopy(BOOTSTRAP)
        elif item == 'vp':
            revision['value_weights'] = dict(revision['value_weights'], vp=VP_WEIGHT)
        elif item == 'ownership-stop':
            require(v3.is_v3(result.get(LOCAL)), 'ownership-stop needs the era heads (--value-revision heads)')
            require(result['value_phase'].get('acceptance_steps') == OWNERSHIP_STOP['interval_steps'],
                    'The ownership stop checks at the value phase\'s 16-step boundaries')
            revision['ownership_stop'] = copy.deepcopy(OWNERSHIP_STOP)
        elif item == 'search-aux':
            from .training_revision_g3270 import applied as teacher_revised
            require(v3.is_v3(result.get(LOCAL)), 'search-aux needs the era heads (--value-revision heads)')
            require(teacher_revised(result), 'search-aux reads the g3270 rotating search-value held-out set (--teacher-revision)')
            revision['search_auxiliary'] = copy.deepcopy(SEARCH_AUX)
        elif item == 'heads':
            old = result.get(LOCAL)
            if v3.is_v3(old): continue  # adopted earlier
            require(old is not None and accepted_warmup(old) and old['mode'] == 'apply',
                    'Era heads replace the applied 2.0.0 local-scoring companion')
            result[LOCAL] = v3.default(generation, old, mode=old['mode'])
        else:
            local = result.get(LOCAL)
            require(v3.is_v3(local), item + ' needs the era heads (--value-revision heads)')
            weights = local['weights']
            if item == 'rail-end':
                weights['rail_end'] = weights['canal'] / 2
            else:
                value = float(item.split('=')[1])
                require(v3.LADDER.index(value) >= v3.LADDER.index(weights['canal']), 'The local-scoring weight only moves up')
                weights.update(canal=value, rail=value, rail_end=value / 2 if weights['rail_end'] else 0.)
            v3.validate_config(local)
    require('search_auxiliary' not in revision or 'ownership_stop' in revision,
            'search-aux early-stops through the ownership stop (--value-revision ownership-stop)')
    result[KEY] = validate(revision)
    return result


def project(old, new):
    """`old` with the new revision and local-scoring settings if `new` only moves forward; else None."""
    from . import local_scoring_v3 as v3
    from .local_scoring_auxiliary import KEY as LOCAL, accepted_warmup
    if KEY not in new and KEY not in old: return old
    if KEY in old and KEY not in new: return None
    try:
        a, b = block(old), block(new)
    except (ValueError, KeyError, TypeError): return None
    forward = ((a['train_metrics_every'], b['train_metrics_every']) in ((1, 1), (1, 10), (10, 10))
               and not (a['validation'] and not b['validation']) and not (a['bootstrap'] and not b['bootstrap'])
               and not (a['value_weights']['vp'] > b['value_weights']['vp'])
               and not ('ownership_stop' in a and 'ownership_stop' not in b)
               and not ('search_auxiliary' in a and 'search_auxiliary' not in b))
    if not forward: return None
    la, lb = old.get(LOCAL), new.get(LOCAL)
    if la != lb:
        if la is None or lb is None: return None
        try:
            if v3.is_v3(lb) and not v3.is_v3(la):
                expected = v3.default(lb['replacement']['generation'], la, mode=la['mode'])
                ok = accepted_warmup(la) and la['mode'] == 'apply' and (lb == expected or v3.config_forward(expected, lb))
            else: ok = v3.config_forward(la, lb)
        except (ValueError, KeyError, TypeError): return None
        if not ok: return None
    result = copy.deepcopy(old)
    result[KEY] = copy.deepcopy(new[KEY])
    if lb is not None: result[LOCAL] = copy.deepcopy(lb)
    return result


# ---- settings read by the training code --------------------------------------------------------

def train_metrics_due(config, generation):
    return generation % block(config)['train_metrics_every'] == 0


def validation_fraction(config):
    v = block(config)['validation']
    return v['fraction'] if v else None


def bootstrap(config):
    return copy.deepcopy(block(config)['bootstrap'])


def value_weights(config):
    w = block(config)['value_weights']
    return (w['rank'], w['vp'])


def ownership_stop(config):
    """The era heads' own held-out CE early stop, or None (heads keep every attempted step)."""
    return copy.deepcopy(block(config).get('ownership_stop'))


def search_auxiliary(config):
    """The g3423 search-table batch for the era heads, or None."""
    return copy.deepcopy(block(config).get('search_auxiliary'))


def previous(config):
    """The recipe before any g3140 item: the revision key dropped and the replaced 2.0.0 companion restored."""
    from . import local_scoring_v3 as v3
    from .local_scoring_auxiliary import KEY as LOCAL
    if KEY not in config and not v3.is_v3(config.get(LOCAL)): return config
    result = copy.deepcopy(config); result.pop(KEY, None)
    if v3.is_v3(result.get(LOCAL)):
        result[LOCAL] = copy.deepcopy(result[LOCAL]['replacement']['previous'])
    return result


def changed(before, after):
    from .local_scoring_auxiliary import KEY as LOCAL
    return (before.get(KEY), before.get(LOCAL)) != (after.get(KEY), after.get(LOCAL))


def tag(config):
    """A filename tag naming every adopted item, so each adoption writes its own recipe file."""
    from . import local_scoring_v3 as v3
    from .local_scoring_auxiliary import KEY as LOCAL
    b = block(config); parts = []
    if b['train_metrics_every'] != 1: parts.append('scan10')
    if b['validation']: parts.append('newval25')
    local = config.get(LOCAL)
    if v3.is_v3(local):
        parts.append('eraheads-w' + str(local['weights']['canal']).replace('.', ''))
        if local['weights']['rail_end']: parts.append('railend')
    if b['bootstrap']: parts.append('blend50')
    if b['value_weights']['vp'] != OFF['value_weights']['vp']: parts.append('vp05')
    if 'ownership_stop' in b: parts.append('cestop16')
    if 'search_auxiliary' in b: parts.append('searchaux')
    return '-'.join(parts) or 'none'


def summary(config):
    from .local_scoring_auxiliary import KEY as LOCAL
    local = config.get(LOCAL, {})
    return dict(block(config), local_scoring=dict(contract=local.get('contract'), weights=local.get('weights')))
