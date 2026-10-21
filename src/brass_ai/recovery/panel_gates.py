"""One-way retirement of diagnostic panels; main PL evidence stays independent."""
import copy
import math

from brass_ai.bank import read, sha
from brass_ai.learning.contracts import digest, project_path, require

CONTRACT = 'reference-panel-gates/1.0.0'
MIGRATION = 'reference-panels-and-il-pfsp/1.0.0'
DEFAULT = dict(contract=CONTRACT, win_share_at_least=.9, il_history_sampling='pfsp')
NAMES = ('v4-panel', 'il-panel')


def validate(config):
    require(type(config) is dict and config == DEFAULT, 'Invalid reference-panel gates')
    return config


def configure(config):
    """Explicit signed continuation; no reset of training or opponent statistics."""
    from .v4_share import configure_90
    result = configure_90(config)
    result['panel_gates'] = copy.deepcopy(DEFAULT)
    if 'entropy_control' in result:
        result['entropy_control']['deployment_temperature_check']['enabled'] = False
    return result


def initial(generation):
    return dict(contract=CONTRACT, origin_generation=generation, last_generation=None,
                retired={name: None for name in NAMES})


def validate_trigger(trigger):
    require(type(trigger) is dict and set(trigger) == {'generation', 'candidate', 'games', 'setups',
        'win_share', 'panels', 'spec', 'reference', 'threshold'} and
        type(trigger['generation']) is int and trigger['generation'] > 0 and
        trigger['candidate'] == f'generation-{trigger["generation"]:06d}' and
        type(trigger['games']) is int and type(trigger['setups']) is int and
        trigger['games'] == 4 * trigger['setups'] and trigger['setups'] > 0 and
        type(trigger['win_share']) in (int, float) and math.isfinite(trigger['win_share']) and
        trigger['threshold'] == .9 and .9 <= trigger['win_share'] <= 1.,
        'Invalid retired-panel trigger')
    for name in ('panels', 'spec'):
        ref = trigger[name]
        require(type(ref) is dict and set(ref) == {'path', 'sha256'} and
            type(ref['path']) is str and type(ref['sha256']) is str and len(ref['sha256']) == 64,
            'Invalid retired-panel receipt')
    require(type(trigger['reference']) is str and len(trigger['reference']) == 64,
            'Invalid retired-panel reference identity')
    return trigger


def validate_state(saved, generation):
    require(type(saved) is dict and set(saved) == set(initial(0)) and saved['contract'] == CONTRACT
        and type(saved['origin_generation']) is int and 0 <= saved['origin_generation'] <= generation
        and set(saved['retired']) == set(NAMES), 'Invalid panel-gate state')
    last = saved['last_generation']
    require(last is None or type(last) is int and 0 < last <= generation, 'Invalid panel-gate progress')
    for trigger in saved['retired'].values():
        if trigger is not None:
            validate_trigger(trigger)
            require(last is not None and trigger['generation'] <= last, 'Panel trigger exceeds committed progress')
    return saved


def reference(rating, name):
    from .plackett_luce import IL, V4
    agent = rating['agents'][IL if name == 'il-panel' else V4]
    # Cache path changes do not change the frozen policy identity.
    return digest(dict(model_id=agent['checkpoint']['model_id'])) if name == 'il-panel' else digest(agent)


def verified_panel(root, rating, slot):
    contents = {}
    for key in ('panels', 'spec'):
        path = project_path(root, slot[key]['path'])
        require(sha(path) == slot[key]['sha256'], 'Panel gate receipt changed')
        contents[key] = read(path)
    spec, panels = contents['spec'], contents['panels']
    require(spec['generation'] == slot['generation'] and spec['config'] == rating['config'] and
        panels['candidate'] == spec['candidate'] == f'generation-{slot["generation"]:06d}',
        'Panel gate candidate/protocol differs')
    from .plackett_luce import IL, V4
    for name in NAMES:
        value = panels['panels'][name]
        if value['games'] == 0:
            require(value.get('status') == 'retired' and value['learner_win_share'] is None,
                    'Missing panel must be explicitly retired')
            require(not any(r['kind']==name for r in spec['lineups']), 'Retired panel has scheduled games')
            continue
        expected = [spec['candidate'], V4, V4, V4] if name == 'v4-panel' else [spec['candidate']] * 2 + [IL] * 2
        lineups = [r for r in spec['lineups'] if r['kind'] == name]
        require(value['games'] == rating['config']['panel_setups'] * 4 == len(lineups) and
            value['setups'] == rating['config']['panel_setups'] and
            all(r['agents'] == expected for r in lineups) and
            {(r['seed'], r['seat']) for r in lineups} ==
                {(seed, seat) for seed in rating['seeds'][:rating['config']['panel_setups']] for seat in range(4)},
            'Panel gate needs complete rotated panel support')
        key = V4 if name == 'v4-panel' else IL
        require(spec['agents'][key] == rating['agents'][key] and
            (name != 'v4-panel' or spec['agents'][key]['policy'].get('search_free') is True),
            'Panel gate frozen reference differs')
        require(type(value['learner_win_share']) in (int, float) and
            math.isfinite(value['learner_win_share']) and 0 <= value['learner_win_share'] <= 1,
            'Invalid panel win share')
    return panels


def verify_retired(root, rating, saved):
    for name, trigger in saved['retired'].items():
        if trigger is not None: verify_evidence(root, rating, name, trigger)


def verify_evidence(root, rating, name, trigger):
    """Verify a frozen panel observation, including after full-slot retention."""
    from .ratings import archived_history
    require(name in NAMES and trigger['reference'] == reference(rating, name), 'Retired panel reference changed')
    slot = next((s for s in rating['slots'] if s['generation'] == trigger['generation']), None)
    if slot is not None:
        require(all(trigger[k] == slot[k] for k in ('panels', 'spec')), 'Retired panel slot differs')
        panels = verified_panel(root, rating, slot)
    else:
        archive = archived_history(root, rating)
        slot = next((s for s in archive['slots'] if s['generation'] == trigger['generation']), None)
        require(slot is not None and all(trigger[k]['sha256'] == slot['original_sha256'][k]
            for k in ('panels', 'spec')), 'Retired panel archive receipt differs')
        panels = slot['panels']
    value = panels['panels'][name]
    require(panels['candidate'] == trigger['candidate'] and value['games'] == trigger['games'] and
        value['setups'] == trigger['setups'] and value['learner_win_share'] == trigger['win_share'],
        'Retired panel evidence differs')


def latest_evidence(root, state, config):
    """Latest complete active panels, or the verified one-way retirement receipt."""
    rating=state['ratings']; saved=prepare(root,state,config)
    result=copy.deepcopy(saved['retired'])
    if not rating['slots']: return result
    slot=max(rating['slots'],key=lambda s:s['generation'])
    panels=verified_panel(root,rating,slot)
    for name in NAMES:
        value=panels['panels'][name]
        if result[name] is None and value['games']:
            result[name]=dict(generation=slot['generation'],candidate=panels['candidate'],
                games=value['games'],setups=value['setups'],win_share=value['learner_win_share'],
                panels=copy.deepcopy(slot['panels']),spec=copy.deepcopy(slot['spec']),
                reference=reference(rating,name))
    return result


def prepare(root, state, config):
    if 'panel_gates' not in config: return None
    validate(config['panel_gates'])
    rating = state['ratings']; generation = state['iteration']
    saved = copy.deepcopy(rating.get('panel_gates', initial(generation)))
    validate_state(saved, generation)
    verify_retired(root, rating, saved)
    slots = [s for s in rating['slots'] if s['generation'] <= generation]
    if not slots: return saved
    slot = max(slots, key=lambda s: s['generation'])
    require(saved['last_generation'] is None or saved['last_generation'] <= slot['generation'],
            'Panel-gate rating progress went backwards')
    if saved['last_generation'] == slot['generation']: return saved
    # Adoption uses the latest completed panel, never a historical peak.
    panels = verified_panel(root, rating, slot)
    for name in NAMES:
        value = panels['panels'][name]
        if value['games']==0:
            require(saved['retired'][name] is not None and value.get('trigger')==saved['retired'][name],
                    'Retired panel lost its committed gate')
        if saved['retired'][name] is None and value['games'] and value['learner_win_share'] >= .9:
            saved['retired'][name] = dict(generation=slot['generation'], candidate=panels['candidate'],
                games=value['games'], setups=value['setups'], win_share=value['learner_win_share'],
                panels=copy.deepcopy(slot['panels']), spec=copy.deepcopy(slot['spec']),
                reference=reference(rating, name), threshold=.9)
    saved['last_generation'] = slot['generation']
    return validate_state(saved, generation)


def il_gate(state, config):
    if 'panel_gates' not in config: return None
    saved = validate_state(state['ratings']['panel_gates'], state['iteration'])
    return copy.deepcopy(saved['retired']['il-panel'])


def status(saved):
    return {name: dict(status='retired' if value else 'active', trigger=copy.deepcopy(value))
            for name, value in saved['retired'].items()} if saved else {}
