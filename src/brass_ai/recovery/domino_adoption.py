"""Generation-bound adoption; archive old PFSP estimates and defer cache expiry."""
import copy
from brass_ai.learning.contracts import digest,require
from brass_ai.step5.league import write_json
from .domino_config import MIGRATION,enabled


def pool_transition(pool):
    from . import pfsp_pool,quality_pool
    pfsp_pool.validate_state(pool,pool['config'])
    updated=copy.deepcopy(pool)
    before={m['model_id']:{k:m[k] for k in ('S','C','alarm_streak')} for m in pool['members']}
    for m in updated['members']:m.update(pfsp_pool.prior(pool['config']))
    pfsp_pool.seal(updated)
    receipt=dict(contract=MIGRATION,from_digest=quality_pool.state_digest(pool),to_digest=quality_pool.state_digest(updated),
        origin=pool['generation'],games_done=pool['games_done'],archived_scores=before,
        scope='New uniform-context PFSP estimator with existing prior/half-life; initial weights change. Membership, admission ages, IL gate and old score backup retained.')
    return updated,receipt


def prepare(root,run,state,raw,config,phase_raw):
    if not enabled(config):return phase_raw,None
    from .domino_auxiliary import KEY
    if KEY in (raw or {}).get('training',{}):
        require(state.get('domino_state_digest')==digest(raw['training'][KEY]['state']), 'DOMiNO state/checkpoint mismatch')
        return phase_raw,None
    a=config['diversity']['adoption']
    if state['iteration']==0:
        require(a is None,'New DOMiNO run cannot use another run adoption')
        return phase_raw,None
    entry=state['history'][-1]
    require(a==dict(generation=state['iteration'],checkpoint_sha256=entry['sha256'],model_id=entry['model_id']),
        'DOMiNO adoption requires the explicitly bound full checkpoint')
    from .actor_value import KEY as PHASE_KEY
    old=raw['training'][PHASE_KEY]
    receipt=dict(contract=MIGRATION,**a,previous_ppg_origin=old.get('cache_origin'),previous_ppg_receipts=copy.deepcopy(old.get('rolling_buffer',[])),
        cache_expiry='Only after first new checkpoint and state commit; value Adam and cadence preserved')
    if 'domino_adoption' in state:require(state['domino_adoption']==receipt,'DOMiNO adoption cannot repeat')
    if 'rolling_buffer' not in old:return phase_raw,receipt
    rewritten=copy.copy(phase_raw);rewritten['training']=dict(phase_raw['training'])
    value=dict(phase_raw['training'][PHASE_KEY]);value.update(cache_origin=state['iteration'],rolling_buffer=[])
    rewritten['training'][PHASE_KEY]=value
    return rewritten,receipt


def archive(run,receipt):
    from brass_ai.bank import read
    from .retention import safe
    path=safe(run,run/f'domino-adoption-g{receipt["generation"]:06d}.json')
    require(not path.exists() or read(path)==receipt,'DOMiNO adoption archive changed')
    if not path.exists():write_json(path,receipt)
