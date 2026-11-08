"""Ordinary collector operational checks; never consumes a training RNG."""
from brass_ai.learning.contracts import digest, require
from brass_ai.recovery.collector_runtime import MODES

CONTRACT = 'brass-collector-validation/1'


def mode():
    from brass_ai.recovery.collector_runtime import option
    return option('collector_validation')


def selection(job, *, ordinal=0, first=False, selected_mode=None):
    """Private scheduler decision, outside the frozen job and its identity."""
    from .lab_collection import frozen_id
    selected_mode = selected_mode or mode()
    require(selected_mode in MODES and type(ordinal) is int and ordinal >= 0,
            'Invalid collector validation selection')
    identity = frozen_id(job)
    branch = int(digest(dict(contract=CONTRACT, job=identity, kind='branch'))[:16], 16) % 16 == 0
    full = selected_mode == 'strict' or first or ordinal % 64 == 0
    return dict(contract=CONTRACT, mode=selected_mode, job=identity, ordinal=ordinal,
                first=bool(first), semantic=full, branch=full or branch,
                family_every=64, branch_every=16)


def plan(jobs, *, selected_mode=None):
    """Stable frozen order, independent of completion order/process count."""
    seen = set(); result = {}
    for ordinal, job in enumerate(jobs):
        stratum = (job['provenance']['kind'], tuple(sorted({p['kind'] for p in job['policies']})))
        result[job['index']] = selection(job, ordinal=ordinal, first=stratum not in seen,
                                         selected_mode=selected_mode)
        seen.add(stratum)
    return result


def check(value, job):
    require(type(value) is dict and set(value) == set(selection(job)), 'Invalid validation coverage')
    require(value == selection(job, ordinal=value['ordinal'], first=value['first'], selected_mode=value['mode']),
            'Validation selection binding changed')
    return value


def coverage(value, *, child=False):
    return dict(policy=value, structural_admitted=True, semantic_audited=value['semantic'],
                branch_audited=bool(child and value['branch']))


def admit(view, choices):
    """Typed qualified Rust producer; schema deep checks belong to the audit."""
    from brass_ai.recovery.admitted_inputs import freeze
    require(type(view['seat']) is int and 0 <= view['seat'] < 4 and
            type(view['context']['revision']) is int and view['context']['revision'] >= 0 and
            0 < len(choices) <= 65536 and all(type(c.get('id')) is str and c['id'] for c in choices),
            'Native decision binding differs')
    import hashlib,json
    def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
    v,c=canonical(view),canonical(choices);hashed=hashlib.sha256()
    for part in (b'{"choices":',c,b',"view":',v,b'}'):hashed.update(part)
    from brass_ai.recovery.replay_binding import count
    count('live_input_hash_executed');count('canonical_input_json_executed')
    return freeze(view, choices, hashed.hexdigest(),canonical=(v,c))


def structural_game(game, arrays):
    """Minimal independent admission for exact, already checked native arrays."""
    import math
    from brass_ai.recovery.replay import CheckedGame
    from brass_ai.learning.replay import split_for, terminal_targets
    require(game['players']==4 and len(game['opponents'])==4 and
            game['split']==split_for(game['setup_group']) and len(game['records'])==len(arrays)>0,
            'Structural game coverage differs')
    terminal_targets(game['result']);previous=-1
    for step,(row,e) in enumerate(zip(game['records'],arrays,strict=True)):
        choices=row['choices'];seat=row['seat'];b=row['behavior'];revision=row['view']['context']['revision']
        require(type(seat) is int and 0<=seat<4 and row['view']['seat']==seat and row['step']==step
                and type(revision) is int and revision>previous,'Structural row identity differs')
        previous=revision
        require(type(row['chosen']) is int and 0<=row['chosen']<len(choices) and row['teacher'] is None,
                'Structural chosen/teacher differs')
        require(b['kind'] in ('model','random','external') and b['policy_id']==game['opponents'][seat] and
                math.isfinite(b['probability']) and 0<b['probability']<=1 and
                math.isfinite(row['seconds']) and row['seconds']>=0 and math.isfinite(b['temperature']) and b['temperature']>=0,
                'Structural behavior differs')
        require(len(choices)>1 or (row['chosen']==0 and b['probability']==1.),'Forced choice differs')
        require(e.state.shape==(3128,) and e.actions.shape==(len(choices),327) and
                e.choice_ids==tuple(c['id'] for c in choices) and len(set(e.choice_ids))==len(e.choice_ids),
                'Structural feature choices/shape differs')
    if 'branch' not in game:require({r['seat'] for r in game['records']}==set(range(4)),'Missing seat trajectories')
    return CheckedGame(game).bind(arrays)
