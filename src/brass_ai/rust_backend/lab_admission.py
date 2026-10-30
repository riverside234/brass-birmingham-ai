"""B03: explicit native family -> ordinary replay admission.

The original family, frozen job and recorder identities are retained verbatim.
A collector cannot self-admit: readers require the separate writer receipt that
the orchestrator binds to its frozen plan. Diagnostic collections never train.
"""
from __future__ import annotations

import copy
import gzip
import json
import os
from pathlib import Path
import random
import uuid

from brass_ai.bank import sha
from brass_ai.learning.contracts import compatibility, digest, keys, require
from brass_ai.recovery.replay import check_game
from .lab_collection import frozen_id, route
from .lab_contract import METHOD, SOURCE
from .lab_families import validate_family

FORMAT = 'four-seat-native-replay/1.0.0'
CONTRACT = 'brass-native-admission/1.0.0'
SCORER = 'brass-native-scoring-producer/1.0.0'
LIMIT = 512 * 1024**2


def hash64(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def scorer(producer):
    return dict(contract=SCORER, core_source=SOURCE, build_sha256=producer['build_sha256'],
                b01_sha256=producer['b01_sha256'], method=METHOD, v205_enabled=False)


def validate_scorer(value):
    keys(value, ('contract', 'core_source', 'build_sha256', 'b01_sha256', 'method', 'v205_enabled'))
    require(value['contract'] == SCORER and value['core_source'] == SOURCE
            and value['method'] == METHOD and value['v205_enabled'] is False
            and all(hash64(value[k]) for k in ('build_sha256', 'b01_sha256')), 'Unadmitted native scoring producer')
    return value


def binding(job, producer, *, generation, model_id, config, scope='ordinary'):
    """Called with scheduler-owned inputs, never with a family's claimed job."""
    from brass_ai.recovery.utility import reward_spec
    keys(producer, ('build_sha256', 'b01_sha256', 'signature', 'profile', 'v205_enabled'))
    require(producer['profile'] == METHOD and producer['v205_enabled'] is False
            and all(hash64(producer[k]) for k in ('build_sha256', 'b01_sha256', 'signature')),
            'Unadmitted collection producer/search method')
    proof = job['provenance']
    require(proof['purpose'] == 'training' and proof['kind'] in ('selfplay', 'historical', 'legacy')
            and proof['spec'] == producer['signature'] and proof['learner_behavior']['model_id'] == model_id,
            'Frozen ordinary job/signature/learner differs')
    require(type(generation) is int and generation >= 0 and scope in ('ordinary', 'diagnostic'), 'Invalid collection scope')
    require(all(p['kind'] in ('neural', 'legacy', 'lab-legacy') for p in job['policies'])
            and all(p['kind'] == 'neural' or p.get('search_free') is True for p in job['policies']),
            'Search/teacher/unsupported policy in ordinary admission')
    require(bool('critic_table' in proof) == bool('critic_data' in config), 'Missing or unexpected critic table')
    require(proof.get('ppo_opening_objective') == config['update'].get('opening_objective'), 'Opening objective differs')
    local = config.get('local_scoring_auxiliary', {}).get('mode', 'off') != 'off'
    require(not local or 'local_scoring' in proof, 'Enabled auxiliary collection lacks recorder binding')
    value = dict(contract=CONTRACT, compatibility=digest(compatibility()), core_source=SOURCE,
        generation=generation, model_id=model_id, reward_id=digest(reward_spec(config)),
        producer=copy.deepcopy(producer), scope=scope, auxiliary=local,
        job=copy.deepcopy({k: v for k, v in job.items() if k != 'path'}), job_id=frozen_id(job))
    value['sha256'] = digest(value)
    return value


def validate_binding(value):
    keys(value, ('contract', 'compatibility', 'core_source', 'generation', 'model_id', 'reward_id',
                 'producer', 'scope', 'auxiliary', 'job', 'job_id', 'sha256'))
    require(value['contract'] == CONTRACT and value['compatibility'] == digest(compatibility())
            and value['core_source'] == SOURCE and value['sha256'] == digest({k: v for k, v in value.items() if k != 'sha256'})
            and value['job_id'] == frozen_id(value['job']) and value['scope'] in ('ordinary', 'diagnostic'),
            'Changed native admission binding')
    p = value['producer']
    keys(p, ('build_sha256', 'b01_sha256', 'signature', 'profile', 'v205_enabled'))
    require(p['profile'] == METHOD and p['v205_enabled'] is False
            and all(hash64(p[k]) for k in ('build_sha256', 'b01_sha256', 'signature')),
            'Foreign producer method/source')
    proof = value['job']['provenance']
    require(proof['purpose'] == 'training' and proof['kind'] in ('selfplay', 'historical', 'legacy')
            and proof['spec'] == p['signature'] and proof['learner_behavior']['model_id'] == value['model_id']
            and type(value['auxiliary']) is bool and type(value['generation']) is int and value['generation'] >= 0
            and hash64(value['reward_id']), 'Invalid frozen admission metadata')
    require(all(s['kind'] in ('neural', 'legacy', 'lab-legacy') and
                (s['kind'] == 'neural' or s.get('search_free') is True) for s in value['job']['policies']),
            'Search/teacher policy cannot enter ordinary replay')
    return value


def branch_trigger(family, job, diagnostic):
    """The diagnostic first-eligible override is never a production sample."""
    parent = family['parent']; child = family['child']; config = job.get('branching')
    require(diagnostic or not parent.get('diagnostic_branch_probe', False), 'Diagnostic branch override cannot train')
    if not config:
        require(child is None, 'Unexpected branch without a frozen schedule'); return
    if diagnostic: return  # Alternative, causal and opening checks still run in validate_family.
    from brass_ai.recovery.branching import eligible, seed_for, alternative
    from brass_ai.step5.policy import exploration_distribution
    expected = None
    for row, evidence in zip(parent['game']['records'], parent['evidence'], strict=True):
        if not eligible(row['view'], row['choices'], row['behavior'].get('action_decision', 0), config['skip_opening_decisions']):
            continue
        seed = seed_for(parent['game']['game_id'], row['step'], config['contract'])
        if random.Random('select:' + seed).random() >= config['rate_per_decision']: continue
        import numpy as np
        q = exploration_distribution(row['choices'], row['intent'], 3, 'action')
        if alternative(np.asarray(evidence['raw_probabilities']), q, row['chosen'], random.Random(seed), contract=config['contract']) is not None:
            expected = row['step']; break
    require((None if child is None else child['game']['branch']['decision']) == expected, 'Frozen branch trigger differs')


def convert(family, admission, *, allow_diagnostic=False, arrays=None):
    admission = validate_binding(admission)
    require(admission['scope'] == 'ordinary' or allow_diagnostic, 'Diagnostic replay is not training data')
    job = admission['job']; proof = job['provenance']; producer = admission['producer']
    checked=validate_family(family, job, producer,arrays=arrays,return_games=True)
    branch_trigger(family, job, admission['scope'] == 'diagnostic')
    expected_route = route(job)[0]
    games = []
    for part, original in zip((family['parent'], family['child']),checked):
        if part is None: continue
        native = part['producer']['contract'] == 'brass-lab-collection/1.0.0'
        require(native == (expected_route == 'rust'), 'Frozen whole-table route changed')
        game = copy.deepcopy(dict(part['game']))
        require(all(row['teacher'] is None for row in game['records']), 'Teacher/search labels cannot enter ordinary replay')
        require('branches' not in game, 'Family contains an unsealed nested child')
        if native and admission['auxiliary']:
            from brass_ai.recovery.local_scoring_targets import bind, FIELD
            require(FIELD not in game, 'Native family cannot supply its own scoring envelope')
            game[FIELD] = bind(copy.deepcopy(part['scoring']), game, scorer(producer))
        elif admission['auxiliary']:
            from brass_ai.recovery.local_scoring_targets import FIELD, identity
            local = proof['local_scoring']
            require(all(local.get(k) == v for k, v in identity().items()) and FIELD in game
                    and game[FIELD]['producer'] == local['producer']
                    and part['producer']['runtime']['sha256'] == local['producer']['runtime_sha256'],
                    'Reference scoring provenance differs')
        game = check_game(game,[original.encoded(r,None) for r in original['records']],validated_inputs=True)
        if 'critic_table' in proof:
            from brass_ai.recovery.critic_data import check_table
            table = copy.deepcopy(proof['critic_table'])
            require(table['generation'] == admission['generation'], 'Stale critic descriptor generation')
            check_table(table, game, proof['learner_behavior']['policy_id'])
            for seat, record in enumerate(table['seats']):
                spec = job['policies'][(seat - job.get('rotation', 0)) % 4]
                require(record['model_id'] == spec.get('model_id'), 'Descriptor model attribution differs')
            game.critic_table = table; game.table_kind = proof['kind']
        if 'ppo_opening_objective' in proof:
            from brass_ai.recovery.opening_objective import validate
            game.ppo_opening_objective = copy.deepcopy(validate(proof['ppo_opening_objective']))
        game.native_admission = copy.deepcopy(admission)
        if 'diversity_assignment' in proof:
            from brass_ai.recovery.domino_sampling import bind
            bind(game,proof)
        games.append(game)
    parent = games[0]
    if len(games) == 2:
        arrays = [parent.encoded(r, None) for r in parent['records']]
        parent['branches'] = [games[1]]
        parent.bind(arrays)
    return parent


def publish(path, family, admission, *, cache=True):
    """Exclusive, one-family publication. Retrying requires the identical receipt."""
    game = convert(family, admission, allow_diagnostic=True)
    path = Path(path)
    header = dict(format=FORMAT, compatibility=compatibility(), admission=copy.deepcopy(admission))
    value = (json.dumps(header, separators=(',', ':'), allow_nan=False) + '\n' +
             json.dumps(family, separators=(',', ':'), allow_nan=False) + '\n').encode()
    require(len(value) <= LIMIT, 'Native replay exceeds bound')
    path.parent.mkdir(parents=True, exist_ok=True)
    receipt = dict(contract=CONTRACT, admission=copy.deepcopy(admission), family_sha256=family['sha256'])
    if path.exists():
        receipt['replay_sha256'] = sha(path)
        read(path, receipt, allow_diagnostic=True)
        if cache:
            from brass_ai.recovery.replay import write_cache,cache_receipt
            write_cache(path,[game]+game.get('branches',[]));receipt['features']=cache_receipt(path)
        return record(path, game, admission, receipt)
    temp = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        with temp.open('xb') as raw:
            with gzip.GzipFile(fileobj=raw, mode='wb', mtime=0, compresslevel=1) as stream: stream.write(value)
            raw.flush(); os.fsync(raw.fileno())
        # Windows rename is no-replace; Linux hard-link provides the same
        # atomic publication rule. A concurrent writer never gets overwritten.
        if os.name == 'nt': temp.rename(path)
        else: os.link(temp, path)
    finally:
        temp.unlink(missing_ok=True)
    receipt['replay_sha256'] = sha(path)
    if cache:
        from brass_ai.recovery.replay import write_cache,cache_receipt
        write_cache(path,[game]+game.get('branches',[]));receipt['features']=cache_receipt(path)
    return record(path, game, admission, receipt)


def record(path, game, admission, receipt):
    return dict(job=admission['job']['index'], game_id=game['game_id'], path=str(path),
                sha256=receipt['replay_sha256'], features=receipt.get('features'), native_admission=receipt)


def read(path, receipt, *, allow_diagnostic=False):
    require(type(receipt) is dict and receipt.get('contract') == CONTRACT and receipt.get('replay_sha256') == sha(path),
            'Native replay requires its independently bound writer receipt')
    with gzip.open(path, 'rb') as stream: data = stream.read(LIMIT + 1)
    require(len(data) <= LIMIT, 'Native replay exceeds bound')
    lines = data.splitlines(); require(len(lines) == 2, 'Incomplete/extra native replay family')
    header, family = map(json.loads, lines)
    keys(header, ('format', 'compatibility', 'admission'))
    require(header['format'] == FORMAT and header['compatibility'] == compatibility()
            and header['admission'] == receipt['admission'] and family['sha256'] == receipt['family_sha256'],
            'Native replay/writer receipt differs')
    from brass_ai.recovery.replay import load_cache
    games=[family['parent']['game']]+([family['child']['game']] if family['child'] is not None else [])
    arrays=load_cache(path,games,receipt.get('features'))
    return convert(family, receipt['admission'], allow_diagnostic=allow_diagnostic,arrays=arrays)


def admit_teacher_rows(rows, *, job, model, history):
    """Separate B02.5 contract; never returns ordinary games or actor samples."""
    from brass_ai.recovery.teacher_labels import admit_rows
    from .lab_teacher_overlay import source_fingerprint
    sources = {SOURCE, source_fingerprint()}
    require(all(not row.get('diagnostic',False) for row in rows), 'Calibration probes cannot enter the teacher buffer')
    require(job.get('kind') == 'teacher_phase' and job.get('ordinary_training') is False
            and job['search']['method'] == METHOD and job['search']['v205_enabled'] is False
            and job['expected']['source_fingerprint'] in sources, 'Unadmitted teacher job/producer')
    return admit_rows(rows, model=model, expected=job['expected'], history=history, admitted_sources=sources)
