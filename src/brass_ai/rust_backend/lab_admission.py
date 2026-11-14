"""B03: explicit native family -> ordinary replay admission.

The original family, frozen job and recorder identities are retained verbatim.
A collector cannot self-admit: readers require the separate writer receipt that
the orchestrator binds to its frozen plan. Diagnostic collections never train.
"""
from __future__ import annotations

import copy
from functools import lru_cache
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import random
import time
import uuid

from brass_ai.bank import sha
from brass_ai.learning.contracts import compatibility, digest, keys, require
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
    from brass_ai.recovery import ordinary_cotton
    cotton = (ordinary_cotton.planned_assignment(config['table_plan'][ordinary_cotton.KEY],
                                                proof['ordinary_game_index'], job['sampling_seed'])
              if ordinary_cotton.enabled(config) and proof['kind'] == 'selfplay' else None)
    require(proof.get(ordinary_cotton.KEY) == cotton and (cotton is None or
            (proof['sampling_seed'] == job['sampling_seed'] and not job.get('branching') and
             all(p.get('exploration_version') == 3 and
            p['policy_id'] == proof['learner_behavior']['policy_id'] for p in job['policies']))),
            'Ordinary cotton job differs from the table plan')
    require(type(generation) is int and generation >= 0 and scope in ('ordinary', 'diagnostic'), 'Invalid collection scope')
    require(all(p['kind'] in ('neural', 'legacy', 'lab-legacy') for p in job['policies'])
            and all(p['kind'] == 'neural' or p.get('search_free') is True for p in job['policies']),
            'Search/teacher/unsupported policy in ordinary admission')
    require(bool('critic_table' in proof) == bool('critic_data' in config), 'Missing or unexpected critic table')
    require(proof.get('ppo_opening_objective') == config['update'].get('opening_objective'), 'Opening objective differs')
    require(proof.get('seat_contexts') == config.get('seat_contexts'), 'Seat-context contract differs')
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


def convert(family, admission, *, allow_diagnostic=False, arrays=None, _owned=False, _coverage=None, _owned_seal=False):
    admission = validate_binding(admission)
    require(admission['scope'] == 'ordinary' or allow_diagnostic, 'Diagnostic replay is not training data')
    job = admission['job']; proof = job['provenance']; producer = admission['producer']
    checked=validate_family(family, job, producer,arrays=arrays,return_games=True,
        semantic=_coverage is None or _coverage['semantic_audited'],
        branch_audit=_coverage is None or _coverage['branch_audited'],owned_seal=_owned and (_owned_seal or _coverage is not None))
    if _coverage is None or _coverage['semantic_audited'] or _coverage['branch_audited']:
        branch_trigger(family, job, admission['scope'] == 'diagnostic')
    return _assemble(family, admission, checked=checked,owned=_owned)


def _assemble(family, admission, *, checked=None, arrays=None, owned=False):
    """Reconstruct admitted games; arrays-only callers require the writer seal.

    This private path is used only after read() verifies the independently held
    validation receipt, authoritative replay bytes and every feature-cache byte.
    It does not accept a family's claim that it has already been validated.
    """
    from brass_ai.recovery.replay import CheckedGame
    job=admission['job'];proof=job['provenance'];producer=admission['producer']
    expected_route = route(job)[0]
    games = [];offset=0
    for index,part in enumerate((family['parent'], family['child'])):
        if part is None: continue
        native = part['producer']['contract'] == 'brass-lab-collection/1.0.0'
        legacy_archive=(not native and any(s['kind'] in ('legacy','lab-legacy') for s in job['policies']))
        require(native == (expected_route == 'rust') or legacy_archive, 'Frozen whole-table route changed')
        # read() owns freshly decoded JSON; publish() uses the game only inside
        # this synchronous call and returns a compact receipt. Copying every
        # board/history adds no isolation on either private path. Public
        # convert() still detaches input; all paths retain row mutation seals.
        game = dict(part['game']) if owned else copy.deepcopy(dict(part['game']))
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
        if checked is None:
            require(arrays is not None,'Missing independently attested features')
            from .lab_features import bind as bind_features
            count=len(game['records']);game=bind_features(game,arrays,start=offset);offset+=count
        else:
            original=checked[index]
            # validate_family already fully checked these exact rows, behavior,
            # features and scoring. Only the game-level scoring envelope above
            # was added (and independently validated by bind). Reuse the row
            # seals instead of walking/validating and hashing every row again.
            game=CheckedGame(game)
            game.features=original.features if owned else dict(original.features)
            game.seals=original.seals if owned else copy.deepcopy(original.seals)
            game.header_seal=digest({k:v for k,v in game.items() if k!='records'})
        if 'seat_contexts' in proof:
            from brass_ai.recovery.neutral_training import check_rows
            check_rows(game,proof)
        from brass_ai.recovery import ordinary_cotton
        if ordinary_cotton.KEY in proof:
            game.cotton_seats, _ = ordinary_cotton.validate_assignment(proof)
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
    require(checked is not None or offset==len(arrays),'Extra attested native features')
    parent = games[0]
    if len(games) == 2:
        parent['branches'] = [games[1]]
        # Branch attachment changes only the parent header, not its rows.
        parent.header_seal=digest({k:v for k,v in parent.items() if k!='records'})
    return parent


def publish(path, family, admission, *, cache=True, validation_receipt=False, timings=None, coverage=None,
            _owned_family=False, _published_game=None):
    """Exclusive, one-family publication. Retrying requires the identical receipt."""
    began=time.perf_counter();timings={} if timings is None else timings
    if coverage is not None:
        from . import lab_perf
        lab_perf.check(coverage['policy'],admission['job'])
        require(coverage==lab_perf.coverage(coverage['policy'],child=family['child'] is not None),'Validation coverage differs')
    game = convert(family, admission, allow_diagnostic=True,_owned=True,_coverage=coverage,_owned_seal=_owned_family)
    from brass_ai.recovery.replay_binding import enabled
    owned_pack=_owned_family and enabled()
    timings['full_validation_and_assembly_seconds']=time.perf_counter()-began;began=time.perf_counter()
    path = Path(path)
    header = dict(format=FORMAT, compatibility=compatibility(), admission=copy.deepcopy(admission))
    # One canonical family encoding serves its original digest AND durable
    # replay bytes. Append sha256 outside that hashed object, preserving the
    # original digest formula and JSON replay format.
    canonical=json.dumps({k:v for k,v in family.items() if k!='sha256'},sort_keys=True,
                         separators=(',',':'),allow_nan=False).encode()
    encoded_header=json.dumps(header,separators=(',',':'),allow_nan=False).encode()
    timings['json_encoding_seconds']=time.perf_counter()-began
    began=time.perf_counter();family_sha=hashlib.sha256(canonical).hexdigest()
    if _owned_family:family['sha256']=family_sha
    else:require(family['sha256']==family_sha,'Family changed during publication')
    timings['family_seal_seconds']=time.perf_counter()-began
    suffix=b',"sha256":'+json.dumps(family_sha).encode()+b'}\n'
    require(len(encoded_header)+1+len(canonical)-1+len(suffix)<=LIMIT,'Native replay exceeds bound')
    from brass_ai.recovery.replay_binding import count
    count('family_json_encoding_executed');count('replay_family_encoding_reused')
    path.parent.mkdir(parents=True, exist_ok=True)
    receipt = dict(contract=CONTRACT, admission=copy.deepcopy(admission), family_sha256=family['sha256'])
    if path.exists():
        receipt['replay_sha256'] = sha(path)
        read(path, receipt, allow_diagnostic=True)
        if cache:
            from . import lab_features
            receipt['features']=lab_features.write(path,[game]+game.get('branches',[]),receipt['replay_sha256'],_owned=owned_pack)
        if validation_receipt:_attest(receipt,coverage=coverage)
        if _published_game is not None:_published_game(game)
        return record(path, game, admission, receipt)
    temp = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    began=time.perf_counter()
    buffer=io.BytesIO()
    with gzip.GzipFile(fileobj=buffer,mode='wb',compresslevel=1,mtime=0,filename='') as stream:
        stream.write(encoded_header);stream.write(b'\n');stream.write(memoryview(canonical)[:-1]);stream.write(suffix)
    packed=buffer.getvalue();del canonical
    replay_hash=hashlib.sha256(packed).hexdigest()
    try:
        with temp.open('xb') as raw:
            raw.write(packed)
            raw.flush(); os.fsync(raw.fileno())
        # Windows rename is no-replace; Linux hard-link provides the same
        # atomic publication rule. A concurrent writer never gets overwritten.
        if os.name == 'nt': temp.rename(path)
        else: os.link(temp, path)
    finally:
        temp.unlink(missing_ok=True)
    receipt['replay_sha256'] = replay_hash
    timings['replay_write_hash_seconds']=time.perf_counter()-began;began=time.perf_counter()
    if cache:
        from . import lab_features
        receipt['features']=lab_features.write(path,[game]+game.get('branches',[]),replay_hash,_owned=owned_pack)
    timings['feature_cache_seconds']=time.perf_counter()-began
    if validation_receipt:_attest(receipt,coverage=coverage)
    if _published_game is not None:_published_game(game)
    return record(path, game, admission, receipt)


def record(path, game, admission, receipt):
    return dict(job=admission['job']['index'], game_id=game['game_id'], path=str(path),
                sha256=receipt['replay_sha256'], features=receipt.get('features'), native_admission=receipt)


@lru_cache(maxsize=1)
def _validation_version():
    """Pin the loaded implementation once per immutable worker/reader process.

    Deployment/source admission requires a fresh process. Replay and feature
    bytes are never cached by stat/mtime; every read still hashes them in full.
    """
    from brass_ai.step5.league import implementation_id
    return implementation_id()


def _attest(receipt,*,coverage=None):
    # Issued only by the full validator, never by the native collector. The
    # scheduler/checkpoint retains this receipt separately from the replay.
    bound=dict(receipt,validation_coverage=coverage) if coverage else receipt
    receipt['validation']=dict(contract='brass-native-validation/2.0.0' if coverage else 'brass-native-validation/1.0.0',
        validator=_validation_version(),receipt_sha256=digest(bound),**(dict(coverage=coverage) if coverage else {}))


def read(path, receipt, *, allow_diagnostic=False, reuse_validation=True, fast_recovery=False):
    with Path(path).open('rb') as stream:
        size=os.fstat(stream.fileno()).st_size
        require(0<size<=LIMIT,'Native replay exceeds bound')
        packed=stream.read(size+1)
    require(len(packed)==size,'Native replay size changed')
    require(len(packed)<=LIMIT and type(receipt) is dict and receipt.get('contract') == CONTRACT,
            'Native replay requires its independently bound writer receipt')
    require(receipt.get('replay_sha256') == hashlib.sha256(packed).hexdigest(),
            'Native replay changed: independently bound writer receipt differs')
    with gzip.GzipFile(fileobj=io.BytesIO(packed),mode='rb') as stream: data = stream.read(LIMIT + 1)
    require(len(data) <= LIMIT, 'Native replay exceeds bound')
    lines = data.splitlines(); require(len(lines) == 2, 'Incomplete/extra native replay family')
    header, family = map(json.loads, lines)
    keys(header, ('format', 'compatibility', 'admission'))
    require(header['format'] == FORMAT and header['compatibility'] == compatibility()
            and header['admission'] == receipt['admission'] and family['sha256'] == receipt['family_sha256'],
            'Native replay/writer receipt differs')
    from . import lab_features
    games=[family['parent']['game']]+([family['child']['game']] if family['child'] is not None else [])
    validation=receipt.get('validation');trusted=False
    if validation is not None:
        v2=validation.get('contract')=='brass-native-validation/2.0.0'
        keys(validation,('contract','validator','receipt_sha256'),('coverage',) if v2 else ())
        body={k:v for k,v in receipt.items() if k!='validation'}
        if v2:body['validation_coverage']=validation['coverage']
        require(validation['contract'] in ('brass-native-validation/1.0.0','brass-native-validation/2.0.0') and
                validation['receipt_sha256']==digest(body),
                'Native validation receipt changed')
        if v2:
            from . import lab_perf
            coverage=validation['coverage'];lab_perf.check(coverage['policy'],receipt['admission']['job'])
            require(coverage==lab_perf.coverage(coverage['policy'],child=family['child'] is not None),'Validation coverage changed')
        trusted=reuse_validation and validation['validator']==_validation_version()
        if v2 and lab_perf.mode()=='strict' and not coverage['semantic_audited']:trusted=False
    # Exact replay + feature bytes were independently checked by this loaded
    # validator. Disable repeated finite/selector/seat/context deep scans on
    # that warm path; cold recovery and unversioned receipts still check all.
    arrays=lab_features.read(path,games,receipt.get('features'),receipt['replay_sha256'],deep=not trusted)
    if trusted and arrays is not None:
        admitted=validate_binding(receipt['admission'])
        require(admitted['scope']=='ordinary' or allow_diagnostic,'Diagnostic replay is not training data')
        from brass_ai.recovery.replay_binding import seal_game
        return seal_game(_assemble(family,admitted,arrays=arrays,owned=True),_owned=True)
    from .lab_replay_compat import reviewed_validator
    coverage=(validation['coverage'] if validation and validation.get('contract')=='brass-native-validation/2.0.0'
              and (validation['validator']==_validation_version() or reviewed_validator(validation['validator']))
              and fast_recovery and lab_perf.mode()!='strict' else None)
    from brass_ai.recovery.replay_binding import seal_game
    return seal_game(convert(family, receipt['admission'], allow_diagnostic=allow_diagnostic,arrays=arrays,_owned=True,_coverage=coverage),_owned=True)


def teacher_sources(jobs):
    """Bind a frozen collection once; cold callers establish their own boundary."""
    from .lab_teacher_overlay import source_fingerprint
    sources = {SOURCE, source_fingerprint()}
    require(all(job.get('kind') == 'teacher_phase' and job.get('ordinary_training') is False
                and job['search']['method'] == METHOD and job['search']['v205_enabled'] is False
                and job['expected']['source_fingerprint'] in sources for job in jobs), 'Unadmitted teacher job/producer')
    return sources


def admit_teacher_rows(rows, *, job, model, history):
    """Separate B02.5 contract; never returns ordinary games or actor samples."""
    from brass_ai.recovery.teacher_labels import admit_rows
    sources = teacher_sources([job])
    require(all(not row.get('diagnostic',False) for row in rows), 'Calibration probes cannot enter the teacher buffer')
    return admit_rows(rows, model=model, expected=job['expected'], history=history, admitted_sources=sources)
