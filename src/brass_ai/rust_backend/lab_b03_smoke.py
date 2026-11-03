"""Reuse sealed B02 games for B03 Python admission; collect no additional games."""
import copy
import gzip
import importlib.util
import json
from pathlib import Path

import numpy as np

from brass_ai.learning.contracts import SOURCE_ROOT, digest, require
from brass_ai.recovery.branching import expand
from brass_ai.recovery.critic_holdout import select
from brass_ai.recovery.replay import check_game
from brass_ai.recovery.updates import prepared
from brass_ai.recovery.utility import reward_spec
from brass_ai.step5.league import implementation_id
from .lab_contract import METHOD, SOURCE, pin, sha
from .lab_collection import frozen_id
from . import lab_admission as admission


def entry(path, run, device):
    path = Path(path).resolve(); receipt = json.loads(path.read_text(encoding='utf-8'))
    require(receipt.get('contract') == 'brass-lab-b02/2.0.0' and receipt.get('passed') is True
            and receipt.get('milestone') == 'B02' and receipt.get('method') == METHOD
            and receipt.get('v205_enabled') is False and receipt.get('search_enabled') is False
            and receipt.get('training_admitted') is False, 'Passing search_v1 B02 qualification required')
    for key in ('b01', 'build', 'smoke'):
        require(pin(receipt[key]['path']) == receipt[key], 'Changed B02 ' + key)
    spec = importlib.util.spec_from_file_location('b03_b02_entry', SOURCE_ROOT / 'scripts/qualify-rust-lab-b02.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    snapshot, _ = module.entry(receipt['b01']['path'], run, device)
    build = json.loads(Path(receipt['build']['path']).read_text(encoding='utf-8'))
    smoke = json.loads(Path(receipt['smoke']['path']).read_text(encoding='utf-8'))
    require(build['shared_core'] == SOURCE and build['b01'] == receipt['b01'] and build['search_method'] == METHOD
            and build.get('trainer_implementation') == implementation_id(),
            'B02 must run with the B03 checkout so its evidence binds this exact trainer')
    require(smoke.get('passed') is True and smoke.get('branch_exit_qualified') is True
            and not smoke['missing_table_kinds'] and 'frozen_jobs' in smoke and 'families' in smoke,
            'B02 lacks sealed admission inputs; rerun the bounded B02 qualification with the B03 update')
    for item in (smoke['frozen_jobs'], *smoke['families'], build['collector'], build['bootstrap']['binary'], *build['bootstrap']['dependencies']):
        require(pin(item['path']) == item, 'Changed B02 admission/runtime input: ' + item['path'])
    jobs = json.loads(Path(smoke['frozen_jobs']['path']).read_text(encoding='utf-8'))['jobs']
    require(len(jobs) == len(smoke['families']) and len({frozen_id(j) for j in jobs}) == len(jobs), 'Missing/duplicate frozen B02 jobs')
    producer = dict(build_sha256=receipt['build']['sha256'], b01_sha256=receipt['b01']['sha256'],
                    signature=snapshot['signature_sha256'], profile=METHOD, v205_enabled=False)
    return snapshot, jobs, smoke['families'], producer


def prepared_rows(games, config, learner):
    result = []
    for game in expand(games):
        if hasattr(game, 'critic_table'):
            seats = {s for s, p in enumerate(game.critic_table['seats']) if p['role'] in ('learner', 'snapshot')}
            choose = lambda r: r['seat'] in seats
        else: choose = lambda r: r['behavior']['policy_id'] == learner
        result += prepared([game], choose, insolvency=config.get('insolvency'), reward=reward_spec(config))
    return result


def compare_preparation(game, family, config, learner):
    """B02 already proved these fixed views/maps equal its C# oracle.

    Exercise legacy versus native Python preparation on that evidence, retaining
    the original receipts. This is not a second native rules-parity measurement.
    """
    from brass_ai.recovery.local_scoring_targets import FIELD, bind
    legacy = copy.deepcopy(dict(game))
    for old, part in zip([legacy] + legacy.get('branches', []), [family['parent'], family['child']]):
        if FIELD in old and part['producer']['contract'] == 'brass-lab-collection/1.0.0':
            recorder = dict(part['scoring'], producer='local-scoring-host/1.0.0')
            old[FIELD] = bind(recorder, old, dict(runtime_sha256='0'*64, host_sha256='0'*64))
    legacy = check_game(legacy)
    for attribute in ('critic_table', 'table_kind', 'ppo_opening_objective'):
        if hasattr(game, attribute): setattr(legacy, attribute, copy.deepcopy(getattr(game, attribute)))
    left = prepared_rows([game], config, learner); right = prepared_rows([legacy], config, learner)
    require(len(left) == len(right), 'Native/reference preparation row count differs')
    for (a, ta, ia, ea), (b, tb, ib, eb) in zip(left, right, strict=True):
        require(ta == tb and ia == ib, 'Native/reference target or trajectory differs')
        for name in ('state', 'actions', 'stable_to_relative'): np.testing.assert_array_equal(getattr(ea, name), getattr(eb, name))
        keep = lambda r: {k: v for k, v in r.items() if k.startswith('_') and k != '_local_scoring'}
        np.testing.assert_equal(keep(a), keep(b))
        if '_local_scoring' in a:
            from brass_ai.recovery.local_scoring_targets import map_for
            require(map_for(a) == map_for(b), 'Native/reference auxiliary targets differ')
    return left


def run(snapshot, jobs, files, producer, output):
    config = snapshot['state']['signature']['config']; games = []; reports = []; identities = set()
    for job, item in zip(jobs, files, strict=True):
        require(pin(item['path']) == item, 'B02 family changed before admission')
        with gzip.open(item['path'], 'rb') as stream: data = stream.read(admission.LIMIT + 1)
        require(len(data) <= admission.LIMIT, 'B02 family exceeds bound')
        family = json.loads(data)
        binding = admission.binding(job, producer, generation=snapshot['generation'], model_id=snapshot['model_id'],
                                    config=config, scope='diagnostic')
        record = admission.publish(Path(output) / (frozen_id(job) + '.jsonl.gz'), family, binding)
        game = admission.read(record['path'], record['native_admission'], allow_diagnostic=True)
        require(game['game_id'] not in identities, 'Repeated family admission'); identities.add(game['game_id'])
        learner = job['provenance']['learner_behavior']['policy_id']
        rows = compare_preparation(game, family, config, learner)
        reports.append(dict(job=job['index'], family_sha256=family['sha256'], receipt=record, prepared_rows=len(rows),
            forced_critic_rows=sum(len(e.choice_ids) == 1 for _, _, _, e in rows),
            actor_rows=sum(r['behavior']['policy_id'] == learner and len(e.choice_ids) > 1 for r, _, _, e in rows),
            snapshot_rows=sum(r.get('_critic_role') == 'snapshot' for r, _, _, _ in rows),
            branch=family['child'] is not None, kind=job['provenance']['kind']))
        games.append(game)
    kinds = {g['game_id']: getattr(g, 'table_kind', job['provenance']['kind'])
             for game, job in zip(games, jobs, strict=True) for g in expand([game])}
    selection = select(kinds, config['value_phase']['holdout_fraction'], config['value_phase']['holdout'])
    from brass_ai.recovery.branching import family as family_id
    held = set(selection['heldout_games'])
    require(all((g in held) == (family_id(g) in held) for g in kinds), 'Whole-family holdout split differs')
    return dict(passed=True, families=reports, holdout_selection=selection, new_games_collected=0,
        scope='B02-sealed diagnostic families; Python replay/target/descriptor/scoring parity; no production admission or optimizer steps')
