"""Bounded P5 evaluation tables using the existing shared network/search services.

P5 outcome/public-review records are separate from training replay. Original Hard
executes whole stock actions which cannot honestly be relabeled as R1 choices.
"""
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from pathlib import Path
import platform
import random
import time

import torch

from brass_ai.bank import sha
from brass_ai.host import RulesHost
from brass_ai.learning.batching import model_id
from brass_ai.learning.contracts import digest, project_path, read_json, require
from brass_ai.learning.replay import terminal_targets, split_for
from brass_ai.learning.reward import CostLedger
from brass_ai.personality.deployment.evaluation import public_behavior
from brass_ai.personality.search.contracts import Profile
from brass_ai.personality.search.engine import SearchSession
from brass_ai.personality.search.inference import SharedInference
from brass_ai.personality.search.readiness import load as load_ready
from brass_ai.recovery.inference import RemoteEvaluator
from brass_ai.recovery.retention import owned, run_lock
from brass_ai.recovery.telemetry import Timings
from brass_ai.search.common import SearchConfig, turn_key
from brass_ai.step5.league import write_json
from brass_ai.step5.policy import NeuralPolicy
from . import panels
from .contracts import CHUNK, RECORD, RESERVATIONS, identity, sealed, read_sealed
from .ui import IntentDisplay, PublicAction, PUBLIC_TEXT


class TrackedChannels:
    def __init__(self, lease):
        self.lease = lease; self.endpoints = []

    def acquire(self):
        endpoint = self.lease.acquire(); self.endpoints.append(endpoint)
        return endpoint

    def close(self):
        for endpoint in self.endpoints: endpoint.close()


def environment(device):
    return dict(system=platform.system(), machine=platform.machine(), processor=platform.processor(),
                node=platform.node(), torch=torch.__version__, device=str(device),
                gpu=torch.cuda.get_device_name(device) if str(device).startswith('cuda') else None)


def reserve(root, directory, out, panel, jobs):
    """Commit panel-wide ownership before any game can run, including failed jobs.

    The short OS-backed lock only serializes reservation writes. A crash after
    the ledger commits leaves jobs reserved; another output cannot retry them.
    """
    root, out = owned(root, out)
    _, directory = owned(root, directory)
    with run_lock(directory):
        require(not out.exists(), 'Immutable chunk: do not overwrite failed or selected results')
        path = directory/'reservations.json'
        ledger = read_sealed(path, RESERVATIONS) if path.exists() else dict(
            format=RESERVATIONS, contract=identity(), panel_id=panel['id'], jobs={})
        require(ledger['panel_id'] == panel['id'], 'Reservation panel changed')
        indices = [j['index'] for j in jobs]
        require(indices and len(set(indices)) == len(indices)
                and all(type(i) is int and 0 <= i < len(panel['jobs']) for i in indices), 'Invalid reservation jobs')
        require(all(str(i) not in ledger['jobs'] for i in indices),
                'Panel job already reserved; preserve the failed attempt and explicitly amend the study')
        claims = dict(ledger['jobs'], **{str(i): out.relative_to(root).as_posix() for i in indices})
        write_json(path, sealed(dict(format=RESERVATIONS, contract=identity(), panel_id=panel['id'], jobs=claims)))
        out.mkdir(parents=True)
        write_json(out/'reservation.json', dict(panel_id=panel['id'], jobs=indices, fixture=panel['fixture']))


def execute(root, panel_directory, output, *, start=0, games=16, tables=24, device='cpu', mono=None,
            fixture_direct=False):
    root, out = owned(root, output)
    require(not out.exists(), 'Immutable chunk: do not overwrite failed or selected results')
    panel_directory = project_path(root, panel_directory)
    panel, value = panels.load(root, panel_directory)
    require(0 <= start < len(panel['jobs']) and 1 <= games <= 4096 and 1 <= tables <= 24, 'Invalid bounded chunk')
    require(tables == panel['tables'], 'Table allowance changed; freeze a new panel for that execution profile')
    require(not fixture_direct or panel['fixture'], 'Direct smoke is fixture-only; real P5 uses declared search')
    jobs = panel['jobs'][start:start+games]; tables = min(tables, len(jobs))
    model, parents, ready, handoff, pool = load_ready(root, value['shared_model']['ready'])
    require(pool == panel['pool_id'] and ready['model']['model_id'] == value['shared_model']['model_id'], 'Frozen model/pool changed')
    execution = dict(handoff['signature']['config']['execution'], processes=tables, hosts_per_process=1,
                     tensor_threads=panel['profiles']['tensor_threads'])
    torch.set_num_threads(panel['profiles']['tensor_threads'])
    reserve(root, panel_directory, out, panel, jobs)
    by_member = {m['id']: m for m in value['members']}; began = time.monotonic()
    service = SharedInference(execution, device=device)
    try:
        ident = service.register(model); parent_ids = [service.register(m) for m in parents]
        require(ident == value['shared_model']['model_id'] and parent_ids == [model_id(m) for m in parents], 'Model registration differs')
        environments = environment(device)
        # Channel allocation mutates the service's lane table. Do it before threads
        # start, not concurrently from per-table lazy session creation.
        channels = []
        for lane in range(tables):
            tiers = {s['tier'] for j in jobs[lane::tables] for s in j['table'] if s['kind']=='personality'}
            count = 1 + panel['profiles']['workers']*(1+len(parent_ids)) if not fixture_direct else 1
            channels.append(dict(opponent=TrackedChannels(service.channels(1)),
                                 tiers={t:TrackedChannels(service.channels(count)) for t in sorted(tiers)}))
    except BaseException:
        # Registration/channel allocation owns queues, pipes and shared buffers
        # even before the inference thread's context has started.
        service.__exit__()
        raise

    def table_loop(lane_index):
        receipts = []; sessions = {}; lane = channels[lane_index]
        opponent_lane = lane['opponent'].acquire()
        timings = Timings()
        policies = {i: NeuralPolicy(None, evaluator=RemoteEvaluator(opponent_lane, p), focus='neutral', exploration=.05)
                    for i, p in enumerate(parent_ids)}
        host = None
        try:
            for job in jobs[lane_index::tables]:
                started = time.monotonic(); stock_calls = 0; current_kind = 'setup'; rows = []; displays = []
                baseline_failures = []; result = None; behavior = {}; telemetry = {}; failure = None
                phases = set(); turn_support = {}; seq = [0]*4; ui_keys = [None]*4
                ui = [IntentDisplay(s) for s in range(4)]
                try:
                    # Each table retains the same native host and single shared weights.
                    if host is None: host = RulesHost(project_path(root, panel['baseline']['host']), mono=mono)
                    with service.cpu.acquire(1, timings): c = host.reset(job['seed'])
                    active_tiers = {s['tier'] for s in job['table'] if s['kind'] == 'personality'}
                    for tier in active_tiers:
                        cfg = SearchConfig(**panel['searches'][tier]).validate()
                        profiles = [Profile.parse(by_member[s['member']]['profile']) if s['kind'] == 'personality' else Profile(alpha=0.)
                                    for s in job['table']]
                        if tier not in sessions:
                            sessions[tier] = SearchSession(root, service, lane['tiers'][tier], ident, parent_ids,
                                cfg, profiles, ready['goal_calibration'], mono=mono, search_enabled=not fixture_direct)
                        session = sessions[tier]; session.profiles = profiles
                        session.begin(job['seed'], job['sampling_seed'])
                        telemetry[tier] = dict(session.telemetry)
                    rngs = [random.Random(digest(dict(seed=job['sampling_seed'], seat=s))) for s in range(4)]
                    ledger = CostLedger(ready['reward'])
                    for step in range(execution['max_decisions']):
                        if c['terminal']: break
                        seat = c['actor']; spec = job['table'][seat]; current_kind = spec['kind']
                        for tier, session in sessions.items():
                            if spec.get('tier') != tier: session.finish_turn()
                        # SearchSession owns admission across both actions/subchoices.
                        lease = nullcontext() if any(s.turn is not None for s in sessions.values()) else service.cpu.acquire(1, timings)
                        with lease:
                            view = host.observe(seat); choices = host.choices()
                        ledger.observe(view); phases.add(view['context']['era'])
                        if spec['kind'] == 'stock':
                            with service.cpu.acquire(1, timings):
                                reply = host.call('stock_step', game_id=c['game_id'], revision=c['revision'])
                            before_finish = reply['action'].get('before_finish')
                            ledger.chosen(before_finish if before_finish is not None else view,
                                          dict(kind='finish' if before_finish is not None else 'liquidate'))
                            stock_calls += 1; c = reply['context']
                            # Original Hard failures never become random legal replacements.
                            continue
                        if spec['kind'] == 'parent':
                            with service.cpu.acquire(1, timings):
                                policy = policies[spec['index']]; policy.utility_remaining[rngs[seat]] = tuple(ledger.remaining)
                                chosen, _, _ = policy.select_batch([(view, choices, rngs[seat])])[0]
                            info = None
                        else:
                            session = sessions[spec['tier']]
                            chosen, ignored_teacher, info = session.select(view, choices, rngs[seat], ledger)
                            # These outcomes never feed PPO, IL, teachers or the descriptor.
                            require(ignored_teacher is None or ignored_teacher['reward_id'] == digest(ready['reward']), 'Search reward differs')
                            key = (seat, *turn_key(view['context']))
                            t = turn_support.setdefault(key, dict(seat=seat, tier=spec['tier'], completed=0, searches=0))
                            t['completed'] += info.get('completed', 0); t['searches'] += int(info['searched'])
                        require(type(chosen) is int and 0 <= chosen < len(choices), 'Illegal selected choice')
                        rows.append(dict(step=len(rows), seat=seat, view=view, choices=choices, chosen=chosen))
                        action = choices[chosen]; ledger.chosen(view, action)
                        lease = nullcontext() if any(s.turn is not None for s in sessions.values()) else service.cpu.acquire(1, timings)
                        with lease: c = host.apply(c, action)
                        # Public indicators are made only AFTER successful public actions.
                        if spec['kind'] == 'personality' and action['kind'] in PUBLIC_TEXT:
                            key = turn_key(view['context'])
                            if key != ui_keys[seat]: seq[seat] += 1; ui_keys[seat] = key
                            member = by_member[spec['member']]
                            group = next(g for g in value['archetypes'] if g['id'] == member['archetype_id'])
                            snapshot = session.intents[seat].snapshot()
                            displays.append(dict(seat=seat, era=key[0], turn=seq[seat], **ui[seat].update(
                                era=key[0], turn=seq[seat], description=group['description'] or 'Fixture style',
                                mode=panel['profiles']['hint_mode'], action=PublicAction(action['kind']), own_intent=snapshot)))
                        for session in sessions.values():
                            if c['terminal'] or (session.turn is not None and
                                (turn_key(c), c['actor'], c['phase']=='debt') != session.turn['key']):
                                session.finish_turn()
                    require(c['terminal'], 'Incomplete P5 game; no win or qualification credit')
                    with service.cpu.acquire(1, timings): result = host.call('result')
                    terminal_targets(result)
                    game = dict(records=rows)
                    behavior = {str(s): public_behavior(game, s, include_boards=True)
                                for s, spec in enumerate(job['table']) if spec['kind'] == 'personality'}
                except Exception as error:
                    failure = dict(kind=current_kind, type=type(error).__name__, message=str(error))
                    if current_kind == 'stock': baseline_failures.append(failure)
                    # Replace a damaged native process before another planned game.
                    if host is not None: host.close(); host = None
                finally:
                    for session in sessions.values(): session.finish_turn()
                metrics = {tier: {k: sessions[tier].telemetry[k]-v for k, v in before.items()}
                           for tier, before in telemetry.items()}
                record = sealed(dict(format=RECORD, contract=identity(), panel_id=panel['id'], job=job,
                    fixture=panel['fixture'], fixture_direct=fixture_direct, result=result, failure=failure,
                    baseline_failures=baseline_failures, stock_actions=stock_calls, both_eras=len(phases)==2,
                    behavior=behavior, intent_display=displays, telemetry=metrics, turns=list(turn_support.values()),
                    seconds=time.monotonic()-started, environment=environments))
                path = out/f'game-{job["index"]:07d}.json'; write_json(path, record)
                receipts.append(dict(job=job['index'], path=path.name, sha256=sha(path), id=record['id']))
                print(f'P5 evaluation: completed {job["index"]} ({"failed" if failure else "ok"}); '
                      f'{"FIXTURE, unqualified" if panel["fixture"] else panel["stage"]}', flush=True)
        finally:
            for session in sessions.values(): session.close()
            if host is not None: host.close()
            lane['opponent'].close()
            for lease in lane['tiers'].values(): lease.close()
        return receipts

    with service:
        with ThreadPoolExecutor(max_workers=tables, thread_name_prefix='p5-table') as executor:
            records = [r for group in executor.map(table_loop, range(tables)) for r in group]
    result = sealed(dict(format=CHUNK, contract=identity(), panel_id=panel['id'], fixture=panel['fixture'],
        fixture_direct=fixture_direct, panel_sha256=sha(panel_directory/'manifest.json'),
        records=sorted(records, key=lambda r:r['job']), seconds=time.monotonic()-began,
        tables=panel['tables'], active_tables=tables, environment=environments, inference=service.report()))
    write_json(out/'manifest.json', result)
    write_json(out/'pins.json', dict(paths=[panel_directory.relative_to(root).as_posix()+'/manifest.json']+
                                   [(out/r['path']).relative_to(root).as_posix() for r in records]))
    return result


def read_chunks(root, directory, chunks):
    panel, value = panels.load(root, directory); seen = set(); sources = []
    def records():
        ledger = read_sealed(project_path(root, directory)/'reservations.json', RESERVATIONS)
        require(ledger['panel_id'] == panel['id'], 'Reservation panel changed')
        for chunk in chunks:
            chunk = project_path(root, chunk); receipt = read_sealed(chunk/'manifest.json', CHUNK)
            require(receipt['panel_id'] == panel['id'] and receipt['fixture'] == panel['fixture']
                    and receipt['tables'] == panel['tables']
                    and receipt['panel_sha256'] == sha(project_path(root, directory)/'manifest.json'), 'Foreign P5 chunk')
            reservation = read_json(chunk/'reservation.json')
            require(reservation == dict(panel_id=panel['id'], fixture=panel['fixture'], jobs=[r['job'] for r in receipt['records']]),
                    'Chunk omitted or substituted a reserved job')
            require(all(ledger['jobs'].get(str(r['job'])) == chunk.relative_to(Path(root).resolve()).as_posix()
                        for r in receipt['records']), 'Chunk does not own these reserved jobs')
            sources.append(dict(path=chunk.relative_to(Path(root).resolve()).as_posix(), sha256=sha(chunk/'manifest.json')))
            for item in receipt['records']:
                path = (chunk/item['path']).resolve()
                require(path.parent == chunk and sha(path) == item['sha256'], 'Game path/evidence changed')
                r = read_sealed(path, RECORD); index = r['job']['index']
                require(index not in seen and 0 <= index < len(panel['jobs']) and r['job'] == panel['jobs'][index]
                        and item['job'] == index and item['id'] == r['id'] and r['panel_id'] == panel['id'], 'Duplicate/unplanned game')
                require(r['fixture'] == panel['fixture'] and r['fixture_direct'] == receipt['fixture_direct']
                        and r['environment'] == receipt['environment'], 'Evidence context changed')
                require(split_for(r['job']['setup_group']) == panel['partition']
                        and r['job']['setup_group'] not in panel['forbidden_groups'], 'Reused/mislabeled evaluation setup')
                seen.add(index); yield r
    return panel, value, records(), seen, sources
