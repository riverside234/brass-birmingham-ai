#!/usr/bin/env python3
"""Start a separate lab run at g4970, or continue it from its last commit."""

import argparse
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path('artifacts/step5/joint7-v11-2-g1190-restart-3')
RUN = Path('artifacts/step5/joint7-v11-2-g4970-rail025-soft8-beta0.02')
GENERATION = 4970
sys.path.insert(0, str(ROOT / 'src'))


def copy_file(source, run, path):
    original = ROOT / path
    destination = run / original.relative_to(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(original, destination)
    return destination.relative_to(ROOT).as_posix()


def restore_ratings(state, source, run):
    from brass_ai.bank import read
    from brass_ai.learning.contracts import digest
    from brass_ai.recovery.plackett_luce import fit
    from brass_ai.recovery.rating_coverage import SEED
    from brass_ai.recovery.ratings import receipt
    from brass_ai.step5.league import write_json

    ratings = state['ratings']
    ratings['agents'] = {name: agent for name, agent in ratings['agents'].items()
                         if agent['kind'] != 'model' or agent['checkpoint']['generation'] <= GENERATION}
    ratings['slots'] = [slot for slot in ratings['slots'] if slot['generation'] <= GENERATION]
    for slot in ratings['slots']:
        for key in ('spec', 'records', 'report', 'panels', 'bootstrap'):
            slot[key]['path'] = copy_file(source, run, slot[key]['path'])

    archive = read(ROOT / ratings['archive']['path'])
    archive['slots'] = [slot for slot in archive['slots'] if slot['generation'] <= GENERATION]
    archive['games'] = [game for game in archive['games'] if game['slot'] <= GENERATION]
    ratings['panel_gates']['last_generation'] = (ratings['slots'] or archive['slots'])[-1]['generation']
    path = run / 'ratings' / f'history-{digest(archive)}.json'
    write_json(path, archive, compact=True)
    ratings['archive'] = receipt(ROOT, path)
    if ratings['slots']:
        ratings['latest_report'] = ratings['slots'][-1]['report']
    else:
        # The g4970 report was compacted. Restore point estimates from the
        # archived games for the first slot's opponent selection.
        config = ratings['config']
        fitted = fit([row for row in archive['games'] if row['kind'] == 'rating'], ratings['agents'], config)
        theta = dict(zip(fitted['agents'], fitted['theta']))
        zero = theta[SEED]
        scores = {name: dict(theta=value - zero, points=(value - zero) * config['elo_scale'])
                  for name, value in theta.items()}
        path = run / 'ratings' / 'restart-points.json'
        write_json(path, dict(generation=GENERATION, ratings=scores), compact=True)
        ratings['latest_report'] = receipt(ROOT, path)


def restore_pool(state, source, run, games_done):
    from brass_ai.recovery import pfsp_pool, quality_pool
    from brass_ai.recovery.v4_history import bot

    pool = state['quality_pool']
    members = [m for m in pool['members'] if m['admitted_iteration'] <= GENERATION]
    prior = pfsp_pool.prior(pool['config'])
    for member in members:
        member.update(prior)
        if bot(member):
            member['profile_seats'] = 0
            continue
        member['path'] = copy_file(source, run, member['path'])
        probe = member['probe']
        probe['path'] = copy_file(source, run, probe['path'])
    pool.update(generation=GENERATION, games_done=games_done, members=members, health_window=[],
                planning=None, events=[event for event in pool['events'] if event['generation'] <= GENERATION])
    pfsp_pool.seal(pool)
    state['population'] = [quality_pool.reference(m) for m in members if not bot(m)]
    state['evaluation_anchor'] = next(m for m in state['population'] if m['generation'] == 0)
    return quality_pool.state_digest(pool)


def prepare(source, run):
    import torch

    from brass_ai.bank import read
    from brass_ai.learning.contracts import digest
    from brass_ai.learning.utility_model import UtilityConfig, UtilityNet
    from brass_ai.recovery import adaptive_teacher_beta as adaptive
    from brass_ai.recovery.branch_restart import ppg_state, ppo_optimizer, rewrite
    from brass_ai.recovery.ratings import receipt
    from brass_ai.recovery.search_value_iteration import stamp
    from brass_ai.step5.league import write_json

    state = read(source / 'state.json')
    ref = state['ratings']['agents'][f'generation-{GENERATION:06d}']['checkpoint']
    checkpoint = torch.load(ROOT / ref['path'], weights_only=True, map_location='cpu')
    saved = checkpoint['training']
    signature = next((t['from_signature'] for t in state['source_transitions']
                      if t['iteration'] >= GENERATION), state['signature'])
    config = signature['config']
    model = UtilityNet(UtilityConfig.from_dict(checkpoint['model_config']))
    phases = ppg_state(config, model, GENERATION, gate=None)
    gate = state['recent_training'][-1]['actor']['gae']['gate']
    # Keep table activations that had already qualified by g4970. The missing
    # optimizer and rolling evidence start with the standard restart state.
    gae = phases['gae']
    gae['stage_origin'] = min(gate['stage_origin'], GENERATION)
    for table, metrics in gate['table_gate'].items():
        if metrics.get('enabled') and metrics['origin'] <= GENERATION:
            gae['table_enabled'][table] = True
            gae['table_origins'][table] = metrics['origin']
    checkpoint['optimizer'] = ppo_optimizer(config, model)

    run.mkdir(parents=True)
    run_name = run.relative_to(ROOT).as_posix()
    old = source.relative_to(ROOT).as_posix() + '/'
    history = rewrite([row for row in state['history'] if row['generation'] <= GENERATION], old, run_name + '/')
    games_done = state['games_done'] - (state['iteration'] - GENERATION) * config['games_per_iteration']
    pool_digest = restore_pool(state, source, run, games_done)
    restore_ratings(state, source, run)

    limit = saved['il_kl_limit']
    probe = read(ROOT / limit['probe']['path'])
    probe['cache']['path'] = copy_file(source, run, probe['cache']['path'])
    probe_path = run / 'il-kl' / 'manifest.json'
    write_json(probe_path, probe)
    limit['probe'] = dict(receipt(ROOT, probe_path), probe_hash=probe['probe_hash'])

    teacher = saved['teacher_phase']
    weight = teacher['weight']
    weight.update(beta=.02, shadow_shares=[], outside_streak=0)
    weight[adaptive.REMEASUREMENT] = dict(requested_generation=GENERATION, previous_beta=weight['beta'])
    weight.pop(adaptive.KEY, None)
    teacher['buffer'].update(receipts=[], retired=[])
    search = saved['search_value']
    search['buffer'].update(receipts=[], retired=[])
    search['label_ages'] = []

    saved.update(fresh_run=run_name, signature=digest(signature), quality_pool_digest=pool_digest,
                 actor_value_phases=phases)
    output = run / f'generation-{GENERATION:04d}.pt'
    torch.save(checkpoint, output)
    entry = dict(receipt(ROOT, output), generation=GENERATION, model_id=checkpoint['model_id'])
    history[-1] = entry
    state.pop('native_backend_switch', None)
    state.update(signature=signature, iteration=GENERATION, history=history, resume=[entry], games_done=games_done,
                 recent_training=[], source_transitions=[], exploration=saved['exploration'], il_kl_limit=limit,
                 native_backend=saved['native_backend'], teacher_phase=teacher,
                 teacher_seed_cursor=teacher['teacher_seed_cursor'], search_value=stamp(search))

    for name in ('vp-history.json', 'teacher-vp-history.json'):
        source_history = source / 'ratings' / name
        if source_history.is_file():
            vp = rewrite(read(source_history), old, run_name + '/')
            vp.update(committed_generation=GENERATION,
                      iterations=[row for row in vp['iterations'] if row['generation'] <= GENERATION])
            write_json(run / 'ratings' / name, vp, compact=True)
    write_json(run / 'state.json', state, compact=True)
    print(f'Prepared {run_name} from the saved g{GENERATION} weights with fresh PPO/PPG Adam and replay. '
          f'Teacher beta is {weight["beta"]:g} for the first iteration while remeasuring.', flush=True)
    return state


def repair_panel_progress(run, state):
    last = state['ratings']['panel_gates']['last_generation']
    if last is None or last <= state['iteration']:
        return state

    from brass_ai.bank import read
    from brass_ai.recovery.retention import run_lock
    from brass_ai.step5.league import write_json

    path = run / 'state.json'
    with run_lock(run):
        state = read(path)
        ratings = state['ratings']
        report = read(ROOT / ratings['latest_report']['path'])
        ratings['panel_gates']['last_generation'] = report['generation']
        write_json(path, state, compact=True)
    return state


def main():
    from brass_ai.bank import read

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--expected-generation', type=int, default=GENERATION)
    parser.add_argument('--iterations', type=int, default=10)
    parser.add_argument('--share-target', type=float, help='Defaults to the current share target in this run')
    parser.add_argument('--workers', type=int, default=64)
    parser.add_argument('--optimizer-revision', action='store_true',
                        help='Adopt the g5402 optimizer revision at this committed generation (set once, then kept)')
    parser.add_argument('--auxiliary-revision', action='store_true',
                        help='Adopt the g5466 auxiliary revision at this committed generation (set once, then kept)')
    parser.add_argument('--clip-revision', action='store_true',
                        help='Adopt the g5685 clip revision at this committed generation (set once, then kept)')
    args = parser.parse_args()

    run = ROOT / RUN
    state_path = run / 'state.json'
    if args.expected_generation == GENERATION and not state_path.is_file():
        state = prepare(ROOT / SOURCE, run)
    else:
        state = repair_panel_progress(run, read(state_path))
    share = args.share_target if args.share_target is not None else state['signature']['config']['teacher_phase']['term']['share_target']
    return subprocess.call([
        sys.executable, str(ROOT / 'scripts' / 'continue-rust-g1190-restart-2.py'),
        '--run', RUN.as_posix(), '--expected-generation', str(args.expected_generation),
        '--iterations', str(args.iterations), '--share-target', str(share), '--workers', str(args.workers),
        '--rail-end-weight', '0.25', '--soft-policy-weight', '8',
        *(['--optimizer-revision'] if args.optimizer_revision else []),
        *(['--auxiliary-revision'] if args.auxiliary_revision else []),
        *(['--clip-revision'] if args.clip_revision else []),
    ], cwd=ROOT)


if __name__ == '__main__':
    sys.exit(main())
