"""Four native reserved turns after a .025 -> .075 teacher reward migration.

Uses complete local smoke replays for observed roots and utility calibration.
It changes no training run and makes no claim about playing strength.
Time/coverage rejections remain diagnostic outcomes, not forced label yields.
"""
import argparse
from collections import Counter
import copy
from dataclasses import replace
from pathlib import Path
import random
import time

import torch

from brass_ai.bank import read, sha
from brass_ai.learning.batching import model_id
from brass_ai.learning.contracts import SOURCE_ROOT, digest, require
from brass_ai.learning.model import ModelConfig
from brass_ai.learning.utility_model import UtilityNet, utility_config
from brass_ai.recovery.config import teacher_time_seconds
from brass_ai.recovery.execution import make_policy
from brass_ai.recovery.inference import InferenceService
from brass_ai.recovery.replay import read_shards
from brass_ai.recovery.resources import TurnLease
from brass_ai.recovery.teacher_schedule import CONTRACT, reserved_rounds
from brass_ai.recovery.telemetry import Timings
from brass_ai.recovery.updates import Dataset, prepared
from brass_ai.recovery.utility import calibrate, calibration_ready, reward_spec
from brass_ai.search.common import turn_key
from brass_ai.searchlab.spiel import BrassSpielGame, action_id
from brass_ai.step5.league import implementation_id, write_json
from smoke_teacher_phases import positions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--replay', type=Path, action='append', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    require(out.is_relative_to(SOURCE_ROOT/'artifacts') and not out.exists(), 'Choose a new artifact directory')
    source = implementation_id(); began = time.monotonic()
    torch.set_num_threads(2); torch.manual_seed(74160)
    cfg = read(SOURCE_ROOT/'configs/step5/joint7-g150-gate72.json')
    reward = reward_spec(cfg); budget = cfg['teacher_work_budget']
    games = read_shards(args.replay, partition='train')
    roots = positions(games, 1, reward)
    model = UtilityNet(utility_config(ModelConfig.preset('small'), reward))
    dataset = Dataset(prepared(games, insolvency=reward['insolvency'], reward=reward), 'cpu', policy=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'])
    calibration = calibrate(model, optimizer, dataset, cfg['update'])
    require(calibration_ready(model), 'Smoke calibration failed')
    # These are frozen test weights, not the user's historical checkpoints.
    old = copy.deepcopy(model); old_reward = copy.deepcopy(reward)
    old_reward['insolvency']['max_cost'] = .025
    old.config = replace(old.config, utility=old_reward); old.utility_calibration = None
    old_identity = model_id(old)
    execution = dict(cfg['execution'], processes=1, heavy_processes=1)
    results = []; requests = Counter()
    with InferenceService(execution, 'cpu') as service:
        identity = service.register(model); require(service.register(old) == old_identity, 'Old model changed')
        evaluate = service._evaluate
        def measured(frames):
            for _, frame in frames:
                role = 'old_opponent' if frame['model'] == old_identity else 'learner'
                requests[role] += 1
                if role == 'old_opponent':
                    require(frame.get('remaining') is None, 'Learner budget leaked into a frozen opponent')
                elif frame.get('remaining') is not None:
                    requests['learner_with_ledger'] += 1
            return evaluate(frames)
        service._evaluate = measured
        channels = service.channels(5); endpoints = [channels.acquire() for _ in range(5)]
        policy = make_policy(dict(kind='masked_ismcts', model_id=identity, opponents=[old_identity],
            search=cfg['search'], focus='neutral', cpu_role='reference', teacher_schedule=CONTRACT,
            teacher_work_budget=budget), SOURCE_ROOT, endpoints, execution, None)
        driver = BrassSpielGame(root=SOURCE_ROOT, fast=True); lease = TurnLease(service.cpu, Timings())
        try:
            for phase, rows in roots.items():
                episode, row, original_ledger = rows[0]
                view = copy.deepcopy(row['view']); seat = view['seat']; seed = 74160
                state = driver.from_observation(view, seed)
                require(state.legal == row['choices'], 'Reconstructed legality differs')
                rng = random.Random(seed)
                sampling = next(i for i in range(10000) if reserved_rounds(seed, i, seat)[phase] == view['context']['round'])
                policy.begin_teacher_game(rng, seat, seed, sampling)
                ledger = copy.deepcopy(original_ledger); key = turn_key(state.context); decisions = []
                lease.update(key+(seat,), 4); started = time.monotonic(); before = requests.copy()
                print(dict(phase=phase, seconds=teacher_time_seconds(budget, phase)), flush=True)
                for _ in range(96):
                    if state.is_terminal() or state.is_chance_node() or state.current_player() != seat or turn_key(state.context) != key: break
                    live, choices = state.views[seat], state.legal
                    policy.admit_teacher_turn(live['context'], rng)
                    ledger.observe(live); policy.set_reward_context(ledger)
                    chosen, _, teacher = policy.select_batch([(live, choices, rng)])[0]
                    require(0 <= chosen < len(choices), 'Illegal teacher action')
                    decisions.append(dict(phase=live['context']['phase'], actions_remaining=live['context']['actions_remaining'],
                                          legal=len(choices), qualified=teacher is not None))
                    ledger.chosen(live, choices[chosen]); state.apply_action(action_id(choices[chosen]['id']))
                else: raise AssertionError('Continuation cap exceeded')
                counts = policy.teacher_report(rng)['phases'][phase]; lease.close()
                require(service.error is None and not service.cpu.aborted.value, 'Inference aborted')
                # A bounded turn may correctly reject its labels. Preserve the
                # reason/work counters instead of aborting before writing them.
                # Inference/protocol failures still raise or fail the gate above.
                result = dict(phase=phase, original_game=episode['game_id'],
                    position=digest(dict(view=row['view'], choices=row['choices'])),
                    elapsed_seconds=time.monotonic()-started, counts=counts, decisions=decisions,
                    requests=dict(requests-before))
                results.append(result); write_json(out/(phase+'.json'), result)
                print(result, flush=True)
            require(requests['old_opponent'] > 0, 'Frozen opponents were not exercised')
            require(requests['learner_with_ledger'] > 0, 'Learner search ledger was not exercised')
            require(model_id(service.models[old_identity]) == old_identity, 'Frozen opponent mutated')
            require(service.cpu.report()['active'] == 0, 'CPU lease leaked')
            inference = service.report()
        finally:
            lease.close(); policy.close(); driver.close()
            for endpoint in endpoints: endpoint.close()
    require(source == implementation_id(), 'Source changed during smoke')
    report = dict(implementation=source, elapsed_seconds=time.monotonic()-began,
        inputs=[dict(path=str(p.resolve()), sha256=sha(p)) for p in args.replay],
        learner=dict(model_id=identity, reward=reward), frozen_opponent=dict(model_id=old_identity, reward=old_reward),
        budget=budget, calibration=calibration, native=results, inference=inference, requests=dict(requests),
        usage=dict(qualified_labels=sum(r['counts']['qualified_samples'] for r in results), distinct_labels_used=0,
                   policy_drift=None, reason='Partial-turn diagnostic probes; no distillation update.'),
        scope='Local CPU, calibrated small test weights, four workers, current phase clocks. Not a lab resume or playing-strength evaluation.')
    write_json(out/'report.json', report)
    print('Smoke report:', out/'report.json', flush=True)


if __name__ == '__main__': main()
