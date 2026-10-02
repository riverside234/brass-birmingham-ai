"""One-second original v3 S1 teachers; neural search remains Masked ISMCTS.

Each S1 replica owns a private frozen engine and a deterministic root partition.
One turn deadline/work allowance covers both actions and any replanning.
Rule-based choices are imitation labels, never NN root-visit distillation labels.
"""
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import time
import weakref

from brass_ai.bank import FrozenOpponent, admission, load_bank, read
from brass_ai.learning.contracts import digest, require
from brass_ai.learning.legacy_policy import LegacyPolicy
from .s1_bank import prepare, STYLES

CONTRACT = 'v3-s1-training/1.0.0'


def legacy_spec(root, parent, budget=None, purpose='training'):
    admission(root, parent, purpose)
    profile = load_bank(root, parent.split('/')[0])['payload']['profiles'][parent.split('/')[1]]
    require(profile['mode'] == 'S1' and profile['milliseconds'] == 1000,
            'Training opponents must use the one-second original v3 S1 bank')
    if budget is None:
        return dict(kind='legacy', policy_id=parent, purpose=purpose)
    from .config import legacy_budget_config
    legacy_budget_config(budget)
    identity = 'v3-s1-forest/' + digest(dict(parent=parent, budget=budget, contract=CONTRACT))
    return dict(kind='lab-legacy', parent=parent, policy_id=identity,
                budget=budget, purpose=purpose)


def teachers(root, budget=None):
    return {legacy_spec(root, p, budget)['policy_id']
            for p in read(Path(root) / 'training/opponents/pool.json')['training']}


class LabLegacyPolicy(LegacyPolicy):
    def __init__(self, root, parent, budget, *, mono=None, purpose='training', diagnostic=False):
        self.root, self.parent, self.budget = Path(root), parent, dict(budget)
        self.policy_id = legacy_spec(root, parent, budget, purpose)['policy_id']
        self.workers = []
        self.contexts = weakref.WeakKeyDictionary()
        try:
            for _ in range(budget['search_workers']):
                self.workers.append(FrozenOpponent(root, parent, purpose=purpose, mono=mono))
        except BaseException:
            for worker in self.workers:
                worker.close()
            raise
        self.opponent = self.workers[0]
        self.pool = ThreadPoolExecutor(max_workers=len(self.workers))
        self.telemetry = dict(backend='S1', turns=0, branches=0, nodes=0, unique_plans=0,
            failures=0, over_budget_turns=0, max_turn_seconds=0.,
            seconds=budget['seconds'], search_workers=len(self.workers),
            work_multiplier=budget['work_multiplier'])

    def decision(self, view, choices, state):
        count = len(self.workers) if view['context']['phase'] == 'action' else 1
        allowance = max(1, self.opponent.profile['work_limit'] *
                        self.budget['work_multiplier'] - state['work'])

        def branch(index):
            profile = dict(self.opponent.profile,
                milliseconds=max(0, 1000 * (state['deadline'] - time.monotonic() - .05)),
                work_limit=max(1, allowance // count + int(index < allowance % count)),
                partition_count=count, partition_index=index)
            return self.workers[index].worker.call('plan', observation=view, choices=choices,
                profile=profile, fixed_work=False, sampling_seed=1, intent=state['intent'])

        # Drain every submitted request even if a sibling fails. Neither a clock
        # nor a cancelled future proves its private engine has stopped running.
        futures = [self.pool.submit(branch, i) for i in range(count)]
        results, errors = [], []
        for future in futures:
            try:
                results.append(future.result())
            except Exception as error:
                errors.append(error)
        if errors:
            raise errors[0]
        require(all(d['mode'] == 'S1' and d['failures'] == 0 for d in results),
                'S1 planning failed; reject the incomplete teacher game')
        plans = {p['signature']: p for d in results
                 for p in [d['selected']] + d.get('alternatives', []) +
                 list((d.get('family_best') or {}).values())}
        require(plans, 'No legal complete S1 incumbent')
        selected = results[0]['selected'] if count == 1 else self.opponent.worker.call(
            'select_plans', plans=list(plans.values()), profile=self.opponent.profile)
        nodes = sum(d['nodes'] for d in results)
        state['work'] += nodes
        self.telemetry['branches'] += count
        self.telemetry['nodes'] += nodes
        self.telemetry['unique_plans'] += len(plans)
        self.last_decision = dict(backend='S1', selected=selected, nodes=nodes,
            unique_plans=len(plans), failures=0, work_allowance=allowance,
            partitions=count, elapsed_ms=[d['elapsed_ms'] for d in results])
        return selected

    def select_batch(self, inputs):
        output = []
        for view, choices, rng in inputs:
            c = view['context']
            key = (c['era'], c['round'], c['turn'], view['seat'])
            state = self.contexts.setdefault(rng,
                dict(turn=None, revision=-1, intent=None, queue=deque(), work=0))
            if state['turn'] != key or c['revision'] <= state['revision']:
                began = time.monotonic()
                state.update(turn=key, revision=-1, queue=deque(), work=0, overrun=False,
                             deadline=began + self.budget['seconds'], began=began)
                self.telemetry['turns'] += 1
            elif c['revision'] != state['revision'] + 1:
                state['queue'].clear()
            ids = [p['id'] for p in choices]
            if state['queue'] and state['queue'][0] not in ids:
                state['queue'].clear()
            if not state['queue']:
                plan = self.decision(view, choices, state)
                state['queue'].extend(x for action in plan['actions'] for x in action)
                state['intent'] = plan.get('intent')
            chosen = state['queue'].popleft()
            require(chosen in ids, 'Illegal S1 continuation')
            elapsed = time.monotonic() - state['began']
            self.telemetry['max_turn_seconds'] = max(self.telemetry['max_turn_seconds'], elapsed)
            if elapsed > self.budget['seconds'] and not state['overrun']:
                self.telemetry['over_budget_turns'] += 1
                state['overrun'] = True
            state['revision'] = c['revision']
            output.append((ids.index(chosen), 1., None))
        return output

    def close(self):
        self.pool.shutdown(wait=True)
        self.contexts.clear()
        for worker in self.workers:
            worker.close()
