"""Search after a reward migration still uses the actual frozen opponents."""
import copy
from dataclasses import replace
import random
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from fixtures import position
from brass_ai.learning.batching import Request, model_id
from brass_ai.learning.contracts import digest
from brass_ai.learning.model import ModelConfig
from brass_ai.learning.reward import CostLedger, specification
from brass_ai.learning.utility_model import UtilityNet, utility_config
from brass_ai.recovery.config import EXECUTION
from brass_ai.recovery.inference import InferenceService, RemoteEvaluator
from brass_ai.search.common import SearchConfig
from brass_ai.search.ismcts import MaskedISMCTS
from brass_ai.step5.policy import NeuralPolicy, context_for


CURRENT = specification(dict(win=1., placement=.1, vp=.0005), dict(event_cost=.005, max_cost=.075))
PREVIOUS = specification(dict(win=1., placement=.1, vp=.0005), dict(event_cost=.005, max_cost=.025))


class ThreeTurns:
    """Authored root -> opponent -> new root frontier; no native rules needed."""
    def __init__(self):
        self.i = 0
        self.recall = [[] for _ in range(4)]
        self.sync()

    def sync(self):
        actor = self.current_player()
        self.views = [position(seat)[0] for seat in range(4)]
        for view in self.views:
            view['context'].update(actor=actor, turn_owner=actor, turn=self.i+1,
                                   round=1 if self.i < 2 else 2, revision=self.i)
        self.context = self.views[actor]['context']
        self.legal = position(actor)[1]

    def current_player(self): return 1 if self.i == 1 else 0
    def is_terminal(self): return False
    def is_chance_node(self): return False
    def apply_action(self, _): self.i += 1; self.sync()


class TeacherOpponentRewardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(2)

    def models(self):
        torch.manual_seed(173)
        learner = UtilityNet(utility_config(ModelConfig.preset('small'), CURRENT))
        opponent = copy.deepcopy(learner)
        opponent.config = replace(opponent.config, utility=copy.deepcopy(PREVIOUS))
        return learner, opponent

    def search(self, base, opponent):
        return MaskedISMCTS(SimpleNamespace(from_observation=lambda *_: ThreeTurns()), base, [opponent],
            SearchConfig(contract='masked-ismcts/1.2.0', reward=CURRENT, simulations=1, pending_paths=1),
            0, context_for('neutral'), random.Random(7), time.monotonic()+30,
            reward_context=dict(spent=[.005, 0., .01, 0.], settled=[]))

    def test_migrated_teacher_with_frozen_old_reward_opponent(self):
        for slot_bytes in (0, 2*1024*1024):
            with self.subTest(shared_slot_bytes=slot_bytes), InferenceService(
                    dict(EXECUTION, processes=1, hosts_per_process=1, shared_slot_bytes=slot_bytes), 'cpu') as service:
                learner, opponent = self.models()
                frozen_identity = model_id(opponent)
                identities = [service.register(m) for m in (learner, opponent)]
                endpoint = service.channels(1).acquire()
                self.addCleanup(endpoint.close)
                policies = [NeuralPolicy(None, focus='neutral', exploration=0.,
                    evaluator=RemoteEvaluator(endpoint, identity)) for identity in identities]
                captured = {identity: [] for identity in identities}
                original = NeuralPolicy.prediction_requests
                def requests(policy, inputs):
                    rows = original(policy, inputs)
                    captured[policy.evaluator.model_id].extend(r.remaining for r in rows)
                    return rows
                bot = self.search(*policies)
                state = ThreeTurns()
                with patch.object(NeuralPolicy, 'prediction_requests', requests):
                    bot.run(state.views[0], state.legal)
                self.assertEqual(bot.completed, 1)
                self.assertEqual(bot.opponent_choices, 1)
                self.assertEqual(len(captured[identities[0]]), 2)
                for budget in captured[identities[0]]:
                    np.testing.assert_allclose(budget, [.07, .075, .065, .075])
                self.assertEqual(captured[identities[1]], [None])
                self.assertIsNone(service.error)
                self.assertFalse(service.cpu.aborted.value)
                self.assertEqual(model_id(service.models[frozen_identity]), frozen_identity)
                self.assertEqual(opponent.config.utility, PREVIOUS)
                self.assertTrue(all(e.pending == 0 for n in bot.tree.values() for e in n.edges.values()))
                # The unused opponent utility context cannot change its actions.
                view, choices = position(1)
                rows = [Request(str(i), frozen_identity, view, choices, time.monotonic()+30,
                                remaining=budget) for i, budget in enumerate((None, (.025,)*4, (0.,)*4))]
                responses = policies[1].evaluator.predict(rows)
                for response in responses[1:]:
                    np.testing.assert_allclose(response['probabilities'], responses[0]['probabilities'], rtol=1e-6, atol=1e-7)
                endpoint.close()

    def test_policy_only_request_clears_stale_budget_and_keeps_cache_separate(self):
        learner, _ = self.models()
        policy = NeuralPolicy(learner, focus='neutral')
        self.addCleanup(policy.close)
        rng = random.Random(13)
        state = ThreeTurns()
        bot = self.search(policy, policy)
        ledger = CostLedger(CURRENT)
        response = ([1/3]*3, dict(utility=[0.]*4, reward_id=digest(CURRENT)))
        valued = bot.request(policy, state, rng, ledger)
        valued_key = next(valued)[0]
        with self.assertRaises(StopIteration): valued.send(response)
        self.assertEqual(policy.utility_remaining[rng], (.075,)*4)
        action_only = bot.request(policy, state, rng)
        action_key = next(action_only)[0]
        self.assertNotEqual(valued_key, action_key)
        self.assertIsNone(policy.prediction_requests([(state.views[0], state.legal, rng)])[0].remaining)
        with self.assertRaises(StopIteration): action_only.send(response)

    def test_root_leaf_reward_match_is_still_required(self):
        learner, _ = self.models()
        policy = NeuralPolicy(learner, focus='neutral')
        self.addCleanup(policy.close)
        state = ThreeTurns()
        bot = self.search(policy, policy)
        trajectory = bot.trajectory(state.views[0], state.legal)
        next(trajectory)
        with self.assertRaisesRegex(ValueError, 'Search leaf reward differs'):
            trajectory.send(([1/3]*3, dict(utility=[0.]*4, reward_id=digest(PREVIOUS))))
        self.assertEqual(bot.completed, 0)

    def test_invalid_explicit_old_model_budget_still_aborts_inference(self):
        with InferenceService(dict(EXECUTION), 'cpu') as service:
            _, opponent = self.models()
            identity = service.register(opponent)
            endpoint = service.channels(1).acquire()
            try:
                client = RemoteEvaluator(endpoint, identity)
                with self.assertRaisesRegex(ValueError, 'Invalid inference utility budget'):
                    client.predict([Request('invalid', identity, *position(), time.monotonic()+30, remaining=(.075,)*4)])
                self.assertTrue(service.cpu.aborted.value)
                self.assertIn('Invalid inference utility budget', str(service.error))
            finally: endpoint.close()


if __name__ == '__main__': unittest.main()
