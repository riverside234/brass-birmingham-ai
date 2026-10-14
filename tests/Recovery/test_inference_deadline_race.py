"""A deadline crossing the final response gate retires search, not its actor."""
import copy
import random
import time
import unittest
from threading import Event, get_ident
from unittest.mock import patch

import numpy as np

from fixtures import position
from brass_ai.learning.batching import Request, accept_response
from brass_ai.learning.contracts import digest
from brass_ai.personality.search.inference import PersonalityPolicy
from brass_ai.recovery.config import teacher_search_config
from brass_ai.search.common import SearchIncomplete
from brass_ai.search.ismcts import MaskedISMCTS
from brass_ai.step5.policy import NeuralPolicy, inference_deadline, predict_groups


def response(request):
    result = dict(status='ok', protocol='four-seat-inference/1.0.0',
                  request_id=request.request_id, model_id=request.expected_model,
                  revision=request.view['context']['revision'], observer=request.view['seat'],
                  choice_ids=[c['id'] for c in request.choices], input_digest=request.input_digest,
                  probabilities=[.6, .3, .1], win_share=[.25]*4, placement=[2.5]*4, vp=[80.]*4)
    if hasattr(request, 'personality'):
        result['context_id'] = digest(request.personality)
    return result


class Queue:
    def __init__(self, endpoint=None, model_id='a'*64):
        self.endpoint = endpoint if endpoint is not None else object()
        self.model_id = model_id
        self.mixed_calls = 0

    def predict(self, requests):
        return [response(r) for r in requests]

    def predict_mixed(self, groups):
        self.mixed_calls += 1
        return [self.predict(requests) for _, requests in groups]


class InferenceDeadlineRaceTests(unittest.TestCase):
    def setUp(self):
        self.view, self.choices = position()
        self.inputs = [(self.view, self.choices, random.Random(7))]
        self.queue = Queue()
        self.policy = NeuralPolicy(None, evaluator=self.queue, focus='neutral', exploration=0.)
        self.request = Request('fixture', self.queue.model_id, self.view, self.choices,
                               time.monotonic()+120)

    def test_expiry_between_policy_and_final_gate_ends_search(self):
        # The exact reported race: live in prediction_outputs, expired inside
        # accept_response. No sleeping or probabilistic timing is involved.
        with inference_deadline([self.policy], self.request.deadline), \
             patch.object(Request, 'stopped', side_effect=[None, 'expired']), \
             self.assertRaises(SearchIncomplete):
            self.policy.prediction_outputs([self.request], [response(self.request)])
        self.assertIsNone(self.policy.request_deadline)

    def test_cancellation_between_policy_and_final_gate_ends_search(self):
        with inference_deadline([self.policy], self.request.deadline), \
             patch.object(Request, 'stopped', side_effect=[None, 'cancelled']), \
             self.assertRaises(SearchIncomplete):
            self.policy.prediction_outputs([self.request], [response(self.request)])

    def test_mixed_model_queue_uses_the_same_deadline_handling(self):
        other = NeuralPolicy(None, evaluator=Queue(self.queue.endpoint, 'b'*64), focus='neutral')
        with inference_deadline([self.policy, other], time.monotonic()+120), \
             patch.object(Request, 'stopped', side_effect=[None, 'expired']), \
             self.assertRaises(SearchIncomplete):
            predict_groups([(self.policy, self.inputs), (other, self.inputs)])
        self.assertEqual(self.queue.mixed_calls, 1)
        self.assertIsNone(other.request_deadline)

    def test_personality_policy_handles_the_same_gate_race(self):
        policy = PersonalityPolicy(self.queue)
        for stopped in ('expired', 'cancelled'):
            with self.subTest(stopped=stopped), \
                 patch.object(Request, 'stopped', side_effect=[None, stopped]), \
                 self.assertRaises(SearchIncomplete):
                policy.predict(self.inputs)

    def test_nonsearch_stopped_predictions_still_fail_validation(self):
        for stopped in ('expired', 'cancelled'):
            with self.subTest(stopped=stopped), patch.object(Request, 'stopped', return_value=stopped), \
                 self.assertRaisesRegex(ValueError, 'Request no longer live'):
                self.policy.prediction_outputs([self.request], [response(self.request)])

    def test_explicit_expired_and_cancelled_responses_end_search(self):
        for status in ('expired', 'cancelled'):
            with self.subTest(status=status), inference_deadline([self.policy], self.request.deadline), \
                 self.assertRaises(SearchIncomplete):
                self.policy.prediction_outputs([self.request], [dict(status=status)])

    def test_bad_responses_are_not_misclassified_when_deadline_follows_validation(self):
        mutations = dict(model_id='b'*64, request_id='wrong', revision=99, observer=3,
                         input_digest='bad', choice_ids=['wrong']*3,
                         probabilities=[float('nan'), .3, .1], win_share=[0.]*4)
        for field, value in mutations.items():
            broken = dict(response(self.request), **{field: value})
            with self.subTest(field=field), inference_deadline([self.policy], self.request.deadline), \
                 patch.object(Request, 'stopped', side_effect=[None, None, 'expired']), \
                 self.assertRaises(ValueError):
                self.policy.prediction_outputs([self.request], [broken])

    def test_valid_probabilities_are_unchanged(self):
        outputs = self.policy.prediction_outputs([self.request], [response(self.request)])
        np.testing.assert_allclose(outputs[0][0], [.6, .3, .1], rtol=0, atol=1e-15)
        accept_response(self.request, outputs[0][1], self.view['context']['revision'], self.queue.model_id)

    def test_search_retires_all_pending_paths_without_backing_up_expired_result(self):
        bot = object.__new__(MaskedISMCTS)
        bot.config = teacher_search_config({'pending_paths': 2})
        bot.time_budget_primary = True
        bot.deadline = time.monotonic()+120
        bot.cancelled = Event(); bot.owner = get_ident(); bot.lanes = [0, 1]
        bot.completed = bot.interrupted = bot.max_pending = 0
        bot.metrics = dict(coalesced=0)
        bot.statistics = lambda choices: copy.deepcopy(bot.metrics)
        retired = []
        def trajectory(view, choices, lane):
            try:
                yield (str(lane), self.policy, view, choices, self.inputs[0][2])
                self.fail('Expired result reached a search trajectory')
            finally:
                retired.append(lane)
        bot.trajectory = trajectory
        with inference_deadline([self.policy], bot.deadline), \
             patch.object(Request, 'stopped', side_effect=[None, 'expired']):
            statistics = bot.run(self.view, self.choices)
        self.assertEqual(sorted(retired), [0, 1])
        self.assertEqual((statistics['attempted'], bot.completed, bot.interrupted), (2, 0, 2))


if __name__ == '__main__':
    unittest.main()
