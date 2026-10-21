"""Keep numeric inference replies in arrays until the final transport copy."""
import copy
import time
import unittest
from unittest.mock import patch

import numpy as np
import torch

from review_fixture import current_position as position
from brass_ai.learning.batching import Evaluator, Request, accept_response
from brass_ai.learning.model import FourSeatNet, ModelConfig
from brass_ai.learning.reward import specification
from brass_ai.learning.utility_model import UtilityNet, utility_config
from brass_ai.recovery.config import EXECUTION
from brass_ai.recovery.encoding import PreparedEncoder
from brass_ai.recovery.inference import InferenceService, RemoteEvaluator
from brass_ai.recovery.shared_transport import SharedSlot


FIELDS = ('probabilities', 'win_share', 'placement', 'vp', 'utility')


class ReplyArrayPerformanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        torch.manual_seed(7301)
        config = ModelConfig.preset('small')
        reward = specification(dict(win=1., placement=.1, vp=.0005),
                               dict(event_cost=.005, max_cost=.025))
        cls.models = [FourSeatNet(config).eval(), UtilityNet(utility_config(config, reward)).eval()]

    def frame(self, identity):
        view, choices = position(3)
        return dict(request_id='reply', model=identity, revision=view['context']['revision'], seat=3,
                    digest='source', deadline=time.monotonic()+60, submitted=time.monotonic(),
                    cancelled=False, encoded=PreparedEncoder().encode(view, choices), remaining=None)

    def test_private_arrays_preserve_list_values_float64_wire_and_detachment(self):
        for model in self.models:
            service = InferenceService(EXECUTION, 'cpu')
            slot = SharedSlot(size=65536)
            try:
                identity = service.register(model)
                frame = self.frame(identity)
                expected = service._evaluate([(0, frame)])[0]
                arrays = service._evaluate([(0, frame)], array_responses=True)[0]
                for key in FIELDS:
                    if key in arrays:
                        self.assertIsInstance(expected[key], list)
                        self.assertIsInstance(arrays[key], np.ndarray)
                        np.testing.assert_array_equal(arrays[key], expected[key])
                packed = slot.pack_responses([arrays], 'a'*32)
                for key in FIELDS:
                    if key in packed[0]:self.assertEqual(packed[0][key]['dtype'], '<f8')
                received = slot.unpack_responses(packed, 'a'*32)[0]
                self.assertEqual(received, expected)
                # Neither later slot reuse nor the producer's buffers may
                # change already delivered Python lists.
                saved = copy.deepcopy(received)
                for key in FIELDS:
                    if key in arrays:arrays[key][:] = 0
                slot.pack_responses([arrays], 'b'*32)
                self.assertEqual(received, saved)
            finally:
                slot.close()
                service.__exit__(None, None, None)

    def test_shared_inline_and_overflow_keep_full_detached_list_replies(self):
        for slot_bytes in (0, 128, 2*1024*1024):
            with self.subTest(slot_bytes=slot_bytes), InferenceService(
                    dict(EXECUTION, processes=1, hosts_per_process=1,
                         shared_slot_bytes=slot_bytes, max_candidate_cells=2), 'cpu') as service:
                identities = [service.register(model) for model in self.models]
                endpoint = service.channels(1).acquire()
                try:
                    for index, (model, identity) in enumerate(zip(self.models, identities)):
                        client = RemoteEvaluator(endpoint, identity)
                        request = Request(str(index), identity, *position(index), time.monotonic()+60)
                        received = client.predict([request])[0]
                        accept_response(request, received, request.view['context']['revision'], identity)
                        reference = Evaluator(model, max_candidate_cells=2)
                        try:expected = reference.predict([request])[0]
                        finally:reference.close()
                        self.assertEqual(received, expected)
                        for key in FIELDS:
                            if key in received:self.assertIsInstance(received[key], list)
                        saved = copy.deepcopy(received)
                        client.predict([request])
                        self.assertEqual(received, saved)
                    self.assertEqual(service.metrics['shared_replies'], 4 if slot_bytes > 128 else 0)
                    self.assertIsNone(service.error)
                finally:endpoint.close()

    def test_array_reply_mode_retains_nonfinite_output_rejection(self):
        service = InferenceService(EXECUTION, 'cpu')
        try:
            identity = service.register(self.models[0])
            frame = self.frame(identity)
            invalid = [np.full((1, len(frame['encoded'].choice_ids)), np.nan, np.float32)]
            invalid += [np.ones((1, 4), np.float32)]*3
            with patch('brass_ai.recovery.inference.predict_encoded', return_value=invalid):
                for arrays in (False, True):
                    with self.assertRaisesRegex(ValueError, 'Nonfinite inference output'):
                        service._evaluate([(0, frame)], array_responses=arrays)
        finally:service.__exit__(None, None, None)

    def test_old_signature_subclass_and_instrumentation_keep_list_contract(self):
        class LegacyService(InferenceService):
            def _evaluate(self, items):
                replies = super()._evaluate(items)
                assert all(isinstance(reply['probabilities'], list) for reply in replies.values())
                return replies
        for instrumented in (False, True):
            for slot_bytes in (0, 128, 65536):
                service_type = InferenceService if instrumented else LegacyService
                with self.subTest(instrumented=instrumented, slot_bytes=slot_bytes), service_type(
                        dict(EXECUTION, shared_slot_bytes=slot_bytes), 'cpu') as service:
                    if instrumented:
                        evaluate = service._evaluate
                        def measured(items):
                            replies = evaluate(items)
                            assert all(isinstance(reply['probabilities'], list) for reply in replies.values())
                            return replies
                        service._evaluate = measured
                    identity = service.register(self.models[0])
                    endpoint = service.channels(1).acquire()
                    try:
                        request = Request('legacy', identity, *position(), time.monotonic()+60)
                        reply = RemoteEvaluator(endpoint, identity).predict([request])[0]
                        accept_response(request, reply, request.view['context']['revision'], identity)
                        self.assertIsInstance(reply['probabilities'], list)
                        self.assertIsNone(service.error)
                    finally:endpoint.close()

    def test_stopped_array_replies_do_not_gain_numeric_payloads(self):
        service = InferenceService(EXECUTION, 'cpu')
        try:
            frame = self.frame('unused')
            frame['deadline'] = time.monotonic()-1
            expired = service._evaluate([(0, frame)], array_responses=True)[0]
            self.assertEqual(expired['status'], 'expired')
            frame['cancelled'] = True
            cancelled = service._evaluate([(0, frame)], array_responses=True)[0]
            self.assertEqual(cancelled['status'], 'cancelled')
            self.assertFalse(set(FIELDS) & set(expired))
            self.assertFalse(set(FIELDS) & set(cancelled))
        finally:service.__exit__(None, None, None)


if __name__ == '__main__':unittest.main()
