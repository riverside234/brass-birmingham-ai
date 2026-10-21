"""Budget parity and first-failure attribution for mixed historical models."""
import threading
import time
import unittest
from unittest.mock import patch

import numpy as np
import torch

from review_fixture import current_position as position
from brass_ai.learning.batching import Evaluator, Request
from brass_ai.learning.model import ModelConfig
from brass_ai.learning.reward import specification
from brass_ai.learning.utility_model import UtilityNet, utility_config
from brass_ai.recovery.config import EXECUTION
from brass_ai.recovery import inference


class InferenceFollowupTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(2)

    def model(self):
        reward = specification(dict(win=1., placement=.1, vp=.001), dict(event_cost=.005, max_cost=.025))
        return UtilityNet(utility_config(ModelConfig.preset('small'), reward))

    def test_local_evaluator_rejects_learner_budget_for_older_opponent(self):
        evaluator = Evaluator(self.model())
        self.addCleanup(evaluator.close)
        request = Request('old-cap', evaluator.model_id, *position(), time.monotonic()+30, remaining=(.075,)*4)
        with self.assertRaisesRegex(ValueError, 'Invalid inference utility budget'):
            evaluator.predict([request])

    def test_local_and_remote_agree_on_valid_default_explicit_and_spent_budgets(self):
        model = self.model(); local = Evaluator(model)
        self.addCleanup(local.close)
        with inference.InferenceService(dict(EXECUTION), 'cpu') as service:
            identity = service.register(model); endpoint = service.channels(1).acquire()
            try:
                remote = inference.RemoteEvaluator(endpoint, identity)
                requests = [Request('valid-'+str(i), identity, *position(i), time.monotonic()+30, remaining=budget)
                    for i, budget in enumerate((None, (.025,)*4, (.02, .01, 0., .025)))]
                for actual, expected in zip(local.predict(requests), remote.predict(requests)):
                    for key in ('probabilities', 'utility', 'win_share', 'placement', 'vp'):
                        np.testing.assert_allclose(actual[key], expected[key], rtol=1e-5, atol=2e-5)
            finally: endpoint.close()

    def pool(self, routing):
        config = dict(EXECUTION, processes=1, hosts_per_process=1, inference_routing=routing)
        with patch.object(torch.cuda, 'is_available', return_value=True), \
             patch.object(torch.cuda, 'device_count', return_value=2), \
             patch.object(inference, '_single_service', side_effect=lambda c,d,p,s: inference.InferenceService(c,'cpu')):
            service = inference.InferenceService(config, 'cpu', inference_devices=['cuda:0','cuda:1'])
        self.addCleanup(service.__exit__, None, None, None)
        return service

    def test_other_replica_broken_reply_cannot_replace_original_error(self):
        for routing in ('demand', 'round_robin'):
            with self.subTest(routing=routing):
                service = self.pool(routing)
                service.services[1]._failed(ValueError('Invalid inference utility budget'))
                service.services[0]._failed(BrokenPipeError('actor exited'))
                self.assertEqual(str(service.error), 'Invalid inference utility budget')
                self.assertEqual(service.report()['error'], 'Invalid inference utility budget')
                self.assertTrue(service.cpu.aborted.value)

    def test_router_queue_failure_cannot_replace_original_replica_error(self):
        import queue
        service = self.pool('demand')
        service.services[1]._failed(ValueError('original replica failure'))
        # Reproduce the router's generic secondary error after that replica
        # stops draining. No sleeps or real GPU are needed.
        message = dict(lane=0, token='a'*32,
                       frames=[dict(model='fixture', deadline=time.monotonic()+30)])
        service.affinity[('fixture',)] = 1
        with patch.object(service.requests, 'get', side_effect=[message, queue.Empty]), \
             patch.object(service.services[1].requests, 'put', side_effect=queue.Full):
            service._dispatch()
        self.assertIn('Inference replica failed while routing', str(service.router_error))
        self.assertEqual(str(service.error), 'original replica failure')

    def test_router_failure_is_kept_when_it_precedes_replica_cleanup(self):
        service = self.pool('demand')
        with patch.object(service.requests, 'get', side_effect=ValueError('invalid router envelope')):
            service._dispatch()
        service.services[0]._failed(BrokenPipeError('actor exited'))
        self.assertEqual(str(service.error), 'invalid router envelope')
        self.assertTrue(service.cpu.aborted.value)

    def test_router_cannot_overtake_a_replica_recording_its_failure(self):
        import queue
        service = self.pool('demand'); replica = service.services[1]
        recording = threading.Event(); release = threading.Event()
        def delayed_receipt(error):
            recording.set()
            release.wait(5)
            service._record_failure(error)
        replica.on_failure = delayed_receipt
        failure = threading.Thread(target=replica._failed,
                                   args=(ValueError('original replica failure'),), daemon=True)
        failure.start()
        try:
            self.assertTrue(recording.wait(5), 'Replica did not reach its failure receipt')
            message = dict(lane=0, token='a'*32,
                           frames=[dict(model='fixture', deadline=time.monotonic()+30)])
            service.affinity[('fixture',)] = 1
            attempts = []
            def full_queue(*args, **kwargs):
                attempts.append(None)
                if len(attempts)>1:
                    release.set(); failure.join(timeout=5)
                    self.assertFalse(failure.is_alive(), 'Replica failed to finish recording')
                raise queue.Full
            with patch.object(service.requests, 'get', return_value=message), \
                 patch.object(replica.requests, 'put', side_effect=full_queue):
                service._dispatch()
        finally:
            release.set(); failure.join(timeout=5)
        self.assertFalse(failure.is_alive())
        self.assertIn('Inference replica failed while routing', str(service.router_error))
        self.assertEqual(str(service.error), 'original replica failure')

    def test_real_replica_budget_failure_survives_a_different_dead_reply(self):
        from test_inference_failure_review import DeadReply
        from brass_ai.recovery.execution import ActorPool
        with self.pool('round_robin') as service:
            identity = service.register(self.model()); lease = service.channels(2)
            endpoints = [lease.acquire(), lease.acquire()]
            try:
                client = inference.RemoteEvaluator(endpoints[1], identity)
                with self.assertRaisesRegex(ValueError, 'Invalid inference utility budget'):
                    client.predict([Request('fatal', identity, *position(), time.monotonic()+30, remaining=(.075,)*4)])
                original = service.services[0].connections[endpoints[0].lane]
                service.services[0].connections[endpoints[0].lane] = DeadReply()
                try: service.services[0]._reply(dict(lane=endpoints[0].lane), {})
                finally: service.services[0].connections[endpoints[0].lane] = original
                self.assertEqual(str(service.error), 'Invalid inference utility budget')
                pool = ActorPool.__new__(ActorPool); pool.service = service
                with patch.object(pool, '_collect', side_effect=ValueError('CPU admission aborted')):
                    with self.assertRaisesRegex(ValueError, 'Inference service failed: Invalid inference utility budget'):
                        pool.collect([dict(index=0)])
            finally:
                for endpoint in endpoints: endpoint.close()


if __name__ == '__main__': unittest.main()
