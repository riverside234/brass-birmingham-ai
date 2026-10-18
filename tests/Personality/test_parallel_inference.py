"""CPU replicas keep named outputs, deadlines and private slot retirement."""
from pathlib import Path
from threading import Barrier, Event, Thread
import time
import unittest
from unittest.mock import patch
import uuid

import numpy as np

from brass_ai.learning.features import Encoded, feature_schema
from brass_ai.personality.laptop.inference import PortableInference
from brass_ai.recovery.inference import Endpoint


class FakeGraph:
    gate = None
    entered = None
    release = None
    fail = False

    def __init__(self, path, backend, threads):
        schema = feature_schema()
        self.threads = threads
        self.inputs = ['states', 'actions', 'legal_mask', 'stable_to_relative']
        self.meta = dict(model_id='fixture', actor=False, config=dict(
            state_dim=schema['state_dim'], action_dim=schema['action_dim'],
            utility=dict(insolvency=dict(max_cost=.075))))

    def run(self, inputs):
        if self.gate is not None:
            self.gate.wait(timeout=5)
        if self.entered is not None:
            self.entered.set()
            if not self.release.wait(timeout=5):
                raise RuntimeError('test gate timed out')
        if self.fail:
            raise RuntimeError('injected graph failure')
        values = np.repeat(inputs['states'][:, :1], 4, axis=1)
        return dict(policy_logits=np.where(inputs['legal_mask'], 0., -np.inf),
                    win_share_4=values, placement_4=values, vp_4=values,
                    remaining_utility_4=values)


class ParallelInferenceTests(unittest.TestCase):
    def setUp(self):
        FakeGraph.gate = FakeGraph.entered = FakeGraph.release = None
        FakeGraph.fail = False
        self.graph = patch('brass_ai.personality.laptop.inference.Graph', FakeGraph)
        self.check = patch('brass_ai.personality.laptop.inference.check_backend', return_value={'passed': True})
        self.graph.start(); self.check.start()
        self.addCleanup(self.graph.stop); self.addCleanup(self.check.stop)

    def service(self, **kwargs):
        service = PortableInference(Path('.'), [dict(path='fixture', model_id='fixture')], **kwargs)
        self.addCleanup(service.__exit__)
        return service

    def channel(self, service):
        c, lane, shared = service._channel()
        endpoint = Endpoint(service.requests, c, lane, shared=shared)
        self.addCleanup(endpoint.close)
        return endpoint

    def submit(self, endpoint, value, *, expired=False, cancelled=False, inline=False):
        schema = feature_schema(); state = np.zeros(schema['state_dim'], np.float32); state[0] = value
        e = Encoded(state, np.zeros((3, schema['action_dim']), np.float32), np.arange(4, dtype=np.int64), ('a', 'b', 'c'))
        token = uuid.uuid4().hex
        frame = dict(request_id=token, model='fixture', revision=17, seat=0, digest='input-'+str(value),
                     deadline=time.monotonic()+(-1 if expired else 10), submitted=time.monotonic(),
                     cancelled=cancelled, encoded=e, remaining=[.075]*4)
        packed = None if inline else endpoint.slot().pack_frames([frame], token)
        endpoint.requests.put(dict(lane=endpoint.lane, token=token, shared=packed is not None,
                                   frames=packed if packed is not None else [frame]))
        return token

    def receive(self, endpoint, token):
        self.assertTrue(endpoint.connection.poll(8), 'private reply missing')
        response = endpoint.connection.recv()
        self.assertEqual(response['token'], token)
        if response.get('shared'):
            response['responses'] = endpoint.slot().unpack_responses(response['responses'], token)
        return response

    def test_independent_single_thread_sessions_overlap_and_keep_private_outputs(self):
        service = self.service(); a = self.channel(service); b = self.channel(service)
        FakeGraph.gate = Barrier(2)
        with service:
            ta = self.submit(a, .125); tb = self.submit(b, .75)
            for endpoint, token, value in ((a, ta, .125), (b, tb, .75)):
                response = self.receive(endpoint, token)['responses'][0]
                self.assertEqual(response['status'], 'ok')
                self.assertEqual(response['vp'], [value]*4)
                np.testing.assert_allclose(response['probabilities'], [1/3]*3)
                self.assertEqual(response['request_id'], token)
        report = service.report()
        self.assertEqual((report['replica_count'], report['session_threads'], report['positions']), (2, 1, 2))
        self.assertEqual([r['positions'] for r in report['replicas']], [1, 1])
        self.assertFalse(any(r.thread.is_alive() for r in service.replicas))
        self.assertTrue(all(s.closed for s in service.slots.values()))
        self.assertTrue(all(not r.graphs for r in service.replicas))

    def test_expired_and_cancelled_requests_never_reach_graph(self):
        service = self.service(); endpoint = self.channel(service); FakeGraph.fail = True
        with service:
            for keyword, expected in (('expired', 'expired'), ('cancelled', 'cancelled')):
                token = self.submit(endpoint, .5, **{keyword: True})
                self.assertEqual(self.receive(endpoint, token)['responses'][0]['status'], expected)
        self.assertIsNone(service.error)
        self.assertEqual(service.report()['positions'], 0)

    def test_graph_failure_replies_once_and_reaches_facade(self):
        service = self.service(); endpoint = self.channel(service); FakeGraph.fail = True
        with service:
            token = self.submit(endpoint, .5, inline=True)
            self.assertIn('injected graph failure', self.receive(endpoint, token)['error'])
            self.assertFalse(endpoint.connection.poll(.05))
        self.assertIn('injected graph failure', str(service.error))

    def test_shutdown_waits_for_all_graph_owners(self):
        service = self.service(); endpoint = self.channel(service)
        FakeGraph.entered = Event(); FakeGraph.release = Event()
        service.__enter__(); self.submit(endpoint, .5)
        self.assertTrue(FakeGraph.entered.wait(5))
        closed = Event(); errors = []
        def close():
            try: service.__exit__()
            except BaseException as error: errors.append(error)
            finally: closed.set()
        closer = Thread(target=close); closer.start()
        try:
            self.assertFalse(closed.wait(.05))
            self.assertFalse(any(s.closed for s in service.slots.values()))
        finally:
            FakeGraph.release.set(); closer.join(8)
        self.assertTrue(closed.is_set()); self.assertEqual(errors, [])
        self.assertTrue(all(s.closed for s in service.slots.values()))

    def test_gpu_keeps_single_session_and_cpu_allowance_is_strict(self):
        for backend in ('openvino-gpu', 'openvino-cpu'):
            service = self.service(backend=backend)
            self.assertEqual((len(service.replicas), service.session_threads), (1, 2))
        for kwargs in (dict(sessions=2, threads=2), dict(sessions=True),
                       dict(backend='openvino-gpu', sessions=2)):
            with self.assertRaises(ValueError): self.service(**kwargs)

    def test_partial_session_and_thread_start_failures_retire_allocations(self):
        service = object.__new__(PortableInference)
        count = 0
        def graph(*args):
            nonlocal count
            count += 1
            if count == 2: raise RuntimeError('second session failed')
            return FakeGraph(*args)
        with patch('brass_ai.personality.laptop.inference.Graph', side_effect=graph):
            with self.assertRaisesRegex(RuntimeError, 'second session failed'):
                PortableInference.__init__(service, Path('.'), [dict(path='fixture', model_id='fixture')])
        self.assertTrue(service.closed)
        self.assertFalse(service.graphs)
        service = self.service()
        # The peer starts first. Failure starting the facade must join that peer.
        with patch.object(service.thread, 'start', side_effect=RuntimeError('start failed')):
            with self.assertRaisesRegex(RuntimeError, 'start failed'): service.__enter__()
        self.assertTrue(service.closed)
        self.assertFalse(any(r.thread.is_alive() for r in service.replicas))


if __name__ == '__main__':
    unittest.main()
