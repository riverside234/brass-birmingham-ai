"""Inference shutdown and failure isolation; no GPU or long games required."""
import multiprocessing
import queue
import threading
import time
import unittest
from unittest.mock import patch

import torch

from review_fixture import current_position as position
from brass_ai.learning.batching import Request
from brass_ai.learning.model import FourSeatNet, ModelConfig
from brass_ai.recovery.config import EXECUTION
from brass_ai.recovery import inference


def close_with_orphaned_input(results):
    service = inference.InferenceService(dict(EXECUTION), 'cpu')
    # No consumer exists, as after an inference-thread failure. This exceeds
    # the OS pipe capacity and reliably leaves an active queue feeder.
    for _ in range(4):
        service.requests.put(b'x' * 1024 * 1024)
    time.sleep(.05)
    service.__exit__(None, None, None)
    results.put(service.closed)


class DeadReply:
    def __init__(self): self.sends = 0
    def send(self, response):
        self.sends += 1
        raise BrokenPipeError('actor exited')
    def close(self): pass


class FullReplica:
    def __init__(self): self.entered = threading.Event()
    def put(self, message, timeout=None):
        self.entered.set()
        # An unbounded put would hang indefinitely. The router must supply a
        # timeout and check its shutdown event between attempts.
        if timeout is None:
            raise AssertionError('uninterruptible router put')
        time.sleep(timeout)
        raise queue.Full


class InferenceFailureReviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(2)

    def test_orphaned_input_feeder_does_not_hang_shutdown(self):
        mp = multiprocessing.get_context('spawn')
        results = mp.Queue()
        process = mp.Process(target=close_with_orphaned_input, args=(results,))
        try:
            process.start()
            self.assertTrue(results.get(timeout=30))
            process.join(timeout=10)
            self.assertEqual(process.exitcode, 0)
        finally:
            if process.is_alive(): process.terminate(); process.join(timeout=5)
            results.cancel_join_thread(); results.close()

    def test_dead_reply_does_not_duplicate_or_starve_other_lanes(self):
        config = dict(EXECUTION, processes=1, hosts_per_process=1, batch_size=8, flush_ms=100.)
        service = inference.InferenceService(config, 'cpu')
        identity = service.register(FourSeatNet(ModelConfig.preset('small')))
        lease = service.channels(3)
        endpoints = [lease.acquire() for _ in range(3)]
        clients = [inference.RemoteEvaluator(endpoint, identity) for endpoint in endpoints]
        tickets = [client.submit([Request('reply-' + str(i), identity, *position(i), time.monotonic() + 60)])
                   for i, client in enumerate(clients)]
        dead = DeadReply()
        replaced = service.connections[endpoints[1].lane]
        service.connections[endpoints[1].lane] = dead
        try:
            with service:
                self.assertEqual(clients[0].poll(tickets[0], wait=True)[0]['status'], 'ok')
                self.assertTrue(endpoints[2].connection.poll(10), 'Healthy lane lost its reply')
                self.assertEqual(clients[2].poll(tickets[2], wait=True)[0]['status'], 'ok')
                self.assertFalse(endpoints[0].connection.poll(.1), 'Successful reply was sent twice')
                self.assertEqual(dead.sends, 1)
                self.assertIn('actor exited', str(service.error))
        finally:
            replaced.close()
            for endpoint in endpoints: endpoint.close()
            service.__exit__(None, None, None)

    def test_endpoint_close_releases_reply_handle(self):
        with inference.InferenceService(dict(EXECUTION), 'cpu') as service:
            endpoint = service.channels(1).acquire()
            endpoint.slot()
            endpoint.close(); endpoint.close()
            self.assertTrue(endpoint.connection.closed)
            self.assertIsNone(endpoint._slot)

    def test_first_inference_error_survives_later_broken_replies(self):
        with inference.InferenceService(dict(EXECUTION), 'cpu') as service:
            service._failed(ValueError('Invalid inference utility budget'))
            service._failed(BrokenPipeError('actor exited'))
            self.assertEqual(str(service.error), 'Invalid inference utility budget')
            self.assertIsNone(service.error.__context__)
            self.assertTrue(service.cpu.aborted.value)

    def test_saturated_replica_routing_is_interruptible(self):
        config = dict(EXECUTION, processes=1, hosts_per_process=1, inference_routing='demand')
        with patch.object(torch.cuda, 'is_available', return_value=True), \
             patch.object(torch.cuda, 'device_count', return_value=2), \
             patch.object(inference, '_single_service', side_effect=lambda c,d,p,s: inference.InferenceService(c,'cpu')):
            service = inference.InferenceService(config, 'cpu', inference_devices=['cuda:0','cuda:1'])
        original = service.services[0].requests
        full = FullReplica()
        service.services[0].requests = full
        try:
            service.requests.put(dict(lane=0, token='a'*32,
                frames=[dict(model='test', deadline=time.monotonic()+60)]))
            service.router.start(); service.router_started = True
            self.assertTrue(full.entered.wait(5))
            service.router_stop.set(); service.router.join(timeout=2)
            self.assertFalse(service.router.is_alive())
            self.assertIsNone(service.router_error)
            self.assertEqual(service.loads, [0,0])
        finally:
            service.router_stop.set()
            service.services[0].requests = original
            service.__exit__(None, None, None)


if __name__ == '__main__': unittest.main()
