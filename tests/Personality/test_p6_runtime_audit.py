"""P6 service failures and resource lifetime; no installed game or real model."""
import io
import json
from pathlib import Path
from threading import Event, RLock
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from brass_ai.learning.contracts import SOURCE_ROOT
from brass_ai.personality.laptop import server
from brass_ai.personality.laptop.contracts import PROTOCOL
from brass_ai.personality.laptop.inference import PortableInference
from brass_ai.personality.laptop.runtime import Runtime
from brass_ai.personality.search.contracts import Profile
from brass_ai.search.common import SearchConfig


class P6RuntimeAuditTests(unittest.TestCase):
    def test_nonobject_requests_fail_without_killing_service(self):
        messages = [[], None, 42, 'not an envelope',
                    dict(protocol=PROTOCOL, id='hello', op='hello'),
                    dict(protocol=PROTOCOL, id='close', op='close')]
        incoming = io.StringIO(''.join(json.dumps(x) + '\n' for x in messages))
        outgoing = io.StringIO()
        with patch.object(server, 'validate', return_value=dict(id='fixture', fixture=True)), \
                patch.object(server, 'Runtime', side_effect=AssertionError('No planning requested')), \
                patch.object(server.sys, 'stdin', incoming), patch.object(server.sys, 'stdout', outgoing):
            server.serve('unused', fixture=True)
        replies = [json.loads(line) for line in outgoing.getvalue().splitlines()]
        self.assertEqual(len(replies), 5)
        for reply in replies[:4]:
            self.assertFalse(reply['ok'])
            self.assertIsNone(reply['id'])
        self.assertEqual(replies[-1]['id'], 'hello')
        self.assertTrue(replies[-1]['ok'])

    def runtime(self):
        runtime = object.__new__(Runtime)
        runtime.root = SOURCE_ROOT
        runtime.mono = None
        runtime.lock = RLock()
        runtime.interrupted = Event()
        runtime.cancelled = Event()
        runtime.active = None
        runtime.sessions = {}
        runtime.channels = {}
        runtime.closed = False
        runtime.game = Mock()
        runtime.package = dict(searches=dict(easy=SearchConfig(workers=2).to_dict()),
                               models=[dict(model_id='fixture-model')], calibration={})
        runtime.service = PortableInference(Path('.'), [])
        return runtime

    def test_failed_forest_closes_persistent_inference_resources(self):
        runtime = self.runtime()
        opened = []

        def fail(root, service, channels, *args, **kwargs):
            opened.append(channels.acquire())
            raise OSError('Injected native forest construction failure')

        try:
            with patch('brass_ai.personality.laptop.runtime.SearchSession', side_effect=fail):
                with self.assertRaisesRegex(OSError, 'Injected native forest'):
                    runtime.session(0, 'easy', Profile())
            self.assertTrue(opened[0].connection.closed)
            self.assertTrue(runtime.service.closed)
            self.assertTrue(all(connection.closed for connection in runtime.service.connections.values()))
            runtime.game.close.assert_called_once_with()
            runtime.close()
            runtime.game.close.assert_called_once_with()
        finally:
            runtime.service.__exit__()

    def test_partial_channel_allocation_retires_service(self):
        runtime = self.runtime()
        allocate = runtime.service._channel
        count = 0

        def fail():
            nonlocal count
            count += 1
            if count == 1:
                return allocate()
            # Let the Windows feeder finish copying the first real handle
            # before teardown closes it. The partial service allocations are
            # still live and must be retired by Runtime.session's failure path.
            runtime.service.leases[-1].get(timeout=5)[0].close()
            raise OSError('Injected partial channel allocation failure')

        try:
            with patch.object(runtime.service, '_channel', side_effect=fail):
                with self.assertRaisesRegex(OSError, 'Injected partial channel'):
                    runtime.session(0, 'easy', Profile())
            self.assertTrue(runtime.service.connections, 'No partial allocation was exercised')
            self.assertTrue(runtime.service.closed)
            self.assertTrue(all(connection.closed for connection in runtime.service.connections.values()))
            runtime.game.close.assert_called_once_with()
        finally:
            runtime.service.__exit__()

    def test_service_recreates_runtime_after_failed_forest(self):
        ready = Event()

        class Output(io.StringIO):
            def flush(self):
                ready.set()

        first = SimpleNamespace(closed=False, interrupted=Event(), cancelled=Event())
        second = SimpleNamespace(closed=False, interrupted=Event(), cancelled=Event())

        def fail(body):
            first.closed = True
            raise OSError('Native forest failed and runtime retired')

        first.plan = Mock(side_effect=fail)
        first.interrupt = Mock()
        first.close = Mock()
        second.plan = Mock(side_effect=lambda body: dict(request_id=body['request_id'], status='ok'))
        second.interrupt = Mock()
        second.close = Mock()

        def incoming():
            for identity in ('failed', 'retry'):
                yield json.dumps(dict(protocol=PROTOCOL, id=identity, op='plan',
                                      body=dict(request_id=identity))) + '\n'
                self.assertTrue(ready.wait(5), 'Worker did not publish its reply')
                ready.clear()
            yield json.dumps(dict(protocol=PROTOCOL, id='close', op='close')) + '\n'

        outgoing = Output()
        with patch.object(server, 'validate', return_value=dict(id='fixture', fixture=True)), \
                patch.object(server, 'Runtime', side_effect=[first, second]) as constructor, \
                patch.object(server.sys, 'stdin', incoming()), patch.object(server.sys, 'stdout', outgoing):
            server.serve('unused', fixture=True)
        replies = [json.loads(line) for line in outgoing.getvalue().splitlines()]
        self.assertEqual([(r['id'], r['ok']) for r in replies], [('failed', False), ('retry', True)])
        self.assertEqual(constructor.call_count, 2)
        second.close.assert_called_once_with()


if __name__ == '__main__':
    unittest.main()
