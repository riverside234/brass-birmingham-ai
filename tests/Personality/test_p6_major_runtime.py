"""P6 control isolation under deterministic late-timer scheduling."""
from pathlib import Path
import tempfile
from threading import Event
import unittest
from unittest.mock import Mock, patch

from brass_ai.learning.contracts import SOURCE_ROOT
from brass_ai.personality.laptop import benchmark


class P6MajorRuntimeTests(unittest.TestCase):
    def test_late_benchmark_control_cannot_interrupt_the_next_case(self):
        root = SOURCE_ROOT/'artifacts/p0p6-audit-20261013/tmp'
        root.mkdir(parents=True, exist_ok=True)

        class LateTimer:
            """Callback begins after plan retires but before cancel can stop it."""
            def __init__(self, delay, callback):
                self.callback = callback

            def start(self):
                pass

            def cancel(self):
                self.callback()

            def join(self):
                pass

        class FixtureRuntime:
            def __init__(self, *args, **kwargs):
                self.interrupted = Event(); self.cancelled = Event()
                self.package = dict(id='authored-package', roster=dict(archetypes=[dict(id=str(i)) for i in range(16)]))
                self.service = Mock(); self.service.report.return_value = {}

            def interrupt(self, *, cancel=False):
                self.interrupted.set()
                if cancel:
                    self.cancelled.set()

            def plan(self, request):
                cancelled = self.cancelled.is_set()
                result = dict(status='cancelled' if cancelled else 'ok', paths=[] if cancelled else [['finish']],
                              metrics=dict(completed=1, qualified_samples=0,
                                           play_now=self.interrupted.is_set(), cancelled=cancelled))
                # Runtime.plan retires workers, then clears controls in finally.
                self.interrupted.clear(); self.cancelled.clear()
                return result

            def close(self):
                pass

        case = dict(view=dict(seat=0, context=dict(era='CanalEra')), choices=[dict(kind='pass')])
        for control in ('play-now', 'cancel'):
            with self.subTest(control=control), tempfile.TemporaryDirectory(dir=root) as tmp, \
                    patch.object(benchmark, 'Runtime', FixtureRuntime), \
                    patch.object(benchmark, 'Timer', LateTimer), \
                    patch.object(benchmark, 'new_save', return_value={}):
                report = benchmark.measure('unused', [dict(case, interrupt=control), case],
                                           Path(tmp)/'benchmark.json', tiers=('easy',), fixture=True)
                result = report['results'][0]
                self.assertTrue(result['available'])
                first, ordinary = result['records']
                self.assertFalse(first['metrics']['play_now'], 'Late callback must not count as observed control')
                self.assertEqual(ordinary['status'], 'ok')
                self.assertFalse(ordinary['metrics']['play_now'])
                self.assertFalse(ordinary['metrics']['cancelled'])


if __name__ == '__main__':
    unittest.main()
