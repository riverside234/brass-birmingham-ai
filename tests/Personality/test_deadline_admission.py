"""Launch admission must never turn a partial rollout into completed evidence."""
from collections import defaultdict
from threading import Event, get_ident
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from brass_ai.search.common import SearchConfig
from brass_ai.search.deadline import DeadlineAdmission
from brass_ai.search.ismcts import MaskedISMCTS


class DeadlineAdmissionTests(unittest.TestCase):
    def test_local_bounded_optimistic_window(self):
        admission = DeadlineAdmission()
        for duration in (1., 2., 3.): admission.observe(duration)
        self.assertTrue(admission.admit(.1))
        admission.observe(100.)
        self.assertEqual(admission.estimate, 1.)
        self.assertFalse(admission.admit(.9)); self.assertTrue(admission.admit(1.))
        for duration in (float('nan'), float('inf'), 0., -1.): admission.observe(duration)
        self.assertEqual(admission.estimate, 1.)
        for _ in range(8): admission.observe(5.)
        self.assertEqual(admission.estimate, 5.)
        self.assertFalse(admission.admit(0.))

    def bot(self, *, admission=True, pending=1):
        clock = SimpleNamespace(now=0.)
        bot = object.__new__(MaskedISMCTS)
        bot.config = SearchConfig(pending_paths=pending, simulations=50)
        bot.deadline = 4.5; bot.cancelled = Event(); bot.owner = get_ident()
        bot.lanes = [None]*pending; bot.deadline_admission = admission
        bot.completed = bot.interrupted = bot.max_pending = 0
        bot.metrics = defaultdict(int); bot.statistics = lambda choices: dict(bot.metrics)
        return bot, clock

    def test_skip_late_launch_without_fabricating_an_attempt(self):
        for admission, expected in ((True, (4, 4, 0)), (False, (5, 4, 1))):
            bot, clock = self.bot(admission=admission)
            def trajectory(*args):
                clock.now += 1.
                bot.check(); bot.completed += 1
                yield from ()
            bot.trajectory = trajectory
            with patch('brass_ai.search.ismcts.time.monotonic', lambda: clock.now):
                stats = bot.run({}, [])
            self.assertEqual((stats['attempted'], bot.completed, bot.interrupted), expected)
            self.assertEqual(stats['deadline_stops'], int(admission))

    def test_pending_paths_drain_when_new_launch_is_withheld(self):
        bot, clock = self.bot(pending=2); retired = []
        def trajectory(*args):
            try:
                yield ('shared', None, {}, [], None)
                clock.now += .4
                bot.check(); bot.completed += 1
            finally:
                retired.append(True)
        bot.trajectory = trajectory
        def predict(parts):
            clock.now += .2
            return [[None for _ in rows] for _, rows in parts]
        with patch('brass_ai.search.ismcts.time.monotonic', lambda: clock.now), \
                patch('brass_ai.search.ismcts.predict_groups', predict):
            stats = bot.run({}, [])
        self.assertEqual(stats['attempted'], len(retired))
        self.assertEqual(stats['attempted'], bot.completed+bot.interrupted)
        self.assertEqual(bot.interrupted, 0)
        self.assertEqual(stats['deadline_stops'], 1)
        self.assertLess(clock.now, bot.deadline)

    def test_cancellation_launches_nothing(self):
        bot, clock = self.bot(); bot.cancelled.set()
        bot.trajectory = lambda *args: self.fail('cancelled search started work')
        with patch('brass_ai.search.ismcts.time.monotonic', lambda: clock.now):
            stats = bot.run({}, [])
        self.assertEqual((stats['attempted'], bot.completed, bot.interrupted), (0, 0, 0))


if __name__ == '__main__':
    unittest.main()
