"""Search attempt receipts remain exact across supported budget modes."""
from dataclasses import replace
from pathlib import Path
from threading import Event, get_ident
import unittest
from unittest.mock import Mock, patch

from brass_ai.search.common import SearchConfig
from brass_ai.search.ismcts import MaskedISMCTS
from brass_ai.search.ismcts import MaskedSearchPolicy
from brass_ai.recovery.parallel import parallelize
from brass_ai.searchlab.spiel import BrassSpielGame


class SearchAuditTests(unittest.TestCase):
    def bot(self, timed=False):
        bot = object.__new__(MaskedISMCTS)
        bot.config = replace(SearchConfig(), simulations=1, pending_paths=1)
        bot.time_budget_primary = timed
        bot.deadline = 12.
        bot.cancelled = Event(); bot.owner = get_ident(); bot.lanes = [None]
        bot.completed = bot.interrupted = bot.max_pending = 0
        bot.metrics = dict(coalesced=0, inference_seconds=0., batches=0, positions=0, max_batch=0)
        bot.statistics = lambda choices: dict(bot.metrics, completed=bot.completed, interrupted=bot.interrupted)
        return bot

    def test_cancel_before_first_launch_never_invents_an_interrupted_attempt(self):
        for timed in (False, True):
            with self.subTest(time_primary=timed):
                bot = self.bot(timed); bot.cancelled.set()
                with patch('brass_ai.search.ismcts.time.monotonic', return_value=0.):
                    result = bot.run({}, [])
                self.assertEqual((result['attempted'], result['completed'], result['interrupted']), (0, 0, 0))

    def test_loop_boundary_expiry_counts_every_retired_pending_attempt(self):
        for timed in (False, True):
            with self.subTest(time_primary=timed):
                bot = self.bot(timed); clock = [0.]; retired = []
                def trajectory(*args):
                    try:
                        yield ('leaf', None, {}, [], None)
                        clock[0] = 13.
                        yield ('late_leaf', None, {}, [], None)
                    finally:
                        retired.append(True)
                bot.trajectory = trajectory
                with patch('brass_ai.search.ismcts.time.monotonic', side_effect=lambda: clock[0]), \
                     patch('brass_ai.search.ismcts.predict_groups', return_value=[[(None, None)]]):
                    result = bot.run({}, [])
                self.assertEqual(retired, [True])
                self.assertEqual((result['attempted'], result['completed'], result['interrupted']), (1, 0, 1))

    def test_adapter_initialization_failure_closes_the_started_host(self):
        for failure in ('composition', 'reset', 'merchant_inventory'):
            with self.subTest(failure=failure):
                reads = [dict(extension='brass-search-chance/1.0.0'),
                         FileNotFoundError('missing card composition') if failure == 'composition' else {}]
                with patch('brass_ai.searchlab.spiel.RulesHost') as host, \
                     patch('brass_ai.searchlab.spiel.read_json', side_effect=reads):
                    if failure == 'reset':
                        host.return_value.reset.side_effect = RuntimeError('initial reset failed')
                    host.return_value.call.return_value = dict(setup=dict(merchants=[]))
                    with self.assertRaises((FileNotFoundError, RuntimeError, ValueError)):
                        BrassSpielGame(root=Path(__file__).resolve().parents[2])
                    host.return_value.close.assert_called_once_with()

    def test_forest_initialization_failure_closes_coordinator_and_created_workers(self):
        policy = object.__new__(MaskedSearchPolicy)
        policy.config = SearchConfig()
        policy.cancelled = Event(); policy.contexts = {}
        policy.game = Mock(); policy.simulation_games = [Mock(), Mock()]
        policy.base = Mock(); policy.rollouts = [Mock()]
        worker = Mock()
        factory = Mock(side_effect=[worker, RuntimeError('replica startup failed')])
        with self.assertRaisesRegex(RuntimeError, 'replica startup failed'):
            parallelize(policy, 4, 'lab', factory)
        worker.close.assert_called_once_with()
        for resource in [policy.game, policy.base] + policy.simulation_games + policy.rollouts:
            resource.close.assert_called_once_with()
        self.assertTrue(policy.cancelled.is_set())


if __name__ == '__main__':
    unittest.main()
