"""Cancellation and finalization checks for time-primary private search forests."""
import copy
import random
from threading import Event, get_ident
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from fixtures import position
from test_teacher_schedule import Clock, evidence
from brass_ai.recovery.config import TEACHER_TIME_BUDGET, teacher_search_config
from brass_ai.recovery.parallel import parallelize
from brass_ai.recovery.teacher_schedule import training_teacher
from brass_ai.search.ismcts import MaskedISMCTS, MaskedSearchPolicy


class SearchRetirementReviewTests(unittest.TestCase):
    def bot(self, clock):
        bot = object.__new__(MaskedISMCTS)
        bot.config = teacher_search_config({'pending_paths': 2})
        bot.time_budget_primary = True
        bot.deadline = clock.now + 12.
        bot.cancelled = Event(); bot.owner = get_ident(); bot.lanes = [None, None]
        bot.completed = bot.interrupted = bot.max_pending = 0
        bot.metrics = dict(coalesced=0)
        bot.statistics = lambda choices: bot.metrics
        return bot

    def test_cancelled_search_does_not_spin_until_the_whole_turn_deadline(self):
        clock = Clock(); clock.now = 100.
        bot = self.bot(clock); bot.cancelled.set()
        launched = []
        def trajectory(*args):
            launched.append(True); clock.now += .01
            bot.check()
            yield from ()
        bot.trajectory = trajectory
        with patch('brass_ai.search.ismcts.time.monotonic', clock):
            result = bot.run({}, [])
        self.assertEqual(launched, [])
        self.assertEqual((result['attempted'], bot.interrupted), (0, 0))
        self.assertEqual(clock.now, 100.)

    def test_cancel_mid_launch_retires_pending_paths_without_new_attempts(self):
        clock = Clock(); clock.now = 100.
        bot = self.bot(clock); launched = []; retired = []
        def trajectory(*args):
            launched.append(True); clock.now += .01
            bot.check()
            try:
                bot.cancelled.set()
                yield ('leaf', None, {}, [], None)
                raise AssertionError('Cancelled path resumed')
            finally:
                retired.append(True)
        bot.trajectory = trajectory
        with patch('brass_ai.search.ismcts.time.monotonic', clock):
            result = bot.run({}, [])
        self.assertEqual((len(launched), len(retired)), (1, 1))
        self.assertEqual((result['attempted'], bot.completed, bot.interrupted), (1, 0, 1))
        self.assertAlmostEqual(clock.now, 100.01)

    def forest(self):
        view, choices = position()
        stats = evidence(view, choices, [])
        stats['opponent_choices'] = 7
        stats['batching'] = dict(attempted=stats['completed'])
        def policy():
            p = object.__new__(MaskedSearchPolicy)
            p.config = teacher_search_config({})
            p.policy_id = 'fixture'; p.game = object(); p.simulation_games = []
            p.cancelled = Event(); p.reward_context = None; p.rollouts = []
            p.last_statistics = None; p.last_continuation = {}
            p.base = SimpleNamespace(request_deadline=None, contexts={}, policy_id='fixture',
                evaluator=SimpleNamespace(model_id='a'*64),
                predict=lambda inputs: [(np.array([.6, .3, .1]), {})])
            p.telemetry = dict(searches=0, simulations=0, incomplete=0, max_pending=0, opponent_choices=0)
            p.context_features = lambda *args: ([0.]*4, [0.]*8)
            return p
        p = parallelize(policy(), 4, 'lab', lambda i: policy())
        self.addCleanup(p.pool.shutdown)
        training_teacher(p, TEACHER_TIME_BUDGET)
        p.incumbent = lambda *args: {}
        return p, view, choices, stats

    def test_private_workers_only_publish_statistics_and_coordinator_builds_one_label(self):
        planner, view, choices, stats = self.forest()
        with patch('brass_ai.search.ismcts.MaskedISMCTS') as bot, \
             patch('brass_ai.step5.league.implementation_id', return_value='b'*64) as fingerprint:
            bot.return_value.run.side_effect = lambda *args: copy.deepcopy(stats)
            selected, _, teacher = planner.decide(view, choices, random.Random(7), float('inf'), [])
        self.assertEqual(fingerprint.call_count, 1)
        self.assertEqual(teacher['samples'], 4*stats['completed'])
        self.assertEqual(selected, 0)
        self.assertEqual(planner.last_statistics['completed'], 4*stats['completed'])

    def test_merged_opponent_work_is_included_in_forest_telemetry(self):
        planner, view, choices, stats = self.forest()
        with patch('brass_ai.search.ismcts.MaskedISMCTS') as bot, \
             patch('brass_ai.step5.league.implementation_id', return_value='b'*64):
            bot.return_value.run.side_effect = lambda *args: copy.deepcopy(stats)
            planner.decide(view, choices, random.Random(7), float('inf'), [])
        self.assertEqual(planner.telemetry['opponent_choices'], 28)

    def test_previous_time_source_resumes_exact_learning_state(self):
        import torch
        from brass_ai.learning.contracts import SOURCE_ROOT
        from brass_ai.learning.checkpoint import resume_optimizer
        from brass_ai.recovery.resume import load_resume, commit_transition, check_signature
        from test_time_resume import signatures
        from test_teacher_work_resume import TeacherWorkResumeTests
        torch.set_num_threads(2)
        fixture = TeacherWorkResumeTests(); self.addCleanup(fixture.doCleanups)
        def same_recipe():
            _, new = signatures(); old = copy.deepcopy(new)
            old['implementation'] = '6a4fce0541ce50802bc009330db2bf7026134ef40ced967f5a3b6863981dba62'
            return old, new
        fixture.signatures = same_recipe
        run, state, new, original, optimizer = fixture.fixture()
        checkpoint = run/'generation-0070.pt'; before = checkpoint.read_bytes()
        model, raw, changed = load_resume(SOURCE_ROOT, run, state, new, reviewed=True)
        self.assertTrue(changed); self.assertFalse(model.utility_calibration['ready'])
        for key, value in original.state_dict().items():
            torch.testing.assert_close(model.state_dict()[key], value, rtol=0, atol=0)
        restored = torch.optim.AdamW(model.parameters(), lr=new['config']['learning_rate'])
        resume_optimizer(raw, restored)
        self.assertTrue(torch.equal(torch.get_rng_state(), raw['rng']))
        self.assertEqual(restored.state_dict()['param_groups'], optimizer.state_dict()['param_groups'])
        for key, values in optimizer.state_dict()['state'].items():
            for name, value in values.items():
                torch.testing.assert_close(restored.state_dict()['state'][key][name], value, rtol=0, atol=0)
        updated = commit_transition(run, state, new)
        for key, value in state.items():
            if key != 'signature': self.assertEqual(updated[key], value, key)
        self.assertEqual(updated['source_transitions'][-1]['config_changes'], {})
        self.assertEqual(checkpoint.read_bytes(), before)
        self.assertFalse(load_resume(SOURCE_ROOT, run, updated, new)[2])
        other = copy.deepcopy(new); other['config']['learning_rate'] *= 2
        with self.assertRaisesRegex(ValueError, 'unchanged config'):
            check_signature(state, other, True)


if __name__ == '__main__':
    unittest.main()
