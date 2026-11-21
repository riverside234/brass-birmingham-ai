"""g4475 teacher training revision: 160 teacher label rows per PPO step and the search-value head's training paused
(the era heads keep the search tables), plus the soft held-out check once after PPO, before PPG."""
import copy
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from brass_ai.recovery import training_revision_g4475 as R


def before():
    from test_teacher_search_g4138 import before as g3270_recipe
    from brass_ai.recovery import training_revision_g4138 as g4138
    return g4138.configure(g3270_recipe(), 4138)


def signature(config):
    from brass_ai.rust_backend.lab_replay_compat import QUALIFIED
    return dict(config=config, implementation=QUALIFIED)


class Recipe(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.before = before(); cls.after = R.configure(cls.before, 4475)

    def test_it_overlays_160_rows_and_pauses_the_search_value_head(self):
        from brass_ai.recovery import training_revision_g3270 as g3270
        from brass_ai.recovery.config import validate_config
        from brass_ai.recovery.teacher_phase_config import buffer_identity
        from brass_ai.rust_backend.lab_teacher_activation import is_on
        before, after = self.before, validate_config(copy.deepcopy(self.after))
        self.assertEqual(after['teacher_phase'], before['teacher_phase'])  # the qualified recipe is untouched
        self.assertEqual(R.previous(after), before)
        self.assertTrue(is_on(before) and is_on(after))  # recipe recognition peels the block
        self.assertEqual(buffer_identity(after), buffer_identity(before))  # labels and beta carry over
        self.assertEqual(R.configure(after, 4476), after)  # adopted once
        options = lambda config: g3270.phase_options(config, config['teacher_phase'])
        self.assertEqual((options(after)['term']['rows_per_step'], options(before)['term']['rows_per_step']), (160, 128))
        self.assertEqual(options(after)['tables'], dict(train=44, held_out=4))  # the g4138 tables stay
        self.assertEqual(options(after)['buffer'], options(before)['buffer'])
        self.assertEqual((R.search_value_paused(after), R.search_value_paused(before)), (True, False))
        self.assertEqual(R.tag(after), 'rows160-svhead-paused')
        bad = copy.deepcopy(after); bad[R.KEY]['rows_per_step'] = 192
        with self.assertRaisesRegex(ValueError, 'g4475 teacher training revision'):validate_config(bad)
        from test_teacher_revision_g3270 import before as without_g3270
        with self.assertRaisesRegex(ValueError, 'g3270 teacher revision'):R.configure(without_g3270(), 4475)

    def test_it_is_adopted_once_and_the_tables_are_unchanged(self):
        from brass_ai.recovery.teacher_phase import plan_jobs
        from brass_ai.rust_backend.lab_shadow_config import signature_transition
        before, after = self.before, self.after
        self.assertTrue(signature_transition(signature(before), signature(after)))
        self.assertTrue(signature_transition(signature(after), signature(after)))
        self.assertFalse(signature_transition(signature(after), signature(before)))
        changed = copy.deepcopy(after); changed[R.KEY]['from_generation'] = 4480
        self.assertFalse(signature_transition(signature(after), signature(changed)))
        # Trainer-only: the same jobs, seeds and identities.
        state = lambda config: dict(iteration=4475, history=[], signature=signature(config))
        jobs = lambda config: plan_jobs(config, state(config), model_id='m'*64, export_hash='a'*64, epoch=4475, ready=True)
        self.assertEqual(jobs(after), jobs(before))


STAGE = dict(stage='C', w=1., scale=1., generation=4474, lam=.95, startup='immediate', g0_passed=True, history=[],
             ordinary=[], baseline=None, full_scale_generation=None, h2h=None)


class SearchValuePause(unittest.TestCase):
    G = 4480

    def phase(self, directory):
        from brass_ai.recovery import training_revision_g3270 as g3270, training_revision_g4138 as g4138
        from brass_ai.recovery.search_value_phase import SearchValuePhase
        from brass_ai.recovery.search_value_training import CONTRACT
        after = R.configure(before(), 4475)
        options = g3270.search_value_options(after, after['teacher_phase']['search_value'])
        return after, SearchValuePhase(Path(directory)/'values', object(), options, 'r',
            saved=dict(stage=dict(STAGE, contract=CONTRACT)), trainable=False,
            revision=g3270.value_settings(after), rebuild=g4138.rebuild(after))

    def publish(self, phase):
        row = lambda g, split, game, i: dict(generation=g, split=split, reward_id='r', game_id=game, index=i, u=[0.]*4)
        for g in range(self.G - 24, self.G + 1):
            for split, tables in (('train', 2), ('heldout', 1)):
                for t in range(tables):
                    game = f'teacher/{g}-{split}-{t}'
                    phase.buffer.publish(g, split, [row(g, split, game, i) for i in range(3)], game_id=game)

    def test_the_paused_head_does_not_train_and_aged_shards_still_retire(self):
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            _, paused = self.phase(a); _, reference = self.phase(b)
            for phase in (paused, reference): self.publish(phase)
            stage = copy.deepcopy(paused.machine.state)
            with patch.object(paused, '_scheduled_after_ppg', side_effect=AssertionError('trained')):
                report = paused.paused(self.G)
            reference.buffer.rows(self.G, 'train')  # what the training pass's read retires
            names = lambda receipts: sorted(r['file'] for r in receipts)
            self.assertEqual(names(paused.buffer.receipts), names(reference.buffer.receipts))
            self.assertEqual(names(paused.buffer.retired), names(reference.buffer.retired))
            self.assertTrue(paused.buffer.retired)  # train shards past 16 iterations and held-out ones past theirs
            self.assertEqual(paused.machine.state, stage)  # no stage advance
            self.assertFalse(report['updated']); self.assertNotIn('update', report)
            self.assertEqual(report['paused']['retired_shards'], len(paused.buffer.retired))
            self.assertEqual(sum(report['paused']['shards'].values()), len(paused.buffer.receipts))

    def test_the_iteration_skips_the_post_ppg_pass_and_keeps_the_era_heads_search_tables(self):
        from brass_ai.recovery.search_value_iteration import SearchValueIteration
        with tempfile.TemporaryDirectory() as directory:
            after, phase = self.phase(directory)
            self.publish(phase)
            iteration = SearchValueIteration.__new__(SearchValueIteration)
            iteration.mode, iteration.config, iteration.phase, iteration.generation = 'on', after, phase, self.G
            iteration.auxiliary = dict(train_iterations=4)
            iteration.report = iteration.pending_generation = iteration._publication = None
            for paused in (False, True):
                iteration.paused = paused
                phases = types.SimpleNamespace()
                with patch('brass_ai.recovery.search_auxiliary.tables', return_value='search-tables') as tables, \
                        patch('brass_ai.recovery.training_revision_g3140.search_rows', return_value=None), \
                        patch('brass_ai.recovery.training_revision_g3140.rail_end_returns_on', return_value=False):
                    iteration.prepare_ppg(phases, self.G + 1)
                self.assertEqual(phases.search_auxiliary['tables'], 'search-tables')  # the era heads' pool is read either way
                self.assertEqual(tables.call_args.args[1:], (self.G,))
                self.assertEqual(phases.post_value_fit, None if paused else iteration.post_fit)
                from brass_ai.recovery.search_value_schedule import beta_floor
                self.assertEqual(phases.beta_floor, beta_floor(after['teacher_phase']['search_value'], 'C'))  # stage kept
            with patch.object(phase, 'after_ppg', side_effect=AssertionError('trained')):
                report = iteration.finish_ppg()
            self.assertFalse(report['updated']); self.assertIn('paused', report)


class SoftAfterPpo(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from test_soft_policy_g3423 import Fixture
        Fixture.setUpClass.__func__(cls)
        cls.start = copy.deepcopy(cls.model.state_dict())

    def run_iteration(self, measure):
        """prepare, two PPO steps, [the after-PPO check], a stand-in PPG change of the trunk, finish."""
        from test_soft_policy_g3423 import soft_settings
        from brass_ai.recovery.teacher_phase import TeacherPhase
        from brass_ai.recovery.value_preservation import TrunkGradients
        self.model.load_state_dict(self.start)
        rows, heldout = self.rows, [r for _, r in self.rows[:4]]
        config = dict(seed=5, teacher_revision=1, soft_policy=soft_settings())

        class Buffer:
            def rows(self, generation): return rows
            def payload(self): return {}

        p = TeacherPhase.__new__(TeacherPhase)
        p.config, p.options, p.generation, p.weight = config, self.options, 3270, dict(beta=.03, calibrated=True)
        p.buffer, p.heldout, p.metrics, p.jobs, p.cursor = Buffer(), heldout, {}, [], 0
        p.receipt, p.reward_id, p.admission_seconds, p.soft_state = None, 'r', 0., None
        torch.manual_seed(11)
        term = p.prepare(self.model); term.plan(2)
        for step in (1, 2):
            term.apply(self.model, TrunkGradients(self.model), step); self.model.zero_grad(set_to_none=True)
        if measure: p.after_ppo(self.model)
        rng = torch.random.get_rng_state()
        with torch.no_grad():  # a stand-in for PPG moving the shared trunk
            for name, parameter in self.model.named_parameters():
                if name.startswith('policy_scorer.'): parameter.add_(.05)
        report = p.finish(self.model, {})
        weights = {k: v.clone() for k, v in self.model.state_dict().items()}
        self.model.load_state_dict(self.start)
        return report, rng, weights, p.soft_state

    def test_the_after_ppo_check_splits_the_change_and_changes_nothing_that_trains(self):
        plain, plain_rng, plain_weights, plain_state = self.run_iteration(False)
        measured, rng, weights, state = self.run_iteration(True)
        soft = measured['heldout']['soft']
        self.assertEqual((soft['after_ppo']['rows'], soft['before']['rows']), (4, 4))
        self.assertAlmostEqual(soft['ppo_delta'] + soft['ppg_delta'], soft['delta'], places=12)
        self.assertNotEqual(soft['after_ppo']['kl'], soft['before']['kl'])  # the head trained in PPO
        self.assertNotEqual(soft['after_ppo']['kl'], soft['after']['kl'])  # the trunk moved in "PPG"
        self.assertEqual(set(measured['heldout']['after_ppo']), set(measured['heldout']['before']))
        self.assertEqual(set(measured['heldout']['ppo_delta']), set(measured['heldout']['before']))
        # Measurement only: the same RNG, weights, soft head state and every other report value.
        self.assertTrue(torch.equal(rng, plain_rng))
        self.assertTrue(all(torch.equal(weights[k], plain_weights[k]) for k in weights))
        self.assertEqual(state, plain_state)
        def untimed(value):
            if isinstance(value, dict): return {k: untimed(v) for k, v in value.items() if 'seconds' not in k}
            if isinstance(value, list): return [untimed(v) for v in value]
            return value
        new = dict(after_ppo=None, ppo_delta=None, ppg_delta=None)
        clean = copy.deepcopy(measured)
        clean['heldout'] = {k: v for k, v in clean['heldout'].items() if k not in new}
        clean['heldout']['soft'] = {k: v for k, v in clean['heldout']['soft'].items() if k not in new}
        self.assertEqual(untimed(clean), untimed(plain))


if __name__ == '__main__':
    unittest.main()
