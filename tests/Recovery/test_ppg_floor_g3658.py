"""g3658: PPG beta floor .125 with a one-time beta reset, and weight=0.5,vp on the current recipe."""
import copy
from pathlib import Path
import tempfile
import unittest

import torch

from brass_ai.learning.contracts import SOURCE_ROOT
from brass_ai.recovery import ppg_g210 as P
from brass_ai.recovery import training_revision_g3140 as G
from brass_ai.recovery import training_revision_g3658 as F
from brass_ai.recovery.config import validate_config


def transition(before, after):
    from brass_ai.rust_backend.lab_shadow_config import signature_transition
    from test_training_g3140 import signature
    return signature_transition(signature(before), signature(after)), signature_transition(signature(after), signature(before))


class Recipe(unittest.TestCase):
    def test_both_floors_move_to_an_eighth_once_and_only_forward(self):
        from brass_ai.recovery import search_value_schedule as S
        from brass_ai.rust_backend.lab_teacher_activation import is_on
        from test_training_g3140 import g2880
        before = g2880()
        after = validate_config(F.configure(before))
        self.assertTrue(F.applied(after)); self.assertFalse(F.applied(before))
        self.assertEqual(sorted(k for k in before if before[k] != after[k]), ['teacher_phase', 'value_phase'])
        self.assertEqual((after['value_phase']['beta_clone_min'], S.beta_floor(after['teacher_phase']['search_value'], 'C')), (.125, .125))
        self.assertEqual(S.beta_floor(before['teacher_phase']['search_value'], 'C'), .25)
        self.assertEqual(F.configure(after), after); self.assertEqual(F.previous(after), before)
        self.assertTrue(is_on(before) and is_on(after))
        self.assertEqual(transition(before, after), (True, False))
        half = copy.deepcopy(before); half['value_phase']['beta_clone_min'] = .5
        with self.assertRaisesRegex(ValueError, 'committed .25 floor'):F.configure(half)
        with self.assertRaises(ValueError):P.validate(dict(after['value_phase'], beta_clone_min=.0625))

    def test_weight_half_and_vp_still_adopt_after_drop_rail_end_with_or_without_the_floor(self):
        from brass_ai.recovery import local_scoring_v3 as v3
        from test_rail_end_g3528 import Recipe as Rows
        dropped = validate_config(G.configure(Rows().adopted(), G.parse('drop-rail-end'), 3528))
        value = validate_config(G.configure(dropped, G.parse('weight=0.5,vp'), 3658))
        self.assertEqual(value['local_scoring_auxiliary']['weights'], dict(canal=.5, rail=.5, rail_end=0.))
        self.assertEqual(G.value_weights(value), (.25, .5))
        self.assertEqual(value['local_scoring_auxiliary']['rail_end_removed'], 3528)
        self.assertTrue(v3.config_forward(dropped['local_scoring_auxiliary'], value['local_scoring_auxiliary']))
        self.assertIn('-w05-norailend-blend50-vp05-', '-' + G.tag(value) + '-')
        self.assertEqual(transition(dropped, value), (True, False))
        both = validate_config(F.configure(value))
        self.assertEqual(transition(dropped, both), (True, False))
        self.assertEqual(validate_config(F.configure(validate_config(G.configure(both, G.parse('weight=0.5,vp'), 3670)))), both)


class Resume(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_the_first_resume_sets_beta_to_an_eighth_and_later_ones_keep_it(self):
        from brass_ai.recovery.actor_value import TrainingPhases, KEY
        from test_gae_value_phase import finish
        from test_ppg_g210 import config, prepare
        from test_value_buffer import commit
        m, _, c = config()
        c['value_phase'] = dict(P.KL_STOP_DEFAULT, max_epochs=1, batch_size=4, clone_rows=64, clone_batch_size=8)
        low = copy.deepcopy(c); low['value_phase']['beta_clone_min'] = .125
        update = dict(categorical_probe=dict(categorical_kl=0.), controller_warmup_done=True, optimizer_steps=1, kl_early_stop=False, gae={})
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT / 'artifacts') as tmp:
            run = Path(tmp); p = TrainingPhases(m, c, root=SOURCE_ROOT, run=run)
            prepare(m, p, 1); p.fit(m); finish(p, update, 1)
            p.beta = .5; payload = p.payload(); commit(run, p, 1)
            adam = copy.deepcopy(p.optimizer.state_dict())
            q = TrainingPhases(m, low, raw=dict(training={KEY: payload}), generation=1, root=SOURCE_ROOT, run=run)
            migration = q.ppg_schedule_migration
            self.assertEqual((q.beta, q.trips), (.125, p.trips))
            self.assertEqual((migration['contract'], migration['beta_before'], migration['beta_after']), (P.EIGHTH_FLOOR_MIGRATION, .5, .125))
            self.assertEqual(q.adoption, migration)
            self.assertEqual(str(q.optimizer.state_dict()), str(adam))
            kept = TrainingPhases(m, c, raw=dict(training={KEY: payload}), generation=1, root=SOURCE_ROOT, run=run)
            self.assertEqual(kept.beta, .5)  # without the revision nothing changes
            prepare(m, q, 2); q.fit(m); finish(q, update, 2)
            later = q.payload(); commit(run, q, 2)
            self.assertEqual(later['options']['value_phase']['beta_clone_min'], .125); self.assertEqual(later['adoption'], migration)
            again = TrainingPhases(m, low, raw=dict(training={KEY: later}), generation=2, root=SOURCE_ROOT, run=run)
            self.assertEqual(again.beta, q.beta); self.assertGreaterEqual(again.beta, .125)
            with self.assertRaisesRegex(ValueError, 'configuration differs'):
                TrainingPhases(m, c, raw=dict(training={KEY: later}), generation=2, root=SOURCE_ROOT, run=run)

    def test_beta_halves_down_to_the_eighth_and_kl_hits_still_raise_it(self):
        options = dict(P.KL_STOP_DEFAULT, max_epochs=1, beta_clone_min=.125)
        clean = dict(kl_guard_tripped=False, optimizer_steps=16, probe_kl=.0001)
        self.assertEqual(P.beta_after_phase(.25, [], clean, options)[0], .125)
        beta, _, feedback = P.beta_after_phase(.125, [], clean, options)
        self.assertEqual((beta, feedback['reason']), (.125, 'at_minimum'))
        hit = dict(kl_guard_tripped=True, optimizer_steps=16, steps_before_kl_cap=8, probe_kl=.003)
        self.assertEqual(P.beta_after_phase(.125, [], hit, options)[0], .1875)

    def test_the_era_heads_without_rail_end_resume_at_weight_half(self):
        from brass_ai.recovery import local_scoring_auxiliary as auxiliary
        from brass_ai.recovery import local_scoring_v3 as v3
        from test_rail_end_g3528 import HeadRemoval
        case = HeadRemoval()
        removed = dict(v3.default(5, weight=.15), rail_end_removed=5)
        a = case.companion(removed, 5)
        half = copy.deepcopy(removed); half['weights'].update(canal=.5, rail=.5)
        b = case.companion(half, 5, raw=dict(training={auxiliary.KEY: a.payload(5)}))
        self.assertEqual(b.config['weights'], dict(canal=.5, rail=.5, rail_end=0.))
        for (name, x), (_, y) in zip(a.heads.named_parameters(), b.heads.named_parameters()):
            self.assertTrue(torch.equal(x, y), name)
        with self.assertRaises(ValueError):  # the weight never moves back down
            case.companion(removed, 5, raw=dict(training={auxiliary.KEY: b.payload(5)}))


if __name__ == '__main__':
    unittest.main()
