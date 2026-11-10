"""The g1307 revision: a 4.0e-6 PPO learning-rate floor with the current rate kept."""
import copy
import unittest

import torch

from brass_ai.bank import read
from brass_ai.learning.contracts import SOURCE_ROOT as ROOT
from brass_ai.recovery import learning_rate, lr_controller as L, training_revision_g1294 as T, training_revision_g1307 as F
from brass_ai.recovery.collector_runtime import recipe
from brass_ai.recovery.config import validate_config
from brass_ai.recovery.search_value_schedule import previous_recipe
from brass_ai.rust_backend import lab_shadow_config as C, lab_teacher_activation as A

EXPORT = ROOT/'results/joint7-v11-2-g1290-trained-restart-g1307-20261110T020738908085Z'


class Recipe(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.state = read(EXPORT/'state.json')
        cls.before = recipe(cls.state['signature']['config'])
        cls.after = F.configure(cls.before)

    def test_only_the_floor_moves(self):
        validate_config(self.after)
        self.assertTrue(A.is_on(self.after))
        self.assertEqual({k for k in self.after if self.after[k] != self.before[k]}, {'ppo_lr_controller'})
        self.assertEqual(self.after['ppo_lr_controller'], dict(self.before['ppo_lr_controller'], min_multiplier=4e-6/6e-6))
        self.assertEqual(self.after['learning_rate']*self.after['ppo_lr_controller']['min_multiplier'], 4e-6)

    def test_the_old_recipe_is_recovered_exactly(self):
        self.assertEqual(F.previous(self.after), self.before)
        self.assertEqual(previous_recipe(self.after), previous_recipe(self.before))

    def test_it_is_applied_once_and_kept_by_the_other_revisions(self):
        self.assertEqual(F.configure(self.after), self.after)
        self.assertEqual(T.configure(self.after, self.state), self.after)
        self.assertEqual(A.configure(self.after, .1, revise_stages=True), self.after)

    def test_it_starts_at_g1307_and_cannot_be_undone(self):
        self.assertTrue(F.due(self.state))
        self.assertFalse(F.due(dict(self.state, iteration=1306)))
        self.assertTrue(F.compatible(self.before, self.after))
        self.assertFalse(F.compatible(self.after, self.before))

    def test_the_reviewed_transition_accepts_it_one_way(self):
        signature = dict(self.state['signature'], config=self.after)
        self.assertTrue(C.signature_transition(self.state['signature'], signature))
        self.assertFalse(C.signature_transition(signature, self.state['signature']))

    def test_another_floor_is_not_reviewed(self):
        bad = copy.deepcopy(self.before)
        bad['ppo_lr_controller']['min_multiplier'] = .25
        with self.assertRaisesRegex(ValueError, 'replaces the reviewed'):
            F.configure(bad)


class Controller(unittest.TestCase):
    def optimizer(self, controller):
        model = torch.nn.Linear(2, 1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=6e-6)
        learning_rate.configure(optimizer, 6e-6, 100, .1, 'selfplay')
        optimizer.param_groups[0][learning_rate.KEY]['completed_steps'] = 62845
        L.configure(optimizer, controller)
        return optimizer

    def test_the_current_rate_is_kept_and_cuts_stop_at_the_floor(self):
        before = recipe(read(EXPORT/'state.json')['signature']['config'])
        after = F.configure(before)
        optimizer = self.optimizer(before['ppo_lr_controller'])
        group = optimizer.param_groups[0]
        self.assertEqual(L.multiplier(group), 1.)
        state = L.configure(optimizer, after['ppo_lr_controller'])
        self.assertEqual(state['multiplier'], 1.)
        self.assertEqual(L.multiplier(group), 1.)
        self.assertEqual(group[L.KEY]['config']['min_multiplier'], F.floor_multiplier(after))
        cut = dict(kl=.02, steps_done=46, steps_planned=46, early_stop=False, warmup_done=True,
                   data_signature='a'*64, selfplay_vp_per_seat=140.,
                   economic_health=dict(contract='selfplay-economic-health/3.0.0', ok=True, reasons=[]))
        rates = []
        current = group[L.KEY]
        for _ in range(4):
            current, report = L.decide(dict(current, cooldown=0), cut)
            rates.append(6e-6*current['multiplier'])
        self.assertAlmostEqual(rates[0], 4.8e-6)
        self.assertEqual(rates[1:], [4e-6]*3)
        self.assertTrue(report['at_min'])

    def test_a_multiplier_below_the_new_floor_is_refused_at_resume(self):
        before = recipe(read(EXPORT/'state.json')['signature']['config'])
        optimizer = self.optimizer(before['ppo_lr_controller'])
        optimizer.param_groups[0][L.KEY]['multiplier'] = .6
        with self.assertRaisesRegex(ValueError, 'outside requested bounds'):
            L.configure(optimizer, F.configure(before)['ppo_lr_controller'])


if __name__ == '__main__':
    unittest.main()
