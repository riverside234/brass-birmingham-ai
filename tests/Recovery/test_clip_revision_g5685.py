"""g5685 clip revision: PPO clip 6.5 and PPG clip 2.4, each with a one-time Adam rescale from the logged norms; the
era heads' own per-phase norm summary is no longer measured."""
import copy
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from brass_ai.recovery import training_revision_g5466 as g5466
from brass_ai.recovery import training_revision_g5685 as R

# g5655-g5684 of the g5685 export: PPO (actor.gradient_norms) iteration, steps, steps above 5.83, mean and std;
# PPG (value_phase.gradient_clipping) iteration, steps, steps above 2.2, mean and P98.
PPO_LOGGED = [
    (5655, 46, 6, 4.769767, 0.7948503), (5656, 46, 5, 4.872655, 0.8218305), (5657, 46, 2, 4.882462, 0.58939),
    (5658, 46, 6, 4.866493, 0.933105), (5659, 46, 3, 4.782688, 0.9888818), (5660, 46, 2, 4.617428, 0.8093523),
    (5661, 46, 4, 4.73125, 0.9000897), (5662, 46, 1, 4.632838, 0.7207744), (5663, 46, 4, 4.764745, 0.7534813),
    (5664, 46, 2, 4.585503, 0.7499877), (5665, 46, 2, 4.601165, 0.6835777), (5666, 46, 4, 4.756682, 0.8254796),
    (5667, 46, 5, 4.760994, 0.8706012), (5668, 46, 3, 4.679926, 0.6719002), (5669, 46, 4, 4.669496, 0.7215545),
    (5670, 46, 3, 4.892133, 0.695989), (5671, 46, 2, 4.599603, 0.8037604), (5672, 46, 5, 4.883024, 0.7120041),
    (5673, 46, 5, 4.74443, 0.9771993), (5674, 46, 3, 4.771132, 0.649709), (5675, 46, 2, 4.755261, 0.7121966),
    (5676, 46, 5, 4.730656, 0.7904312), (5677, 46, 4, 4.730065, 0.7772324), (5678, 46, 3, 4.737875, 0.787195),
    (5679, 20, 4, 5.308473, 0.8052287), (5680, 46, 9, 5.00537, 0.9716492), (5681, 46, 2, 4.669383, 0.6928973),
    (5682, 46, 7, 4.967797, 0.8751031), (5683, 46, 8, 4.994348, 0.9906171), (5684, 46, 7, 5.028575, 1.129053),
]
PPG_LOGGED = [
    (5655, 16, 0, 1.316058, 2.058503), (5656, 32, 4, 1.499624, 2.450552), (5657, 16, 2, 1.462532, 2.373588),
    (5658, 48, 3, 1.339585, 2.810837), (5659, 32, 0, 1.32343, 1.966148), (5660, 32, 1, 1.289369, 2.042099),
    (5661, 16, 3, 1.614525, 2.752994), (5662, 32, 0, 1.214719, 2.118504), (5663, 16, 1, 1.104511, 2.081274),
    (5664, 32, 2, 1.29675, 2.407083), (5665, 32, 3, 1.373108, 3.06866), (5666, 16, 0, 1.217725, 1.818159),
    (5667, 48, 4, 1.436056, 3.08243), (5668, 16, 1, 1.374366, 2.152454), (5669, 16, 1, 1.402129, 2.090539),
    (5670, 16, 0, 1.271256, 1.945982), (5671, 48, 2, 1.380863, 2.992203), (5672, 16, 0, 1.303244, 1.844155),
    (5673, 54, 4, 1.375687, 2.481468), (5674, 16, 0, 1.41363, 1.963998), (5675, 32, 4, 1.420429, 2.530497),
    (5676, 16, 0, 1.170906, 1.667068), (5677, 48, 3, 1.289565, 2.353834), (5678, 32, 5, 1.576081, 3.500547),
    (5679, 16, 0, 1.279781, 1.817481), (5680, 54, 2, 1.304008, 2.373455), (5681, 16, 1, 1.264274, 2.160614),
    (5682, 32, 0, 1.187561, 1.831499), (5683, 16, 1, 1.359408, 2.182312), (5684, 53, 4, 1.30172, 2.573523),
]
PPO = [dict(iteration=i, steps=n, clipped_steps=c, mean=m, std=s) for i, n, c, m, s in PPO_LOGGED]
PPG = [dict(iteration=i, steps=n, clipped_steps=c, mean=m, p98=q) for i, n, c, m, q in PPG_LOGGED]
NORMS = dict(ppo=PPO, ppg=PPG)


def before_recipe():
    """The run's recipe shape at g5685: g5402 (PPO clip 5.83) and g5466 (PPG clip 2.2) adopted."""
    from test_auxiliary_revision_g5466 import NORMS as G5466_NORMS, before_recipe as g5402_recipe
    return g5466.configure(g5402_recipe(), 5466, gradient_norms=G5466_NORMS)


def signature(config):
    from test_training_g3140 import signature as sign
    return sign(config)


def actor(r):
    return dict(gradient_norms=dict(count=r['steps'], mean=r['mean'], std=r['std'], min=1., median=r['mean'],
                                    p98=r['mean'] + 2 * r['std'], max=9.), gradient_clip_steps=r['clipped_steps'])


def clipping(r, max_norm=2.2):
    return dict(max_norm=max_norm, steps=r['steps'], clipped_steps=r['clipped_steps'],
                clipped_percent=100. * r['clipped_steps'] / r['steps'],
                pre_clip_norm=dict(mean=r['mean'], median=r['mean'] * .9, p98=r['p98'], max=4.), scope='logging')


class Recipe(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.before = before_recipe(); cls.after = R.configure(cls.before, 5685, gradient_norms=NORMS)

    def test_it_moves_both_clips_and_nothing_else(self):
        from brass_ai.recovery.config import validate_config
        from brass_ai.rust_backend.lab_teacher_activation import is_on
        before, after = self.before, validate_config(copy.deepcopy(self.after))
        self.assertEqual((before['update']['grad_clip'], after['update']['grad_clip']), (5.83, 6.5))
        self.assertEqual((R.ppg_grad_clip(before), R.ppg_grad_clip(after)), (2.2, 2.4))
        changed = sorted(k for k in set(before) | set(after) if before.get(k) != after.get(k))
        self.assertEqual(changed, ['clip_revision', 'update'])
        self.assertEqual({k for k in before['update'] if before['update'][k] != after['update'][k]}, {'grad_clip'})
        self.assertEqual(R.previous(after), before)
        self.assertEqual(R.configure(after, 5686, gradient_norms=None), after)  # adopted once
        self.assertTrue(is_on(after)); self.assertEqual(R.tag(after), 'clip65-ppgclip24')
        self.assertTrue(g5466.rail_end_in_ppo(after) and g5466.fixed_head_rate(after))  # g5466 kept
        with self.assertRaisesRegex(ValueError, 'follows the g5402'):
            R.configure(g5466.previous(before), 5685, gradient_norms=NORMS)
        early = dict(NORMS, ppg=[dict(r, iteration=r['iteration'] - 300) for r in PPG])  # before g5466: clip 1.0
        with self.assertRaisesRegex(ValueError, 'trained at PPO clip 5.83 and PPG clip 2.2'):
            R.configure(before, 5685, gradient_norms=early)
        bad = copy.deepcopy(after); bad['update']['grad_clip'] = 5.83
        with self.assertRaisesRegex(ValueError, 'PPO clip'): validate_config(bad)
        bad = copy.deepcopy(after); bad[R.KEY]['ppg_grad_clip'] = 3.
        with self.assertRaisesRegex(ValueError, 'Invalid g5685 clip revision'): validate_config(bad)
        bad = copy.deepcopy(after); bad[R.KEY]['ppo_adam_rescale']['second_moment'] *= 2
        with self.assertRaisesRegex(ValueError, 'Invalid g5685 PPO Adam rescale'): validate_config(bad)

    def test_it_is_adopted_forward_only(self):
        from brass_ai.rust_backend.lab_shadow_config import is_shadow, signature_transition
        before, after = self.before, self.after
        self.assertTrue(signature_transition(signature(before), signature(after)))
        self.assertTrue(signature_transition(signature(after), signature(after)))
        self.assertFalse(signature_transition(signature(after), signature(before)))
        other = R.configure(before, 5685, gradient_norms=dict(NORMS, ppo=[dict(r, mean=r['mean'] * 1.01) for r in PPO]))
        self.assertFalse(signature_transition(signature(after), signature(other)))
        self.assertEqual(is_shadow(after), is_shadow(before))


class Fit(unittest.TestCase):
    def test_the_rescales_fit_the_logged_norms(self):
        ppo, ppg = R.ppo_adam_rescale(PPO), R.ppg_adam_rescale(PPG)
        self.assertEqual((ppo['steps'], ppo['old_clip'], ppo['new_clip']), (1354, 5.83, 6.5))
        self.assertEqual((ppg['steps'], ppg['old_clip'], ppg['new_clip']), (865, 2.2, 2.4))
        self.assertAlmostEqual(ppo['logged_clipped_percent'], 100. * 122 / 1354)
        self.assertAlmostEqual(ppg['logged_clipped_percent'], 100. * 51 / 865)
        for r in (ppo, ppg):
            self.assertAlmostEqual(r['predicted_clipped_percent'], r['logged_clipped_percent'], delta=1.5)
        self.assertAlmostEqual(ppo['first_moment'], 1.00746, places=4)
        self.assertAlmostEqual(ppo['second_moment'], 1.01869, places=4)
        self.assertAlmostEqual(ppg['first_moment'], 1.00765, places=4)
        self.assertAlmostEqual(ppg['second_moment'], 1.02304, places=4)
        # Against a sample of the same normals, each iteration weighted by its steps.
        generator = np.random.default_rng(0)
        for rescale, records, spread, old, new in ((ppo, PPO, lambda r: r['std'], 5.83, 6.5),
                                                   (ppg, PPG, lambda r: (r['p98'] - r['mean']) / g5466.UPPER['p98'], 2.2, 2.4)):
            n = np.concatenate([generator.normal(r['mean'], spread(r), r['steps'] * 4000) for r in records])
            self.assertAlmostEqual(rescale['first_moment'], np.minimum(n, new).mean() / np.minimum(n, old).mean(), places=3)
            self.assertAlmostEqual(rescale['second_moment'],
                                   np.square(np.minimum(n, new)).mean() / np.square(np.minimum(n, old)).mean(), places=3)
        # A fit that cannot reproduce the logged clip rate is refused.
        block = dict(copy.deepcopy(R.SETTINGS), from_generation=5685, ppg_adam_rescale=ppg,
                     ppo_adam_rescale=R.ppo_adam_rescale([dict(r, clipped_steps=r['steps']) for r in PPO]))
        with self.assertRaisesRegex(ValueError, 'Invalid g5685 PPO Adam rescale'): R.validate(block)
        with self.assertRaisesRegex(ValueError, 'Invalid committed PPO gradient norms'):
            R.ppo_adam_rescale([dict(PPO[0], std=-1.)])

    def test_the_norms_come_from_the_runs_committed_reports(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp); rows = {r['iteration']: dict(actor=actor(r)) for r in PPO}
            for r in PPG: rows[r['iteration']]['value_phase'] = dict(gradient_clipping=clipping(r))
            history = {f'iteration-{i}/report.json': dict(sha256='0' * 64, summary=dict(iteration=i, **row))
                       for i, row in rows.items() if i < 5681}
            (run / 'retention-history.json').write_text(json.dumps(history))
            for i in (5681, 5682):
                (run / f'iteration-{i}').mkdir()
                (run / f'iteration-{i}' / 'report.json').write_text(json.dumps(dict(iteration=i, **rows[i])))
            state = dict(iteration=5685, recent_training=[dict(iteration=i, **rows[i]) for i in (5683, 5684)])
            self.assertEqual(R.committed_norms(run, state), NORMS)
            # An iteration without PPG steps is skipped: an earlier one, still among the latest 40, fills the window.
            history['iteration-5670/report.json']['summary']['value_phase']['gradient_clipping']['steps'] = 0
            history['iteration-5650/report.json'] = dict(summary=dict(iteration=5650, actor=actor(PPO[0]),
                value_phase=dict(gradient_clipping=clipping(PPG[0]))))
            (run / 'retention-history.json').write_text(json.dumps(history))
            read = R.committed_norms(run, state)
            self.assertEqual([r['iteration'] for r in read['ppg']][:2], [5650, 5655])
            self.assertNotIn(5670, [r['iteration'] for r in read['ppg']]); self.assertEqual(read['ppo'], PPO)
            del history['iteration-5650/report.json']; (run / 'retention-history.json').write_text(json.dumps(history))
            with self.assertRaisesRegex(ValueError, 'PPG Adam rescale needs 30 .* found 29'): R.committed_norms(run, state)
            history['iteration-5670/report.json']['summary']['value_phase']['gradient_clipping'] = clipping(PPG[15], 1.)
            (run / 'retention-history.json').write_text(json.dumps(history))
            with self.assertRaisesRegex(ValueError, 'clipped PPG at 2.2'): R.committed_norms(run, state)


class Rescale(unittest.TestCase):
    """Each Adam of the adoption generation: moments rescaled once, with the receipt in its group."""

    @classmethod
    def setUpClass(cls):
        cls.config = R.configure(before_recipe(), 5685, gradient_norms=NORMS)

    def trained(self):
        torch.manual_seed(0); model = torch.nn.Linear(3, 2)
        optimizer = torch.optim.Adam(model.parameters(), lr=3.9e-6)
        for _ in range(3):
            for p in model.parameters(): p.grad = torch.randn_like(p)
            optimizer.step()
        return model, optimizer

    def test_each_adoption_checkpoint_is_rescaled_once(self):
        for rescale, name, key in ((R.rescale_ppo, 'ppo', R.PPO_GROUP_KEY), (R.rescale_ppg, 'ppg', R.PPG_GROUP_KEY)):
            model, optimizer = self.trained()
            if name == 'ppg':  # the run's PPG Adam already carries g5466's receipt: that rescale stays a no-op
                optimizer.param_groups[0][g5466.GROUP_KEY] = dict(generation=5466)
                self.assertIsNone(g5466.rescale_ppg(optimizer, self.config, 5685))
            moments = copy.deepcopy(optimizer.state_dict()['state'])
            receipt = rescale(optimizer, self.config, 5685)
            scale = self.config[R.KEY][name + '_adam_rescale']
            self.assertEqual((receipt['phase'], receipt['first_moment'], receipt['second_moment'], receipt['parameters'],
                              receipt['new_clip']), (name, scale['first_moment'], scale['second_moment'], 2, scale['new_clip']))
            for index, saved in optimizer.state_dict()['state'].items():
                self.assertTrue(torch.equal(saved['step'], moments[index]['step']))
                torch.testing.assert_close(saved['exp_avg'], moments[index]['exp_avg'] * scale['first_moment'])
                torch.testing.assert_close(saved['exp_avg_sq'], moments[index]['exp_avg_sq'] * scale['second_moment'])
            # The receipt travels with the checkpoint: a later load is not rescaled again.
            saved = copy.deepcopy(optimizer.state_dict())
            again = torch.optim.Adam(model.parameters(), lr=1.); again.load_state_dict(saved)
            self.assertEqual(again.param_groups[0][key], receipt)
            self.assertIsNone(rescale(again, self.config, 5686))
            self.assertIsNone(rescale(self.trained()[1], before_recipe(), 5685))  # not adopted
            with self.assertRaisesRegex(ValueError, 'runs once'): rescale(self.trained()[1], self.config, 5686)

    def test_the_trainer_hooks_call_the_rescales(self):
        from unittest import mock
        from brass_ai.recovery import resume
        _, optimizer = self.trained(); config = {'learning_rate': 4.8e-6, 'weight_decay': 0., R.KEY: self.config[R.KEY]}
        with mock.patch('brass_ai.recovery.training_revision_g5402.rebase_ppo', lambda *a: None):
            result = resume.apply_optimizer_config(optimizer, config, generation=5685)
        self.assertEqual(result['clip_revision']['phase'], 'ppo')
        self.assertIn(R.PPO_GROUP_KEY, optimizer.param_groups[0])


class EraHeadNorms(unittest.TestCase):
    def test_the_era_heads_keep_no_per_phase_norm_summary(self):
        from brass_ai.recovery import local_scoring_auxiliary as auxiliary
        from brass_ai.recovery import local_scoring_v3 as v3
        from test_gae_value_phase import fixture
        companion = auxiliary.Companion(fixture(games=1)[0], v3.default(1), raw=dict(model_id='m'), generation=1)
        parameters = list(companion.heads.parameters()); torch.manual_seed(5); norms = []
        for phase in ('ppo', 'ppg', 'ppg'):
            companion.begin(phase); companion.pending = True
            for p in parameters: p.grad = torch.randn_like(p) * 1e-2
            norms.append(float(torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(p.grad) for p in parameters]))))
            companion.step(phase)
        report = companion.report()
        self.assertNotIn('gradient_norms', report); self.assertFalse(hasattr(companion, 'phase_norms'))
        self.assertAlmostEqual(report['gradient_norm_max'], max(norms), places=5)  # the existing maximum stays


if __name__ == '__main__':
    unittest.main()
