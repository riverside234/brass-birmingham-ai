"""g5402 optimizer revision: PPO grad clip 5.83 with PPG's own 1.0, the PPO rate cut 20% as the new base, the soft
head on the PPO rate of each step, the era heads on the PPG rate (the PPO rate in PPO), and the PPG clip report."""
import copy
import unittest
from unittest import mock

import numpy as np
import torch

from brass_ai.recovery import training_revision_g5402 as R
from test_soft_policy_g3423 import Fixture as SoftFixture

# g5400/g5401's PPO pre-clip norms (state.recent_training of the g5402 export): every step clipped at 1.0.
NORMS = [dict(iteration=5400, count=46, mean=4.64852119528729, std=0.6129372497697125),
         dict(iteration=5401, count=46, mean=4.88137848480888, std=0.7739770402474277)]


def before_recipe():
    """The run's recipe shape at g5402: the g4770 era-head revision on the applied controller with the 4e-6 floor."""
    from test_era_head_revision_g4770 import dropped
    from brass_ai.recovery import training_revision_g4770 as g4770
    return g4770.configure(dropped(), 4770)


def signature(config):
    from test_training_g3140 import signature as sign
    return sign(config)


class Recipe(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.before = before_recipe(); cls.after = R.configure(cls.before, 5402, multiplier=1., gradient_norms=NORMS)

    def test_it_sets_the_ppo_clip_and_cuts_the_rate_in_force_to_a_new_base(self):
        from brass_ai.recovery import training_revision_g1307 as floor
        from brass_ai.recovery.config import validate_config
        from brass_ai.rust_backend.lab_teacher_activation import is_on
        before, after = self.before, validate_config(copy.deepcopy(self.after))
        self.assertEqual((before['update']['grad_clip'], after['update']['grad_clip']), (1., 5.83))
        self.assertEqual((before['learning_rate'], after['learning_rate']), (6e-6, 4.8e-6))
        self.assertEqual(after['ppo_lr_controller']['min_multiplier'], 4e-6 / 4.8e-6)
        self.assertTrue(floor.applied(after))  # the 4e-6 floor stays
        self.assertEqual(R.ppg_grad_clip(after), 1.); self.assertIsNone(R.ppg_grad_clip(before))
        self.assertTrue(R.soft_follows_ppo(after) and R.era_heads_follow(after))
        self.assertFalse(R.soft_follows_ppo(before) or R.era_heads_follow(before))
        changed = sorted(k for k in set(before) | set(after) if before.get(k) != after.get(k))
        self.assertEqual(changed, ['learning_rate', 'optimizer_revision', 'ppo_lr_controller', 'update'])
        self.assertEqual({k for k in after['update'] if after['update'][k] != before['update'][k]}, {'grad_clip'})
        self.assertEqual(R.previous(after), before)
        self.assertEqual(R.configure(after, 5403, multiplier=None, gradient_norms=None), after)  # adopted once
        self.assertTrue(is_on(after)); self.assertEqual(R.tag(after), 'clip583-lr080-linked-heads')
        # With the controller at 1.25 the rate in force is 7.5e-6, and 80% of it is the base.
        raised = R.configure(before, 5402, multiplier=1.25, gradient_norms=NORMS)
        self.assertEqual((raised['learning_rate'], raised[R.KEY]['before']['multiplier']), (6e-6, 1.25))
        bad = copy.deepcopy(after); bad['update']['grad_clip'] = 1.
        with self.assertRaisesRegex(ValueError, 'differs from the PPO clip'): validate_config(bad)
        bad = copy.deepcopy(after); bad[R.KEY]['after']['learning_rate'] = 5e-6
        with self.assertRaisesRegex(ValueError, '80% of the committed rate'): validate_config(bad)
        with self.assertRaisesRegex(ValueError, 'g5402 optimizer revision follows'):
            R.configure(dict(before, update=dict(before['update'], grad_clip=2.)), 5402, multiplier=1., gradient_norms=NORMS)

    def test_it_is_adopted_forward_only(self):
        from brass_ai.rust_backend.lab_shadow_config import signature_transition
        before, after = self.before, self.after
        self.assertTrue(signature_transition(signature(before), signature(after)))
        self.assertTrue(signature_transition(signature(after), signature(after)))
        self.assertFalse(signature_transition(signature(after), signature(before)))
        other = R.configure(before, 5402, multiplier=1.25, gradient_norms=NORMS)
        self.assertFalse(signature_transition(signature(after), signature(other)))

    def test_the_committed_multiplier_comes_from_the_last_update(self):
        schedule = dict(base_lr=6e-6, steps=100, completed_steps=19880)
        state = dict(iteration=5402, signature=dict(config=self.before), recent_training=[
            dict(iteration=5401, actor=dict(learning_rate_schedule=schedule,
                                            lr_controller=dict(mode='apply', warmup_done=True, multiplier_after=1.25)))])
        self.assertEqual(R.committed_multiplier(state), 1.25)
        stale = copy.deepcopy(state); stale['recent_training'][0]['iteration'] = 5400
        with self.assertRaisesRegex(ValueError, 'recent_training'): R.committed_multiplier(stale)

    def test_the_committed_norms_come_from_fully_clipped_updates(self):
        def row(norms):
            n = norms['count']
            return dict(iteration=norms['iteration'], actor=dict(optimizer_steps=n, gradient_clip_steps=n, gradient_norms=dict(
                count=n, mean=norms['mean'], std=norms['std'], min=3.5, p05=3.7, median=4.7, p95=6.3, max=7.3)))
        state = dict(iteration=5402, recent_training=[row(r) for r in NORMS])
        self.assertEqual(R.committed_gradient_norms(state), NORMS)
        partly = copy.deepcopy(state); partly['recent_training'][1]['actor']['gradient_clip_steps'] = 40
        with self.assertRaisesRegex(ValueError, 'every step clipped'): R.committed_gradient_norms(partly)
        rescale = self.after[R.KEY]['adam_rescale']
        self.assertEqual((rescale['steps'], rescale['old_clip'], rescale['new_clip']), (92, 1., 5.83))
        # The normal fit's E[min(n, 5.83)] and E[min(n, 5.83)^2] against a sample of the same normal.
        sample = np.minimum(np.random.default_rng(0).normal(rescale['norm_mean'], rescale['norm_std'], 2_000_000), 5.83)
        self.assertAlmostEqual(rescale['first_moment'], sample.mean(), places=2)
        self.assertAlmostEqual(rescale['second_moment'], np.square(sample).mean(), places=1)
        bad = copy.deepcopy(self.after[R.KEY]); bad['adam_rescale']['second_moment'] *= 2
        with self.assertRaisesRegex(ValueError, 'Adam rescale'): R.validate(bad)


class Rebase(unittest.TestCase):
    """The committed PPO Adam of the adoption generation: base and multiplier rebased, moments rescaled once."""

    @classmethod
    def setUpClass(cls):
        cls.before = before_recipe(); cls.after = R.configure(cls.before, 5402, multiplier=1., gradient_norms=NORMS)

    def trained(self, config, *, multiplier=1.):
        from brass_ai.recovery import learning_rate as L, lr_controller as C
        torch.manual_seed(0)
        model = torch.nn.Linear(3, 2)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
        w = config['learning_rate_warmup']
        L.configure(optimizer, config['learning_rate'], w['selfplay_steps'], w['start_factor'], 'selfplay')
        C.configure(optimizer, config['ppo_lr_controller'])
        group = optimizer.param_groups[0]; group[L.KEY]['completed_steps'] = group[L.KEY]['steps']  # warm-up done
        for _ in range(3):  # Adam moments to carry over
            for p in model.parameters(): p.grad = torch.randn_like(p)
            L.step(optimizer, ppo=True)
        group[C.KEY].update(multiplier=multiplier, window=[dict(kl=.007, early_stop=False)] * 3)
        return optimizer

    def test_the_adoption_checkpoint_is_rebased_and_rescaled_once(self):
        from brass_ai.recovery import learning_rate as L, lr_controller as C
        from brass_ai.recovery.resume import apply_optimizer_config
        optimizer = self.trained(self.before)
        moments = copy.deepcopy(optimizer.state_dict()['state'])
        completed = optimizer.param_groups[0][L.KEY]['completed_steps']
        self.assertEqual(L.upcoming(optimizer, ppo=True), 6e-6)
        change = apply_optimizer_config(optimizer, self.after, generation=5402)['optimizer_revision']
        self.assertEqual((change['before']['rate'], change['after']['rate'], change['after']['cooldown']), (6e-6, 4.8e-6, 3))
        group = optimizer.param_groups[0]
        self.assertEqual((group[L.KEY]['base_lr'], group[L.KEY]['completed_steps']), (4.8e-6, completed))
        self.assertEqual((group[C.KEY]['multiplier'], group[C.KEY]['window'], group[C.KEY]['cooldown']), (1., [], 3))
        self.assertEqual(group[C.KEY]['config']['min_multiplier'], 4e-6 / 4.8e-6)
        self.assertEqual((group['lr'], L.upcoming(optimizer, ppo=True)), (4.8e-6, 4.8e-6))
        rescale = self.after[R.KEY]['adam_rescale']
        self.assertEqual((change['adam_rescale']['first_moment'], change['adam_rescale']['second_moment'],
                          change['adam_rescale']['parameters']), (rescale['first_moment'], rescale['second_moment'], 2))
        for index, saved in optimizer.state_dict()['state'].items():
            self.assertTrue(torch.equal(saved['step'], moments[index]['step']))
            torch.testing.assert_close(saved['exp_avg'], moments[index]['exp_avg'] * rescale['first_moment'])
            torch.testing.assert_close(saved['exp_avg_sq'], moments[index]['exp_avg_sq'] * rescale['second_moment'])
        # A later checkpoint is already rebased (and not rescaled again); any other point refuses the rebase.
        settled = copy.deepcopy(optimizer.state_dict()['state'])
        self.assertNotIn('optimizer_revision', apply_optimizer_config(optimizer, self.after, generation=5403))
        self.assertTrue(all(torch.equal(optimizer.state_dict()['state'][i]['exp_avg_sq'], s['exp_avg_sq']) for i, s in settled.items()))
        with self.assertRaisesRegex(ValueError, 'runs once'):
            apply_optimizer_config(self.trained(self.before), self.after, generation=5403)
        with self.assertRaisesRegex(ValueError, 'runs once'):
            apply_optimizer_config(self.trained(self.before, multiplier=1.25), self.after, generation=5402)
        # As after a controller cut: the cooldown runs, and one cut more reaches the 4e-6 floor.
        state = group[C.KEY]
        measured = dict(kl=.02, steps_done=46, steps_planned=46, early_stop=False, warmup_done=True, hold=False,
                        data_signature='a' * 64, selfplay_vp_per_seat=10.,
                        economic_health=dict(contract='selfplay-economic-health/3.0.0', ok=True))
        after, report = C.decide(state, measured)
        self.assertEqual((report['decision'], report['reason']), ('down', 'kl_above_upper_band'))
        self.assertAlmostEqual(after['multiplier'] * 4.8e-6, 4e-6, places=18)
        _, report = C.decide(state, dict(measured, kl=.006))
        self.assertEqual(report['decision'], 'cooldown')

    def test_the_rescale_keeps_the_step_size_across_the_clip_change(self):
        """Adam trained on norm-4.8 gradients clipped to 1.0, then the clip moves to 5.83: without the rescale the
        next update's steps are several times larger; with it they stay the same size (the rate is held fixed)."""
        def run(rescale):
            generator = torch.Generator().manual_seed(2)
            p = torch.nn.Parameter(torch.zeros(256)); optimizer = torch.optim.Adam([p], lr=1e-3)
            drift = torch.randn(256, generator=generator)
            def step(clip):
                g = drift + torch.randn(256, generator=generator)
                norm = float(torch.normal(4.77, .75, (1,), generator=generator).clamp_min(1.5))
                p.grad = g / g.norm() * min(norm, clip); before = p.detach().clone(); optimizer.step()
                return float((p.detach() - before).norm())
            for _ in range(3000): step(1.)
            today = np.mean([step(1.) for _ in range(46)])
            if rescale:
                factors = R.adam_rescale([dict(iteration=0, count=1000, mean=4.77, std=.75)])
                s = optimizer.state[p]; s['exp_avg'].mul_(factors['first_moment']); s['exp_avg_sq'].mul_(factors['second_moment'])
            return np.mean([step(5.83) for _ in range(46)]) / today
        self.assertGreater(run(False), 2.5)
        self.assertAlmostEqual(run(True), 1., delta=.1)


class SoftHead(SoftFixture):
    def test_a_linked_head_steps_at_the_ppo_rate_of_the_step(self):
        from brass_ai.recovery import learning_rate as L, lr_controller as C
        from brass_ai.recovery.soft_policy import SoftPolicy
        from test_soft_policy_g3423 import soft_settings
        ppo = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))], lr=4.8e-6)
        L.configure(ppo, 4.8e-6, 0, .1, 'selfplay'); C.configure(ppo, C.DEFAULT_CONTROLLER)
        ppo.param_groups[0][C.KEY]['multiplier'] = 1.25
        rate = L.upcoming(ppo, ppo=True); self.assertAlmostEqual(rate, 6e-6, places=18)
        own = SoftPolicy(self.model, soft_settings(), None, generation=3270)
        linked = SoftPolicy(self.model, soft_settings(), None, generation=3270); linked.follow(ppo)
        start = own.head.detach().clone()
        gradient = torch.randn(own.head.shape, generator=torch.Generator().manual_seed(3)) * .01
        own.step(gradient, float(gradient.norm())); linked.step(gradient, float(gradient.norm()))
        self.assertEqual((own.last_rate, linked.last_rate), (1e-4, rate))
        torch.testing.assert_close(linked.head.detach() - start, (own.head.detach() - start) * (rate / 1e-4))
        self.assertEqual(linked.report()['learning_rate'], dict(source='ppo-step-rate', last=rate, configured=1e-4))
        self.assertEqual(own.report()['learning_rate']['source'], 'settings')


class EraHeads(unittest.TestCase):
    def test_linked_heads_take_the_main_rate_of_the_step(self):
        from brass_ai.recovery import local_scoring_auxiliary as auxiliary
        from brass_ai.recovery import local_scoring_v3 as v3
        from test_gae_value_phase import fixture
        companion = auxiliary.Companion(fixture(games=1)[0], v3.default(1), raw=dict(model_id='m'), generation=1)
        lr = lambda phase: companion.optimizers[phase].param_groups[0]['lr']
        companion.begin('ppg', rate_factor=.5); self.assertEqual(lr('ppg'), 1e-4 * .5)
        companion.linked_rates = True
        companion.begin('ppg', rate_factor=.5, learning_rate=3.12e-6); self.assertEqual(lr('ppg'), 3.12e-6 * .5)
        companion.begin('ppo', learning_rate=4.8e-6); self.assertEqual(lr('ppo'), 4.8e-6)
        with self.assertRaisesRegex(ValueError, 'main network rate'): companion.begin('ppg', rate_factor=.5)
        self.assertEqual(companion.report()['companion_lr_source'], 'main-network-step-rate')


class Clipping(unittest.TestCase):
    def test_the_report_counts_every_attempted_step(self):
        from brass_ai.recovery.value_phase import clipping_report
        values = (.5, 1., 1.5, 3.)
        r = clipping_report([torch.tensor(v) for v in values], 1.)
        self.assertEqual((r['max_norm'], r['steps'], r['clipped_steps'], r['clipped_percent']), (1., 4, 2, 50.))
        # g5466: the upper percentile is P98 (was P95), with the maximum beside it.
        self.assertEqual(r['pre_clip_norm'], dict(mean=1.5, median=1.25, p98=float(np.quantile(values, .98)), max=3.))
        empty = clipping_report([], 1.)
        self.assertEqual((empty['steps'], empty['clipped_percent'], empty['pre_clip_norm']['median']), (0, None, None))

    def test_ppg_clips_at_its_own_value_and_the_era_heads_follow_its_rate(self):
        """Through ppg_g210.fit (the g4770 phase fixture) with a PPG clip of .5 against the update's 1.0."""
        from brass_ai.recovery import local_scoring_auxiliary as auxiliary
        import test_era_head_revision_g4770 as g4770_tests
        original = auxiliary.Companion.begin; seen = []
        def begin(self, phase, **kwargs):
            original(self, phase, **kwargs)
            seen.append((phase, kwargs.get('rate_factor', 1.), kwargs.get('learning_rate'),
                         self.optimizers[phase].param_groups[0]['lr']))
        with mock.patch.object(auxiliary.Companion, 'begin', begin), \
                mock.patch.object(auxiliary.Companion, 'linked_rates', True), \
                mock.patch.object(R, 'ppg_grad_clip', lambda config: .5):
            r = g4770_tests.Phase.run_phase(None)
        clipping = r['gradient_clipping']
        self.assertEqual((clipping['max_norm'], clipping['steps']), (.5, r['attempted_steps']))  # rolled-back steps too
        self.assertGreater(clipping['steps'], 0); self.assertIsNotNone(clipping['pre_clip_norm']['p98'])
        self.assertAlmostEqual(clipping['clipped_percent'], 100. * clipping['clipped_steps'] / clipping['steps'])
        ppg = [s for s in seen if s[0] == 'ppg']
        self.assertTrue(ppg and all(s[2] == r['peak_learning_rate'] for s in ppg))
        for _, factor, base, used in ppg: self.assertEqual(used, base * min(1., factor))
        # The heads' rate is the value phase's own rate at each of its attempted steps.
        self.assertEqual([s[3] for s in ppg[:r['attempted_steps']]], r['attempted_learning_rates'])


if __name__ == '__main__':
    unittest.main()
