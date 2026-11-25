"""g5466 auxiliary revision: the Rail-end heads frozen and preserved in PPO like Canal/Rail, PPG clip 2.2 with a
one-time PPG Adam rescale, and the era heads on the fixed PPO rate in PPG."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

from brass_ai.recovery import training_revision_g5466 as R

# g5436-g5465's PPG norm summaries (value_phase.gradient_clipping of the g5466 export): iteration, attempted steps,
# clipped steps at 1.0, pre-clip mean and P95.
LOGGED = [
    (5436, 53, 43, 1.48998, 2.345724), (5437, 53, 39, 1.359215, 2.27239), (5438, 53, 43, 1.341818, 2.143059),
    (5439, 16, 16, 1.324195, 1.729008), (5440, 16, 16, 1.599566, 2.147005), (5441, 48, 37, 1.395407, 2.09214),
    (5442, 53, 36, 1.528155, 2.878137), (5443, 16, 15, 1.502559, 2.123973), (5444, 16, 15, 1.539266, 2.081609),
    (5445, 53, 45, 1.402967, 1.910184), (5446, 54, 38, 1.497412, 2.223564), (5447, 48, 29, 1.185351, 1.779715),
    (5448, 16, 13, 1.52401, 2.550985), (5449, 32, 30, 1.446392, 2.623939), (5450, 16, 15, 1.568719, 2.956899),
    (5451, 16, 14, 1.273815, 1.94258), (5452, 53, 44, 1.532281, 2.639037), (5453, 16, 14, 1.505294, 2.102943),
    (5454, 48, 39, 1.411441, 2.447676), (5455, 16, 13, 1.319155, 1.755342), (5456, 16, 14, 1.420207, 1.831609),
    (5457, 16, 14, 1.437779, 1.911318), (5458, 32, 25, 1.36839, 2.121999), (5459, 16, 13, 1.327335, 1.904496),
    (5460, 32, 30, 1.766953, 3.002973), (5461, 16, 15, 1.535021, 2.555811), (5462, 16, 13, 1.334058, 1.858616),
    (5463, 16, 14, 1.366931, 2.179756), (5464, 16, 14, 1.56454, 2.564984), (5465, 48, 36, 1.337037, 2.213308),
]
NORMS = [dict(iteration=i, steps=n, clipped_steps=c, mean=m, p95=q) for i, n, c, m, q in LOGGED]


def clipping(record, max_norm=1.):
    return dict(max_norm=max_norm, steps=record['steps'], clipped_steps=record['clipped_steps'],
                clipped_percent=100. * record['clipped_steps'] / record['steps'],
                pre_clip_norm=dict(mean=record['mean'], median=record['mean'] * .9, p95=record['p95']), scope='logging')


def before_recipe():
    """The run's recipe shape at g5466: the g4770 era-head and g5402 optimizer revisions."""
    from test_optimizer_revision_g5402 import NORMS as PPO_NORMS, before_recipe as g4770_recipe
    from brass_ai.recovery import training_revision_g5402 as g5402
    return g5402.configure(g4770_recipe(), 5402, multiplier=1., gradient_norms=PPO_NORMS)


def signature(config):
    from test_training_g3140 import signature as sign
    return sign(config)


class Recipe(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.before = before_recipe(); cls.after = R.configure(cls.before, 5466, gradient_norms=NORMS)

    def test_it_moves_the_ppg_clip_and_links_the_heads(self):
        from brass_ai.recovery.config import validate_config
        from brass_ai.rust_backend.lab_teacher_activation import is_on
        before, after = self.before, validate_config(copy.deepcopy(self.after))
        self.assertEqual((R.ppg_grad_clip(before), R.ppg_grad_clip(after)), (1., 2.2))
        self.assertEqual(after['update']['grad_clip'], 5.83)  # PPO keeps its clip
        self.assertTrue(R.rail_end_in_ppo(after) and R.fixed_head_rate(after))
        self.assertFalse(R.rail_end_in_ppo(before) or R.fixed_head_rate(before))
        changed = sorted(k for k in set(before) | set(after) if before.get(k) != after.get(k))
        self.assertEqual(changed, ['auxiliary_revision'])
        self.assertEqual(R.previous(after), before)
        self.assertEqual(R.configure(after, 5467, gradient_norms=None), after)  # adopted once
        self.assertTrue(is_on(after)); self.assertEqual(R.tag(after), 'ppgclip22-railendppo-headrate')
        from brass_ai.recovery import training_revision_g5402 as g5402
        with self.assertRaisesRegex(ValueError, 'follows the g5402 optimizer revision'):
            R.configure(g5402.previous(before), 5466, gradient_norms=NORMS)
        bad = copy.deepcopy(after); bad[R.KEY]['ppg_grad_clip'] = 3.
        with self.assertRaisesRegex(ValueError, 'Invalid g5466 auxiliary revision'): validate_config(bad)
        bad = copy.deepcopy(after); bad[R.KEY]['adam_rescale']['second_moment'] *= 2
        with self.assertRaisesRegex(ValueError, 'Invalid g5466 PPG Adam rescale'): validate_config(bad)

    def test_it_is_adopted_forward_only(self):
        from brass_ai.rust_backend.lab_shadow_config import is_shadow, signature_transition
        before, after = self.before, self.after
        self.assertTrue(signature_transition(signature(before), signature(after)))
        self.assertTrue(signature_transition(signature(after), signature(after)))
        self.assertFalse(signature_transition(signature(after), signature(before)))
        other = R.configure(before, 5466, gradient_norms=[dict(r, mean=r['mean'] * 1.01) for r in NORMS])
        self.assertFalse(signature_transition(signature(after), signature(other)))
        self.assertEqual(is_shadow(after), is_shadow(before))

    def test_the_rescale_fits_the_logged_norms(self):
        rescale = self.after[R.KEY]['adam_rescale']
        self.assertEqual((rescale['steps'], rescale['old_clip'], rescale['new_clip'], len(rescale['norms'])), (916, 1., 2.2, 30))
        self.assertAlmostEqual(rescale['logged_clipped_percent'], 100. * 742 / 916)
        self.assertAlmostEqual(rescale['predicted_clipped_percent'], rescale['logged_clipped_percent'], delta=1.)
        self.assertAlmostEqual(rescale['first_moment'], 1.4965, places=3)
        self.assertAlmostEqual(rescale['second_moment'], 2.4248, places=3)
        # Against a sample of the same normals, each iteration weighted by its steps.
        generator = np.random.default_rng(0); sample = []
        for r in NORMS:
            sample.append(generator.normal(r['mean'], (r['p95'] - r['mean']) / R.Z95, r['steps'] * 4000))
        n = np.concatenate(sample)
        first = np.minimum(n, 2.2).mean() / np.minimum(n, 1.).mean()
        second = np.square(np.minimum(n, 2.2)).mean() / np.square(np.minimum(n, 1.)).mean()
        self.assertAlmostEqual(rescale['first_moment'], first, places=2)
        self.assertAlmostEqual(rescale['second_moment'], second, places=2)
        # A record logged after g5466 carries P98 instead of P95: its z is 2.054, so the same spread fits the same way.
        z98 = R.UPPER['p98']
        moved = [dict({k: v for k, v in r.items() if k != 'p95'}, p98=r['mean'] + (r['p95'] - r['mean']) / R.Z95 * z98)
                 for r in NORMS]
        again = R.adam_rescale(moved)
        self.assertAlmostEqual(again['first_moment'], rescale['first_moment'], places=9)
        self.assertAlmostEqual(again['second_moment'], rescale['second_moment'], places=9)
        with self.assertRaisesRegex(ValueError, 'Invalid committed PPG gradient norms'):
            R.adam_rescale([dict(NORMS[0], p98=3.)])  # both percentiles in one record
        # A fit that cannot reproduce the logged clip rate is refused.
        unclipped = [dict(r, clipped_steps=0) for r in NORMS]
        with self.assertRaisesRegex(ValueError, 'PPG Adam rescale'):
            R.validate(dict(copy.deepcopy(R.SETTINGS), from_generation=5466, adam_rescale=R.adam_rescale(unclipped)))

    def test_the_norms_come_from_the_runs_committed_reports(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            history = {f'iteration-{r["iteration"]}/report.json': dict(sha256='0' * 64, summary=dict(
                iteration=r['iteration'], value_phase=dict(gradient_clipping=clipping(r)))) for r in NORMS[:-3]}
            (run / 'retention-history.json').write_text(json.dumps(history))
            for r in NORMS[-3:]:
                (run / f'iteration-{r["iteration"]}').mkdir()
                (run / f'iteration-{r["iteration"]}' / 'report.json').write_text(json.dumps(dict(
                    iteration=r['iteration'], value_phase=dict(gradient_clipping=clipping(r)))))
            state = dict(iteration=5466, recent_training=[dict(iteration=r['iteration'], value_phase=dict(
                gradient_clipping=clipping(r))) for r in NORMS[-2:]])
            self.assertEqual(R.committed_ppg_norms(run, state), NORMS)
            # The latest iteration logged after g5466: P98 in place of P95.
            latest = dict(clipping(NORMS[-1]), pre_clip_norm=dict(mean=NORMS[-1]['mean'], median=1., p98=2.6, max=3.))
            newer = dict(state, recent_training=[state['recent_training'][0], dict(iteration=5465, value_phase=dict(
                gradient_clipping=latest))])
            read = R.committed_ppg_norms(run, newer)
            self.assertEqual(read[-1], dict(iteration=5465, steps=48, clipped_steps=36, mean=NORMS[-1]['mean'], p98=2.6))
            self.assertEqual(read[:-1], NORMS[:-1])
            # An iteration without PPG steps is skipped; a missing one leaves too few.
            history.pop('iteration-5450/report.json')
            (run / 'retention-history.json').write_text(json.dumps(history))
            with self.assertRaisesRegex(ValueError, 'found 29'): R.committed_ppg_norms(run, state)
            history['iteration-5450/report.json'] = dict(summary=dict(iteration=5450, value_phase=dict(
                gradient_clipping=clipping(NORMS[14], max_norm=2.2))))
            (run / 'retention-history.json').write_text(json.dumps(history))
            with self.assertRaisesRegex(ValueError, 'clipped PPG at 1.0'): R.committed_ppg_norms(run, state)


class Rescale(unittest.TestCase):
    """The PPG Adam of the adoption generation: moments rescaled once, with the receipt in its group."""

    @classmethod
    def setUpClass(cls):
        cls.config = R.configure(before_recipe(), 5466, gradient_norms=NORMS)

    def trained(self):
        torch.manual_seed(0); model = torch.nn.Linear(3, 2)
        optimizer = torch.optim.Adam(model.parameters(), lr=3.9e-6)
        for _ in range(3):
            for p in model.parameters(): p.grad = torch.randn_like(p)
            optimizer.step()
        return model, optimizer

    def test_the_adoption_checkpoint_is_rescaled_once(self):
        model, optimizer = self.trained()
        moments = copy.deepcopy(optimizer.state_dict()['state'])
        receipt = R.rescale_ppg(optimizer, self.config, 5466)
        scale = self.config[R.KEY]['adam_rescale']
        self.assertEqual((receipt['first_moment'], receipt['second_moment'], receipt['parameters'], receipt['new_clip']),
                         (scale['first_moment'], scale['second_moment'], 2, 2.2))
        for index, saved in optimizer.state_dict()['state'].items():
            self.assertTrue(torch.equal(saved['step'], moments[index]['step']))
            torch.testing.assert_close(saved['exp_avg'], moments[index]['exp_avg'] * scale['first_moment'])
            torch.testing.assert_close(saved['exp_avg_sq'], moments[index]['exp_avg_sq'] * scale['second_moment'])
        # The receipt travels with the checkpoint: a later load is not rescaled again.
        saved = copy.deepcopy(optimizer.state_dict())
        again = torch.optim.Adam(model.parameters(), lr=1.); again.load_state_dict(saved)
        self.assertIsNone(R.rescale_ppg(again, self.config, 5467))
        self.assertTrue(all(torch.equal(again.state_dict()['state'][i]['exp_avg_sq'], s['exp_avg_sq'])
                            for i, s in saved['state'].items()))
        self.assertIsNone(R.rescale_ppg(self.trained()[1], before_recipe(), 5466))  # not adopted
        with self.assertRaisesRegex(ValueError, 'runs once'): R.rescale_ppg(self.trained()[1], self.config, 5467)

    def test_the_rescale_keeps_the_step_size_across_the_clip_change(self):
        """Adam trained on norms like the logged ones clipped at 1.0, then the clip moves to 2.2: without the rescale
        the next phase's steps are about 1.5x the size Adam settles at under the new clip; with it they are that size."""
        def steps(clip_before, rescale, *, warm=3000, measured=46):
            generator = torch.Generator().manual_seed(2)
            p = torch.nn.Parameter(torch.zeros(256)); optimizer = torch.optim.Adam([p], lr=1e-3)
            drift = torch.randn(256, generator=generator)
            def step(clip):
                g = drift + torch.randn(256, generator=generator)
                norm = float(torch.normal(1.43, .5, (1,), generator=generator).clamp_min(.05))
                p.grad = g / g.norm() * min(norm, clip); before = p.detach().clone(); optimizer.step()
                return float((p.detach() - before).norm())
            for _ in range(warm): step(clip_before)
            if rescale:
                factors = R.adam_rescale([dict(iteration=0, steps=1000, clipped_steps=800, mean=1.43, p95=1.43 + .5 * R.Z95)])
                s = optimizer.state[p]; s['exp_avg'].mul_(factors['first_moment']); s['exp_avg_sq'].mul_(factors['second_moment'])
            return np.mean([step(2.2) for _ in range(measured)])
        settled = steps(2.2, False)  # Adam's own size under the new clip
        self.assertGreater(steps(1., False) / settled, 1.3)
        self.assertAlmostEqual(steps(1., True) / settled, 1., delta=.1)


class RailEnd(unittest.TestCase):
    """PPO's frozen outputs and preservation loss cover the restored Rail-end outputs only with the revision."""

    def preserve(self, on):
        """The frozen outputs, and which head parameters PPO's preservation loss reaches once the heads have moved."""
        from brass_ai.learning.contracts import SOURCE_ROOT
        from brass_ai.recovery import local_scoring_auxiliary as auxiliary
        from brass_ai.recovery import local_scoring_v3 as v3
        from brass_ai.recovery import ppg_g210
        from brass_ai.recovery import training_revision_g3140 as G
        from brass_ai.recovery import training_revision_g4770 as g4770
        from brass_ai.recovery.actor_value import TrainingPhases
        from brass_ai.recovery.updates import autocast
        from brass_ai.recovery.value_phase import value_parts
        from brass_ai.recovery.value_preservation import FrozenValues
        from test_ppg_g210 import config, prepare
        torch.manual_seed(3); torch.set_num_threads(2)
        m, _, c = config()
        c['value_phase'] = dict(ppg_g210.KL_STOP_DEFAULT, batch_size=4, clone_rows=64, clone_batch_size=8)
        c[G.KEY] = dict(G.OFF, validation=copy.deepcopy(G.VALIDATION), bootstrap=copy.deepcopy(G.BOOTSTRAP),
                        ownership_stop=copy.deepcopy(G.OWNERSHIP_STOP), search_auxiliary=copy.deepcopy(G.SEARCH_AUX),
                        search_rows=copy.deepcopy(G.SEARCH_ROWS))
        c['local_scoring_auxiliary'] = dict(v3.default(1), rail_end_removed=1, rail_end_restored=2,
                                            weights=dict(canal=.5, rail=.5, rail_end=.25))
        c[g4770.KEY] = dict(copy.deepcopy(g4770.SETTINGS), from_generation=2)
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT / 'artifacts') as tmp:
            p = TrainingPhases(m, c, root=SOURCE_ROOT, run=Path(tmp))
            companion = p.local_scoring = auxiliary.Companion(m, c['local_scoring_auxiliary'], raw=dict(model_id='m'), generation=1)
            companion.rail_end_ppo = on
            prepare(m, p, 1, companion=True)
            ds = p.critic_dataset
            frozen = FrozenValues(m, ds, c['update'], companion)
            with torch.no_grad():  # the heads move after the snapshot, as PPO's steps move them
                for parameter in companion.heads.parameters(): parameter.add_(.05 * torch.randn_like(parameter))
            companion.heads.zero_grad(set_to_none=True)
            for _, ix, inputs in ds.value_batches(value_parts(ds, range(len(ds.data)), c['update'], m), m):
                with autocast(c['update'], ds.device): loss = frozen.losses(m, inputs, ix)['local']
                loss.sum().backward()
            grads = {n: (q.grad is not None and float(q.grad.abs().max()) > 0) for n, q in companion.heads.named_parameters()}
            return companion, sorted(frozen.values), grads

    def test_rail_end_outputs_are_frozen_and_preserved_like_canal_and_rail(self):
        companion, frozen, grads = self.preserve(False)
        self.assertNotIn('rail_end_slots', companion.preservation_weights())
        self.assertFalse(any(k.startswith('rail_end_') for k in frozen))
        self.assertFalse(grads['owner.rail_end_slots.weight'] or grads['vp.rail_end_links.weight'])
        self.assertTrue(grads['owner.canal_slots.weight'])
        self.assertFalse(companion.report()['rail_end_heads']['in_ppo_preservation'])
        companion, frozen, grads = self.preserve(True)
        weights = companion.preservation_weights()
        self.assertEqual({k: weights[k] for k in ('canal_slots', 'rail_end_slots', 'rail_end_links')},
                         dict(canal_slots=(.5, .5), rail_end_slots=(.25, .25), rail_end_links=(.25, .25)))
        self.assertTrue({'rail_end_slots_owner', 'rail_end_slots_vp', 'rail_end_links_owner', 'rail_end_links_vp'} <= set(frozen))
        self.assertTrue(grads['owner.rail_end_slots.weight'] and grads['vp.rail_end_links.weight'] and grads['owner.canal_slots.weight'])
        self.assertTrue(companion.report()['rail_end_heads']['in_ppo_preservation'])


class Norms(unittest.TestCase):
    def test_the_era_heads_report_their_own_norms_per_phase_with_p98(self):
        from brass_ai.recovery import local_scoring_auxiliary as auxiliary
        from brass_ai.recovery import local_scoring_v3 as v3
        from test_gae_value_phase import fixture
        companion = auxiliary.Companion(fixture(games=1)[0], v3.default(1), raw=dict(model_id='m'), generation=1)
        parameters = list(companion.heads.parameters()); torch.manual_seed(5)
        def step(phase, scale):
            companion.begin(phase); companion.pending = True
            for p in parameters: p.grad = torch.randn_like(p) * scale
            expected = float(torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(p.grad) for p in parameters])))
            companion.step(phase); return expected
        ppo = [step('ppo', 1e-3) for _ in range(5)]; ppg = [step('ppg', 1e-1) for _ in range(3)]
        report = companion.report()['gradient_norms']
        for phase, values in (('ppo', ppo), ('ppg', ppg)):
            r = report[phase]
            self.assertEqual((r['steps'], r['max_norm']), (len(values), 1.))
            self.assertAlmostEqual(r['pre_clip_norm']['p98'], float(np.quantile(values, .98)), places=5)
            self.assertAlmostEqual(r['pre_clip_norm']['max'], max(values), places=5)
            self.assertEqual(r['clipped_steps'], sum(v > 1. for v in values))
        self.assertGreater(report['ppg']['clipped_steps'], 0)  # these PPG norms are above the heads' 1.0
        self.assertNotIn('p95', report['ppo']['pre_clip_norm'])

    def test_ppo_reports_p98_of_its_gradient_norms(self):
        from brass_ai.recovery.review_metrics import distribution
        values = list(range(1, 101))
        ppo = distribution(values, upper=98)
        self.assertEqual(ppo['p98'], float(np.quantile(values, .98))); self.assertNotIn('p95', ppo)
        self.assertIn('p95', distribution(values))  # the other distributions keep P95


class Phase(unittest.TestCase):
    def test_ppg_clips_at_its_revised_value_and_the_heads_take_the_fixed_ppo_rate(self):
        """Through ppg_g210.fit (the g4770 phase fixture): every era-head PPG step, head-only ones included, at the PPO
        rate with no schedule factor, while the main network keeps PPG's warm-up/decay."""
        from brass_ai.recovery import local_scoring_auxiliary as auxiliary
        import test_era_head_revision_g4770 as g4770_tests
        original = auxiliary.Companion.begin; seen = []
        def begin(self, phase, **kwargs):
            original(self, phase, **kwargs)
            seen.append((phase, kwargs.get('rate_factor'), kwargs.get('learning_rate'),
                         self.optimizers[phase].param_groups[0]['lr']))
        with mock.patch.object(auxiliary.Companion, 'begin', begin), \
                mock.patch.object(auxiliary.Companion, 'linked_rates', True), \
                mock.patch.object(R, 'fixed_head_rate', lambda config: True), \
                mock.patch.object(R, 'ppg_grad_clip', lambda config: .5):
            r = g4770_tests.Phase.run_phase(None)
        self.assertEqual(r['gradient_clipping']['max_norm'], .5)
        rate = r['era_head_rate']['rate']
        self.assertEqual(r['era_head_rate'], dict(source='ppo-last-applied', rate=rate, fixed=True))
        self.assertAlmostEqual(r['peak_learning_rate'], rate * .65, places=18)  # PPG's own peak is .65 of it
        ppg = [s for s in seen if s[0] == 'ppg']
        # The joint steps and any head-only steps after the stop.
        self.assertEqual(len(ppg), r['attempted_steps'] + r['ownership_stop']['head_only_steps'])
        self.assertTrue(all(s[1] is None and s[2] == rate and s[3] == rate for s in ppg))
        self.assertGreater(len(set(r['attempted_learning_rates'])), 1)  # the main network's rate is scheduled


if __name__ == '__main__':
    unittest.main()
