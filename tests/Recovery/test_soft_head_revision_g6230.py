"""g6230 soft-head revision: soft targets from the current logits and search bonus over the visited legal actions only,
weight 0 for rows with at most one visited action, the soft held-out KL once after PPO on the same targets, the soft head at 1e-4
for 100 iterations, and PPG clip 2.5 with a one-time Adam rescale."""
import copy
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from brass_ai.recovery import training_revision_g5466 as g5466
from brass_ai.recovery import training_revision_g5685 as g5685
from brass_ai.recovery import training_revision_g6230 as R

# g6200-g6229 of the g6230 export (value_phase.gradient_clipping): iteration, steps, steps above 2.4, mean and P98.
LOGGED = [
    (6200, 32, 0, 1.408958, 2.196035), (6201, 48, 4, 1.518143, 3.344812), (6202, 32, 0, 1.264056, 2.121048),
    (6203, 54, 4, 1.502988, 2.710006), (6204, 16, 0, 1.413003, 2.061505), (6205, 48, 2, 1.35232, 2.443624),
    (6206, 16, 3, 1.796967, 3.693563), (6207, 16, 2, 1.276091, 2.497849), (6208, 16, 0, 1.288459, 2.207528),
    (6209, 16, 1, 1.493946, 2.623999), (6210, 16, 2, 1.663263, 2.533936), (6211, 32, 2, 1.569749, 2.681955),
    (6212, 16, 0, 1.263463, 1.851828), (6213, 54, 5, 1.467753, 3.844661), (6214, 54, 1, 1.388123, 1.940421),
    (6215, 54, 5, 1.367326, 2.711671), (6216, 32, 3, 1.435522, 2.594578), (6217, 16, 1, 1.486277, 2.705012),
    (6218, 48, 4, 1.432557, 2.594902), (6219, 32, 1, 1.461521, 2.56757), (6220, 32, 5, 1.58892, 3.13216),
    (6221, 32, 10, 1.947187, 3.921217), (6222, 32, 0, 1.325766, 2.190703), (6223, 16, 0, 1.369889, 1.956329),
    (6224, 54, 7, 1.582464, 3.050968), (6225, 16, 0, 1.608789, 2.237271), (6226, 32, 3, 1.671254, 2.733782),
    (6227, 32, 1, 1.269458, 2.368795), (6228, 32, 4, 1.624579, 3.58011), (6229, 16, 1, 1.717973, 2.426817),
]
NORMS = [dict(iteration=i, steps=n, clipped_steps=c, mean=m, p98=q) for i, n, c, m, q in LOGGED]


def before_recipe():
    """The run's recipe shape at g6230: the g5685 clip revision with the g3423 soft head."""
    from test_clip_revision_g5685 import NORMS as G5685_NORMS, before_recipe as g5466_recipe
    from brass_ai.recovery import training_revision_g3423 as g3423
    return g5685.configure(g3423.configure(g5466_recipe(), 3423), 5685, gradient_norms=G5685_NORMS)


def signature(config):
    from test_training_g3140 import signature as sign
    return sign(config)


def clipping(r, max_norm=2.4):
    return dict(max_norm=max_norm, steps=r['steps'], clipped_steps=r['clipped_steps'],
                clipped_percent=100. * r['clipped_steps'] / r['steps'],
                pre_clip_norm=dict(mean=r['mean'], median=r['mean'] * .9, p98=r['p98'], max=4.), scope='logging')


class Recipe(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.before = before_recipe(); cls.after = R.configure(cls.before, 6230, gradient_norms=NORMS)

    def test_it_adds_one_block_and_moves_the_ppg_clip(self):
        from brass_ai.recovery.config import validate_config
        from brass_ai.rust_backend.lab_teacher_activation import is_on
        before, after = self.before, validate_config(copy.deepcopy(self.after))
        self.assertEqual((R.ppg_grad_clip(before), R.ppg_grad_clip(after)), (2.4, 2.5))
        self.assertEqual(after['update']['grad_clip'], 6.5)  # PPO keeps its clip
        changed = sorted(k for k in set(before) | set(after) if before.get(k) != after.get(k))
        self.assertEqual(changed, ['soft_head_revision'])
        self.assertEqual(R.previous(after), before)
        self.assertEqual(R.configure(after, 6231, gradient_norms=None), after)  # adopted once
        self.assertTrue(is_on(after)); self.assertEqual(R.tag(after), 'softvisited-lr1e4x100-ppgclip25')
        self.assertTrue(R.masked_soft_targets(after)); self.assertFalse(R.masked_soft_targets(before))
        self.assertEqual([R.soft_rate(after, g) for g in (6230, 6329, 6330)], [1e-4, 1e-4, None])
        self.assertIsNone(R.soft_rate(before, 6230)); self.assertEqual(R.soft_rate_until(after), 6330)
        with self.assertRaisesRegex(ValueError, 'follows the g5685 clip revision'):
            R.configure(g5685.previous(before), 6230, gradient_norms=NORMS)
        with self.assertRaisesRegex(ValueError, 'trained at PPG clip 2.4'):
            R.configure(before, 6230, gradient_norms=[dict(r, iteration=r['iteration'] - 600) for r in NORMS])
        bad = copy.deepcopy(after); bad[R.KEY]['soft_min_visited'] = 1
        with self.assertRaisesRegex(ValueError, 'Invalid g6230 soft-head revision'): validate_config(bad)
        bad = copy.deepcopy(after); bad[R.KEY]['ppg_adam_rescale']['first_moment'] *= 2
        with self.assertRaisesRegex(ValueError, 'Invalid g6230 PPG Adam rescale'): validate_config(bad)

    def test_it_is_adopted_forward_only(self):
        from brass_ai.rust_backend.lab_shadow_config import is_shadow, signature_transition
        before, after = self.before, self.after
        self.assertTrue(signature_transition(signature(before), signature(after)))
        self.assertTrue(signature_transition(signature(after), signature(after)))
        self.assertFalse(signature_transition(signature(after), signature(before)))
        other = R.configure(before, 6230, gradient_norms=[dict(r, mean=r['mean'] * 1.01) for r in NORMS])
        self.assertFalse(signature_transition(signature(after), signature(other)))
        self.assertEqual(is_shadow(after), is_shadow(before))


class PpgClip(unittest.TestCase):
    def test_the_rescale_fits_the_logged_norms(self):
        r = R.ppg_adam_rescale(NORMS)
        self.assertEqual((r['steps'], r['old_clip'], r['new_clip']), (942, 2.4, 2.5))
        self.assertAlmostEqual(r['logged_clipped_percent'], 100. * 71 / 942)
        self.assertAlmostEqual(r['predicted_clipped_percent'], r['logged_clipped_percent'], delta=1.)
        self.assertAlmostEqual(r['first_moment'], 1.00480, places=4)
        self.assertAlmostEqual(r['second_moment'], 1.01395, places=4)

    def test_the_norms_come_from_the_runs_committed_reports(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            history = {f'iteration-{r["iteration"]}/report.json': dict(summary=dict(
                iteration=r['iteration'], value_phase=dict(gradient_clipping=clipping(r)))) for r in NORMS[:-2]}
            (run / 'retention-history.json').write_text(json.dumps(history))
            state = dict(iteration=6230, recent_training=[dict(iteration=r['iteration'], value_phase=dict(
                gradient_clipping=clipping(r))) for r in NORMS[-2:]])
            self.assertEqual(R.committed_ppg_norms(run, state), NORMS)
            history['iteration-6210/report.json']['summary']['value_phase']['gradient_clipping'] = clipping(NORMS[10], 2.2)
            (run / 'retention-history.json').write_text(json.dumps(history))
            with self.assertRaisesRegex(ValueError, 'clipped PPG at 2.4'): R.committed_ppg_norms(run, state)

    def test_the_ppg_adam_is_rescaled_once_after_the_earlier_receipts(self):
        config = R.configure(before_recipe(), 6230, gradient_norms=NORMS)
        def trained():
            torch.manual_seed(0); model = torch.nn.Linear(3, 2); optimizer = torch.optim.Adam(model.parameters(), lr=1e-6)
            for _ in range(3):
                for p in model.parameters(): p.grad = torch.randn_like(p)
                optimizer.step()
            group = optimizer.param_groups[0]  # the run's PPG Adam carries the g5466 and g5685 receipts
            group[g5466.GROUP_KEY] = dict(generation=5466); group[g5685.PPG_GROUP_KEY] = dict(generation=5685)
            return model, optimizer
        model, optimizer = trained()
        self.assertIsNone(g5466.rescale_ppg(optimizer, config, 6230))
        self.assertIsNone(g5685.rescale_ppg(optimizer, config, 6230))
        moments = copy.deepcopy(optimizer.state_dict()['state'])
        receipt = R.rescale_ppg(optimizer, config, 6230); scale = config[R.KEY]['ppg_adam_rescale']
        self.assertEqual((receipt['contract'], receipt['old_clip'], receipt['new_clip']), (R.CONTRACT, 2.4, 2.5))
        for index, saved in optimizer.state_dict()['state'].items():
            torch.testing.assert_close(saved['exp_avg_sq'], moments[index]['exp_avg_sq'] * scale['second_moment'])
        again = torch.optim.Adam(model.parameters(), lr=1.); again.load_state_dict(copy.deepcopy(optimizer.state_dict()))
        self.assertIsNone(R.rescale_ppg(again, config, 6231))
        with self.assertRaisesRegex(ValueError, 'g6230 PPG Adam rescale runs once'): R.rescale_ppg(trained()[1], config, 6231)
        self.assertIsNone(R.rescale_ppg(trained()[1], before_recipe(), 6230))  # not adopted


class Targets(unittest.TestCase):
    def test_the_support_and_the_target_cover_the_visited_actions_only(self):
        from brass_ai.recovery.soft_policy import masked_target, visited_support
        label = dict(legal_ids=['a', 'b', 'c', 'd'], counts=[3, 0, 1, 2])
        support, info = visited_support(label)
        self.assertEqual(support.tolist(), [1., 0., 1., 1.]); self.assertEqual((info['visited'], info['eligible']), (3, True))
        logits, bonus = np.array([1., 2., -1., .5]), np.array([.2, 5., -.1, .3])
        target, dropped = masked_target(logits, bonus, support, 4.)
        z = (logits + bonus) / 4.; visited = support > 0
        expected = np.zeros(4); expected[visited] = np.exp(z[visited]) / np.exp(z[visited]).sum()
        np.testing.assert_allclose(target, expected, rtol=1e-12); self.assertEqual(target[1], 0.)
        self.assertAlmostEqual(dropped, np.exp(z[1]) / np.exp(z).sum())  # the masked-out mass
        for counts in ([5, 0, 0, 0], [0, 0, 0, 0]):
            _, info = visited_support(dict(label, counts=counts))
            self.assertFalse(info['eligible']); self.assertEqual(info['visited'], sum(c > 0 for c in counts))


class Term(unittest.TestCase):
    """Through TeacherTerm.apply (the g3423 soft-head fixture) with the head on visited-only targets."""

    @classmethod
    def setUpClass(cls):
        from test_soft_policy_g3423 import Fixture
        from test_teacher_revision_g3270 import label_row
        Fixture.setUpClass.__func__(cls)
        # Some rows with one visited action (a forced choice): weight 0.
        cls.rows = list(cls.rows) + [(i % 8, label_row(cls.model, seat=i % 4, round=2 + i % 6, width=3, counts=(4, 0, 0),
                                                       means=(.5, 0., 0.))) for i in range(4)]

    def soft(self, masked=True):
        from test_soft_policy_g3423 import at_multiplier
        soft = at_multiplier(self.model, 6); soft.masked = masked
        return soft

    def term(self, soft, batches=None):
        from brass_ai.recovery.teacher_term import TeacherTerm
        term = TeacherTerm(self.model, self.options, self.rows, seed=5, generation=3270,
                           state=dict(beta=.03, calibrated=True), revision=self.revision, soft=soft)
        term.plan(3)
        if batches is not None: term.batches = batches
        return term

    def expected(self, term, indices, *, stored=False):
        """The step's soft loss by hand: per eligible row, KL(softmax((logits + bonus) / T) over the visited actions ||
        head softmax over them) / rows, from the current logits (or, `stored`, the label's collection logits)."""
        from brass_ai.recovery import teacher_term as M
        from brass_ai.recovery.soft_policy import capture_hidden, single
        rows = [term.rows[i] for i in indices]; total = 0.
        with M.policy_context(self.model), capture_hidden(self.model) as hidden, torch.no_grad():
            logits = M.forward_policy(self.model, rows).double().cpu().numpy()
            head = torch.matmul(single(hidden).float(), term.soft.head).double().cpu().numpy()
        for k, i in enumerate(indices):
            label = term.rows[i].label; counts = np.asarray(label['counts']); visited = counts > 0; n = len(counts)
            if visited.sum() < 2: continue
            base = np.asarray(label['logits']) if stored else logits[k, :n]
            z = (base + term.bonus[i])[visited] / 4.; p = np.exp(z - z.max()); p /= p.sum()
            q = head[k, :n][visited]; q = q - q.max(); logq = q - np.log(np.exp(q).sum())
            total += float(np.sum(p * (np.log(p) - logq)))
        return total / len(indices)

    def test_the_head_trains_on_current_logit_targets_over_the_visited_actions(self):
        from brass_ai.recovery.value_preservation import TrunkGradients
        soft = self.soft(); term = self.term(soft); indices = term.batches[0]
        # Move the policy after the rows were drawn, so its logits differ from the stored collection logits.
        scorer = self.model.policy_scorer[2].weight; original = scorer.detach().clone()
        with torch.no_grad(): scorer.mul_(1.5)
        try:
            expected, stored = self.expected(term, indices), self.expected(term, indices, stored=True)
            term.apply(self.model, TrunkGradients(self.model), 1); self.model.zero_grad(set_to_none=True)
        finally:
            with torch.no_grad(): scorer.copy_(original)
        self.assertAlmostEqual(soft.steps[-1]['loss'], expected, places=5)
        self.assertGreater(abs(expected - stored), 1e-4)  # the current logits, not the collection ones
        self.assertEqual(soft.iteration_steps, 1)
        report = soft.report()['masked_targets']
        self.assertEqual(report['weight_zero_rows'], sum(1 for v in term.soft_support.values() if not v[1]['eligible']))
        self.assertGreater(report['weight_zero_rows'], 0)

    def test_a_step_with_only_weight_zero_rows_takes_no_soft_step(self):
        from brass_ai.recovery.value_preservation import TrunkGradients
        soft = self.soft(); head = soft.head.detach().clone(); term = self.term(soft)
        forced = [i for i, r in enumerate(term.rows) if sum(c > 0 for c in r.label['counts']) < 2]
        self.assertEqual(len(forced), 4); term.batches = [forced]
        term.apply(self.model, TrunkGradients(self.model), 1); self.model.zero_grad(set_to_none=True)
        self.assertEqual((soft.iteration_steps, soft.skipped_steps, len(soft.steps)), (0, 1, 0))
        self.assertTrue(torch.equal(soft.head.detach(), head))
        self.assertEqual(term.steps[-1]['rows'], len(forced))  # the teacher's own step still ran

    def test_the_host_path_gives_the_same_loss(self):
        from brass_ai.recovery.value_preservation import TrunkGradients
        losses = []
        for staged in (True, False):
            soft = self.soft(); term = self.term(soft)
            if not staged: term.staging = dict(fallback=True)
            term.apply(self.model, TrunkGradients(self.model), 1); self.model.zero_grad(set_to_none=True)
            losses.append(soft.steps[-1]['loss'])
        self.assertAlmostEqual(losses[0], losses[1], places=6)

    def test_the_unrevised_head_keeps_the_g3270_target(self):
        from brass_ai.recovery.value_preservation import TrunkGradients
        soft = self.soft(masked=False); term = self.term(soft)
        term.apply(self.model, TrunkGradients(self.model), 1); self.model.zero_grad(set_to_none=True)
        self.assertEqual(term.soft_support, {}); self.assertNotIn('masked_targets', soft.report())


class Phase(unittest.TestCase):
    """The teacher phase: the soft rate window and the one soft held-out check, after PPO."""

    @classmethod
    def setUpClass(cls):
        Term.setUpClass.__func__(cls)

    def phase(self, config):
        from brass_ai.recovery.teacher_phase import TeacherPhase
        rows = self.rows; heldout = [r for _, r in self.rows[:4]] + [r for _, r in self.rows[-2:]]

        class Buffer:
            def rows(self, generation): return rows
            def payload(self): return {}

        p = TeacherPhase.__new__(TeacherPhase)
        p.config, p.options, p.generation, p.weight = config, self.options, 3270, dict(beta=.03, calibrated=True)
        p.buffer, p.heldout, p.metrics, p.jobs, p.cursor = Buffer(), heldout, {}, [], 0
        p.receipt, p.reward_id, p.admission_seconds, p.soft_state = None, 'r', 0., None
        return p

    def test_one_soft_check_after_ppo_on_targets_built_the_same_way(self):
        from test_soft_policy_g3423 import soft_settings
        from brass_ai.recovery.teacher_term import policy_logits
        from brass_ai.recovery.value_preservation import TrunkGradients
        block = dict(from_generation=3270)
        config = dict(seed=5, teacher_revision=1, soft_policy=soft_settings(), soft_head_revision=block)
        p = self.phase(config)
        calls = []
        from brass_ai.recovery import soft_policy
        heldout, masked = soft_policy.SoftPolicy.heldout, soft_policy.SoftPolicy.heldout_masked
        from unittest import mock
        with mock.patch.object(soft_policy.SoftPolicy, 'heldout', lambda s, *a: calls.append('old') or heldout(s, *a)), \
                mock.patch.object(soft_policy.SoftPolicy, 'heldout_masked', lambda s, *a: calls.append('masked') or masked(s, *a)):
            term = p.prepare(self.model); term.plan(2)
            self.assertTrue(p.soft.masked)
            for a, b in zip(p.pre_logits, policy_logits(self.model, p.heldout), strict=True): np.testing.assert_array_equal(a, b)
            for step in (1, 2):
                term.apply(self.model, TrunkGradients(self.model), step); self.model.zero_grad(set_to_none=True)
            p.after_ppo(self.model)
            report = p.finish(self.model, {})
        self.assertEqual(calls, ['masked'])  # once, after PPO
        soft = report['heldout']['soft']
        self.assertEqual(set(soft) - {'scope'}, {'after_ppo', 'after_ppo_seconds'})
        self.assertEqual((soft['after_ppo']['rows'], soft['after_ppo']['skipped_rows']), (4, 2))
        self.assertIsNotNone(soft['after_ppo']['dropped_unvisited_mass']['median'])
        self.assertIn('after_ppo', report['heldout']); self.assertIn('ppo_delta', report['heldout'])
        rate = report['term']['soft']['learning_rate']
        self.assertEqual((rate['source'], rate['last']), ('g6230-fixed-0.0001-until-g3370', 1e-4))
        # Manual KL of one held-out row against softmax((current logits + bonus) / T) over its visited actions.
        from brass_ai.recovery import teacher_term as M
        r = p.heldout[0]; scale = self.revision['label_scale']
        _, one = p.soft.heldout_masked(self.model, [r], scale)
        with M.policy_context(self.model), soft_policy.capture_hidden(self.model) as hidden, torch.no_grad():
            logits = M.forward_policy(self.model, [r]).double().cpu().numpy()[0]
            head = torch.matmul(soft_policy.single(hidden).float(), p.soft.head).double().cpu().numpy()[0]
        n = len(r.label['legal_ids']); use = np.asarray(r.label['counts']) > 0
        z = (logits[:n] + M.search_bonus(r.label, scale))[use] / 4.; target = np.exp(z - z.max()); target /= target.sum()
        q = head[:n][use] - head[:n][use].max(); logq = q - np.log(np.exp(q).sum())
        self.assertAlmostEqual(one['kl'], float(np.sum(target * (np.log(target) - logq))), places=6)


if __name__ == '__main__':
    unittest.main()
