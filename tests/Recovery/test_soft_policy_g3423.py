"""g3423 soft policy head (step S1 of ai-research/SOFT_POLICY_TARGET_PLAN.md): the separate gradient
path during warm-up, the hidden-returning forward, the copied head, gradient scaling, the checkpoint
state and the recipe."""
import copy
import json
import unittest

import numpy as np
import torch

from brass_ai.recovery import training_revision_g3423 as R


def soft_settings(generation=3270):
    return dict(copy.deepcopy(R.SETTINGS), from_generation=generation)


def at_multiplier(model, trained, generation=3270):
    """A soft head whose warm-up has seen `trained` iterations with head steps."""
    from brass_ai.recovery.soft_policy import SoftPolicy, identity
    state = SoftPolicy(model, soft_settings(generation), None, generation=generation).finish()
    state = {k: v for k, v in state.items() if k != 'identity'}
    state.update(trained_iterations=trained)
    return SoftPolicy(model, soft_settings(generation), dict(state, identity=identity(state)), generation=generation + 1)


class Fixture(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from test_b025 import config, model_fixture
        from test_teacher_revision_g3270 import label_row
        from brass_ai.recovery import training_revision_g1491 as g1491
        from brass_ai.recovery import training_revision_g3270 as g3270
        cls.model = model_fixture()
        options = copy.deepcopy(config()['teacher_phase'])
        options.update(mode='on', buffer=copy.deepcopy(g3270.SETTINGS['buffer']))
        options['term']['rows_per_step'] = 4
        options[g1491.KEY] = g1491.block(variant=g1491.SEARCH_REFRESH)
        cls.options = options
        # Visited moves with different completed values, so the copied head does not already fit.
        cls.rows = [(i % 8, label_row(cls.model, seat=i % 4, round=1 + i % 8, width=3,
                                      counts=(2 + i % 3, 1, 0), means=(.6, .1 * (1 + i % 5), 0.))) for i in range(12)]
        cls.revision = g3270.term_settings(dict(teacher_revision=1))

    def term(self, soft=None):
        from brass_ai.recovery.teacher_term import TeacherTerm
        term = TeacherTerm(self.model, self.options, self.rows, seed=5, generation=3270,
                           state=dict(beta=.03, calibrated=True), revision=self.revision, soft=soft)
        term.plan(3)
        return term

    def step(self, soft=None):
        """One `apply` step on a clean model: (gradient of every model parameter, the teacher's step record)."""
        from brass_ai.recovery.value_preservation import TrunkGradients
        self.model.zero_grad(set_to_none=True)
        term = self.term(soft)
        actor = TrunkGradients(self.model)
        generator = torch.Generator().manual_seed(7)
        for a in actor.actor: a.copy_(torch.randn(a.shape, generator=generator))  # a stand-in PPO actor gradient
        term.apply(self.model, actor, 1)
        grads = {n: None if p.grad is None else p.grad.detach().clone() for n, p in self.model.named_parameters()}
        self.model.zero_grad(set_to_none=True)
        return grads, term.steps[-1], term

    def batch(self, soft):
        """The first planned batch's logits, staged bonus and hidden layer, as `apply` builds them."""
        from brass_ai.recovery import teacher_term as M
        from brass_ai.recovery.soft_policy import capture_hidden, single
        term = self.term(); rows = [term.rows[i] for i in term.batches[0]]
        with M.policy_context(self.model), capture_hidden(self.model) as hidden:
            logits = M.forward_policy(self.model, rows).float()
        bonus = M.target_tensor(rows, logits, [term.bonus[i] for i in term.batches[0]])
        return logits, bonus, single(hidden)


class WarmUp(Fixture):
    def test_search_rows_train_the_head_during_warm_up(self):
        from brass_ai.recovery.soft_policy import SoftPolicy
        off, off_step, _ = self.step()
        soft = SoftPolicy(self.model, soft_settings(), None, generation=3270)
        self.assertEqual((soft.multiplier, soft.trunk, soft.settings['ppo_row_factor']), (0., False, 0.))
        before = soft.head.detach().clone()
        on, on_step, term = self.step(soft)
        # 1. The head has a gradient of its own and moves after its Adam step (it has no bias).
        record = soft.steps[-1]
        self.assertGreater(record['head_norm'], 0.)
        self.assertFalse(torch.equal(soft.head.detach(), before))
        self.assertEqual((soft.iteration_steps, soft.t), (1, 1))
        # 2. Every model parameter's gradient is bit-identical to the step with the soft term off.
        self.assertEqual(off.keys(), on.keys())
        for name in off:
            self.assertEqual(off[name] is None, on[name] is None, name)
            if off[name] is not None:
                self.assertTrue(torch.equal(off[name], on[name]), name)
        self.assertTrue(any(n.startswith('policy_scorer.0') and g is not None for n, g in on.items()))
        # 3. The teacher's own loss, norm and share are identical.
        for key in ('loss', 'raw_share', 'share', 'cosine', 'actor_norm', 'teacher_norm'):
            self.assertEqual(off_step[key], on_step[key], key)
        self.assertEqual((record['raw_trunk_norm'], record['share']), (0., 0.))
        report = term.report()['soft']
        self.assertEqual((report['stage'], report['steps'], report['head_steps']), ('head_only', 1, 1))
        self.assertGreater(report['head_change'], 0.)

    def test_control_at_multiplier_one_the_scorer_gradient_changes(self):
        off, off_step, _ = self.step()
        soft = at_multiplier(self.model, 6)
        self.assertEqual((soft.multiplier, soft.trunk), (1., True))
        on, on_step, _ = self.step(soft)
        self.assertFalse(torch.equal(off['policy_scorer.0.weight'], on['policy_scorer.0.weight']))
        self.assertTrue(torch.equal(off['policy_scorer.2.weight'], on['policy_scorer.2.weight']))  # the main readout gets none
        self.assertEqual(off_step['loss'], on_step['loss'])  # the teacher's own term is still unchanged
        record = soft.steps[-1]
        self.assertGreater(record['raw_trunk_norm'], 0.)
        self.assertAlmostEqual(record['trunk_norm'], record['raw_trunk_norm'] * 8 * .03, places=12)
        self.assertGreater(record['share'], 0.)

    def test_shadow_mode_measures_without_stepping(self):
        from brass_ai.recovery.soft_policy import SoftPolicy
        from brass_ai.recovery.teacher_term import TeacherTerm
        from brass_ai.recovery.value_preservation import TrunkGradients
        soft = SoftPolicy(self.model, soft_settings(), None, generation=3270); before = soft.head.detach().clone()
        term = TeacherTerm(self.model, dict(copy.deepcopy(self.options), mode='shadow'), self.rows, seed=5, generation=3270,
                           revision=self.revision, soft=soft)
        term.plan(3); term.apply(self.model, TrunkGradients(self.model), 1); self.model.zero_grad(set_to_none=True)
        self.assertTrue(torch.equal(soft.head.detach(), before))
        self.assertEqual((len(soft.steps), soft.iteration_steps), (1, 0))
        self.assertEqual(soft.finish()['trained_iterations'], 0)  # no head step, no warm-up progress

    def test_the_stratified_long_and_short_path_is_refused(self):
        from brass_ai.recovery.soft_policy import SoftPolicy
        from brass_ai.recovery.teacher_term import TeacherTerm
        soft = SoftPolicy(self.model, soft_settings(), None, generation=3270)
        with self.assertRaisesRegex(ValueError, 'long-label path'):
            TeacherTerm(self.model, self.options, self.rows, seed=5, generation=3270, revision=None, soft=soft)


class Head(Fixture):
    def test_the_hidden_forward_keeps_the_main_logits_bit_identical(self):
        from brass_ai.recovery import teacher_term as M
        from brass_ai.recovery.soft_policy import SoftPolicy, capture_hidden, single
        rows = [r for _, r in self.rows]
        with M.policy_context(self.model), torch.no_grad():
            plain = M.forward_policy(self.model, rows)
            with capture_hidden(self.model) as hidden:
                hooked = M.forward_policy(self.model, rows)
        self.assertTrue(torch.equal(plain, hooked))
        self.assertEqual(single(hidden).shape[-1], self.model.policy_scorer[2].in_features)
        soft = SoftPolicy(self.model, soft_settings(), None, generation=3270)
        main, measured = soft.heldout(self.model, rows)
        for a, b in zip(main, M.policy_logits(self.model, rows), strict=True):
            self.assertTrue(np.array_equal(a, b))
        self.assertEqual(measured['rows'], len(rows))
        self.assertGreater(measured['kl'], 0.)

    def test_the_copied_head_starts_at_the_main_logits_over_t(self):
        from brass_ai.recovery.soft_policy import SoftPolicy
        soft = SoftPolicy(self.model, soft_settings(), None, generation=3270)
        logits, bonus, hidden = self.batch(soft)
        mask = torch.isfinite(logits)
        # Without the main readout's bias every move is offset by the same -bias/T: the same softmax.
        offset = -float(self.model.policy_scorer[2].bias.detach()) / 4
        np.testing.assert_allclose(soft.logits(hidden, mask)[mask].detach().numpy(), (logits[mask] / 4 + offset).detach().numpy(),
                                   rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(soft.logits(hidden, mask).softmax(-1).detach().numpy(), (logits / 4).softmax(-1).detach().numpy(),
                                   rtol=1e-5, atol=1e-7)
        # No search bonus: the softened current policy, which the copy already predicts.
        self.assertLess(float(soft.loss(logits, torch.zeros_like(bonus), hidden).detach()), 1e-6)
        # A bonus on the visited moves only: the unvisited move still gets a gradient (section 2).
        self.assertGreater(float(soft.loss(logits, bonus, hidden).detach()), 1e-5)
        q = soft.logits(hidden, mask); q.retain_grad()
        target = ((logits.detach() + bonus) / 4).softmax(-1)
        (-(target * q.log_softmax(-1).masked_fill(~mask, 0.)).sum(-1).mean()).backward()
        unvisited = q.grad[:, 2]  # counts (k, 1, 0): the third move is never visited
        self.assertTrue(bool((unvisited.abs() > 1e-7).all()))
        # Their order among themselves is right: the ratio target / prediction is one constant per row.
        ratio = target / q.detach().softmax(-1)
        self.assertTrue(bool(((ratio[:, 2] - 1).abs() > 1e-6).any()))
        soft.head.grad = None

    def test_the_trunk_gradient_scales_with_the_multiplier(self):
        from brass_ai.recovery.teacher_term import ACTOR_PREFIXES
        soft = at_multiplier(self.model, 6)
        parameters = tuple(p for n, p in self.model.named_parameters() if n.startswith(ACTOR_PREFIXES))
        results = {}
        for value in (.5, 1.):
            soft.multiplier = value
            logits, bonus, hidden = self.batch(soft)
            head, actor = soft.gradients(soft.loss(logits, bonus, hidden), parameters)
            results[value] = (head, actor)
        torch.testing.assert_close(results[.5][0], results[1.][0], rtol=0, atol=0)  # the head itself is not scaled
        compared = 0
        for half, full in zip(results[.5][1], results[1.][1]):
            if full is None: continue
            torch.testing.assert_close(half, full * .5, rtol=1e-5, atol=1e-9); compared += 1
        self.assertGreater(compared, 0)


class State(Fixture):
    def test_the_state_round_trips_through_json_and_resumes_the_ramp(self):
        from brass_ai.recovery.soft_policy import SoftPolicy, carry, multiplier
        soft = SoftPolicy(self.model, soft_settings(), None, generation=3270)
        self.step(soft); self.step(soft)
        state = json.loads(json.dumps(soft.finish(), allow_nan=False))
        self.assertEqual((state['generation'], state['trained_iterations'], state['head_steps'], state['adam']['step']),
                         (3271, 1, 2, 2))
        again = SoftPolicy(self.model, soft_settings(), state, generation=3271)
        for a, b in ((again.head, soft.head), (again.m, soft.m), (again.v, soft.v), (again.origin_head, soft.origin_head)):
            self.assertTrue(torch.equal(a.detach(), b.detach()))
        self.assertEqual((again.t, again.trained, again.multiplier, again.fresh), (2, 1, 0., False))
        self.assertEqual(again.finish(), carry(state, 3272))  # no head step: only the generation moves
        moved = carry(state, 3272)
        self.assertEqual((moved['generation'], moved['head']), (3272, state['head']))
        self.assertIsNone(carry(None, 3272))
        with self.assertRaisesRegex(ValueError, 'progress differs'):
            SoftPolicy(self.model, soft_settings(), state, generation=3272)
        corrupt = copy.deepcopy(state); corrupt['head'][0] += 1.
        with self.assertRaisesRegex(ValueError, 'Corrupt'):
            SoftPolicy(self.model, soft_settings(), corrupt, generation=3271)
        with self.assertRaisesRegex(ValueError, 'adoption generation'):
            SoftPolicy(self.model, soft_settings(), None, generation=3271)
        with self.assertRaisesRegex(ValueError, 'step S4'):
            SoftPolicy(self.model, dict(soft_settings(), ppo_row_factor=1.), None, generation=3270)
        self.assertEqual([multiplier(k, R.SETTINGS['warmup']) for k in range(8)], [0., 0., .2, .4, .6, .8, 1., 1.])


class Lifecycle(Fixture):
    def test_the_head_trains_inside_the_ppo_update(self):
        import random
        from fixtures import game
        from test_b025 import current_board
        from brass_ai.recovery.config import UPDATE_V21
        from brass_ai.recovery.soft_policy import SoftPolicy
        from brass_ai.recovery.teacher_term import TeacherTerm
        from brass_ai.recovery.updates import ppo_update
        from brass_ai.step5.policy import NeuralPolicy
        model = copy.deepcopy(self.model); policy = NeuralPolicy(model, focus='neutral', exploration=.05)
        self.addCleanup(policy.close)
        g = game(); g['split'] = 'train'
        for i, row in enumerate(g['records']):
            current_board(row['view'])
            index, p, _ = policy.select_batch([(row['view'], row['choices'], random.Random(i))])[0]
            row.update(chosen=index, behavior=dict(kind='model', policy_id=policy.policy_id, probability=p, temperature=1.))
        cfg = dict(UPDATE_V21, epochs=2, effective_batch_size=2, critic_batch_size=2, microbatch_size=2, target_kl=10.)
        soft = SoftPolicy(model, soft_settings(), None, generation=3270); before = soft.head.detach().clone()
        term = TeacherTerm(model, self.options, self.rows, seed=5, generation=3270, state=dict(beta=.03, calibrated=True),
                           revision=self.revision, soft=soft)
        report = ppo_update(model, torch.optim.AdamW(model.parameters(), lr=1e-5), [g], policy, cfg, teacher_term=term)
        steps = min(report['optimizer_steps'], term.plan_report['batches'])
        self.assertGreater(steps, 0)
        self.assertEqual((len(soft.steps), soft.iteration_steps), (steps, steps))
        self.assertFalse(torch.equal(soft.head.detach(), before))
        out = term.report()['soft']
        self.assertEqual((out['stage'], out['head_steps']), ('head_only', steps))
        self.assertIsNotNone(out['kl']['first'])
        self.assertTrue(all(s['raw_trunk_norm'] == 0. for s in soft.steps))  # head-only: nothing reaches the trunk

    def test_the_teacher_phase_carries_the_head_from_one_iteration_to_the_next(self):
        from brass_ai.recovery.teacher_phase import TeacherPhase
        from brass_ai.recovery.value_preservation import TrunkGradients
        rows, heldout = self.rows, [r for _, r in self.rows[:4]]
        config = dict(seed=5, teacher_revision=1, soft_policy=soft_settings())

        class Buffer:
            def rows(self, generation): return rows
            def payload(self): return {}

        def phase(generation, soft_state):
            p = TeacherPhase.__new__(TeacherPhase)
            p.config, p.options, p.generation, p.weight = config, self.options, generation, dict(beta=.03, calibrated=True)
            p.buffer, p.heldout, p.metrics, p.jobs, p.cursor = Buffer(), heldout, {}, [], 0
            p.receipt, p.reward_id, p.admission_seconds, p.soft_state = None, 'r', 0., soft_state
            return p

        first = phase(3270, None)
        term = first.prepare(self.model); term.plan(2)
        for step in (1, 2):
            term.apply(self.model, TrunkGradients(self.model), step); self.model.zero_grad(set_to_none=True)
        report = first.finish(self.model, {})
        soft = report['heldout']['soft']
        self.assertEqual((soft['before']['rows'], report['term']['soft']['head_steps']), (4, 2))
        self.assertNotEqual(soft['before']['kl'], soft['after']['kl'])  # the head moved; the policy did not
        self.assertEqual(report['heldout']['delta']['all'], 0.)
        payload = first.payload(3271)
        self.assertEqual(json.loads(json.dumps(payload, allow_nan=False)), payload)
        state = payload['soft_policy']
        self.assertEqual((state['generation'], state['trained_iterations'], state['head_steps']), (3271, 1, 2))
        second = phase(3271, state)
        second.prepare(self.model)
        self.assertEqual(second.soft.head.detach().tolist(), state['head'])
        self.assertEqual((second.soft.trained, second.soft.multiplier, second.soft.fresh), (1, 0., False))
        self.assertEqual(second.payload(3272)['soft_policy']['generation'], 3272)  # carried even before it finishes


class Recipe(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from test_teacher_revision_g3270 import before
        from brass_ai.recovery import training_revision_g3270 as g3270
        cls.base = before(); cls.before = g3270.configure(cls.base, 3270); cls.after = R.configure(cls.before, 3423)

    def test_it_is_adopted_once_on_the_g3270_recipe(self):
        from brass_ai.recovery.config import validate_config
        from brass_ai.rust_backend.lab_teacher_activation import is_on
        before, after = self.before, validate_config(copy.deepcopy(self.after))
        self.assertEqual(R.previous(after), before)
        self.assertEqual(after['teacher_phase'], before['teacher_phase'])
        self.assertTrue(is_on(before) and is_on(after))  # recipe recognition peels the block
        self.assertEqual(R.configure(after, 3424), after)
        self.assertEqual(R.settings(after)['from_generation'], 3423)
        self.assertIsNone(R.settings(before))
        self.assertEqual(R.tag(after), 'soft-t4-w8-s1')
        bad = copy.deepcopy(after); bad[R.KEY]['weight'] = 16.
        with self.assertRaisesRegex(ValueError, 'g3423 soft policy'):validate_config(bad)
        with self.assertRaisesRegex(ValueError, 'g3270 teacher revision'):R.configure(self.base, 3423)

    def test_signature_transitions_only_adopt_or_keep(self):
        from brass_ai.rust_backend.lab_replay_compat import QUALIFIED
        from brass_ai.rust_backend.lab_shadow_config import signature_transition
        sig = lambda config: dict(config=config, implementation=QUALIFIED)
        self.assertTrue(signature_transition(sig(self.before), sig(self.after)))
        self.assertTrue(signature_transition(sig(self.after), sig(self.after)))
        self.assertFalse(signature_transition(sig(self.after), sig(self.before)))
        changed = copy.deepcopy(self.after); changed[R.KEY]['from_generation'] = 3430
        self.assertFalse(signature_transition(sig(self.after), sig(changed)))


if __name__ == '__main__':
    unittest.main()
