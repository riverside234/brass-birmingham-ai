"""The g1294 revision: entropy raise, keep-at-KL-cap PPG, 64 search-head steps."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from brass_ai.bank import read
from brass_ai.learning.contracts import SOURCE_ROOT as ROOT
from brass_ai.recovery import entropy_schedule as S, ordinary, ppg_g210 as G, training_revision_g1294 as T, value_phase as V
from brass_ai.recovery.actor_value import KEY, TrainingPhases
from brass_ai.recovery.collector_runtime import recipe
from brass_ai.recovery.config import validate_config
from brass_ai.recovery.retention import _same_payload
from brass_ai.recovery.search_value_config import MAX_STEPS, defaults, max_steps_options, validate as validate_head
from brass_ai.recovery.search_value_schedule import previous_recipe
from brass_ai.recovery.search_value_training import Companion, unconditional_pass
from brass_ai.rust_backend import lab_shadow_config as C, lab_teacher_activation as A
from test_ppg_g210 import config as ppg_config, prepare

EXPORT = ROOT/'results/joint7-v11-2-g1290-trained-restart-g1294-20261110T000540895274Z'


class Recipe(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.state = read(EXPORT/'state.json')
        cls.before = recipe(cls.state['signature']['config'])
        cls.after = T.configure(cls.before, cls.state)

    def test_only_the_three_requested_settings_change(self):
        self.assertTrue(A.is_on(self.before))
        validate_config(self.after)
        self.assertTrue(A.is_on(self.after))
        changed = {k for k in self.after if self.after[k] != self.before[k]}
        self.assertEqual(changed, {'update', 'value_phase', 'teacher_phase'})
        self.assertEqual(self.after['value_phase'], dict(self.before['value_phase'], contract=G.KL_STOP_CONTRACT))
        update = copy.deepcopy(self.after['update'])
        block = update[S.KEY].pop(S.RAISE)
        self.assertEqual(update, self.before['update'])
        self.assertEqual(block['start_generation'], 1295)
        head = self.after['teacher_phase']['search_value']
        self.assertEqual((head['steps_per_iteration'], head['trunk']['intervals']), (64, 4))

    def test_the_old_recipe_is_recovered_exactly(self):
        self.assertEqual(T.previous(self.after), self.before)
        self.assertEqual(previous_recipe(self.after), previous_recipe(self.before))

    def test_the_revision_is_applied_once_and_keeps_its_start(self):
        later = copy.deepcopy(self.state)
        later['iteration'] += 5
        self.assertEqual(T.configure(self.after, later), self.after)
        self.assertEqual(A.configure(self.after, .1, revise_stages=True), self.after)

    def test_it_cannot_be_undone_or_retimed(self):
        self.assertTrue(T.compatible(self.before, self.after))
        self.assertFalse(T.compatible(self.after, self.before))
        moved = copy.deepcopy(self.after)
        moved['update'][S.KEY][S.RAISE]['start_generation'] += 1
        self.assertFalse(T.compatible(self.after, moved))

    def test_an_older_boundary_keeps_its_recipe(self):
        earlier = dict(self.state, iteration=1284)
        self.assertFalse(T.due(earlier))
        with self.assertRaisesRegex(ValueError, 'g1294 or later'):
            T.configure(self.before, earlier)

    def test_an_unfinished_anneal_refuses_the_revision(self):
        with patch.object(S, 'restore_run', return_value=dict(config=dict(mode='on', ramp_iterations=100), progress=99,
                                                              insolvency_steps_applied=9)):
            with patch.object(S, 'waiting', return_value=False), self.assertRaisesRegex(ValueError, 'completed entropy anneal'):
                T.configure(self.before, self.state)


class Entropy(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        state = read(EXPORT/'state.json')
        cls.state = state
        cls.config = T.configure(recipe(state['signature']['config']), state)
        cls.schedule = S.restore_run(cls.config, state)
        cls.c = cls.schedule['config']

    def test_the_weight_rises_linearly_to_exactly_the_target(self):
        expected = {1294: .001, 1295: .00104, 1344: .003, 1393: .00496, 1394: .005, 1395: .005, 2000: .005}
        for generation, value in expected.items():
            for weight in S.weights(self.c, 100, generation).values():
                self.assertAlmostEqual(weight, value, places=12)

    def test_the_applied_weight_is_the_first_step_after_adoption(self):
        self.assertEqual(self.schedule['last_generation'], 1294)
        self.assertAlmostEqual(self.schedule['applied']['canal_action'], .00104, places=12)

    def test_every_committed_iteration_counts_even_when_unhealthy(self):
        s, report = S.advance(self.schedule, self.config, generation=1295, healthy=False)
        self.assertEqual(report[S.RAISE]['step'], 1)
        self.assertEqual(report['reason'], 'complete')
        self.assertAlmostEqual(report['next_alpha']['rail_other'], .00108, places=12)
        S.validate_state(s, 1295)

    def test_the_schedule_round_trips_through_a_restart(self):
        s, _ = S.advance(self.schedule, self.config, generation=1295, healthy=True)
        restored = S.restore(self.config, s, 1295)
        self.assertEqual(restored['applied'], s['applied'])
        self.assertEqual(restored['config'], s['config'])

    def test_ppo_and_its_collection_receipt_agree(self):
        c = self.config['update'][S.KEY]
        alphas = S.weights(c, 100, 1344)
        from brass_ai.learning.contracts import digest
        receipt = dict(contract=c['contract'], config_id=digest(c), generation=1344, progress=100, alphas=alphas)
        behavior = type('Behavior', (), dict(entropy_schedule=receipt, entropy_alphas=alphas))()
        self.assertEqual(S.behavior_weights(self.config['update'], behavior), alphas)

    def test_only_the_reviewed_ramp_is_valid(self):
        for field, value in (('ramp_iterations', 50), ('start_generation', 0), ('final', dict.fromkeys(S.GROUPS, .01))):
            bad = copy.deepcopy(self.c)
            bad[S.RAISE][field] = value
            with self.assertRaises(ValueError):
                S.validate_raise(bad)

    def test_the_raise_must_start_in_the_future(self):
        saved = self.state['exploration'][S.KEY]
        late = copy.deepcopy(self.config)
        late['update'][S.KEY][S.RAISE]['start_generation'] = 1294
        with self.assertRaisesRegex(ValueError, 'prospectively'):
            S.restore(late, saved, 1294)

    def test_an_adopted_raise_cannot_be_dropped(self):
        s, _ = S.advance(self.schedule, self.config, generation=1295, healthy=True)
        with self.assertRaises(ValueError):
            S.restore(S.without_raise(self.config), s, 1295)


class SearchHead(unittest.TestCase):
    def test_the_revised_recipe_allows_exactly_64_steps(self):
        state = read(EXPORT/'state.json')
        head = max_steps_options(recipe(state['signature']['config']))['teacher_phase']['search_value']
        validate_head(head)
        self.assertEqual(head['steps_per_iteration'], MAX_STEPS)
        for field, value in (('steps_per_iteration', 48), ('steps_per_iteration', 128)):
            with self.assertRaises(ValueError):
                validate_head(dict(head, **{field: value}))
        bad = copy.deepcopy(head)
        bad['trunk']['intervals'] = 2
        with self.assertRaises(ValueError):
            validate_head(bad)

    def test_original_recipe_stays_at_32(self):
        self.assertEqual(defaults()['steps_per_iteration'], 32)
        self.assertEqual(defaults()['trunk']['intervals'], 2)

    def test_the_stage_passes_its_step_budget_to_the_head_and_trunk(self):
        from brass_ai.recovery.search_value_phase import SearchValuePhase
        phase = SearchValuePhase.__new__(SearchValuePhase)
        phase.options = dict(steps_per_iteration=64, trunk=dict(intervals=4), schedule={})
        phase.report = {}
        calls = {}
        phase.machine = type('Machine', (), dict(state=dict(lam=1., buffer_rebuild=None), advance=lambda self, *a, **k: {}, choose_lambda=lambda self, v: None))()
        phase.buffer = type('Buffer', (), dict(rows=lambda self, generation, split: [dict(split='train', game_id='g', index=0, terminal_utility=[0.]*4, u=[0.]*4, r=[0.]*4)] if split == 'train' else []))()
        phase.companion = type('Companion', (), dict(head_only=lambda self, rows, generation, steps: calls.update(steps=steps) or dict(steps=steps), seed=1))()
        phase.trunk_scale = lambda: 0.
        with patch('brass_ai.recovery.search_value_schedule.diagnostics', return_value={}):
            phase._scheduled_after_ppg(1294, guarded=None)
        self.assertEqual(calls['steps'], 64)

    def test_a_head_only_pass_runs_all_64_steps(self):
        from test_b025 import model_fixture
        from test_b045 import value_fixture
        model = model_fixture()
        row, _, _ = value_fixture(model)
        companion = Companion(model)
        report = companion.head_only([row], generation=1, steps=64)
        self.assertEqual(report['steps'], 64)
        with self.assertRaises(ValueError):
            companion.head_only([row], generation=2, steps=65)

    def test_the_trunk_pass_stops_within_64_steps_and_uses_the_full_rate_schedule(self):
        from test_b025 import model_fixture
        from test_b045 import value_fixture
        model = model_fixture()
        row, _, _ = value_fixture(model)
        companion = Companion(model)
        rates = []
        def rate(step, total):
            rates.append((step, total))
            return 1e-6
        optimizer = torch.optim.Adam([p for n, p in model.named_parameters() if not n.startswith('search_value_head.')], lr=1e-6)
        guard = dict(value_optimizer=optimizer, cloning_loss=lambda: sum(p.sum()*0 for p in model.parameters()),
                     measure_joint_kl=lambda: 0., beta=.25, rate_for_step=rate, kl_cap=.003)
        result = unconditional_pass(companion, [row], scale=.1, intervals=4, **guard)
        self.assertEqual(result['steps'], 64)
        self.assertEqual(len(result['intervals']), 4)
        self.assertEqual(rates[0], (1, 64))
        self.assertEqual(rates[-1], (64, 64))


class KlCap(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def phases(self, tmp, batch_size=2):
        m, opt, c = ppg_config()
        c['value_phase'] = dict(G.KL_STOP_DEFAULT, batch_size=batch_size, clone_rows=64, clone_batch_size=8)
        G.validate(c['value_phase'])
        p = TrainingPhases(m, c, root=ROOT, run=Path(tmp))
        prepare(m, p, 1)
        return m, p

    def run_fit(self, m, p, losses, kl):
        """losses maps the optimizer step count to the held-out loss seen there."""
        metric = V.metrics
        capture = {}
        real_step = p.optimizer.step
        def count():
            return max((int(s['step']) for s in p.optimizer.state.values()), default=0)
        def step(*a, **kw):
            out = real_step(*a, **kw)
            capture[count()] = copy.deepcopy(m.state_dict())
            return out
        def metrics(net, ds, ids, update, **kw):
            result = metric(net, ds, ids, update, **kw)
            if not kw.get('unknown'):  # This recipe validates on the two-age buffer.
                result['loss'] = losses(count())
            return result
        checks = iter(kl)  # Later diagnostic probes all read zero.
        with patch.object(p.optimizer, 'step', step), patch.object(V, 'metrics', metrics), \
                patch.object(V.RawPolicyReference, 'measure', side_effect=lambda *a: next(checks, 0.)):
            report = p.fit(m)
        return report, capture

    def test_hitting_the_cap_stops_and_keeps_the_update(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'artifacts') as tmp:
            m, p = self.phases(tmp)
            report, capture = self.run_fit(m, p, lambda n: 1.-.001*n, [.0035])
            self.assertEqual(report['attempted_steps'], 32)
            self.assertEqual(report['optimizer_steps'], 32)
            self.assertTrue(report['kl_guard_tripped'])
            self.assertEqual(report['early_stop_reason'], 'policy_kl_cap')
            self.assertEqual(report['acceptance'], 'kl-cap-stops-and-keeps-interval/1.0.0')
            self.assertTrue(report['interval_reports'][-1]['kept_at_kl_cap'])
            self.assertEqual(report['steps_before_kl_cap'], 16)
            self.assertAlmostEqual(report['probe_kl'], .0035)
            self.assertTrue(_same_payload(capture[32], m.state_dict()))

    def test_the_kept_update_stands_even_if_held_out_loss_rose(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'artifacts') as tmp:
            m, p = self.phases(tmp)
            report, capture = self.run_fit(m, p, lambda n: 1. if n == 0 else .9 if n == 16 else .95, [.0035])
            self.assertEqual(report['optimizer_steps'], 32)
            self.assertTrue(report['interval_reports'][-1]['accepted'])
            self.assertIsNone(report['interval_reports'][-1]['rejection_reason'])
            self.assertTrue(_same_payload(capture[32], m.state_dict()))

    def test_a_held_out_loss_rise_without_a_cap_hit_still_rolls_back(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'artifacts') as tmp:
            m, p = self.phases(tmp)
            report, capture = self.run_fit(m, p, lambda n: 1. if n == 0 else .9 if n == 16 else .95, [.001, .001])
            self.assertEqual(report['optimizer_steps'], 16)
            self.assertFalse(report['kl_guard_tripped'])
            self.assertEqual(report['early_stop_reason'], 'heldout_loss_increased')
            self.assertEqual(report['interval_reports'][1]['rejection_reason'], 'heldout_loss_increased')
            self.assertTrue(_same_payload(capture[16], m.state_dict()))

    def test_a_retained_state_over_the_cap_is_kept_but_counts_as_a_trip(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'artifacts') as tmp:
            m, p = self.phases(tmp)
            report, capture = self.run_fit(m, p, lambda n: 1. if n == 0 else .9 if n == 16 else .95, [.001, .0035])
            self.assertEqual(report['optimizer_steps'], 16)
            self.assertTrue(report['kl_guard_tripped'])
            self.assertEqual([(v['step'], v['scope']) for v in report['kl_checks']], [(32, 'periodic'), (16, 'retention')])
            self.assertTrue(_same_payload(capture[16], m.state_dict()))

    def test_a_rejected_first_interval_keeps_nothing(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'artifacts') as tmp:
            m, p = self.phases(tmp)
            before = copy.deepcopy(m.state_dict())
            report, _ = self.run_fit(m, p, lambda n: 1. if n == 0 else 1.1, [])
            self.assertEqual(report['optimizer_steps'], 0)
            self.assertFalse(report['kl_guard_tripped'])
            self.assertTrue(_same_payload(before, m.state_dict()))

    def test_an_early_hit_still_raises_beta_as_before(self):
        c = G.KL_STOP_DEFAULT
        report = dict(kl_guard_tripped=True, optimizer_steps=32, steps_before_kl_cap=16, probe_kl=.0035)
        beta, trips, feedback = G.beta_after_phase(.25, [], report, c)
        self.assertEqual(beta, .375)
        self.assertTrue(feedback['early_kl_trigger'])
        self.assertEqual(feedback['retained_steps'], 16)
        later = dict(kl_guard_tripped=True, optimizer_steps=64, steps_before_kl_cap=48, probe_kl=.0035)
        self.assertEqual(G.beta_after_phase(.25, [], later, c)[0], .25)

    def test_adoption_keeps_adam_replay_and_beta(self):
        from brass_ai.learning.model import ModelConfig
        from brass_ai.learning.utility_model import UtilityNet, utility_config
        from brass_ai.recovery.utility import reward_spec
        old = recipe(read(EXPORT/'state.json')['signature']['config'])
        model = UtilityNet(utility_config(ModelConfig.preset('small'), reward_spec(old)))
        with tempfile.TemporaryDirectory(dir=ROOT/'artifacts') as tmp:
            run = Path(tmp)
            phase = TrainingPhases(model, old, root=ROOT, run=run)
            p = next(model.parameters())
            p.grad = torch.ones_like(p)
            phase.optimizer.step()
            phase.optimizer.zero_grad(set_to_none=True)
            phase.buffer.pending = []
            phase.beta = .5
            phase.trips = [True, False]
            raw = dict(model_config=model.config.to_dict(), training={KEY: phase.payload()})
            before = copy.deepcopy((model.state_dict(), phase.optimizer.state_dict(), torch.get_rng_state(), phase.state, phase.critic_state))
            new = G.kl_stop_options(old)
            revised = TrainingPhases(model, new, raw=raw, root=ROOT, run=run)
            self.assertEqual(revised.beta, .5)
            self.assertEqual(revised.trips, phase.trips)
            self.assertEqual(revised.ppg_schedule_migration['contract'], G.KL_STOP_MIGRATION)
            self.assertTrue(_same_payload(before, (model.state_dict(), revised.optimizer.state_dict(), torch.get_rng_state(), revised.state, revised.critic_state)))
            revised.buffer.pending = []
            again = TrainingPhases(model, new, raw=dict(training={KEY: revised.payload()}), root=ROOT, run=run)
            self.assertIsNone(again.ppg_schedule_migration)

    def test_the_previous_recipe_keeps_its_rollback(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'artifacts') as tmp:
            m, p = self.phases(tmp)
            p.options['value_phase'] = dict(G.TEACHER_FLOOR_DEFAULT, batch_size=2, clone_rows=64, clone_batch_size=8)
            before = copy.deepcopy(m.state_dict())
            report, _ = self.run_fit(m, p, lambda n: 1.-.001*n, [.0035, .0035])
            self.assertEqual(report['optimizer_steps'], 0)
            self.assertTrue(report['kl_guard_tripped'])
            self.assertTrue(_same_payload(before, m.state_dict()))


class Planning(unittest.TestCase):
    def test_the_local_scoring_host_is_verified_once_per_plan(self):
        state = read(EXPORT/'state.json')
        report = read(EXPORT/'iteration-1293'/'report.json')
        config = state['signature']['config']
        fast = next(m for m in state['quality_pool']['members'] if 'policies' in m)['policies']
        service = type('Service', (), dict(models={}, defer_checkpoint=lambda self, a, b: None))()
        calls = []
        with tempfile.TemporaryDirectory(dir=ROOT/'artifacts') as tmp:
            host = Path(tmp)
            (host/'runtime.json').write_text('{}')
            (host/'BrassHost.exe').write_bytes(b'host')
            def install(root):
                calls.append(root)
                return host
            from brass_ai.recovery import collector_runtime as R
            from brass_ai.recovery.curriculum import boost_window
            with patch('brass_ai.recovery.host_runtime.local_scoring_host', install), \
                    patch.object(ordinary, 'verify_pool_checkpoints', lambda root, members, verified: verified.update(m['model_id'] for m in members if 'path' in m)), \
                    patch('brass_ai.rust_backend.lab_cpu_models.bundle_ready', return_value=True), \
                    R.scope(dict(R.DEFAULTS, native_inference=R.cpu_block())):
                plan = ordinary.plan_collection(ROOT, Path('x'), state, config, state['signature'], service, state['history'][-1]['model_id'],
                    sorted(fast), fast, fast, boost_window(state, config), fingerprints=None, exploration=report['exploration_rollout'])
        self.assertEqual(len(calls), 1)
        receipts = [j['provenance']['local_scoring'] for j in plan['jobs']]
        self.assertEqual(len(receipts), config['games_per_iteration'])
        self.assertTrue(all(r == receipts[0] for r in receipts))
        self.assertIsNot(receipts[0], receipts[1])


if __name__ == '__main__':
    unittest.main()
