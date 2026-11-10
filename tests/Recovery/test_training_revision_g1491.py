"""The g1491 revision: long+short teacher labels, 48 tables, one PPO epoch, a two-iteration beta re-calibration."""
import copy
import importlib.util
import math
import unittest

import numpy as np
import torch

from brass_ai.bank import read
from brass_ai.learning.contracts import SOURCE_ROOT as ROOT, digest
from brass_ai.recovery import (lr_controller as L, teacher_gradient_qualification as Q, teacher_labels as TL,
                               training_revision_g1294 as T, training_revision_g1307 as F, training_revision_g1491 as R)
from brass_ai.recovery.collector_runtime import recipe
from brass_ai.recovery.config import validate_config
from brass_ai.recovery.search_value_schedule import previous_recipe, validation_plan
from brass_ai.recovery.teacher_phase import committed_weight, plan_jobs
from brass_ai.recovery.teacher_phase_config import buffer_identity
from brass_ai.rust_backend import lab_shadow_config as C, lab_teacher_activation as A

EXPORT = ROOT/'results/joint7-v11-2-g1290-restart-g1290-g1491-20261110T191929753967Z'
PHASES = ('canal_early', 'canal_late', 'rail_early', 'rail_late')


def committed():
    state = read(EXPORT/'state.json')
    return state, recipe(state['signature']['config'])


class Recipe(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.state, cls.before = committed()
        cls.after = R.configure(cls.before, cls.state)

    def test_the_committed_recipe_is_the_reviewed_teacher_on_recipe(self):
        self.assertEqual(self.state['iteration'], 1491)
        self.assertTrue(A.is_on(self.before))
        self.assertEqual(self.before['teacher_phase']['term']['share_target'], .4)

    def test_exactly_the_requested_fields_change(self):
        validate_config(self.after)
        self.assertEqual({k for k in self.after if self.after[k] != self.before[k]}, {'teacher_phase', 'update'})
        self.assertEqual({k for k in self.after['update'] if self.after['update'][k] != self.before['update'][k]}, {'epochs'})
        self.assertEqual((self.before['update']['epochs'], self.after['update']['epochs']), (2, 1))
        old, new = self.before['teacher_phase'], self.after['teacher_phase']
        self.assertEqual({k for k in set(old) | set(new) if old.get(k) != new.get(k)},
                         {'mode', 'tables', 'buffer', 'term', R.KEY})
        self.assertEqual((old['mode'], new['mode']), ('on', 'shadow'))
        self.assertEqual(new['tables'], dict(train=48, held_out=4))
        self.assertEqual(new['buffer'], dict(iterations=2, age_weights=[4, 3], max_staleness_ratio=1.))
        self.assertEqual(new['term'], dict(old['term'], rows_per_step=2048, shadow_iterations=2))
        self.assertEqual(new[R.KEY]['labels'], dict(searches='long-and-short', long_weight=.6, short_weight=.4))
        from brass_ai.recovery.teacher_cotton import settings
        self.assertEqual(new[R.KEY]['exploration'], dict(contract='root-visit-count-sampling/1.0.0', temperature=1.,
                                                         opening_actions_per_era=7, long_search='always',cotton=settings()))
        # Search budgets and K stay in the operational CPU block; the job profiles are unchanged.
        self.assertEqual((new['search'], new['fast_search'], new['search_value']), (old['search'], old['fast_search'], old['search_value']))

    def test_it_is_a_reviewed_shadow_recipe_projecting_to_the_committed_one(self):
        self.assertTrue(C.is_shadow(self.after))
        self.assertFalse(A.is_on(self.after))
        expected = copy.deepcopy(self.before)
        expected['teacher_phase'].update(mode='shadow')
        expected['teacher_phase']['term']['shadow_iterations'] = 2
        self.assertEqual(R.previous(self.after), expected)
        self.assertEqual(previous_recipe(self.after), previous_recipe(expected))

    def test_it_is_adopted_once_and_kept_by_the_other_revisions(self):
        self.assertEqual(R.configure(self.after, self.state), self.after)
        self.assertEqual(T.configure(self.after, self.state), self.after)
        self.assertEqual(F.configure(self.after), self.after)
        self.assertTrue(R.due(self.state))
        self.assertFalse(R.due(dict(self.state, iteration=1490)))
        with self.assertRaisesRegex(ValueError, 'starts at g1491'):
            R.configure(self.before, dict(self.state, iteration=1490))

    def test_other_values_are_not_reviewed(self):
        for path, value in [('teacher_phase.tables.train', 40), ('teacher_phase.buffer.age_weights', [1, 1]),
                            ('teacher_phase.term.rows_per_step', 1024), ('teacher_phase.term.shadow_iterations', 4),
                            ('update.epochs', 2), (f'teacher_phase.{R.KEY}.labels.long_weight', .5),
                            (f'teacher_phase.{R.KEY}.exploration.opening_actions_per_era', 8)]:
            bad = copy.deepcopy(self.after); target = bad; keys = path.split('.')
            for k in keys[:-1]: target = target[k]
            target[keys[-1]] = value
            with self.subTest(path=path), self.assertRaises(ValueError):
                validate_config(bad)

    def test_new_labels_retire_the_old_labels_and_beta(self):
        self.assertNotEqual(buffer_identity(self.after), buffer_identity(self.before))
        self.assertEqual(committed_weight(self.before, self.state), self.state['teacher_phase']['weight'])
        self.assertEqual(committed_weight(self.after, self.state), {})
        # Planning at g1491 must not restore the old four-iteration qualification into a two-iteration window.
        plan = validation_plan(self.after, self.state)
        self.assertTrue(plan['due'] and plan['forced_by_beta_qualification'])

    def test_jobs_carry_the_revision_and_the_new_table_counts(self):
        jobs, _ = plan_jobs(self.after, self.state, model_id='m'*64, export_hash='a'*64, epoch=1491, ready=True)
        self.assertEqual([j['split'] for j in jobs].count('train'), 48)
        self.assertEqual([j['split'] for j in jobs].count('heldout'), 4)
        for job in jobs:
            self.assertEqual(job['teacher_revision'], R.job_block(self.after,split=job['split']))
            self.assertEqual(R.validate_job(job), job['teacher_revision'])
        from brass_ai.recovery.teacher_cotton import settings
        self.assertEqual(R.job_block(self.after), dict(contract=R.CONTRACT, short_labels=True,
            exploration=dict(contract='root-visit-count-sampling/1.0.0', temperature=1., opening_actions_per_era=7, long_search=True,cotton=settings())))
        old, _ = plan_jobs(self.before, self.state, model_id='m'*64, export_hash='a'*64, epoch=1491, ready=True)
        self.assertEqual(len(old), 28)
        self.assertTrue(all('teacher_revision' not in j for j in old))
        changed = copy.deepcopy(jobs[0]); changed['teacher_revision']['exploration']['opening_actions_per_era'] = 9
        with self.assertRaisesRegex(ValueError, 'Unreviewed g1491'):
            R.validate_job(changed)


class Transitions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.state, cls.before = committed()
        cls.shadow = R.configure(cls.before, cls.state)
        cls.on = A.configure(cls.shadow, .4, revise_stages=True)

    def signature(self, config):
        return dict(self.state['signature'], config=config)

    def test_the_on_recipe_goes_back_to_shadow_one_way(self):
        self.assertTrue(C.signature_transition(self.state['signature'], self.signature(self.shadow)))
        self.assertFalse(C.signature_transition(self.signature(self.shadow), self.state['signature']))
        other = copy.deepcopy(self.shadow); other['teacher_phase']['term']['share_target'] = .3
        self.assertTrue(C.signature_transition(self.state['signature'], self.signature(other)))
        # Changing anything else at the boundary is refused.
        bad = copy.deepcopy(self.shadow); bad['update']['clip'] = .3
        self.assertFalse(C.signature_transition(self.state['signature'], self.signature(bad)))

    def test_calibration_then_teaching_on(self):
        self.assertTrue(A.is_on(self.on))
        self.assertEqual({k for k in self.on['teacher_phase'] if self.on['teacher_phase'][k] != self.shadow['teacher_phase'][k]}, {'mode'})
        self.assertEqual({k for k in self.on if self.on[k] != self.shadow[k]}, {'teacher_phase'})
        self.assertTrue(C.signature_transition(self.signature(self.shadow), self.signature(self.on)))
        self.assertTrue(C.signature_transition(self.signature(self.on), self.signature(self.on)))
        self.assertFalse(C.signature_transition(self.signature(self.on), self.signature(self.shadow)))
        self.assertFalse(R.holds_ppo_lr(self.on))
        self.assertTrue(R.holds_ppo_lr(self.shadow))
        self.assertFalse(R.holds_ppo_lr(self.before))

    def test_the_collection_changes_at_g1491_and_stays_through_the_switch(self):
        old, shadow, on = (C.collection_signature(self.signature(c)) for c in (self.before, self.shadow, self.on))
        self.assertNotEqual(old, shadow)
        self.assertEqual(shadow, on)


class Qualification(unittest.TestCase):
    """Two consecutive full 48+4 shadow iterations fix beta; teaching then resumes at Stage C."""

    @classmethod
    def setUpClass(cls):
        cls.state, cls.before = committed()
        cls.shadow = R.configure(cls.before, cls.state)
        cls.on = A.configure(cls.shadow, .4, revise_stages=True)

    def measure(self, generation, previous, raw, tables=None):
        context = dict(tables=tables or {'train': 48, 'heldout': 4}, labels_by_phase={p: 30 for p in PHASES},
                       model_id='m%d' % generation, export_hash='h%d' % generation)
        term = type('Term', (), {})()
        term.config = self.shadow['teacher_phase']; term.mode = 'shadow'; term.generation = generation
        term.qualification = previous; term.qualification_context = context; term.beta = 1.; term.calibrated = False
        term.steps = [dict(raw_share=v, cosine=.01, actor_norm=1., teacher_norm=v, share=v) for v in raw]
        state = {}
        Q.finish(term, state)
        return state['gradient_qualification']

    def test_two_full_iterations_qualify_and_authorize_teaching(self):
        first = self.measure(1491, None, [4., 5., 6.])
        self.assertFalse(first['passed'])
        self.assertEqual(first['required_iterations'], 2)
        second = self.measure(1492, first, [5., 7.])
        self.assertTrue(second['passed'])
        self.assertEqual(second['raw_median'], 5.)
        self.assertEqual(second['beta'], Q.fixed_beta(.4, 5.))
        self.assertEqual(second['qualified_generation'], 1493)
        state = copy.deepcopy(self.state)
        state.update(iteration=1493, signature=dict(state['signature'], config=self.shadow))
        # The completed anneal advances its generation each iteration.
        state['exploration']['entropy_schedule']['last_generation'] = state['exploration']['generation'] = 1493
        state['history'] = state['history']+[dict(generation=g, model_id='m%d' % g, sha256='h%d' % g, path='x') for g in (1491, 1492)]
        state['teacher_phase']['weight'] = dict(gradient_qualification=second)
        self.assertEqual(state['search_value']['stage']['stage'], 'C')
        q = A.authorize(state, self.on)
        self.assertEqual(q['beta'], second['beta'])
        # An unrevised first activation still requires Stage A.
        state['signature']['config'] = self.before
        old = copy.deepcopy(self.before); old['teacher_phase']['mode'] = 'shadow'
        state['signature']['config'] = old
        with self.assertRaises(ValueError):
            A.authorize(state, self.before)

    def test_a_28_table_iteration_does_not_qualify_the_new_recipe(self):
        q = self.measure(1491, None, [4., 5.], tables={'train': 28, 'heldout': 4})
        self.assertEqual(q['observations'], [])
        self.assertIn('48+4', q['reason'])


class Admission(unittest.TestCase):
    def test_48_4_tables_count_long_and_short_labels_separately(self):
        from types import SimpleNamespace
        from unittest.mock import Mock
        from brass_ai.recovery.teacher_phase import TeacherPhase
        state, before = committed()
        after = R.configure(before, state)
        phase = TeacherPhase.__new__(TeacherPhase)
        phase.generation, phase.reward_id, phase.config, phase.state, phase.jobs = 1491, 'reward', after, state, []
        results, rows = [], {}
        for i in range(52):
            key = str(i)
            phase.jobs.append(dict(expected={'job_id': key, 'model_id': 'model'}, split='train' if i < 48 else 'heldout',
                                   reservations={'0': {'canal_early': 1}}, teacher_revision=R.job_block(after)))
            label = dict(kind='root', phase='canal_early', seat=0, completed_trajectories=64, effective_k=8, completed_rounds=3,
                         deadline_hit=False, overrun_seconds=0., fallback_played=False, explored=True)
            long = SimpleNamespace(row={'view': {'context': {'round': 1}}}, label=label)
            short = SimpleNamespace(row={'view': {'context': {'round': 2}}}, label=dict(label, profile='fast', explored=False))
            rows[key] = [long, short, SimpleNamespace(row=short.row, label=dict(short.label, explored=True))]
            results.append(dict(job_id=key, metrics={'reserved_turns': 16, 'turns': []}))
        phase.buffer = Mock(); phase.search_value = Mock()
        verifier = Mock(); verifier.take.side_effect = lambda result, job, model: (rows[result['job_id']], {})
        phase.admit_collection(results, None, verifier=verifier)
        m = phase.metrics
        self.assertEqual(m['tables'], {'train': 48, 'heldout': 4})
        self.assertEqual((m['labels'], m['long_labels'], m['short_labels']['labels']), (156, 52, 104))
        self.assertEqual(m['labels_by_phase'], {'canal_early': 52})
        self.assertEqual(m['explored_labels'], dict(long=52, short=52))
        self.assertTrue(m['heldout_validation']['forced_by_beta_qualification'])
        # A short label on a reserved turn is refused.
        rows['0'][1] = SimpleNamespace(row={'view': {'context': {'round': 1}}}, label=rows['0'][1].label)
        with self.assertRaisesRegex(ValueError, 'reserved long-search turn'):
            phase.admit_collection(results, None, verifier=verifier)


class Controller(unittest.TestCase):
    def measurements(self, kl, **extra):
        return dict(kl=kl, steps_done=23, steps_planned=23, early_stop=False, warmup_done=True, data_signature='a'*64,
                    selfplay_vp_per_seat=140., economic_health=dict(contract='selfplay-economic-health/3.0.0', ok=True, reasons=[]),
                    **extra)

    def test_calibration_holds_the_rate_and_keeps_its_kl_out_of_the_window(self):
        _, before = committed()
        state = L.initial_state(before['ppo_lr_controller'])
        state['window'] = [dict(kl=.002, early_stop=False), dict(kl=.002, early_stop=False)]
        held, report = L.decide(state, self.measurements(.001, hold=True))
        self.assertEqual((report['decision'], report['reason']), ('hold', 'teacher_beta_calibration'))
        self.assertEqual(held['multiplier'], 1.)
        self.assertEqual(held['window'], state['window'])
        # Even a KL far above the band does not cut while beta is being measured.
        held, report = L.decide(state, self.measurements(.05, hold=True))
        self.assertEqual((held['multiplier'], held['cooldown']), (1., 0))
        # Without the hold the same low KL completes the window and raises.
        raised, report = L.decide(state, self.measurements(.001))
        self.assertEqual(report['decision'], 'up')


class Labels(unittest.TestCase):
    def label(self, **changes):
        from test_b025 import example, model_fixture
        row, _ = example(model_fixture())
        row.label.update(changes)
        return row

    def test_short_labels_record_their_own_k(self):
        row = self.label(profile='fast', requested_k=8)
        self.assertIsNotNone(TL.validate_improved_policy(row.label, row.row['choices']))
        for bad in (dict(profile='full'), dict(profile='fast', requested_k=3), dict(requested_k=8),
                    dict(profile='fast', requested_k=2, effective_k=4), dict(explored='yes')):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                candidate = self.label(**dict(dict(profile='fast', requested_k=8), **bad)) if 'profile' in bad or bad.get('explored') else self.label(**bad)
                TL.validate_improved_policy(candidate.label, candidate.row['choices'])
        explored = self.label(explored=True)
        self.assertIsNotNone(TL.validate_improved_policy(explored.label, explored.row['choices']))

    def test_short_labels_live_only_on_unreserved_turns_of_a_revised_job(self):
        job = dict(reservations={'0': dict(canal_early=3)}, teacher_revision=dict(short_labels=True))
        long = dict(seat=0, phase='canal_early'); short = dict(long, profile='fast')
        TL.check_turn(long, 3, job)
        TL.check_turn(short, 2, job)
        for label, round_, current in ((long, 2, job), (short, 3, job), (short, 2, dict(job, teacher_revision=None))):
            with self.subTest(label=label, round=round_), self.assertRaises(ValueError):
                TL.check_turn(label, round_, current)


class Term(unittest.TestCase):
    """L = .6 mean(L_long) + .4 mean(L_short), sampled per pool and chunked exactly."""

    def fixture(self, short, long=4, rows_per_step=6):
        from test_b025 import config, example, model_fixture
        model = model_fixture()
        options = copy.deepcopy(config()['teacher_phase'])
        options.update(mode='on', buffer=copy.deepcopy(R.BUFFER))
        options['term']['rows_per_step'] = rows_per_step
        options[R.KEY] = R.block()
        rows = []
        for i in range(long+short):
            row, _ = example(model, seat=i % 4, round=1+i % 8)
            if i >= long: row.row['teacher'].update(profile='fast', requested_k=8)
            rows.append((i % 2, row))
        from brass_ai.recovery.teacher_term import TeacherTerm
        return model, options, rows, TeacherTerm(model, options, rows, seed=5, generation=12, state=dict(beta=1., calibrated=True))

    def test_each_pool_gets_its_share_of_the_loss(self):
        model, options, rows, term = self.fixture(short=5)
        indices, weights, count = term.stratified(3)
        self.assertEqual(count, round(.6*6))
        self.assertEqual(len(indices), 6)
        self.assertTrue(all(term.profiles[i] == 'full' for i in indices[:count]))
        self.assertTrue(all(term.profiles[i] == 'fast' for i in indices[count:]))
        self.assertAlmostEqual(sum(weights[:count]), .6)
        self.assertAlmostEqual(sum(weights[count:]), .4)
        self.assertEqual(term.stratified(3), (indices, weights, count))
        self.assertNotEqual(term.stratified(4)[0], indices)

    def test_a_missing_pool_hands_the_other_the_whole_loss(self):
        _, _, _, term = self.fixture(short=0)
        indices, weights, count = term.stratified(1)
        self.assertEqual(count, len(indices))
        self.assertAlmostEqual(sum(weights), 1.)

    def test_the_applied_gradient_is_the_weighted_formula_in_any_chunking(self):
        from brass_ai.recovery import teacher_term as M
        from brass_ai.recovery.value_preservation import TrunkGradients
        model, options, rows, term = self.fixture(short=5)
        indices, weights, count = term.stratified(2)
        selected = [term.rows[i] for i in indices]
        parameters = tuple(p for n, p in model.named_parameters() if n.startswith(M.ACTOR_PREFIXES) and p.requires_grad)
        with M.policy_context(model):
            logits = M.forward_policy(model, selected).float()
            target = torch.zeros_like(logits)
            for i, r in enumerate(selected): target[i, :len(r.label['target'])] = target.new_tensor(r.label['target'])
            logp = logits.log_softmax(-1).masked_fill(~torch.isfinite(logits), 0.)
            losses = (torch.special.xlogy(target, target)-target*logp).sum(-1)
            formula = .6*losses[:count].mean()+.4*losses[count:].mean()
            expected = torch.autograd.grad(formula, parameters, allow_unused=True)
        original = M.chunks
        try:
            M.chunks = lambda rows: original(rows, max_rows=2)
            model.zero_grad(set_to_none=True)
            term.apply(model, TrunkGradients(model), 2)
        finally:
            M.chunks = original
        self.assertAlmostEqual(term.steps[-1]['loss'], float(formula), places=5)
        for p, g in zip(parameters, expected):
            if g is None: continue
            np.testing.assert_allclose(p.grad.detach().numpy(), g.numpy(), rtol=1e-4, atol=1e-6)
        report = term.report()
        self.assertEqual(report['rows_by_profile'], {'full': 4, 'fast': 5})
        self.assertEqual(report['long_weight'], .6)

    def test_chunks_respect_the_microbatch_bounds(self):
        from brass_ai.recovery.teacher_term import chunks
        _, _, rows, term = self.fixture(short=5)
        parts = chunks(term.rows, max_rows=3, max_cells=10**9)
        self.assertEqual(sorted(i for p in parts for i in p), list(range(len(term.rows))))
        self.assertTrue(all(len(p) <= 3 for p in parts))
        width = max(len(r.label['legal_ids']) for r in term.rows)
        self.assertTrue(all(len(p)*max(len(term.rows[i].label['legal_ids']) for i in p) <= max(width, 2*width)
                            for p in chunks(term.rows, max_rows=512, max_cells=2*width)))

    def test_short_labels_need_the_revision(self):
        from brass_ai.recovery.teacher_term import TeacherTerm
        model, options, rows, _ = self.fixture(short=2)
        options.pop(R.KEY)
        with self.assertRaisesRegex(ValueError, 'need the g1491 recipe'):
            TeacherTerm(model, options, rows, seed=5, generation=12)

    def test_heldout_reports_short_labels_separately(self):
        from brass_ai.recovery.teacher_term import heldout_measure
        model, _, rows, _ = self.fixture(short=3)
        out = heldout_measure(model, [r for _, r in rows])
        self.assertEqual(out['all']['rows'], 4)
        self.assertEqual(out['short']['rows'], 3)


class Buffer(unittest.TestCase):
    def test_two_iteration_window(self):
        from brass_ai.recovery.teacher_buffer import TeacherBuffer
        buffer = TeacherBuffer.__new__(TeacherBuffer)
        buffer.identity, buffer.c_scale, buffer.iterations, buffer.retired = 'id', 1., 2, []
        buffer.pending = [dict(profile_digest='id', generation=g, c_scale=1.) for g in (1489, 1490, 1491)]
        self.assertEqual([r['generation'] for r in buffer.window(1491)], [1490, 1491])
        self.assertEqual([r['generation'] for r in buffer.retired], [1489])


class Command(unittest.TestCase):
    def module(self):
        spec = importlib.util.spec_from_file_location('continue_g1491', ROOT/'scripts/continue-rust-g1491.py')
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        return module

    def test_calibration_progress_from_the_committed_state(self):
        m = self.module()
        state, before = committed()
        shadow = R.configure(before, state)
        on = A.configure(shadow, .4, revise_stages=True)
        self.assertEqual(m.calibrated(state), (False, 2))
        state = dict(state, signature=dict(state['signature'], config=shadow), iteration=1492,
                     teacher_phase=dict(state['teacher_phase'], weight=dict(gradient_qualification=dict(
                         passed=False, observations=[dict(generation=1491)]))))
        self.assertEqual(m.calibrated(state), (False, 1))
        state['teacher_phase']['weight']['gradient_qualification'] = dict(passed=True, observations=[])
        self.assertEqual(m.calibrated(state), (True, 0))
        self.assertEqual(m.calibrated(dict(state, signature=dict(state['signature'], config=on))), (True, 0))

    def test_its_defaults_are_the_requested_runtime(self):
        from brass_ai.recovery.collector_runtime import TEACHER_BUDGET, cpu_block
        m = self.module(); shadow = m.shared()
        args = shadow.parser(default_mode='on', default_stage_revision=True, workers=64, teacher_fast=128,
                             inference='cpu', run=m.RUN, iterations=10).parse_args(['--expected-generation', '1491'])
        self.assertEqual((args.workers, args.teacher_fast, args.teacher_fast_k, args.teacher_full, args.inference),
                         (64, 128, 8, 640, 'cpu'))
        self.assertEqual(shadow.inference_choice(args), cpu_block(workers=64, teacher_budget=dict(TEACHER_BUDGET, fast=128)))
        # An interrupted g1491 iteration finishes with the settings its tables were collected with.
        recovery = copy.copy(args); recovery.teacher_fast = m.PREVIOUS_TEACHER_FAST
        self.assertEqual(shadow.inference_choice(recovery), cpu_block(workers=64))

    def test_an_interrupted_iteration_is_detected_from_its_sealed_work(self):
        import tempfile
        from pathlib import Path
        m = self.module()
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            (run/'teacher-labels').mkdir(); (run/'search-values').mkdir()
            self.assertFalse(m.interrupted(run, 1491))
            (run/'teacher-labels/generation-001490-train.jsonl.gz').write_bytes(b'')
            self.assertFalse(m.interrupted(run, 1491))
            (run/'teacher-labels/generation-001491-train.jsonl.gz').write_bytes(b'')
            self.assertTrue(m.interrupted(run, 1491))


if __name__ == '__main__':
    unittest.main()
