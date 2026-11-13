"""The teacher-on migration, beta controller, and dedicated return validation."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

from brass_ai.bank import read, sha
from brass_ai.learning.batching import model_id
from brass_ai.learning.checkpoint import load_checkpoint
from brass_ai.learning.contracts import SOURCE_ROOT as ROOT, digest
from brass_ai.learning.search_value import width
from brass_ai.recovery import adaptive_teacher_beta as beta, search_head_refresh as refresh
from brass_ai.recovery import search_value_schedule as schedule, teacher_cotton as cotton
from brass_ai.recovery.retention import _same_payload
from brass_ai.recovery.search_value_iteration import stamp
from brass_ai.recovery.teacher_term import TeacherTerm
from brass_ai.step5.league import write_json


class Beta(unittest.TestCase):
    def test_convergence_bounds_and_missing_pairs(self):
        state = beta.initial(1., [.05, None, None])
        for _ in range(130):
            state = beta.advance(state, 2., 5., .05)
        self.assertAlmostEqual(state['beta'] * 5. / 2., .05, places=5)
        for actor, teacher in ((0., 1.), (2., None), (float('nan'), 2.)):
            self.assertIs(beta.advance(state, actor, teacher, .05), state)
        for expected, actor, teacher in ((.01, 1., 1000.), (10., 1000., 1.)):
            state = beta.initial(1., [.05, None, None])
            for _ in range(160):
                state = beta.advance(state, actor, teacher, .05)
                self.assertTrue(.01 <= state['beta'] <= 10.)
            self.assertAlmostEqual(state['beta'], expected, places=5)

    def test_teacher_on_remeasurement_and_exact_controller_resume(self):
        from test_utility_width import recipe
        from brass_ai.recovery.teacher_gradient_qualification import finish
        from types import SimpleNamespace

        config = dict(recipe()['teacher_phase'], mode='on', **{beta.KEY: beta.settings()})
        context = dict(tables=dict(train=32, heldout=4), labels_by_phase={
            p: 1 for p in ('canal_early', 'canal_late', 'rail_early', 'rail_late')})
        old_measurement = SimpleNamespace(config=config, mode='shadow', generation=1529, qualification=None,
            beta=1., calibrated=False, qualification_context=context,
            steps=[dict(raw_share=2., actor_norm=1., teacher_norm=2., cosine=0.)])
        weight = {}
        finish(old_measurement, weight)
        q = weight['gradient_qualification']

        model = torch.nn.Linear(1, 1)

        def term(generation, state, phase=config):
            return TeacherTerm(model, phase, [], seed=7, generation=generation, state=state)

        measurement = term(1730, dict(weight, **{beta.REMEASUREMENT: dict(requested_generation=1730)}))
        self.assertEqual((measurement.mode, measurement.beta_used), ('on', q['beta']))
        waiting = measurement.finish()
        self.assertTrue(beta.pending(waiting))
        self.assertEqual(waiting['beta'], q['beta'])
        measurement = term(1730, waiting)
        measurement.qualification_context = context
        measurement.steps = [dict(raw_share=1., share=q['beta'], actor_norm=2., raw_teacher_norm=2.)]
        measured = measurement.finish()
        self.assertEqual(measured['gradient_qualification'], q)
        self.assertEqual(measured[beta.REMEASUREMENT]['measured_generation'], 1730)
        self.assertEqual(measured[beta.KEY]['beta'], .05)
        self.assertEqual(measured[beta.KEY]['updates'], 0)

        def update(controller):
            controller.steps = [dict(raw_share=3., share=.15, actor_norm=2., raw_teacher_norm=6., cosine=0., loss=1.),
                                dict(raw_share=2., share=.10, actor_norm=4., raw_teacher_norm=8., cosine=0., loss=1.),
                                dict(raw_share=None, share=None, actor_norm=None, raw_teacher_norm=1000., cosine=None, loss=1.)]
            return controller.finish()

        next_term = term(1731, measured)
        adjusted = update(next_term)
        self.assertEqual(next_term.beta_used, .05)
        self.assertNotEqual(adjusted['beta'], next_term.beta_used)
        report = next_term.report()[beta.KEY]
        self.assertEqual((report['paired_steps'], report['actor_norm_mean'], report['raw_teacher_norm_mean']), (2, 3., 7.))
        self.assertEqual(report['ema_share'], .05 * 7 / 3)
        resumed = term(1732, json.loads(json.dumps(adjusted)))
        self.assertEqual(resumed.beta_used, adjusted['beta'])
        self.assertEqual(update(resumed), update(term(1732, adjusted)))
        self.assertEqual(term(1732, adjusted).finish()[beta.KEY], adjusted[beta.KEY])

        changed_phase = dict(config, term=dict(config['term'], share_target=.08))
        changed = term(1740, dict(adjusted, outside_streak=3), changed_phase)
        self.assertEqual(changed.beta_used, adjusted['beta'])
        self.assertFalse(changed.remeasure_pending)
        missing = changed.finish()
        self.assertEqual(missing['outside_streak'], 0)
        self.assertEqual(missing[beta.REMEASUREMENT], measured[beta.REMEASUREMENT])
        self.assertEqual(missing['gradient_qualification']['observations'], q['observations'])
        self.assertEqual(missing[beta.KEY], dict(adjusted[beta.KEY], basis=[
            .08, missing['gradient_qualification']['sha256'], 1730]))
        changed_resume = term(1741, json.loads(json.dumps(missing)), changed_phase)
        self.assertEqual(changed_resume.beta_used, adjusted['beta'])
        self.assertEqual(update(changed_resume)[beta.KEY], beta.advance(missing[beta.KEY], 3., 7., .08))
        self.assertEqual(changed_resume.report()[beta.KEY]['share_target'], .08)
        self.assertEqual(term(1741, missing).beta_used, adjusted['beta'])

        fixed = term(1731, weight, {k: v for k, v in config.items() if k != beta.KEY})
        self.assertEqual(update(fixed)['beta'], q['beta'])
        self.assertNotIn(beta.KEY, fixed.report())


class SearchRefresh(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def refreshed_checkpoint(self):
        from brass_ai.learning.utility_width import widen, optimizer_state
        from brass_ai.recovery.utility_width_upgrade import configure
        from test_utility_width import Migration

        run, state = Migration.fixture(self)
        generation = state['iteration']
        previous = state['history'][-1]
        original_path = ROOT / previous['path']
        model, raw = load_checkpoint(original_path)
        options = configure(state['signature']['config'])
        config = dict(options, teacher_phase=dict(options['teacher_phase'], mode='on'))
        shapes = widen(model)
        with torch.no_grad():
            model.search_value_head[-1].bias.fill_(.37)
        saved = raw['training']['search_value']
        search = dict(saved, stage=dict(saved['stage'], stage='C', w=1., scale=1., lam=1.,
                                        stage_a_iterations=100, stage_b_iterations=50))
        backend = dict(state['native_backend'], config_sha256=digest(config), teacher=dict(
            state['native_backend']['teacher'], stage_schedule=digest(config['teacher_phase']['search_value']['schedule'])))
        signature = dict(state['signature'], config=config)
        ppg = raw['training']['actor_value_phases']
        training = dict(raw['training'], signature=digest(signature), search_value=search, native_backend=backend,
                        actor_value_phases=dict(ppg, beta_clone=.5, optimizer=optimizer_state(ppg['optimizer'], model, shapes)))
        raw = dict(raw, training=training, model_config=model.config.to_dict(), model_id=model_id(model),
                   state_dict=model.state_dict(), optimizer=optimizer_state(raw['optimizer'], model, shapes))
        torch.save(raw, original_path)
        entry = dict(previous, sha256=sha(original_path), model_id=raw['model_id'])
        state.update(signature=signature, native_backend=backend, search_value=stamp(search),
                     history=[*state['history'][:-1], entry], resume=[entry])
        write_json(run/'state.json', state)

        updated = refresh.prepare(ROOT, run, expected_generation=generation, adaptive_beta=True)
        return run, state, updated, raw

    def test_checkpoint_keeps_ordinary_weights_and_restarts_search(self):
        from brass_ai.recovery.actor_value import TrainingPhases
        from brass_ai.recovery.rating_coverage import G1290_BASELINE
        from brass_ai.recovery.resume import load_resume
        from brass_ai.recovery.teacher_phase import TeacherPhase
        from brass_ai.rust_backend.lab_teacher_activation import authorize

        run, state, updated, raw = self.refreshed_checkpoint()
        generation = state['iteration']
        entry = state['history'][-1]
        training = raw['training']
        config = updated['signature']['config']
        self.assertEqual(config['rating_coverage'], G1290_BASELINE)
        current, converted, _ = load_resume(ROOT, run, updated, updated['signature'])
        self.assertEqual((updated['iteration'], width(current), current.config.utility_hidden), (generation, 32, 256))
        self.assertEqual(sha(ROOT/entry['path']), entry['sha256'])
        self.assertTrue(all(torch.equal(t, raw['state_dict'][k]) for k, t in current.state_dict().items()
                            if not k.startswith('search_value_head.')))
        self.assertEqual(torch.count_nonzero(current.search_value_head[-1].bias).item(), 0)
        self.assertEqual(torch.count_nonzero(current.search_value_head[-1].weight).item(), 0)
        self.assertTrue(_same_payload(raw['optimizer'], converted['optimizer']))
        self.assertTrue(_same_payload(training['actor_value_phases']['optimizer'],
                                      converted['training']['actor_value_phases']['optimizer']))
        self.assertTrue(torch.equal(raw['rng'], converted['rng']))
        self.assertEqual(TrainingPhases(current, config, raw=converted, generation=generation, root=ROOT, run=run).beta, .25)
        teacher = TeacherPhase(ROOT, run, config, updated, current, converted)
        self.assertTrue(teacher.ready)
        self.assertEqual(config['teacher_phase']['term']['rows_per_step'], 96)
        self.assertIn(beta.KEY, config['teacher_phase'])
        self.assertEqual(authorize(updated, config), state['teacher_phase']['weight']['gradient_qualification'])
        stage = teacher.search_value.phase.machine.state
        self.assertEqual((stage['stage'], stage['w'], stage['lam'], stage['stage_a_iterations']), ('A', 0., .8, 0))
        self.assertEqual(teacher.search_value.phase.companion.steps, 0)
        opts = config['teacher_phase']['search_value']
        self.assertEqual([schedule.beta_floor(opts, s) for s in ('A', 'B', 'C')], [.25, .25, .5])
        jobs = teacher.begin(updated, current, export_hash=updated['history'][-1]['sha256'], epoch=generation)
        self.assertEqual([sum(j['split']==p for j in jobs) for p in ('train', 'heldout')], [32, 4])
        self.assertTrue(all(j['search_value']['w']==0 for j in jobs))
        self.assertTrue(all(cotton.job_settings(j)==cotton.settings(develop=True) for j in jobs if j['split']=='train'))
        self.assertTrue(all(not cotton.enabled(j) for j in jobs if j['split']=='heldout'))
        from brass_ai.recovery.collector_runtime import cpu_block, from_config, recipe as signed_recipe
        spec = importlib.util.spec_from_file_location('refresh_prepare_cli', ROOT/'scripts/continue-rust-shadow.py')
        script = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(script)
        with mock.patch('brass_ai.rust_backend.lab_shadow.teacher_build', return_value=dict(core_source='core')), \
             mock.patch('brass_ai.rust_backend.lab_cpu_runtime.resolve'), \
             mock.patch('brass_ai.step5.league.write_json') as publish:
            _, path, remaining = script.prepare(ROOT, run, generation, 50, .05, mode='on', revise_stages=True,
                                                skip_utility_qualification=True, adaptive_teacher_beta=True, inference=cpu_block())
        self.assertEqual(remaining, 50)
        self.assertEqual(publish.call_args.args[0], path)
        self.assertEqual(signed_recipe(publish.call_args.args[1]), config)
        self.assertEqual(from_config(publish.call_args.args[1])['native_inference'], cpu_block())
        self.assertEqual(refresh.prepare(ROOT, run, expected_generation=generation), updated)

    def check_buffer_retune(self, *, split_c_scale):
        from brass_ai.recovery import teacher_search_retune as retune, training_revision_g1491 as g1491
        from brass_ai.recovery.collector_runtime import DEFAULTS, cpu_block, scope
        from brass_ai.recovery.resume import load_resume
        from brass_ai.recovery.search_value_rows import ValueBuffer
        from brass_ai.recovery.teacher_buffer import TeacherBuffer
        from brass_ai.recovery.teacher_labels import IDENTITY, LabelRow
        from brass_ai.recovery.teacher_phase import TeacherPhase

        run, _, before, _ = self.refreshed_checkpoint()
        generation = before['iteration']
        if split_c_scale:
            before = retune.prepare(ROOT, run, expected_generation=generation)
        entry = before['history'][-1]
        model, raw = load_checkpoint(ROOT/entry['path'])
        training = raw['training']
        search = training['search_value']
        if split_c_scale:
            mixed = g1491.configure_teacher_games(before['signature']['config'], 0)
            before['signature'] = dict(before['signature'], config=mixed)
            before['native_backend'] = dict(before['native_backend'], config_sha256=digest(mixed))
            training.update(signature=digest(before['signature']), native_backend=before['native_backend'])
            search['stage'].update(stage='C', w=1., scale=1., stage_a_iterations=100, stage_b_iterations=50)
        else:
            search['stage']['stage_a_iterations'] = 26
        reward_id = search['reward_id']

        def publish(buffer, g, split, games):
            for i in range(games):
                game = f'{g}-{split}-{i}'
                rows = [dict(generation=g, split=split, game_id=game, index=j, reward_id=reward_id,
                             terminal_utility=[1.]*4, u=[1.]*4, r=[0.]*4) for j in range(1, 65)]
                buffer.publish(g, split, rows, game_id=game)

        values = ValueBuffer(run/'search-values', reward_id)
        publish(values, generation-1, 'train', 1)
        publish(values, generation-1, 'heldout', 4)
        search['buffer'] = values.payload()
        teacher = training['teacher_phase']
        labels = TeacherBuffer(ROOT, run, teacher['buffer']['identity'])
        label = dict.fromkeys(IDENTITY)
        label.update(generation=generation-1, model_id=entry['model_id'], reward_id=reward_id,
                     c2=.15 if split_c_scale else .1,
                     seat=0, turn_key=[], revision=0, kind='root')
        labels.publish(generation-1, 'train', [LabelRow(dict(teacher=label), None)],
                       model_id=entry['model_id'], reward_id=reward_id)
        teacher['buffer'] = labels.payload()
        teacher['weight']['beta'] = .089
        torch.save(raw, ROOT/entry['path'])
        entry = dict(entry, sha256=sha(ROOT/entry['path']))
        before.update(history=[*before['history'][:-1], entry], resume=[entry], teacher_phase=teacher,
                      search_value=stamp(search))
        write_json(run/'state.json', before)

        after = retune.prepare(ROOT, run, expected_generation=generation, split_c_scale=split_c_scale)
        self.assertEqual(retune.prepare(ROOT, run, expected_generation=generation, split_c_scale=split_c_scale), after)
        self.assertEqual(after['history'][-1]['model_id'], entry['model_id'])
        self.assertEqual(sha(ROOT/entry['path']), entry['sha256'])
        config = after['signature']['config']
        expected_profile = dict(full_k=8, c_scale=.2, cotton_c_scale=.1, fast_c_scale=.5) if split_c_scale else dict(
            full_k=8, c_scale=.15, fast_c_scale=.5)
        self.assertEqual(g1491.count_search(config), expected_profile)
        self.assertEqual(config['teacher_phase']['search_value']['max_epochs'], 1)
        self.assertEqual(config['teacher_phase']['mode'], 'on')
        current, converted, _ = load_resume(ROOT, run, after, after['signature'])
        for key in ('optimizer', 'rng', 'cuda_rng'):
            self.assertTrue(_same_payload(raw[key], converted[key]), key)
        self.assertTrue(_same_payload(raw['state_dict'], converted['state_dict']))
        self.assertTrue(_same_payload(training['actor_value_phases'], converted['training']['actor_value_phases']))
        self.assertTrue(_same_payload(search['companion'], converted['training']['search_value']['companion']))

        block = cpu_block(teacher_budget=dict(cpu_block()['teacher_budget'], **expected_profile))
        with scope(dict(DEFAULTS, native_inference=block)):
            phase = TeacherPhase(ROOT, run, config, after, current, converted)
        self.assertTrue(phase.ready)
        self.assertEqual(phase.buffer.rows(generation), [])
        self.assertEqual(phase.buffer.retired, labels.payload()['receipts'])
        self.assertEqual(phase.search_value.phase.buffer.retired, values.payload()['receipts'])
        term = TeacherTerm(current, config['teacher_phase'], [], seed=1, generation=generation, state=phase.weight)
        self.assertEqual((term.mode, term.beta_used, term.remeasure_pending), ('on', .089, True))
        self.assertEqual(schedule.validation_plan(config, after)['reason'], 'beta-remeasurement')
        jobs = phase.begin(after, current, export_hash=after['history'][-1]['sha256'], epoch=generation)
        self.assertEqual([sum(j['split']==s for j in jobs) for s in ('train', 'heldout')], [32, 4])
        if split_c_scale:
            from brass_ai.recovery.collector_runtime import teacher_job_budget
            for split, tables in (('train', 16), ('heldout', 2)):
                cotton_jobs = [j for j in jobs if j['split'] == split and cotton.enabled(j)]
                self.assertEqual(len(cotton_jobs), tables)
                self.assertTrue(all(cotton.job_settings(j)['probability'] == .25 for j in cotton_jobs))
            for job in jobs:
                g1491.validate_job(job)
                budget = teacher_job_budget(block['teacher_budget'], job)
                self.assertEqual(budget['c_scale'], .10 if cotton.enabled(job) else .20)
                self.assertEqual(budget['fast_c_scale'], .5)
                self.assertEqual(job['search_value']['w'], 0.)
        term.qualification_context = dict(tables=dict(train=32, heldout=4), labels_by_phase={
            p: 1 for p in ('canal_early', 'canal_late', 'rail_early', 'rail_late')})
        term.steps = [dict(raw_share=2., share=.178, actor_norm=1., raw_teacher_norm=2.)]
        measured = term.finish()
        self.assertEqual(measured[beta.REMEASUREMENT]['measured_generation'], generation)
        self.assertEqual(measured[beta.KEY]['beta'], .025)
        self.assertEqual(measured[beta.KEY]['updates'], 0)

        value = phase.search_value.phase
        guarded = {} if split_c_scale else None
        with mock.patch('brass_ai.recovery.search_value_phase.predict_rows', return_value=([[0.]*4]*256, [[0.]*4]*256)), \
             mock.patch.object(value.companion, 'head_only_heldout', return_value=dict(steps=1)) as head_fit, \
             mock.patch('brass_ai.recovery.search_value_training.unconditional_pass', return_value=dict(steps=1)) as joint_fit:
            for g in range(generation, generation+16):
                publish(value.buffer, g, 'train', 1)
                if (g+1)%5 == 0:
                    publish(value.buffer, g, 'heldout', 4)
                result = value.after_ppg(g, mode='on', guarded=guarded)
                if split_c_scale:
                    self.assertEqual((result['stage']['stage'], result['stage']['w'], result['stage']['scale']), ('C', 1., 1.))
                else:
                    self.assertEqual(result['stage']['stage_a_iterations'], 26 + int(g == generation+15))
            fit = joint_fit if split_c_scale else head_fit
            fit.assert_called_once()
            self.assertTrue(fit.call_args.kwargs['one_epoch'])
            self.assertEqual(result['stage']['buffer_rebuild']['ready_generation'], generation+7)
            pools = (fit.call_args.args[1], fit.call_args.kwargs['heldout']) if split_c_scale else fit.call_args.args
            self.assertTrue(all(r['generation'] >= generation for pool in pools for r in pool))

    def test_retune_retires_only_teacher_data_and_resumes_a_after_refill(self):
        self.check_buffer_retune(split_c_scale=False)

    def test_mixed_scale_retune_retires_both_buffers_and_refills_in_stage_c(self):
        self.check_buffer_retune(split_c_scale=True)

    def test_one_epoch_fits_each_row_at_most_once_and_rolls_back_a_worse_holdout(self):
        from brass_ai.recovery.search_value_training import Companion
        from test_b025 import model_fixture
        from test_b045 import value_fixture

        model = model_fixture()
        row, _, _ = value_fixture(model)
        companion = Companion(model)
        rows = [row]*513
        heldout = [dict(row, split='heldout', game_id='heldout')]
        validation = iter([1., .9, 1.1])
        with mock.patch('brass_ai.recovery.search_value_retention.HeldoutLoss', return_value=validation.__next__), \
             mock.patch.object(companion, '_head_steps', wraps=companion._head_steps) as train:
            report = companion.head_only_heldout(rows, heldout, generation=1760, intervals=4, interval_steps=1, one_epoch=True)
        first, last = [call.args[1][0] for call in train.call_args_list]
        self.assertEqual([len(first), len(last)], [512, 1])
        np.testing.assert_array_equal(np.sort(np.r_[first,last]), np.arange(len(rows)))
        self.assertEqual((report['max_epochs'], report['max_steps'], report['steps'], companion.steps), (1, 2, 1, 1))
        self.assertEqual(report['early_stop_reason'], 'heldout_loss_increased')
        self.assertEqual(report['heldout_loss_after'], .9)
        head_only = companion.head_only(rows, generation=1761, steps=64, one_epoch=True)
        self.assertEqual((head_only['steps'], head_only['max_steps']), (2, 2))

    def test_one_epoch_caps_joint_trunk_pass_and_its_learning_rate_schedule(self):
        from brass_ai.learning.search_value import ordinary_parameters
        from brass_ai.recovery.search_value_training import Companion, unconditional_pass
        from test_b025 import model_fixture
        from test_b045 import value_fixture

        model = model_fixture()
        row, _, _ = value_fixture(model)
        companion = Companion(model)
        optimizer = torch.optim.Adam(ordinary_parameters(model), lr=1e-5)
        rate = mock.Mock(return_value=1e-5)
        report = unconditional_pass(companion, [row]*3, optimizer, scale=.1, cloning_loss=lambda:next(model.parameters()).sum()*0,
            measure_joint_kl=lambda:.001, beta=.25, rate_for_step=rate, intervals=4, one_epoch=True)
        self.assertEqual((report['max_epochs'], report['max_steps'], report['steps'], companion.steps), (1, 1, 1, 1))
        rate.assert_called_once_with(1,1)

    def test_holdouts_stay_disjoint_over_the_full_twenty_iteration_window(self):
        from brass_ai.recovery.search_value_phase import SearchValuePhase
        from brass_ai.recovery.search_value_rows import ValueBuffer
        from brass_ai.recovery.search_value_retention import recent_holdout
        from test_utility_width import learned, recipe

        before = recipe()
        config = refresh.configure(dict(before, teacher_phase=dict(before['teacher_phase'], mode='on')))
        model = learned()
        reward_id = digest(model.config.utility)
        with tempfile.TemporaryDirectory(dir=ROOT/'artifacts') as directory:
            run = Path(directory)
            buffer = ValueBuffer(run/'search-values', reward_id)

            def publish(buffer, generation, split, count=4):
                for i in range(count):
                    game = f'{generation}-{split}-{i}'
                    rows = [dict(generation=generation, split=split, game_id=game, index=j, reward_id=reward_id,
                                 terminal_utility=[1.]*4, u=[1.]*4, r=[0.]*4, target=[0.]*4) for j in range(1, 65)]
                    buffer.publish(generation, split, rows, game_id=game)

            publish(buffer, 1719, 'heldout')
            publish(buffer, 1729, 'heldout')
            publish(buffer, 1729, 'train', count=1)
            buffer.rows(1729)
            saved = refresh.refresh_buffer(run, buffer.payload(), 1730)
            self.assertEqual(len(saved['receipts']), 8)
            self.assertTrue(all(r['split']=='heldout' for r in saved['receipts']))
            phase = SearchValuePhase(run/'search-values', model, config['teacher_phase']['search_value'], reward_id,
                                     saved=dict(buffer=saved))
            publish(phase.buffer, 1730, 'heldout')
            publish(phase.buffer, 1730, 'train', count=1)
            with mock.patch('brass_ai.recovery.search_value_phase.predict_rows', return_value=([[0.]*4]*256, [[0.]*4]*256)), \
                 mock.patch.object(phase.companion, 'head_only_heldout', return_value=dict(steps=1)) as fit:
                first = phase.after_ppg(1730, mode='on')
                self.assertEqual(first['retention_holdout']['available_games'], 12)
                self.assertEqual(first['stage']['stage_a_iterations'], 0)
                fit.assert_not_called()
                publish(phase.buffer, 1734, 'heldout')
                publish(phase.buffer, 1734, 'train', count=1)
                trained = phase.after_ppg(1734, mode='on')
                train, heldout = fit.call_args.args
                self.assertEqual(trained['stage']['stage_a_iterations'], 1)
                self.assertEqual(len({r['game_id'] for r in heldout}), 16)
                self.assertTrue({r['game_id'] for r in train}.isdisjoint(r['game_id'] for r in heldout))
                self.assertEqual(train[0]['target_lambda'], .8)
                self.assertAlmostEqual(train[0]['target'][0], .8**63, places=12)
                self.assertTrue(all(r['target']==r['u'] and r['split']=='heldout' for r in heldout))
                publish(phase.buffer, 1739, 'heldout')
                _, report = recent_holdout(phase.buffer.rows(1739, 'heldout'), 1739)
                self.assertEqual(report['generations'], [1729, 1730, 1734, 1739])

    def test_continuation_is_one_on_invocation_with_the_requested_budget(self):
        from test_utility_width import recipe
        spec = importlib.util.spec_from_file_location('search_refresh_cli', ROOT/'scripts/continue-rust-g1491.py')
        script = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(script)
        shared = script.shared()
        config = recipe()
        config = dict(config, teacher_phase=dict(config['teacher_phase'], mode='on'))
        before = dict(iteration=1730, signature=dict(config=config))
        self.assertNotIn(beta.KEY, refresh.configure(config)['teacher_phase'])
        after = dict(before, signature=dict(config=refresh.configure(config, adaptive_beta=True)), teacher_phase=dict(
            weight={beta.REMEASUREMENT: dict(requested_generation=1730)}))
        args = ['continue', '--expected-generation', '1730', '--iterations', '50', '--share-target', '.05',
                '--skip-utility-qualification', '--refresh-search-head', '--adaptive-teacher-beta']
        with mock.patch('sys.argv', args), mock.patch.object(script, 'shared', return_value=shared), \
             mock.patch.object(script, 'state_of', return_value=before), mock.patch.object(script, 'ensure_cpu'), \
             mock.patch.object(refresh, 'prepare', return_value=after), mock.patch.object(shared, 'continuation') as train:
            script.main(share_target_required=True)
        self.assertEqual(train.call_args.kwargs, dict(mode='on', expected=1730, iterations=50, revise_stages=True))
        train.assert_called_once()

        # A normal continuation can supply a new target on an already adaptive run.
        weight = {beta.REMEASUREMENT: dict(requested_generation=1730, measured_generation=1730)}
        resumed = dict(after, iteration=1740, teacher_phase=dict(weight=weight))
        args = ['continue', '--expected-generation', '1740', '--iterations', '10', '--share-target', '.08',
                '--skip-utility-qualification', '--c-scale', '0.1']
        with mock.patch('sys.argv', args), mock.patch.object(script, 'shared', return_value=shared), \
             mock.patch.object(script, 'state_of', return_value=resumed), mock.patch.object(script, 'ensure_cpu'), \
             mock.patch.object(refresh, 'prepare') as migrate, mock.patch.object(shared, 'continuation') as train:
            script.main(share_target_required=True)
        migrate.assert_not_called()
        train.assert_called_once()
        self.assertEqual(train.call_args.kwargs, dict(mode='on', expected=1740, iterations=10, revise_stages=True))
        self.assertEqual(train.call_args.args[0].share_target, .08)

    def test_retuned_search_settings_are_persistent_and_keep_teaching_on(self):
        from brass_ai.recovery import teacher_search_retune as retune
        from test_utility_width import recipe

        spec = importlib.util.spec_from_file_location('teacher_search_retune_cli', ROOT/'scripts/continue-rust-g1491.py')
        script = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(script)
        shared = script.shared()
        config = recipe()
        config = refresh.configure(dict(config, teacher_phase=dict(config['teacher_phase'], mode='on')), adaptive_beta=True)
        before = dict(iteration=1760, signature=dict(config=config))
        after = dict(before, signature=dict(config=retune.configure(config)), teacher_phase=dict(
            weight={beta.REMEASUREMENT: dict(requested_generation=1760, previous_beta=.089)}))
        resumed = dict(after, iteration=1770, teacher_phase=dict(weight={
            beta.REMEASUREMENT: dict(requested_generation=1760, measured_generation=1760)}))
        for g, state, flags in ((1760, before, ['--retune-teacher-search']), (1770, resumed, [])):
            args = ['continue', '--expected-generation', str(g), '--iterations', '10', '--share-target', '.15',
                    '--skip-utility-qualification', *flags]
            with mock.patch('sys.argv', args), mock.patch.object(script, 'shared', return_value=shared), \
                 mock.patch.object(script, 'state_of', return_value=state), mock.patch.object(script, 'ensure_cpu'), \
                 mock.patch.object(retune, 'prepare', return_value=after) as migrate, \
                 mock.patch.object(shared, 'continuation') as train:
                script.main(teacher_fast=96, share_target_required=True)
            self.assertEqual(migrate.call_count, int(bool(flags)))
            self.assertEqual(train.call_args.kwargs, dict(mode='on', expected=g, iterations=10, revise_stages=True))
            self.assertEqual(shared.inference_choice(train.call_args.args[0])['teacher_budget'],
                             dict(mode='count', full=640, full_k=8, fast=96, fast_k=8, c_scale=.15, fast_c_scale=.5))

    def test_split_teacher_scales_resume_from_the_requested_boundary(self):
        from brass_ai.recovery import teacher_search_retune as retune, training_revision_g1491 as revision
        from test_utility_width import recipe

        spec = importlib.util.spec_from_file_location('teacher_split_cli', ROOT/'scripts/continue-rust-g1491.py')
        script = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(script)
        shared = script.shared()
        config = recipe()
        config = refresh.configure(dict(config, teacher_phase=dict(config['teacher_phase'], mode='on')), adaptive_beta=True)
        config = revision.configure_teacher_games(retune.configure(config), 0)
        after = retune.configure(config, split_c_scale=True)
        for generation, current, flags in ((2020, config, ['--split-teacher-c-scale']), (2080, after, [])):
            state = dict(iteration=generation, signature=dict(config=current))
            migrated = dict(state, signature=dict(config=after))
            argv = ['continue', '--expected-generation', str(generation), '--iterations', '60', '--share-target', '.30',
                    '--skip-utility-qualification', '--teacher-search-head-weight', '0', *flags]
            with mock.patch('sys.argv', argv), mock.patch.object(script, 'shared', return_value=shared), \
                 mock.patch.object(script, 'state_of', return_value=state), mock.patch.object(script, 'ensure_cpu'), \
                 mock.patch.object(retune, 'prepare', return_value=migrated) as migrate, \
                 mock.patch.object(shared, 'continuation') as train:
                script.main(teacher_fast=96, share_target_required=True)
            if flags:
                self.assertEqual(migrate.call_args.kwargs, dict(expected_generation=generation, split_c_scale=True))
            else:
                migrate.assert_not_called()
            train.assert_called_once()
            self.assertEqual(train.call_args.kwargs, dict(mode='on', expected=generation, iterations=60, revise_stages=True))
            self.assertEqual(shared.inference_choice(train.call_args.args[0])['teacher_budget'],
                             dict(mode='count', full=640, full_k=8, fast=96, fast_k=8,
                                  c_scale=.2, cotton_c_scale=.1, fast_c_scale=.5))
