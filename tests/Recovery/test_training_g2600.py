"""g2600 recipe: T1.3 openings, long C_scale .20, PPG floor .25 / one epoch, and the PPO staging fixes."""
import copy
import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

from brass_ai.learning.contracts import SOURCE_ROOT as ROOT
from brass_ai.recovery import teacher_search_retune as retune, training_revision_g1491 as R
from brass_ai.recovery import training_revision_g2090 as O, training_revision_g2400 as N, training_revision_g2600 as G


def g2400(weight=0):
    from test_training_g2400 import recipe
    return R.configure_teacher_games(N.configure(recipe()), weight)


class Recipe(unittest.TestCase):
    def test_values_validation_and_idempotence(self):
        from brass_ai.recovery.config import validate_config
        from brass_ai.recovery.exploration import THREE_OPENING
        from brass_ai.recovery.opening_objective import FORK_T15
        from brass_ai.rust_backend.lab_teacher_activation import is_on
        before = g2400()
        after = retune.configure(before, g2600=True)
        validate_config(after)
        self.assertTrue(is_on(after) and G.applied(after) and N.applied(after) and O.applied(after))
        self.assertFalse(G.applied(before))
        self.assertIs(G.configure(after), after)
        self.assertIs(O.configure(after), after)
        self.assertEqual((after['opening'], after['update']['opening_objective']), (THREE_OPENING, FORK_T15))
        self.assertEqual(R.count_search(after), dict(full_k=8, c_scale=.2, fast_c_scale=.7))
        self.assertEqual((after['value_phase']['max_epochs'], after['value_phase']['beta_clone_min']), (1, .25))
        search = after['teacher_phase']['search_value']
        self.assertEqual((search['schedule']['ppg_beta_floor_after_stage_b'], search['max_epochs']), (.25, 1))
        self.assertEqual(after['teacher_phase'][R.KEY]['exploration']['game_opening'], dict(temperature=1.3, actions_per_seat=3))
        # Only PPO opening, PPG floor/epochs, the teacher opening and long C_scale change.
        changed = {k for k in set(before) | set(after) if before.get(k) != after.get(k)}
        self.assertEqual(changed, {'opening', 'update', 'value_phase', 'teacher_phase'})
        self.assertEqual({k for k in before['update'] if before['update'][k] != after['update'][k]}, {'opening_objective'})
        with self.assertRaisesRegex(ValueError, 'g2400 cotton-network'):
            from test_training_g2400 import recipe
            G.configure(recipe())

    def test_w1_teacher_jobs_use_t13_and_the_new_scales(self):
        from brass_ai.recovery.collector_runtime import teacher_job_budget
        from brass_ai.recovery.config import validate_config
        from brass_ai.recovery.teacher_phase import plan_jobs
        from brass_ai.recovery import teacher_cotton as C
        config = R.configure_teacher_games(retune.configure(g2400(), g2600=True), 1)
        validate_config(config)
        self.assertTrue(G.applied(config))
        self.assertEqual(R.teacher_search_head_weight(config), 1)
        budget = dict(mode='count', full=640, fast=96, fast_k=8, **R.G2600_SEARCH)
        cotton = 0
        for generation in (2600, 2604):
            state = dict(iteration=generation, signature=dict(config=config), search_value=dict(stage='C', w=1.))
            jobs, _ = plan_jobs(config, state, model_id='m', export_hash='e', epoch=generation, ready=True)
            self.assertEqual({j['split'] for j in jobs}, {'train', 'heldout'} if generation == 2604 else {'train'})
            for job in jobs:
                R.validate_job(job)
                self.assertEqual(job['search_value']['w'], 1.)
                self.assertEqual(job['teacher_revision']['exploration']['game_opening'], G.GAME_OPENING)
                self.assertEqual(teacher_job_budget(budget, job), budget)
                cotton += C.enabled(job)
        self.assertGreater(cotton, 0)

    def test_switch_needs_the_migration_but_the_blend_can_change_after_it(self):
        from brass_ai.rust_backend.lab_replay_compat import QUALIFIED
        from brass_ai.rust_backend.lab_shadow_config import signature_transition
        def sig(config):return dict(config=config, implementation=QUALIFIED)
        before = g2400()
        zero = retune.configure(before, g2600=True)
        one = R.configure_teacher_games(zero, 1)
        self.assertFalse(signature_transition(sig(before), sig(zero)))
        self.assertFalse(signature_transition(sig(zero), sig(before)))
        self.assertTrue(signature_transition(sig(zero), sig(one)))
        self.assertTrue(signature_transition(sig(one), sig(zero)))


class Labels(unittest.TestCase):
    def test_t13_cotton_probability_and_job_temperature(self):
        from test_teacher_cotton import Labels as Fixture
        from brass_ai.recovery import teacher_cotton as C, teacher_labels as L
        label, choices = Fixture.example(Fixture())
        weights = [90 ** (1/1.3), 10 ** (1/1.3)]
        label.update(sampling_temperature=1.3, phase='canal_early', seat=0)
        label['cotton_exploration'].update(contract=C.SEATED_CONTRACT, probability=.7*weights[1]/sum(weights)+.3)
        L.validate_improved_policy(label, choices)
        job = dict(split='train', reservations={'0': {'canal_early': 2}}, search_value=None,
                   teacher_revision=dict(cotton_seats=[0], exploration=dict(game_opening=R.T13_GAME_OPENING,
                                         cotton=C.settings(contract=C.SEATED_CONTRACT))))
        L.check_turn(label, 2, job)
        t2 = copy.deepcopy(job); t2['teacher_revision']['exploration']['game_opening'] = R.GAME_OPENING
        with self.assertRaisesRegex(ValueError, 'opening temperature'):
            L.check_turn(label, 2, t2)
        stale = copy.deepcopy(label); stale['cotton_exploration']['probability'] = .7*.25+.3
        with self.assertRaisesRegex(ValueError, 'behavior probability'):
            L.validate_improved_policy(stale, choices)
        other = copy.deepcopy(label); other['sampling_temperature'] = 1.5
        with self.assertRaisesRegex(ValueError, 'sampling temperature'):
            L.validate_improved_policy(other, choices)

    def test_t13_jobs_need_single_seat_sampling(self):
        from brass_ai.recovery.teacher_phase import plan_jobs
        config = retune.configure(g2400(), g2600=True)
        state = dict(iteration=2600, signature=dict(config=config), search_value=dict(stage='C', w=1.))
        job = next(j for j in plan_jobs(config, state, model_id='m', export_hash='e', epoch=2600, ready=True)[0]
                   if not j['teacher_revision']['cotton_seats'])
        R.validate_job(job)
        seated = copy.deepcopy(job); seated['teacher_revision']['exploration'].pop('normal_sampling')
        with self.assertRaisesRegex(ValueError, 'Unreviewed g1491 teacher job revision'):
            R.validate_job(seated)


class Staging(unittest.TestCase):
    def test_sealed_rows_skip_the_main_thread_audit(self):
        from test_throughput_g470 import probe_fixture
        from brass_ai.recovery import ppo_preparation as P
        from brass_ai.recovery.replay_binding import checked_encoded, seal_encoded
        from brass_ai.recovery.retention import _same_payload
        _, data, _, _, _ = probe_fixture((3, 1, 2, 3), microbatch=2)
        parts = [[3, 1], [0, 2]]
        expected = list(data.policy_batches(parts, .08)) + list(data.policy_batches(parts, 0, value_only=True))
        for sealed in (False, True):
            if sealed:
                data.encoded = [seal_encoded(e) for e in data.encoded]
                self.assertTrue(all(checked_encoded(e) for e in data.encoded))
            with tempfile.TemporaryDirectory(dir=ROOT/'artifacts') as folder, P.PreparationPool(4) as pool:
                with mock.patch.object(P, 'transport_directory', return_value=(Path(folder), 'ram')), \
                     P.Preparation(pool, directory=folder) as prep:
                    prep.input_audit = mock.Mock(wraps=P.InputAudit())
                    actual = list(data.policy_batches(parts, .08, prep)) + \
                             list(data.policy_batches(parts, 0, prep, value_only=True))
                    self.assertTrue(_same_payload(actual, expected))
                    # Unsealed rows are admitted once each (4 full, 4 value-only); sealed rows not at all.
                    self.assertEqual(prep.input_audit.check.call_count, 0 if sealed else 8)

    def test_wide_teacher_steps_use_one_forward(self):
        from test_b025 import config
        from types import SimpleNamespace
        from brass_ai.recovery import teacher_term as T
        from brass_ai.recovery.teacher_labels import LabelRow
        from brass_ai.recovery.value_preservation import TrunkGradients
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__();self.state_encoder=torch.nn.Linear(2,1,bias=False)
                with torch.no_grad():self.state_encoder.weight.zero_()
        model=Model();rows=[];width=512
        for i in range(96):
            x=np.stack((np.linspace(-1,1,width),np.sin(np.arange(width)+i)),axis=-1).astype(np.float32)
            target=np.exp(x[:,0]+x[:,1]*.2);target/=target.sum()
            label=dict(target=target.tolist(),logits=[0.]*width,legal_ids=list(map(str,range(width))),phase='canal_early')
            rows.append(LabelRow(dict(teacher=label),SimpleNamespace(choice_ids=label['legal_ids'],features=x)))
        options=config()['teacher_phase'];options=dict(options,mode='on',term=dict(options['term'],rows_per_step=96))
        cells=[]
        def forward(net,part,**kwargs):
            cells.append(len(part)*width)
            return net.state_encoder(torch.as_tensor(np.stack([r.encoded.features for r in part]))).squeeze(-1)
        with mock.patch.object(T,'forward_policy',side_effect=forward):
            term=T.TeacherTerm(model,options,[(0,r) for r in rows],seed=4,generation=2600,state=dict(beta=1.,calibrated=True))
            logits=forward(model,rows);target=T.target_tensor(rows,logits)
            loss=(torch.special.xlogy(target,target)-target*logits.log_softmax(-1)).sum(-1).mean()
            expected=torch.autograd.grad(loss,tuple(model.parameters()))
            cells.clear();term.apply(model,TrunkGradients(model),1)
        self.assertEqual(cells,[96*width])
        self.assertAlmostEqual(term.steps[0]['loss'],float(loss.detach()),places=6)
        for p,want in zip(model.parameters(),expected):torch.testing.assert_close(p.grad,want,atol=1e-6,rtol=1e-5)


class Command(unittest.TestCase):
    def test_cli_migrates_once_and_keeps_w1_and_the_new_budget(self):
        spec = importlib.util.spec_from_file_location('g2600_cli', ROOT/'scripts/continue-rust-g1491.py')
        script = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(script)
        shared = script.shared()
        before = g2400()
        migrated = retune.configure(before, g2600=True)
        for generation, current, flags in ((2600, before, ['--update-g2600-recipe']), (2610, migrated, [])):
            state = dict(iteration=generation, signature=dict(config=current))
            after = dict(state, signature=dict(config=migrated))
            argv = ['continue', '--run', 'artifacts/step5/joint7-v11-2-g1190-restart-3', '--expected-generation', str(generation),
                    '--iterations', '120', '--share-target', '0.15', '--skip-utility-qualification',
                    '--teacher-search-head-weight', '1', *flags]
            with mock.patch('sys.argv', argv), mock.patch.object(script, 'shared', return_value=shared), \
                 mock.patch.object(script, 'state_of', return_value=state), mock.patch.object(script, 'ensure_cpu'), \
                 mock.patch.object(retune, 'prepare', return_value=after) as migrate, \
                 mock.patch.object(shared, 'continuation') as train:
                script.main(teacher_fast=96, share_target_required=True)
            if flags:
                self.assertEqual(migrate.call_args.kwargs, dict(expected_generation=generation, g2600=True))
            else:
                migrate.assert_not_called()
            train.assert_called_once()
            args = train.call_args.args[0]
            self.assertEqual(args.teacher_search_head_weight, 1)
            self.assertEqual(train.call_args.kwargs, dict(mode='on', expected=generation, iterations=120, revise_stages=True))
            self.assertEqual(shared.inference_choice(args)['teacher_budget'],
                             dict(mode='count', full=640, full_k=8, fast=96, fast_k=8, c_scale=.2, fast_c_scale=.7))


if __name__ == '__main__':
    unittest.main()
