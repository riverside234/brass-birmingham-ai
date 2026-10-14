"""Recovery archives only uncommitted evidence and preserves the resume point."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from brass_ai.bank import read, sha
from brass_ai.learning.contracts import SOURCE_ROOT
from brass_ai.recovery.interrupted import archive_uncommitted_iteration
from brass_ai.recovery.retention import run_lock
from brass_ai.step5.league import implementation_id, write_json
import test_audit_resume as audit_fixture
import test_teacher_work_resume as resume_fixture


class InterruptedIterationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        fixture = resume_fixture.TeacherWorkResumeTests()
        self.addCleanup(fixture.doCleanups)
        def signatures():
            _, new = audit_fixture.AuditResumeTests().signatures()
            return copy.deepcopy(new), new
        fixture.signatures = signatures
        self.run, self.state, _, _, _ = fixture.fixture()
        self.pending = self.run/'iteration-0070'
        self.pending.mkdir()
        (self.pending/'ordinary.jsonl.gz').write_bytes(b'preserved ordinary replay')
        (self.pending/'worker').mkdir()
        (self.pending/'worker/partial.tmp').write_bytes(b'preserved unfinished teacher output')

    def recover(self, **kwargs):
        return archive_uncommitted_iteration(SOURCE_ROOT, self.run, expected_iteration=70, **kwargs)

    def test_dry_run_preserves_files_state_and_checkpoint(self):
        before = {p: sha(p) for p in self.run.rglob('*') if p.is_file()}
        result = self.recover()
        self.assertEqual(result['status'], 'ready-to-archive')
        self.assertEqual(result['iteration'], 70)
        self.assertEqual(result['games_done'], 22400)
        self.assertEqual({p: sha(p) for p in self.run.rglob('*') if p.is_file()}, before)
        self.assertFalse((self.run/'.interrupted-iterations').exists())

    def test_apply_preserves_bytes_and_can_be_retried(self):
        state_bytes = (self.run/'state.json').read_bytes()
        checkpoint_bytes = (self.run/'generation-0070.pt').read_bytes()
        result = self.recover(apply=True)
        archive = self.run/result['archive']
        self.assertEqual(result['status'], 'archived')
        self.assertFalse(self.pending.exists())
        self.assertEqual((archive/'state-before.json').read_bytes(), state_bytes)
        for entry in result['files']:
            self.assertEqual(sha(archive/self.pending.name/entry['path']), entry['sha256'])
        self.assertEqual(read(archive/'manifest.json'), result)
        self.assertEqual((self.run/'state.json').read_bytes(), state_bytes)
        self.assertEqual((self.run/'generation-0070.pt').read_bytes(), checkpoint_bytes)
        self.assertEqual(self.recover(apply=True)['status'], 'nothing-to-archive')

    def test_expected_iteration_and_active_run_guard(self):
        with self.assertRaisesRegex(ValueError, 'Committed iteration differs'):
            archive_uncommitted_iteration(SOURCE_ROOT, self.run, expected_iteration=69, apply=True)
        with run_lock(self.run), self.assertRaisesRegex(ValueError, 'active'):
            self.recover(apply=True)
        self.assertTrue(self.pending.is_dir())

    def test_uncommitted_checkpoint_or_report_requires_separate_commit_recovery(self):
        for name in ('generation-0071.pt', 'generation-0071.pt.tmp', 'iteration-0070/report.json'):
            path = self.run/name
            with self.subTest(name=name):
                path.write_bytes(b'partial commit')
                try:
                    with self.assertRaisesRegex(ValueError, 'checkpoint exists|report exists'):
                        self.recover(apply=True)
                    self.assertTrue(self.pending.is_dir())
                    self.assertFalse((self.run/'.interrupted-iterations').exists())
                finally:
                    path.unlink()

    def test_changed_checkpoint_and_unknown_source_fail_before_archiving(self):
        checkpoint = self.run/'generation-0070.pt'
        original = checkpoint.read_bytes()
        checkpoint.write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'Snapshot changed'):
            self.recover(apply=True)
        checkpoint.write_bytes(original)
        self.state['signature']['implementation'] = '0'*64
        write_json(self.run/'state.json', self.state)
        with self.assertRaisesRegex(ValueError, 'not been reviewed'):
            self.recover(apply=True)
        self.assertTrue(self.pending.is_dir())

    def test_linked_partial_paths_fail_before_archiving(self):
        from brass_ai.recovery import retention
        original = retention.link
        with patch.object(retention, 'link', side_effect=lambda p: p.name == 'worker' or original(p)), \
             self.assertRaisesRegex(ValueError, 'Linked retention target'):
            self.recover(apply=True)
        self.assertTrue(self.pending.is_dir())

    def test_rollout_previous_checkpoint_outside_population_is_checked_before_archive(self):
        from brass_ai.recovery.champion_mix import promoted
        previous = dict(self.state['champion'])
        checkpoint = self.run/'previous-champion.pt'
        original = (self.run/'generation-0070.pt').read_bytes()
        checkpoint.write_bytes(original)
        previous.update(path=checkpoint.relative_to(SOURCE_ROOT).as_posix(), sha256=sha(checkpoint))
        promoted(self.state, previous)
        write_json(self.run/'state.json', self.state)
        before = (self.run/'state.json').read_bytes()
        for content in (None, b'changed required opponent'):
            with self.subTest(content=content):
                if content is None: checkpoint.unlink()
                else: checkpoint.write_bytes(content)
                with self.assertRaises((FileNotFoundError, ValueError)):
                    self.recover(apply=True)
                self.assertTrue(self.pending.is_dir())
                self.assertFalse((self.run/'.interrupted-iterations').exists())
                self.assertEqual((self.run/'state.json').read_bytes(), before)
        checkpoint.write_bytes(original)
        self.assertEqual(self.recover(apply=True)['status'], 'archived')

    def test_failed_rename_preserves_both_original_evidence_and_recovery_manifest(self):
        from pathlib import Path
        with patch.object(Path, 'rename', side_effect=OSError('injected move failure')), \
             self.assertRaisesRegex(OSError, 'injected move failure'):
            self.recover(apply=True)
        self.assertTrue((self.pending/'ordinary.jsonl.gz').is_file())
        archives = list((self.run/'.interrupted-iterations').iterdir())
        self.assertEqual(len(archives), 1)
        self.assertEqual(read(archives[0]/'manifest.json')['status'], 'moving')
        self.assertEqual(self.recover(apply=True)['status'], 'archived')

    def test_reviewed_upgrade_can_resume_after_archive_with_exact_optimizer_and_rng(self):
        from brass_ai.learning.checkpoint import save_checkpoint, resume_optimizer
        from brass_ai.learning.contracts import digest
        from brass_ai.recovery.resume import load_resume, commit_transition
        previous = '56f229de63ee148b7bd9e41245ae4d0b8953d59d85c16440d7b4e00c6d52f4ac'
        # Create an actual older-source checkpoint, then keep its full training
        # state while retiring only the collection after it.
        model, raw, _ = load_resume(SOURCE_ROOT, self.run, self.state, self.state['signature'])
        optimizer = torch.optim.AdamW(model.parameters())
        resume_optimizer(raw, optimizer)
        old = copy.deepcopy(self.state['signature']); old['implementation'] = previous
        path = self.run/'generation-0070.pt'
        save_checkpoint(path, model, optimizer, step=70, training=dict(raw['training'], signature=digest(old)))
        self.state['signature'] = old
        for entry in [*self.state['history'], *self.state['population'], *self.state['resume'],
                      self.state['evaluation_anchor'], self.state['champion']]:
            entry['sha256'] = sha(path)
        write_json(self.run/'state.json', self.state)
        before = path.read_bytes()
        target = dict(old, implementation=implementation_id())
        with self.assertRaisesRegex(ValueError, 'Uncommitted iteration'):
            load_resume(SOURCE_ROOT, self.run, self.state, target, reviewed=True)
        self.recover(apply=True)
        restored_model, checkpoint, changed = load_resume(SOURCE_ROOT, self.run, self.state, target, reviewed=True)
        self.assertTrue(changed)
        restored_optimizer = torch.optim.AdamW(restored_model.parameters())
        resume_optimizer(checkpoint, restored_optimizer)
        self.assertTrue(torch.equal(torch.get_rng_state(), checkpoint['rng']))
        for key, values in optimizer.state_dict()['state'].items():
            for name, value in values.items():
                torch.testing.assert_close(value, restored_optimizer.state_dict()['state'][key][name], rtol=0, atol=0)
        updated = commit_transition(self.run, self.state, target)
        for key in ('iteration', 'games_done', 'population', 'champion', 'next_search_games'):
            self.assertEqual(updated[key], self.state[key])
        self.assertEqual(path.read_bytes(), before)
        self.assertFalse(load_resume(SOURCE_ROOT, self.run, updated, target)[2])


class TeacherCrashResumeTests(unittest.TestCase):
    def test_retry160_after_cap_migration_keeps_raw_checkpoint_and_rollout(self):
        from brass_ai.learning.batching import model_id
        from brass_ai.learning.checkpoint import save_checkpoint, resume_optimizer
        from brass_ai.learning.contracts import compatibility, digest
        from brass_ai.learning.model import ModelConfig
        from brass_ai.learning.utility_model import UtilityNet, utility_config
        from brass_ai.recovery import champion_mix
        from brass_ai.recovery.retention import _same_payload
        from brass_ai.recovery.resume import load_resume, commit_transition
        from brass_ai.recovery.utility import reward_spec, migrate, calibration_ready
        torch.set_num_threads(2)
        cfg = read(SOURCE_ROOT/'configs/step5/joint7-g150-gate72.json'); cfg['model_size'] = 'small'
        oldcfg = copy.deepcopy(cfg)
        oldcfg['insolvency']['max_cost'] = oldcfg['search']['reward']['insolvency']['max_cost'] = .025
        old = dict(config=oldcfg, compatibility=compatibility(), legacy_pool={}, teachers={},
            implementation='56f229de63ee148b7bd9e41245ae4d0b8953d59d85c16440d7b4e00c6d52f4ac')
        failed_source = dict(old, config=cfg, implementation='699bab353343458523eda073f07172ec4e70f679e4e1d1595c7d052a2d78e61c')
        target = dict(failed_source, implementation=implementation_id())
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as tmp:
            run = Path(tmp)
            model = UtilityNet(utility_config(ModelConfig.preset('small'), reward_spec(oldcfg)))
            optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['learning_rate'])
            sum(p.square().mean() for p in model.parameters()).backward(); optimizer.step(); optimizer.zero_grad()
            path = run/'generation-0160.pt'
            identity = save_checkpoint(path, model, optimizer, step=160,
                training=dict(signature=digest(old), fresh_run=run.relative_to(SOURCE_ROOT).as_posix()))
            entry = dict(path=path.relative_to(SOURCE_ROOT).as_posix(), model_id=identity, sha256=sha(path), generation=160)
            state = dict(format='step5-recovery-league/2.2.0', iteration=160, games_done=51200, signature=old,
                history=[entry], population=[entry], evaluation_anchor=entry, champion=entry, next_search_games=51360,
                search_seed_cursor=1234, next_seed=18575575, context_counter=51200, ordinary_seconds=12345., search_seconds=54321.)
            write_json(run/'state.json', state)
            # The failed attempt adopted new source/reward but never committed
            # a generation: its on-disk checkpoint is still the original .025.
            first, raw, changed = load_resume(SOURCE_ROOT, run, state, failed_source, reviewed=True)
            self.assertTrue(changed)
            restored_optimizer = torch.optim.AdamW(first.parameters()); resume_optimizer(raw, restored_optimizer)
            migrate(first, restored_optimizer, reward_spec(cfg))
            state = commit_transition(run, state, failed_source)
            champion_mix.initialize(SOURCE_ROOT, run, state)
            write_json(run/'state.json', state)
            pending = run/'iteration-0160'; pending.mkdir()
            (pending/'ordinary.jsonl.gz').write_bytes(b'partial iteration evidence')
            state_bytes = (run/'state.json').read_bytes(); checkpoint_bytes = path.read_bytes()
            with self.assertRaisesRegex(ValueError, 'Uncommitted iteration'):
                load_resume(SOURCE_ROOT, run, state, target, reviewed=True)
            recovery = archive_uncommitted_iteration(SOURCE_ROOT, run, expected_iteration=160, apply=True)
            self.assertEqual((run/'state.json').read_bytes(), state_bytes)
            self.assertEqual((run/recovery['archive']/'iteration-0160/ordinary.jsonl.gz').read_bytes(), b'partial iteration evidence')
            learner, checkpoint, changed = load_resume(SOURCE_ROOT, run, state, target, reviewed=True)
            self.assertTrue(changed); self.assertEqual(model_id(learner), identity)
            resumed = torch.optim.AdamW(learner.parameters()); resume_optimizer(checkpoint, resumed)
            rng = torch.get_rng_state().clone()
            migrate(learner, resumed, reward_spec(cfg))
            self.assertEqual(learner.config.utility['insolvency']['max_cost'], .075)
            self.assertFalse(calibration_ready(learner))
            self.assertTrue(_same_payload(model.state_dict(), learner.state_dict()))
            self.assertTrue(_same_payload(optimizer.state_dict(), resumed.state_dict()))
            self.assertTrue(torch.equal(rng, checkpoint['rng']))
            self.assertTrue(torch.equal(rng, torch.get_rng_state()))
            updated = commit_transition(run, state, target)
            for key in state:
                if key not in ('signature', 'source_transitions'):
                    self.assertEqual(updated[key], state[key], key)
            self.assertEqual(path.read_bytes(), checkpoint_bytes)
            self.assertFalse(load_resume(SOURCE_ROOT, run, updated, target)[2])


if __name__ == '__main__':
    unittest.main()
