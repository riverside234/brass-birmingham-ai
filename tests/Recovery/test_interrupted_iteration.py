"""Recovery archives only uncommitted evidence and preserves the resume point."""
import copy
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


if __name__ == '__main__':
    unittest.main()
