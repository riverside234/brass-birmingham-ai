"""Tiny Git clones, exact backup/retry and protected-data restore regressions."""
import hashlib
import shutil
import subprocess
import unittest
from unittest.mock import patch

import test_step5_transfer as fixtures

transfer=fixtures.transfer


class CloneRestore(unittest.TestCase):
    def setUp(self):
        self.case=fixtures.TransferTests();self.case.setUp();self.addCleanup(self.case.doCleanups)
        self.root,self.area,self.run=self.case.root,self.case.area,self.case.run
        self.config='configs/step5/current.json'
        self.case.put(self.config,dict(value='Git clone default'))
        self.git('init','-q',str(self.root))
        self.git('-C',str(self.root),'config','core.autocrlf','false')
        for name in ('src','scripts','configs','training','lab-native','requirements-step5.txt'):
            self.git('-C',str(self.root),'add','--',name)
        self.git('-C',str(self.root),'-c','user.name=Fixture','-c','user.email=fixture@example.invalid','commit','-qm','Authored transfer fixture')
        self.dest=self.area/'clone'

    def git(self,*args):
        return subprocess.check_output(['git',*args],stderr=subprocess.PIPE)

    def clone(self):
        # Native Git checkout from an independent copy of the tiny fixture's
        # object database. Windows sandbox disallows the MSYS transport helper
        # used by local git clone; object reads/checkouts exercise real Git.
        self.dest.mkdir()
        shutil.copytree(self.root/'.git',self.dest/'.git')
        self.git('-C',str(self.dest),'checkout-index','-a')

    def export(self,*,dirty=True):
        if dirty:self.case.put(self.config,dict(value='Exact lab launch config'))
        return transfer.export(self.root,self.run,self.case.bundle,archive_format='zip',part_mb=0)

    def test_cloned_default_config_is_backed_up_then_restored_exactly_and_retry_verifies(self):
        self.export();self.clone();before=(self.dest/self.config).read_bytes()
        result=transfer.restore(self.dest,self.case.bundle,relocate=True)
        self.assertTrue(result['verified']);self.assertEqual((self.dest/self.config).read_bytes(),(self.root/self.config).read_bytes())
        backups=result['repository_reconciliation']['backups'];entry=next(e for e in backups if e['path']==self.config)
        self.assertEqual((self.dest/entry['backup']).read_bytes(),before)
        self.assertEqual(entry['sha256'],hashlib.sha256(before).hexdigest())
        self.assertTrue(transfer.restore(self.dest,self.case.bundle,verify_only=True)['verified'])
        self.assertEqual(transfer.restore(self.dest,self.case.bundle)['repository_reconciliation'],result['repository_reconciliation'])

    def test_external_git_bytes_and_crlf_conflicts_are_recovered_without_reexport(self):
        self.export(dirty=False);self.clone()
        script=self.dest/'scripts/step5.py';original=(self.root/'scripts/step5.py').read_bytes()
        script.write_bytes(original.replace(b'\n',b'\r\n'))
        result=transfer.restore(self.dest,self.case.bundle,relocate=True)
        self.assertEqual(script.read_bytes(),original)
        entry=next(e for e in result['repository_reconciliation']['backups'] if e['path']=='scripts/step5.py')
        self.assertEqual((self.dest/entry['backup']).read_bytes(),original.replace(b'\n',b'\r\n'))
        self.assertTrue(transfer.restore(self.dest,self.case.bundle,verify_only=True)['verified'])

    def test_existing_identical_clone_files_are_reused_without_backups(self):
        self.export(dirty=False);self.clone()
        result=transfer.restore(self.dest,self.case.bundle,relocate=True)
        self.assertTrue(result['verified']);self.assertEqual(result['repository_reconciliation']['backups'],[])

    def test_private_checkpoint_and_state_conflicts_are_reported_together_before_any_mutation(self):
        self.export();self.clone()
        original=(self.dest/self.config).read_bytes();private=self.dest/'artifacts/step5/selfplay';private.mkdir(parents=True)
        (private/'state.json').write_bytes(b'newer destination state');(private/'generation-0002.pt').write_bytes(b'newer checkpoint')
        with self.assertRaisesRegex(ValueError,'2 blocking') as error:transfer.restore(self.dest,self.case.bundle,relocate=True)
        self.assertIn('state.json',str(error.exception));self.assertIn('generation-0002.pt',str(error.exception))
        self.assertEqual((self.dest/self.config).read_bytes(),original)
        self.assertEqual((private/'generation-0002.pt').read_bytes(),b'newer checkpoint')
        self.assertFalse((self.dest/'.step5-relocation.json').exists())
        self.assertFalse(list((self.dest/'artifacts').glob('step5-transfer/restore-*')))

    def test_strict_mode_preserves_previous_refusal_and_untracked_files_remain_protected(self):
        self.case.put('scripts/untracked.py',b'archived untracked helper')
        self.export();self.clone();before=(self.dest/self.config).read_bytes()
        with self.assertRaisesRegex(ValueError,'Conflicting destination'):
            transfer.restore(self.dest,self.case.bundle,relocate=True,strict_repo_files=True)
        self.assertEqual((self.dest/self.config).read_bytes(),before)
        file=self.dest/'scripts/untracked.py';file.write_bytes(b'destination work')
        with self.assertRaisesRegex(ValueError,'private/untracked file: scripts/untracked.py'):
            transfer.restore(self.dest,self.case.bundle,relocate=True)
        self.assertEqual(file.read_bytes(),b'destination work');self.assertEqual((self.dest/self.config).read_bytes(),before)

    def test_interrupted_replace_recovers_from_immutable_backup_without_overwriting_new_edits(self):
        from step5_transfer_restore import RestorePlan
        self.export();self.clone();before=(self.dest/self.config).read_bytes();install=RestorePlan.install
        def fail(plan,record,temporary):
            if record['path']==self.config:raise RuntimeError('Authored interruption before replace')
            return install(plan,record,temporary)
        with patch.object(RestorePlan,'install',fail),self.assertRaisesRegex(RuntimeError,'Authored interruption'):
            transfer.restore(self.dest,self.case.bundle,relocate=True)
        self.assertEqual((self.dest/self.config).read_bytes(),before)
        result=transfer.restore(self.dest,self.case.bundle,relocate=True)
        entry=next(e for e in result['repository_reconciliation']['backups'] if e['path']==self.config)
        self.assertEqual((self.dest/entry['backup']).read_bytes(),before)
        (self.dest/self.config).write_bytes(b'new user edit after restore')
        with self.assertRaisesRegex(ValueError,'changed after its restore backup'):transfer.restore(self.dest,self.case.bundle)
        self.assertEqual((self.dest/self.config).read_bytes(),b'new user edit after restore')

    def test_missing_external_file_can_come_only_from_exact_committed_git_object(self):
        self.export(dirty=False);self.clone();file=self.dest/'scripts/step5.py';file.unlink()
        self.assertTrue(transfer.restore(self.dest,self.case.bundle,relocate=True)['verified'])
        self.assertEqual(file.read_bytes(),(self.root/'scripts/step5.py').read_bytes())

    def test_new_transport_tool_is_kept_and_archived_version_remains_hash_verified(self):
        tool='scripts/step5-transfer.py';self.case.put(tool,b'# original archived transfer tool\n')
        self.git('-C',str(self.root),'add','--',tool)
        self.git('-C',str(self.root),'-c','user.name=Fixture','-c','user.email=fixture@example.invalid','commit','-qm','Original transport fixture')
        self.export(dirty=False);self.clone();new=b'# current fixed transfer tool\n';(self.dest/tool).write_bytes(new)
        result=transfer.restore(self.dest,self.case.bundle,relocate=True)
        self.assertEqual((self.dest/tool).read_bytes(),new)
        entry=next(e for e in result['repository_reconciliation']['preserved_transport'] if e['path']==tool)
        self.assertEqual((self.dest/entry['snapshot']).read_bytes(),(self.root/tool).read_bytes())
        self.assertTrue(transfer.restore(self.dest,self.case.bundle,verify_only=True)['verified'])
        (self.dest/entry['snapshot']).write_bytes(b'corrupt archived tool')
        with self.assertRaisesRegex(ValueError,'Conflicting destination'):transfer.restore(self.dest,self.case.bundle,verify_only=True)

    def test_backup_corruption_cannot_be_silently_accepted_on_retry(self):
        self.export();self.clone();result=transfer.restore(self.dest,self.case.bundle,relocate=True)
        entry=next(e for e in result['repository_reconciliation']['backups'] if e['path']==self.config)
        (self.dest/entry['backup']).write_bytes(b'corrupt user backup')
        with self.assertRaisesRegex(ValueError,'backup missing/changed'):
            transfer.restore(self.dest,self.case.bundle,verify_only=True)

    def test_backup_failure_precedes_every_repository_replacement_and_retry_recovers(self):
        from step5_transfer_restore import RestorePlan
        second='configs/step5/second.json';self.case.put(second,dict(value='second Git default'))
        self.git('-C',str(self.root),'add','--',second)
        self.git('-C',str(self.root),'-c','user.name=Fixture','-c','user.email=fixture@example.invalid','commit','-qm','Second fixture config')
        self.case.put(second,dict(value='second archive config'));self.export();self.clone()
        original={n:(self.dest/n).read_bytes() for n in (self.config,second)}
        write=RestorePlan.write_once;backups=0
        def fail(plan,path,data):
            nonlocal backups
            if path.name.startswith('b-'):
                backups+=1
                if backups==2:raise OSError('Authored backup failure')
            return write(plan,path,data)
        with patch.object(RestorePlan,'write_once',fail),self.assertRaisesRegex(OSError,'Authored backup failure'):
            transfer.restore(self.dest,self.case.bundle,relocate=True)
        for name,before in original.items():self.assertEqual((self.dest/name).read_bytes(),before)
        result=transfer.restore(self.dest,self.case.bundle,relocate=True)
        for entry in result['repository_reconciliation']['backups']:
            self.assertEqual((self.dest/entry['backup']).read_bytes(),original[entry['path']])
            self.assertEqual((self.dest/entry['path']).read_bytes(),(self.root/entry['path']).read_bytes())

    def test_native_retry_at_same_absolute_root_still_installs_fresh_buffer_instruction(self):
        from test_native_transfer import NativeTransfer
        from brass_ai.recovery import transfer_resume
        case=NativeTransfer();case.setUp();self.addCleanup(case.doCleanups)
        case.export();case.move_source()
        # The complete original state has arrived, but publication is interrupted
        # just before the one-time destination instruction. A retry must retain
        # that this was an empty destination, even at the same absolute path.
        with patch.object(transfer_resume,'install_instruction',side_effect=RuntimeError('Authored instruction interruption')):
            with self.assertRaisesRegex(RuntimeError,'Authored instruction interruption'):
                transfer.restore(case.root,case.bundle)
        self.assertTrue((case.run/'state.json').is_file());self.assertFalse((case.run/transfer_resume.MARKER).exists())
        self.assertTrue(transfer.restore(case.root,case.bundle)['verified'])
        self.assertTrue((case.run/transfer_resume.MARKER).is_file())


if __name__=='__main__':unittest.main()
