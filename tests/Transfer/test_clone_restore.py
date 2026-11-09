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

    def mixed_manifests(self):
        # Git normalizes the exact two kinds of bytes that a previous exporter
        # incorrectly treated as available through a clean Git checkout.
        self.case.put('src/.gitattributes',b'*.json text\n')
        names=('src/BrassLocalAI/Unity/manifest.json','src/BrassPatch/Persistence/manifest.json')
        lines=[b'fixture line '+str(i).encode()+b'\n' for i in range(30)]
        lf=b''.join(lines)
        for name in names:self.case.put(name,lf)
        self.git('-C',str(self.root),'add','--','src')
        self.git('-C',str(self.root),'-c','user.name=Fixture','-c','user.email=fixture@example.invalid','commit','-qm','Normalized manifests')
        mixed={names[0]:b''.join(lines[:9])+b''.join(lines[9:]).replace(b'\n',b'\r\n'),
               names[1]:b''.join(lines[:9]).replace(b'\n',b'\r\n')+b''.join(lines[9:])}
        for name,value in mixed.items():self.case.put(name,value)
        self.assertEqual(self.git('-C',str(self.root),'diff','--name-only','HEAD','--','src'),b'')
        return mixed

    def legacy_export(self):
        commit=self.git('-C',str(self.root),'rev-parse','HEAD').decode().strip()
        tracked={p.decode() for p in self.git('-C',str(self.root),'ls-files','-z').split(b'\0') if p}
        with patch.object(transfer,'git_baseline',return_value=(commit,tracked)):self.export(dirty=False)
        return transfer.inspect_bundle(self.case.bundle)

    def test_legacy_omitted_mixed_manifests_recover_exact_hashes_and_verify_at_new_root(self):
        mixed=self.mixed_manifests();manifest=self.legacy_export()
        self.assertTrue(all(r['external'] for r in manifest['files'] if r['path'] in mixed))
        self.clone();before={n:(self.dest/n).read_bytes() for n in mixed}
        result=transfer.restore(self.dest,self.case.bundle,relocate=True)
        self.assertTrue(result['verified']);self.assertTrue((self.dest/'.step5-relocation.json').is_file())
        for name,data in mixed.items():
            self.assertEqual((self.dest/name).read_bytes(),data)
            backup=next(e for e in result['repository_reconciliation']['backups'] if e['path']==name)
            self.assertEqual((self.dest/backup['backup']).read_bytes(),before[name])
        self.assertTrue(transfer.restore(self.dest,self.case.bundle,verify_only=True)['verified'])
        self.assertTrue(transfer.restore(self.dest,self.case.bundle)['verified'])

    def test_new_exports_bundle_git_normalized_mixed_endings_even_when_git_is_clean(self):
        mixed=self.mixed_manifests()
        # Also cover arbitrary interleaving, which cannot be recovered from
        # only a normalized blob and an old SHA without enumerating choices.
        name='src/BrassPatch/Other/manifest.json';data=b'first\r\nsecond\nthird\r\nfourth\n'
        self.case.put(name,data.replace(b'\r\n',b'\n'))
        self.git('-C',str(self.root),'add','--',name)
        self.git('-C',str(self.root),'-c','user.name=Fixture','-c','user.email=fixture@example.invalid','commit','-qm','Interleaved manifest')
        self.case.put(name,data);mixed[name]=data
        self.assertEqual(self.git('-C',str(self.root),'diff','--name-only','HEAD','--','src'),b'')
        self.export(dirty=False);manifest=transfer.inspect_bundle(self.case.bundle)
        for record in manifest['files']:
            if record['path'] in mixed:
                self.assertFalse(record['external'])
                self.assertIn(transfer.payload_name(manifest,record),self.case.archive_bytes())
        self.clone();self.assertTrue(transfer.restore(self.dest,self.case.bundle,relocate=True)['verified'])
        for name,data in mixed.items():self.assertEqual((self.dest/name).read_bytes(),data)

    def test_new_exports_prove_uniform_crlf_is_recoverable_and_bundle_content_edits(self):
        self.case.put('src/.gitattributes',b'*.py text\n')
        script='scripts/step5.py';source='src/brass_ai/beliefs.py'
        self.git('-C',str(self.root),'add','--','src')
        self.git('-C',str(self.root),'-c','user.name=Fixture','-c','user.email=fixture@example.invalid','commit','-qm','CRLF fixture')
        data=(self.root/source).read_bytes().replace(b'\n',b'\r\n');self.case.put(source,data)
        self.case.put(script,b'# source content edited\n')
        self.export(dirty=False);records={r['path']:r for r in transfer.inspect_bundle(self.case.bundle)['files']}
        self.assertTrue(records[source]['external']);self.assertFalse(records[script]['external'])
        self.clone();result=transfer.restore(self.dest,self.case.bundle,relocate=True)
        self.assertTrue(result['verified']);self.assertEqual((self.dest/source).read_bytes(),data)
        self.assertEqual((self.dest/script).read_bytes(),(self.root/script).read_bytes())

    def test_legacy_mixed_ending_recovery_never_accepts_a_similar_hash(self):
        mixed=self.mixed_manifests()
        # Same byte length and normalized content as a single-transition file,
        # but different CR positions: candidate size alone must not suffice.
        for name in mixed:
            lines=(self.root/name).read_bytes().replace(b'\r\n',b'\n').splitlines(keepends=True)
            self.case.put(name,b''.join(line.replace(b'\n',b'\r\n') if i%2==0 else line for i,line in enumerate(lines)))
        self.legacy_export();self.clone();before={n:(self.dest/n).read_bytes() for n in mixed}
        with self.assertRaisesRegex(ValueError,'2 blocking') as error:
            transfer.restore(self.dest,self.case.bundle,relocate=True)
        self.assertIn('exact archive bytes',str(error.exception))
        for name,data in before.items():self.assertEqual((self.dest/name).read_bytes(),data)
        self.assertFalse((self.dest/'.step5-relocation.json').exists())
        self.assertFalse((self.dest/'artifacts').exists())

    def test_verify_after_failed_restore_reports_incomplete_restore_not_wrong_destination(self):
        self.export();self.clone();private=self.dest/'artifacts/step5/selfplay/state.json'
        private.parent.mkdir(parents=True);private.write_bytes(b'conflicting private state')
        with self.assertRaisesRegex(ValueError,'blocking'):transfer.restore(self.dest,self.case.bundle,relocate=True)
        with self.assertRaisesRegex(ValueError,'Restore has not completed') as error:
            transfer.restore(self.dest,self.case.bundle,verify_only=True)
        self.assertIn(str(self.dest),str(error.exception));self.assertIn('rerun restore --relocate successfully',str(error.exception))
        self.assertEqual(private.read_bytes(),b'conflicting private state')
        self.assertFalse((self.dest/'.step5-relocation.json').exists())

    def test_git_omission_proof_is_bound_to_record_bytes_not_only_clean_status(self):
        from step5_transfer_restore import reproducible_git_files
        name='scripts/step5.py';data=(self.root/name).read_bytes()
        record=dict(bytes=len(data),sha256=hashlib.sha256(data).hexdigest())
        commit=self.git('-C',str(self.root),'rev-parse','HEAD').decode().strip()
        self.assertEqual(reproducible_git_files(self.root,commit,[name],expected={name:record}),{name})
        # Simulate a file changing between record hashing and Git proof. Even
        # the exact Git blob must not authorize omitting different saved bytes.
        for field,value in (('sha256','0'*64),('bytes',len(data)+1)):
            with self.subTest(field=field):
                changed=dict(record,**{field:value})
                self.assertEqual(reproducible_git_files(self.root,commit,[name],expected={name:changed}),set())
        self.assertEqual((self.root/name).read_bytes(),data)

    def test_legacy_mixed_recovery_preserves_missing_final_newline_and_bare_carriage_return(self):
        from step5_transfer_restore import git_bytes
        name='src/BrassPatch/manifest.json';original=b'first\nsecond\nthird\rvalue'
        self.case.put(name,original)
        self.git('-C',str(self.root),'add','--',name)
        self.git('-C',str(self.root),'-c','user.name=Fixture','-c','user.email=fixture@example.invalid','commit','-qm','Final-newline fixture')
        manifest=dict(git_commit=self.git('-C',str(self.root),'rev-parse','HEAD').decode().strip())
        for data in (original,b'first\r\nsecond\nthird\rvalue',b'first\nsecond\r\nthird\rvalue'):
            with self.subTest(data=data):
                record=dict(path=name,bytes=len(data),sha256=hashlib.sha256(data).hexdigest())
                self.assertEqual(git_bytes(self.root,manifest,record),data)
                self.assertIsNone(git_bytes(self.root,manifest,dict(record,sha256='0'*64)))

    def test_checkout_code_keeps_latest_files_and_bypasses_missing_archived_code_bytes(self):
        mixed=self.mixed_manifests()
        for name in mixed:
            data=(self.root/name).read_bytes().replace(b'\r\n',b'\n')
            self.case.put(name,data.replace(b'\n',b'\r\n',1))
        self.case.put(self.config,dict(value='Archived launch config'))
        self.legacy_export();self.clone()
        current={n:(self.dest/n).read_bytes() for n in (*mixed,self.config)}
        # No archived Git blob is needed for existing checkout code in this
        # mode. Private checkpoint/config payloads still restore exactly.
        with patch('step5_transfer_restore.git_bytes',side_effect=AssertionError('Unexpected archived code lookup')):
            result=transfer.restore(self.dest,self.case.bundle,relocate=True,use_checkout_code=True)
        self.assertEqual(result['code_verification'],'checkout')
        report=result['repository_reconciliation'];self.assertEqual(report['backups'],[])
        self.assertTrue(report['checkout_code']['enabled']);self.assertGreater(report['checkout_code']['files'],0)
        self.assertTrue(set(current).issubset({r['path'] for r in report['checkout_code']['differing_files']}))
        for name,data in current.items():self.assertEqual((self.dest/name).read_bytes(),data)
        for name in ('state.json','generation-0002.pt'):
            self.assertEqual((self.dest/'artifacts/step5/selfplay'/name).read_bytes(),(self.run/name).read_bytes())
        self.assertTrue(transfer.restore(self.dest,self.case.bundle,verify_only=True,use_checkout_code=True)['verified'])
        with self.assertRaisesRegex(ValueError,'Conflicting destination'):
            transfer.restore(self.dest,self.case.bundle,verify_only=True)

    def test_checkout_code_still_refuses_private_checkpoint_and_state_conflicts(self):
        self.export();self.clone();private=self.dest/'artifacts/step5/selfplay';private.mkdir(parents=True)
        for name in ('state.json','generation-0002.pt'):(private/name).write_bytes(b'private destination data')
        current=(self.dest/self.config).read_bytes()
        with self.assertRaisesRegex(ValueError,'2 blocking'):
            transfer.restore(self.dest,self.case.bundle,relocate=True,use_checkout_code=True)
        self.assertEqual((self.dest/self.config).read_bytes(),current)
        for name in ('state.json','generation-0002.pt'):self.assertEqual((private/name).read_bytes(),b'private destination data')
        self.assertFalse((self.dest/'.step5-relocation.json').exists())

    def test_checkout_code_does_not_bypass_untracked_file_conflicts(self):
        name='scripts/untracked.py';self.case.put(name,b'archived untracked code\n')
        self.export();self.clone();(self.dest/name).write_bytes(b'local untracked work\n')
        with self.assertRaisesRegex(ValueError,'private/untracked file: '+name):
            transfer.restore(self.dest,self.case.bundle,relocate=True,use_checkout_code=True)
        self.assertEqual((self.dest/name).read_bytes(),b'local untracked work\n')

    def test_checkout_code_detects_edits_after_planning_before_relocation_publication(self):
        from step5_transfer_restore import RestorePlan
        self.export();self.clone();install=RestorePlan.install;changed=False
        def edit(plan,record,temporary):
            nonlocal changed
            if not changed:
                file=self.dest/self.config;file.write_bytes(file.read_bytes()+b'concurrent edit\n');changed=True
            return install(plan,record,temporary)
        with patch.object(RestorePlan,'install',edit),self.assertRaisesRegex(ValueError,'changed before final verification'):
            transfer.restore(self.dest,self.case.bundle,relocate=True,use_checkout_code=True)
        self.assertTrue(changed);self.assertFalse((self.dest/'.step5-relocation.json').exists())

    def test_checkout_code_retains_training_compatibility_and_can_retry_after_source_correction(self):
        self.export();self.clone();name='src/brass_ai/beliefs.py';source=(self.dest/name).read_bytes()
        (self.dest/name).write_bytes(source+b'# incompatible learner source\n')
        with self.assertRaisesRegex(ValueError,'training source must still be compatible'):
            transfer.restore(self.dest,self.case.bundle,relocate=True,use_checkout_code=True)
        self.assertFalse((self.dest/'.step5-relocation.json').exists())
        self.assertEqual((self.dest/name).read_bytes(),source+b'# incompatible learner source\n')
        (self.dest/name).write_bytes(source)
        self.assertTrue(transfer.restore(self.dest,self.case.bundle,relocate=True,use_checkout_code=True)['verified'])

    def test_checkout_code_recovers_missing_required_files_instead_of_ignoring_them(self):
        self.export(dirty=False);self.clone();name='scripts/step5.py';(self.dest/name).unlink()
        result=transfer.restore(self.dest,self.case.bundle,relocate=True,use_checkout_code=True)
        self.assertTrue(result['verified']);self.assertEqual((self.dest/name).read_bytes(),(self.root/name).read_bytes())

    def test_checkout_code_and_strict_repo_files_are_mutually_exclusive(self):
        with self.assertRaisesRegex(ValueError,'mutually exclusive'):
            transfer.restore(self.dest,self.case.bundle,strict_repo_files=True,use_checkout_code=True)
        self.assertFalse(self.dest.exists());self.assertFalse(self.case.bundle.exists())

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
