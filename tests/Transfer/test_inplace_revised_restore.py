"""Same-project rollback + complete revised-reward reset, without training."""
import copy
import importlib.util
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from test_native_transfer import NativeTransfer,ROOT,transfer
from brass_ai.learning import post_transfer_reward as P
from brass_ai.recovery import transfer_resume as R

spec=importlib.util.spec_from_file_location('inplace_revised_driver',ROOT/'scripts/continue-rust-revised.py')
driver=importlib.util.module_from_spec(spec);spec.loader.exec_module(driver)


class BundleDiscovery(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(dir=ROOT/'artifacts');self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.exports=self.root/'artifacts/step5-transfer';self.run='artifacts/step5/example'

    def bundle(self,name,*,generation=1284,body=b'sealed archive fixture',receipt=True):
        folder=self.exports/name;folder.mkdir(parents=True)
        part=folder/'selfplay-migration.tar.zst.part0001';part.write_bytes(body)
        index=folder/'selfplay-migration.tar.zst.parts.json'
        data=dict(format=transfer.PARTS_FORMAT,bytes=len(body),sha256=hashlib.sha256(body).hexdigest(),
            parts=[dict(name=part.name,bytes=len(body),sha256=hashlib.sha256(body).hexdigest())])
        data['id']=transfer.digest(data);index.write_bytes(transfer.encode(data))
        if receipt:
            result=dict(iteration=generation,run=self.run,bundle='/original/root/'+name+'/'+index.name,
                bundle_sha256=transfer.sha(index))
            (folder/'export-result.json').write_bytes(transfer.encode(result))
        return index

    def select(self,requested=None,**kwargs):
        return driver.select_bundle(self.root,requested,self.run,1284,**kwargs)

    def test_source_export_is_found_without_a_download_variable(self):
        index=self.bundle('step5-transfer-1234-1')
        with patch.dict('os.environ',{},clear=True):self.assertEqual(self.select(),index)
        self.assertFalse((self.root/'.training-rollback').exists())

    def test_wrong_generation_and_run_export_receipts_are_excluded(self):
        target=self.bundle('g1284');self.bundle('g1286',generation=1286,body=b'newer archive')
        other=self.bundle('other-run',body=b'other run')
        result=transfer.read(other.parent/'export-result.json');result['run']='artifacts/step5/different'
        (other.parent/'export-result.json').write_bytes(transfer.encode(result))
        self.assertEqual(self.select(),target)

    def test_empty_shell_expansion_has_an_actionable_error_without_mutation(self):
        with self.assertRaisesRegex(ValueError,'DOWNLOAD may be unset'):
            self.select(self.root/'missing/selfplay-migration.tar.zst.parts.json')
        self.assertFalse((self.root/'.training-rollback').exists())
        self.assertFalse(self.exports.exists())

    def test_duplicate_downloads_of_same_archive_are_not_ambiguous_or_scanned(self):
        a=self.bundle('a');self.bundle('b')
        original=Path.open
        def guarded(path,*args,**kwargs):
            if path.name.endswith('.part0001'):raise AssertionError('Discovery must not read large part payloads')
            return original(path,*args,**kwargs)
        with patch.object(Path,'open',guarded):self.assertEqual(self.select(),a)

    def test_distinct_eligible_exports_require_an_explicit_selection(self):
        a=self.bundle('a');b=self.bundle('b',body=b'distinct frozen archive')
        with self.assertRaisesRegex(ValueError,'Multiple distinct exports') as error:self.select()
        self.assertIn(str(a),str(error.exception));self.assertIn(str(b),str(error.exception))
        self.assertEqual(self.select(b),b)

    def test_incomplete_export_is_not_selected(self):
        index=self.bundle('incomplete');index.with_name('selfplay-migration.tar.zst.part0001').unlink()
        with self.assertRaisesRegex(ValueError,'Missing/truncated part'):self.select()
        self.assertFalse((self.root/'.training-rollback').exists())

    def test_manual_zip_and_parts_exports_without_output_receipts_are_supported(self):
        index=self.bundle('manual',receipt=False)
        self.assertEqual(self.select(),index)
        index.unlink();index.with_name('selfplay-migration.tar.zst.part0001').unlink()
        archive=index.parent/'manual.zip';archive.write_bytes(b'archive checked in prepare, not during discovery')
        self.assertEqual(self.select(),archive)

    def test_directory_override_and_unsplit_filename_resolve_existing_parts_index(self):
        index=self.bundle('manual')
        self.assertEqual(self.select(search_dir=index.parent),index)
        self.assertEqual(self.select(index.with_name('selfplay-migration.tar.zst')),index)


class InplaceRevisedRestore(unittest.TestCase):
    def setUp(self):
        self.case=NativeTransfer();self.case.setUp();self.addCleanup(self.case.doCleanups)
        self.root,self.run=self.case.root,self.case.run
        cfg=self.case.state['signature']['config']
        cfg.update(reward=copy.deepcopy(P.BEFORE),insolvency=copy.deepcopy(P.OLD_COST),search=dict(reward=P.specification(False)))
        self.case.put('distribution/rust-lab-g1250/inputs.zip',b'sealed source input fixture')
        self.case.put('artifacts/step5/selfplay/state.json',self.case.state)
        self.origin=(self.run/'state.json').read_bytes();self.case.export()
        self.new=copy.deepcopy(self.case.state);self.new['iteration']=3
        entry=self.case.entry('generation-0003.pt',b'newer checkpoint with optimizer and RNG',3)
        self.new['history'].append(entry);self.new['resume']=[entry]
        self.case.put('artifacts/step5/selfplay/state.json',self.new)
        self.current=(self.run/'state.json').read_bytes()
        self.dispatch=patch.object(driver,'transfer_module',return_value=transfer);self.dispatch.start();self.addCleanup(self.dispatch.stop)

    def prepare(self):return driver.prepare(self.root,self.case.bundle,self.run,2,3)

    def test_original_project_root_checkpoint_and_newer_backup_are_preserved_with_fresh_instruction(self):
        result=self.prepare();backup=Path(result['backup'])
        self.assertEqual(result['root'],str(self.root));self.assertEqual((self.run/'state.json').read_bytes(),self.origin)
        self.assertEqual((backup/'state.json').read_bytes(),self.current)
        self.assertEqual((backup/'generation-0003.pt').read_bytes(),b'newer checkpoint with optimizer and RNG')
        self.assertEqual((self.run/'generation-0002.pt').read_bytes(),b'checkpoint-with-Adam-and-RNG')
        marker=R.instruction(self.root,self.run);self.assertEqual(marker['reward_revision'],P.REVISION)
        self.assertEqual(marker['origin_generation'],2)
        for name in ('.ppg-eight-cache','.ppg-value-cache','teacher-labels','search-values'):
            self.assertFalse((self.run/name).exists());self.assertTrue((backup/name).exists() or name=='.ppg-value-cache')
        self.assertFalse((backup/'.active-run.json').exists());self.assertFalse((self.run/'.active-run.json').exists())
        self.assertEqual(self.prepare()['backup'],str(backup))

    def test_interrupted_restore_recovers_without_second_backup(self):
        with patch.object(R,'install_instruction',side_effect=RuntimeError('Authored instruction interruption')):
            with self.assertRaisesRegex(RuntimeError,'Authored instruction interruption'):self.prepare()
        backups=list((self.root/'.training-rollback').glob('*-g3-*'));self.assertEqual(len(backups),1)
        self.assertEqual((backups[0]/'state.json').read_bytes(),self.current)
        self.prepare()
        self.assertEqual(len(list((self.root/'.training-rollback').glob('*-g3-*'))),1)
        self.assertEqual((self.run/'state.json').read_bytes(),self.origin)

    def test_retry_after_new_commit_never_rolls_back_or_resets_again(self):
        self.prepare();state=transfer.read(self.run/'state.json');state['iteration']=3
        state['history'].append(self.case.entry('generation-0003.pt',b'new-reward committed checkpoint',3))
        self.case.put('artifacts/step5/selfplay/state.json',state);before=(self.run/'state.json').read_bytes()
        self.case.put('artifacts/step5/selfplay/teacher-labels/fresh.json',dict(new_reward=True))
        result=self.prepare();self.assertEqual(result['generation'],3)
        self.assertEqual((self.run/'state.json').read_bytes(),before)
        self.assertTrue((self.run/'teacher-labels/fresh.json').is_file())

    def test_wrong_current_generation_is_rejected_before_a_backup_or_restore(self):
        self.new['iteration']=4;self.case.put('artifacts/step5/selfplay/state.json',self.new)
        with self.assertRaisesRegex(ValueError,'Current committed generation differs'):self.prepare()
        self.assertFalse((self.root/'.training-rollback').exists())
        self.assertEqual(transfer.read(self.run/'state.json')['iteration'],4)

    def test_wrong_archive_generation_is_rejected_before_any_mutation(self):
        with self.assertRaisesRegex(ValueError,'Archive must contain'):
            driver.prepare(self.root,self.case.bundle,self.run,1,3)
        self.assertEqual((self.run/'state.json').read_bytes(),self.current)
        self.assertFalse((self.root/'.training-rollback').exists())

    def test_changed_newer_backup_blocks_retry_without_touching_restored_state(self):
        result=self.prepare();backup=Path(result['backup']);(backup/'generation-0003.pt').write_bytes(b'changed backup')
        with self.assertRaisesRegex(ValueError,'Preserved newer run changed'):self.prepare()
        self.assertEqual((self.run/'state.json').read_bytes(),self.origin)

    def test_revision_summary_includes_rewards_stage_c_holdout_and_all_schedule_settings(self):
        state=transfer.read(ROOT/'results/joint7-v11-2-g1290-trained-restart-g1284-20261108T193427364056Z/state.json')
        summary=driver.revision_summary(state,.25)
        self.assertEqual(summary['reward'],P.AFTER);self.assertEqual(summary['insolvency'],P.ZERO_COST)
        self.assertEqual(summary['stage_c_holdout'],dict(games=8,buffer_iterations=8,stop_on_loss_increase=True,interval_steps=16))
        self.assertEqual(summary['stages'],dict(A=10,B=20,C=10,trunk_increment=.1,kl_cap=.003,head_to_head_games=0))
        self.assertEqual(summary['fixed_beta'],state['teacher_phase']['weight']['gradient_qualification']['beta'])
        self.assertTrue(summary['stage_counter_paused_during_rebuild']);self.assertEqual(summary['ppg_beta_floor'],.25)


if __name__=='__main__':unittest.main()
