"""Same-project rollback + complete revised-reward reset, without training."""
import copy
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

from test_native_transfer import NativeTransfer,ROOT,transfer
from brass_ai.learning import post_transfer_reward as P
from brass_ai.recovery import transfer_resume as R

spec=importlib.util.spec_from_file_location('inplace_revised_driver',ROOT/'scripts/continue-rust-revised.py')
driver=importlib.util.module_from_spec(spec);spec.loader.exec_module(driver)


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
