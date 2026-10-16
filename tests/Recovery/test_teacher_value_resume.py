"""Completed IL handoff and explicit continuation preserve training state."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from brass_ai.bank import read,sha
from brass_ai.learning.contracts import SOURCE_ROOT,compatibility,digest
from brass_ai.recovery.config import TEACHER_VALUE
from brass_ai.recovery.resume import check_signature,load_resume,commit_transition
from brass_ai.learning.checkpoint import resume_optimizer
from brass_ai.step5.league import implementation_id
import test_training_performance_resume as fixtures

PREVIOUS='014ac0ca4f920053eef265cb58b65f58901b7d35b2f6b1c1a1eaf8dbe92a5809'
AUXILIARY_SOURCE='061e72ea27f3fd4ca8f113971cf8e8711211cd8adaae116d8a8dbcea42421a7f'


class TeacherValueResumeTests(unittest.TestCase):
    state=fixtures.TrainingPerformanceResumeTests.state
    fixture=fixtures.TrainingPerformanceResumeTests.fixture
    test_guards_unrelated_config_sources_and_contracts=fixtures.TrainingPerformanceResumeTests.test_review_flag_settings_unknown_sources_and_old_contracts_are_rejected
    test_pending_work_still_blocks=fixtures.TrainingPerformanceResumeTests.test_uncommitted_collection_and_checkpoint_are_not_reused_or_modified

    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def signatures(self):
        config=read(SOURCE_ROOT/'configs/step5/joint7-latest.json');config['model_size']='small'
        previous=dict(config=copy.deepcopy(config),compatibility=compatibility(),implementation=PREVIOUS,legacy_pool={},teachers={})
        previous['config'].pop('teacher_value')
        current=copy.deepcopy(previous);current['implementation']=implementation_id()
        current['config']['teacher_value']=dict(TEACHER_VALUE,enabled=True)
        return previous,current

    def test_same_source_opt_in_out_and_diagnostic_changes_require_review(self):
        _,current=self.signatures();previous=copy.deepcopy(current)
        previous['config']['teacher_value']=dict(TEACHER_VALUE,enabled=False)
        state=self.state(previous)
        with self.assertRaisesRegex(ValueError,'resume-reviewed-source'):check_signature(state,current)
        self.assertTrue(check_signature(state,current,reviewed=True))
        disabled=copy.deepcopy(current);disabled['config']['teacher_value']['enabled']=False
        self.assertTrue(check_signature(self.state(current),disabled,reviewed=True))
        diag=copy.deepcopy(current);diag['config']['teacher_value'].update(diagnostics=True,probe_positions=16)
        self.assertTrue(check_signature(self.state(current),diag,reviewed=True))
        diag['config']['teacher_value']['probe_positions']=129
        with self.assertRaisesRegex(ValueError,'bounded'):check_signature(self.state(current),diag,reviewed=True)

    def test_preserves_checkpoint_adam_rng_progress_population(self):
        from test_teacher_value import equal_tree
        run,state,current,original,optimizer=self.fixture()
        checkpoint=run/'generation-0070.pt';before=checkpoint.read_bytes();old_state=(run/'state.json').read_bytes()
        model,raw,changed=load_resume(SOURCE_ROOT,run,state,current,reviewed=True)
        self.assertTrue(changed);self.assertEqual(old_state,(run/'state.json').read_bytes())
        restored=torch.optim.AdamW(model.parameters(),lr=current['config']['learning_rate']);resume_optimizer(raw,restored)
        equal_tree(self,original.state_dict(),model.state_dict());equal_tree(self,optimizer.state_dict(),restored.state_dict())
        self.assertTrue(torch.equal(raw['rng'],torch.get_rng_state()));self.assertFalse(model.utility_calibration['ready'])
        self.assertTrue(raw['training']['utility_calibration']['ready'])
        updated=commit_transition(run,state,current)
        for key,value in state.items():
            if key!='signature':self.assertEqual(value,updated[key],key)
        self.assertEqual(updated['source_transitions'][-1]['config_changes'],
            dict(teacher_value=dict(before=None,after=current['config']['teacher_value'])))
        restarted,again,changed=load_resume(SOURCE_ROOT,run,updated,current)
        self.assertFalse(changed);self.assertFalse(restarted.utility_calibration['ready'])
        equal_tree(self,original.state_dict(),restarted.state_dict())
        self.assertEqual(raw['model_id'],again['model_id']);self.assertEqual(before,checkpoint.read_bytes())

    def test_completed_previous_il_admission_and_old_review_prefix_unchanged(self):
        sources=read(SOURCE_ROOT/'configs/learning/imitation-handoff-sources.json')['compatible_sources']
        self.assertTrue(any(e['implementation']==PREVIOUS and e['learning_compatibility']==digest(compatibility()) for e in sources))
        # Previous loader-review regression pins all earlier review records.
        from test_bounded_imitation_resume import BoundedLoaderResumeTests
        BoundedLoaderResumeTests().test_existing_source_reviews_preserved_and_completed_il_admitted()

    def test_completed_il_from_pre_default_source_starts_enabled_without_retraining(self):
        from latest_model_fixture import imitation_fixture
        from brass_ai.recovery.demonstrations import completed_imitation
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as directory:
            with patch('latest_model_fixture.implementation_id',return_value=AUXILIARY_SOURCE):
                run,original,config=imitation_fixture(Path(directory))
            before={p.name:sha(p) for p in run.iterdir() if p.is_file()}
            model,origin=completed_imitation(SOURCE_ROOT,run)
            for name,tensor in original.state_dict().items():self.assertTrue(torch.equal(tensor,model.state_dict()[name]))
            self.assertEqual(config['imitation_checkpoint'],origin)
            self.assertTrue(config['teacher_value']['enabled'])
            self.assertFalse(config['teacher_value']['diagnostics'])
            self.assertEqual(before,{p.name:sha(p) for p in run.iterdir() if p.is_file()})


if __name__=='__main__':unittest.main()
