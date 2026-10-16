"""Exact preceding-v11 source adoption; retain pending/source/config guards."""
import copy
import unittest
from pathlib import Path
import tempfile
from unittest.mock import patch
import torch

from brass_ai.bank import read,sha
from brass_ai.learning.contracts import SOURCE_ROOT,compatibility
from brass_ai.step5.league import implementation_id
import test_training_performance_resume as fixtures

PREVIOUS='c4b6b899d4c28aa83fc91706dbf2812233c0ca2ad3f59b4bee07e9ace3c12168'


class V11SelectorResumeTests(unittest.TestCase):
    state=fixtures.TrainingPerformanceResumeTests.state
    fixture=fixtures.TrainingPerformanceResumeTests.fixture
    test_checkpoint_optimizer_rng_progress_population=fixtures.TrainingPerformanceResumeTests.test_reviewed_commit_preserves_weights_adam_rng_population_and_counters
    test_guard_unrelated_changes=fixtures.TrainingPerformanceResumeTests.test_review_flag_settings_unknown_sources_and_old_contracts_are_rejected
    test_pending_old_collections_rejected=fixtures.TrainingPerformanceResumeTests.test_uncommitted_collection_and_checkpoint_are_not_reused_or_modified

    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def signatures(self):
        config=read(SOURCE_ROOT/'configs/step5/joint7-latest.json');config['model_size']='small'
        current=dict(config=config,compatibility=compatibility(),implementation=implementation_id(),legacy_pool={},teachers={})
        previous=copy.deepcopy(current);previous['implementation']=PREVIOUS
        return previous,current

    def test_completed_previous_v11_il_retains_weights_and_origin(self):
        from latest_model_fixture import imitation_fixture
        from brass_ai.recovery.demonstrations import completed_imitation
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as directory:
            with patch('latest_model_fixture.implementation_id',return_value=PREVIOUS):
                run,original,config=imitation_fixture(Path(directory))
            before={p.name:sha(p) for p in run.iterdir() if p.is_file()}
            model,origin=completed_imitation(SOURCE_ROOT,run)
            self.assertEqual(origin,config['imitation_checkpoint'])
            for name,value in original.state_dict().items():torch.testing.assert_close(value,model.state_dict()[name],rtol=0,atol=0)
            self.assertEqual(before,{p.name:sha(p) for p in run.iterdir() if p.is_file()})


if __name__=='__main__':unittest.main()
