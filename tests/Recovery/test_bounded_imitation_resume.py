"""Preserve exact v9 continuation while changing only standalone IL storage."""
import copy
import unittest

import torch

from brass_ai.bank import read
from brass_ai.learning.contracts import SOURCE_ROOT,compatibility,digest
from brass_ai.step5.league import implementation_id
import test_training_performance_resume as fixtures


PREVIOUS='e794a86e13b891d56fcb21a8a7580a8af1c754bc54065a8b636dd5b04f6792f8'


class BoundedLoaderResumeTests(unittest.TestCase):
    state=fixtures.TrainingPerformanceResumeTests.state
    fixture=fixtures.TrainingPerformanceResumeTests.fixture
    test_checkpoint_optimizer_rng_progress_and_population=fixtures.TrainingPerformanceResumeTests.test_reviewed_commit_preserves_weights_adam_rng_population_and_counters
    test_source_configuration_and_contract_guards=fixtures.TrainingPerformanceResumeTests.test_review_flag_settings_unknown_sources_and_old_contracts_are_rejected
    test_pending_work_still_blocks_resume=fixtures.TrainingPerformanceResumeTests.test_uncommitted_collection_and_checkpoint_are_not_reused_or_modified

    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def signatures(self):
        config=read(SOURCE_ROOT/'configs/step5/joint7-latest.json');config['model_size']='small'
        current=dict(config=config,compatibility=compatibility(),implementation=implementation_id(),legacy_pool={},teachers={})
        previous=copy.deepcopy(current);previous['implementation']=PREVIOUS
        return previous,current

    def test_existing_source_reviews_preserved_and_completed_il_admitted(self):
        raw=read(SOURCE_ROOT/'configs/learning/selfplay-resume-sources.json')
        counts=dict(reason=95,config_changes=79,config_migrations=13,config_migration=3,learning_compatibility=3)
        old=dict(format=raw['format'],transitions=raw['transitions'][:2140],
            definitions={name:{str(i):raw['definitions'][name][str(i)] for i in range(n)} for name,n in counts.items()})
        self.assertEqual(digest(old),'6cbea850777c59249618fb608ae8844401c78ecb77caab56e371b13a6d8be58a')
        sources=read(SOURCE_ROOT/'configs/learning/imitation-handoff-sources.json')['compatible_sources']
        self.assertTrue(any(e['implementation']==PREVIOUS and e['learning_compatibility']==digest(compatibility()) for e in sources))


if __name__=='__main__':unittest.main()
