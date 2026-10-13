"""Same-config continuation across the bounded validation-cache optimization."""
import copy
import unittest

import torch

from brass_ai.bank import read
from brass_ai.learning.contracts import SOURCE_ROOT,compatibility
from brass_ai.recovery.resume import check_signature
from brass_ai.step5.league import implementation_id
import test_time_resume as resume_checks

PREVIOUS='c875dde1baa158108f2a23d05cf0b60b82b0592462e3c85b33d3d5510a8e6f73'


class BoundaryResumeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def signatures(self,name='joint7-g150-gate72.json'):
        cfg=read(SOURCE_ROOT/'configs/step5'/name);cfg['model_size']='small'
        new=dict(config=cfg,compatibility=compatibility(),implementation=implementation_id(),legacy_pool={},teachers={})
        old=copy.deepcopy(new);old['implementation']=PREVIOUS
        return old,new

    def fixture(self):
        from test_teacher_work_resume import TeacherWorkResumeTests
        fixture=TeacherWorkResumeTests();fixture.signatures=self.signatures
        self.addCleanup(fixture.doCleanups)
        return fixture.fixture()

    test_weights_adam_rng_counters_and_search_debt_survive=resume_checks.TimeResumeTests.test_weights_adam_rng_counters_and_search_debt_survive
    test_partial_iteration_and_old_teacher_targets_cannot_be_reused=resume_checks.TimeResumeTests.test_partial_iteration_and_old_teacher_targets_cannot_be_reused

    def test_same_settings_preserved_and_new_budget_or_learning_changes_rejected(self):
        for path in (SOURCE_ROOT/'configs/step5').glob('*.json'):
            if read(path).get('format')!='step5-training/2.2.0':continue
            old,new=self.signatures(path.name)
            state=dict(format='step5-recovery-league/2.2.0',iteration=70,signature=old)
            self.assertEqual(old['config'],new['config'])
            self.assertTrue(check_signature(state,new,reviewed=True))
            with self.assertRaisesRegex(ValueError,'resume-reviewed-source'):check_signature(state,new)
            for section,key in (('search','seconds'),('teacher_work_budget','seconds_per_phase'),
                    ('legacy_opponent_work_budget','seconds_per_turn'),('update','target_kl'),
                    ('execution','search_workers_per_turn')):
                changed=copy.deepcopy(new)
                if key=='seconds_per_phase':changed['config'][section][key]['rail_early']+=1
                else:changed['config'][section][key]*=2
                with self.assertRaises(ValueError):check_signature(state,changed,reviewed=True)


if __name__=='__main__':unittest.main()
