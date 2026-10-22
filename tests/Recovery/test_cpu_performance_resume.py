"""Exact continuation through CPU input optimizations and additive timings."""
import copy
import unittest

from brass_ai.bank import read
from brass_ai.learning.contracts import SOURCE_ROOT,compatibility
from brass_ai.step5.league import implementation_id
import test_audit_resume as audit_checks
import test_time_resume as time_checks


# These fixtures construct the CURRENT model layout. The original source used
# a different learning contract and must not be authorized by relabeling it.
ARCHIVED_PREVIOUS='6446834c236ef81c720c1f166a5f281798c64a2a4c7cba9419382e18531dc4ad'
PREVIOUS='afcb66986a870567dbbd22ea45c5f43467ca0f7829948f04b561a9f8019e556b'


class CpuPerformanceResumeTests(audit_checks.AuditResumeTests):
    def test_archived_source_cannot_be_reinterpreted_as_current_layout(self):
        from brass_ai.recovery.resume import check_signature
        old,new=self.signatures();old['implementation']=ARCHIVED_PREVIOUS
        with self.assertRaisesRegex(ValueError,'not been reviewed'):
            check_signature(dict(format='step5-recovery-league/2.2.0',iteration=70,signature=old),new,reviewed=True)

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

    test_partial_iteration_and_old_teacher_targets_cannot_be_reused=time_checks.TimeResumeTests.test_partial_iteration_and_old_teacher_targets_cannot_be_reused

    def test_same_settings_preserved_and_new_budget_or_learning_changes_rejected(self):
        from brass_ai.recovery.resume import check_signature
        # Cover the archived explicit-budget recipe and the actual current
        # PPO-only recipe. Scanning every unrelated historical JSON repeatedly
        # expands the complete review registry and is not a focused regression.
        for name in ('joint7-g150-gate72.json','joint7-v11-2-grouped-lr-critic-ppo-only-g460.json'):
            old,new=self.signatures(name)
            state=dict(format='step5-recovery-league/2.2.0',iteration=470,signature=old)
            self.assertEqual(old['config'],new['config'])
            self.assertTrue(check_signature(state,new,reviewed=True))
            with self.assertRaisesRegex(ValueError,'resume-reviewed-source'):check_signature(state,new)
            for section,key in (('search','seconds'),('teacher_work_budget','seconds_per_phase'),
                    ('legacy_opponent_work_budget','seconds_per_turn'),('update','target_kl'),
                    ('execution','search_workers_per_turn')):
                changed=copy.deepcopy(new)
                if key=='seconds_per_phase':changed['config'][section][key]['rail_early']+=1
                else:changed['config'][section][key]*=2
                with self.subTest(name=name,field=key),self.assertRaises(ValueError):
                    check_signature(state,changed,reviewed=True)


if __name__=='__main__':unittest.main()
