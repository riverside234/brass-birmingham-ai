"""Exact continuation through CPU input optimizations and additive timings."""
import copy
import unittest

from brass_ai.bank import read
from brass_ai.learning.contracts import SOURCE_ROOT,compatibility
from brass_ai.step5.league import implementation_id
import test_audit_resume as audit_checks
import test_boundary_performance_resume as boundary_checks
import test_time_resume as time_checks


PREVIOUS='6446834c236ef81c720c1f166a5f281798c64a2a4c7cba9419382e18531dc4ad'


class CpuPerformanceResumeTests(audit_checks.AuditResumeTests):
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
    test_same_settings_preserved_and_new_budget_or_learning_changes_rejected=boundary_checks.BoundaryResumeTests.test_same_settings_preserved_and_new_budget_or_learning_changes_rejected


if __name__=='__main__':unittest.main()
