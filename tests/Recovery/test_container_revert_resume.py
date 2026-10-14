"""Exact continuation after the explicitly accepted container-cache tradeoff."""
import unittest
import test_cpu_performance_resume as performance_checks
from brass_ai.learning.contracts import SOURCE_ROOT, read_json, digest
from brass_ai.recovery.source_reviews import expand_source_reviews


class ContainerRevertResumeTests(performance_checks.CpuPerformanceResumeTests):
    def test_all_pre_revert_source_approvals_are_preserved(self):
        approvals = expand_source_reviews(read_json(SOURCE_ROOT / 'configs/learning/selfplay-resume-sources.json'))
        self.assertEqual(digest(approvals['transitions'][:1344]),
                         '0c07068d848245b9d786cb8dfebc3543751b95acde4887408d6ac887ac5c3530')
        origins = read_json(SOURCE_ROOT / 'configs/learning/imitation-handoff-sources.json')
        self.assertEqual(digest(origins['compatible_sources'][:28]),
                         'a5aec412e2b3594779b77e0f6777fd0a33188fe04f2561af0b9afcba8e5a727d')

    def signatures(self, name='joint7-g150-gate72.json'):
        old, new = super().signatures(name)
        old['implementation'] = 'c3e45029ffbbfab1d3a0410d4cd5eb597ff7c3275ede907882132393cdcadcc4'
        return old, new


if __name__ == '__main__': unittest.main()
