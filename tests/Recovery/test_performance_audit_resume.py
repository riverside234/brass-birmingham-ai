"""Same-run continuation from the completed Step 5 performance implementation."""
import unittest
import test_cpu_performance_resume as performance_checks
from brass_ai.learning.contracts import SOURCE_ROOT,read_json,digest
from brass_ai.recovery.source_reviews import expand_source_reviews


class PerformanceAuditResumeTests(performance_checks.CpuPerformanceResumeTests):
    def test_all_pre_audit_source_approvals_are_preserved(self):
        approvals=expand_source_reviews(read_json(SOURCE_ROOT/'configs/learning/selfplay-resume-sources.json'))
        self.assertEqual(digest(approvals['transitions'][:1240]),
                         '2befe06854bb268fda4a12820919563f9d10243cd73457aa314014425aa05b84')
        origins=read_json(SOURCE_ROOT/'configs/learning/imitation-handoff-sources.json')
        self.assertEqual(digest(origins['compatible_sources'][:27]),
                         'a60c26a6fb6d9434cd77df2abcef370f0c70823a404aad221bd00ddda6485269')

    def signatures(self, name='joint7-g150-gate72.json'):
        old, new = super().signatures(name)
        old['implementation'] = 'c45e7b5338ad4ce4367ec42cbb2723135e1a0f23c302b2ec0aa4e0d4c9717c2c'
        return old, new


if __name__ == '__main__': unittest.main()
