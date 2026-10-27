"""Fail-closed interpretation tests for the L14 evidence reporter."""
from pathlib import Path
import runpy
import unittest

ROOT=Path(__file__).resolve().parents[2]
report=runpy.run_path(str(ROOT/'scripts/report-rust-backend.py'))


class ReportTests(unittest.TestCase):
    def test_minimum_absolute_and_relative_benefit_are_both_required(self):
        policy=dict(minimum_main_stall_reduction_ms=4,minimum_main_stall_reduction_fraction=.3)
        f=report['benefit']
        self.assertTrue(f(100,65,policy))
        self.assertFalse(f(10,7,policy))
        self.assertFalse(f(100,90,policy))
        self.assertFalse(f(0,0,policy))
        self.assertFalse(f(100,105,policy))

    def test_tail_screen_retains_worst_observation(self):
        result=report['describe']([1]*39+[100])
        self.assertEqual(result,dict(n=40,median=1.0,p95=1,maximum=100))
        self.assertIsNone(report['describe']([]))

    def test_bootstrap_uses_fixture_blocks_reproducibly(self):
        f=report['bootstrap']
        self.assertEqual(f([-3,-5,-7],100),f([-3,-5,-7],100))
        self.assertEqual(f([-5]*8,100)['low'],-5)
        self.assertEqual(f([-5]*8,100)['high'],-5)
        self.assertIsNone(f([]))

    def test_missing_evidence_is_not_a_pass(self):
        with self.assertRaises((FileNotFoundError,AssertionError)):
            report['check_receipt'](dict(path='artifacts/does-not-exist-l14.json',sha256='0'*64))


if __name__=='__main__':unittest.main()
