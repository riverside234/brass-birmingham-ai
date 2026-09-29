"""Portable headless bank checks; requires private pinned DLLs, not a Unity install.

Fixtures use diagnostic work limits; this is a correctness/platform gate, not a
claim of production timing or strength. The Windows overlay has separate checks.
"""
import argparse
import copy
from datetime import datetime,timezone
from pathlib import Path
import sys
import unittest
SOURCE=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(SOURCE/'src'),str(SOURCE/'tests/legacy-ai')]
from brass_ai.bank import LAB_ROOT,FrozenOpponent,load_bank,below,sha
from brass_ai.qualification import write_new
from test_legacy import LegacyTests
from test_v3 import V3Tests

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=LAB_ROOT);p.add_argument('--bank',required=True)
    p.add_argument('--mono');p.add_argument('--output',required=True);a=p.parse_args();root=a.root.resolve()
    profiles=load_bank(root,a.bank)['payload']['profiles']
    def start(cls):
        opponent=FrozenOpponent(root,a.bank+('/balanced-v3' if 'balanced-v3' in profiles else '/balanced-v2'),purpose='diagnostic',mono=a.mono)
        cls.w=opponent.worker
        original=cls.w.profile
        # Existing v1 fixtures exercise backwards compatibility; production v2
        # settings always come from the immutable manifest, not mutable JSON.
        cls.w.profile=lambda name='balanced-v1':copy.deepcopy(profiles[name]) if name in profiles else original(name)
    LegacyTests.setUpClass=classmethod(start)
    if 'balanced-v3' in profiles:
        V3Tests.setUpClass=classmethod(start)
    suite=unittest.defaultTestLoader.discover(str(SOURCE/'tests/legacy-ai'),pattern='test*.py')
    if 'balanced-v3' not in profiles:
        # The archived v2 binary predates the v3 flags; its original contract
        # suite remains reproducible without pretending it implements v3.
        def legacy_only(items):
            for item in items:
                if isinstance(item,unittest.TestSuite): yield from legacy_only(item)
                elif not isinstance(item,V3Tests): yield item
        suite=unittest.TestSuite(legacy_only(suite))
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    report=dict(format='frozen-bank-platform-check/1.0.0',bank=a.bank,platform=sys.platform,
        checked_at_utc=datetime.now(timezone.utc).isoformat(),passed=result.wasSuccessful(),tests=result.testsRun,
        failures=len(result.failures),errors=len(result.errors),
        manifest_sha256=sha(root/'training/opponents/bank'/a.bank/'manifest.json'),
        scope='Headless correctness only; rerun the declared tournament for platform-specific strength/latency.')
    write_new(below(root,a.output),report)
    return 0 if result.wasSuccessful() else 1

if __name__=='__main__':raise SystemExit(main())
