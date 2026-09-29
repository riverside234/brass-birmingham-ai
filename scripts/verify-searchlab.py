"""Qualify the additional R4/R5 experiments on this machine, separately from R2/R3."""
import argparse
import json
from pathlib import Path
import sys
import time
import unittest
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tests/SearchLab')]
from brass_ai.learning.contracts import LAB_ROOT, require
from brass_ai.searchlab.runtime import gate_identity
from brass_ai.step5.league import write_json

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=LAB_ROOT);p.add_argument('--integration',action='store_true');p.add_argument('--mono')
    p.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    a=p.parse_args();began=time.monotonic()
    identity=gate_identity(a.root,a.device,a.mono) if a.integration else None
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.discover(str(ROOT/'tests/SearchLab'),pattern='test_*.py'))
    if not result.wasSuccessful():raise SystemExit(1)
    if a.integration:
        from integration import run
        evidence=run(a.root.resolve(),a.mono,a.device)
        require(identity==gate_identity(a.root,a.device,a.mono),'Implementation/environment changed during qualification; rerun')
        report=dict(format='searchlab-correctness/1.0.0',passed=True,identity=identity,
            unit_tests=result.testsRun,evidence=evidence,seconds=time.monotonic()-began,promotion=False)
        write_json(a.root/f'artifacts/searchlab/correctness-{a.device}.json',report)
        print(json.dumps(dict(passed=True,tests=result.testsRun,identity=report['identity'],promotion=False)),flush=True)
