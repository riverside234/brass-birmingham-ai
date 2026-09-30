"""Focused recovery checks; use --integration for real small four-player collections."""
from pathlib import Path
import argparse
import json
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tests/Recovery'),str(ROOT/'tests/R2')]

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--integration',action='store_true');p.add_argument('--root',type=Path,default=ROOT)
    p.add_argument('--mono');p.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    a=p.parse_args()
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.discover(str(ROOT/'tests/Recovery'),pattern='test_*.py'))
    if not result.wasSuccessful(): raise SystemExit(1)
    if a.integration:
        from integration import run
        print(json.dumps(run(a.root,a.device,a.mono),indent=2))
