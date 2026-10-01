"""Focused Masked ISMCTS checks; no budget qualification/tournament."""
import argparse
import sys
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--integration',action='store_true')
    args=parser.parse_args()
    suite=unittest.defaultTestLoader.discover(str(ROOT/'tests/MaskedISMCTS'),pattern='test_core.py')
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():raise SystemExit(1)
    if args.integration:
        sys.path.insert(0,str(ROOT/'tests/MaskedISMCTS'))
        from integration import run
        print(run(ROOT))

if __name__=='__main__':main()
