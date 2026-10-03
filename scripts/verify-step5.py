"""Focused Step 5 checks; full CPU smoke commands are in the lab guide."""
from pathlib import Path
import argparse
import json
import sys
import unittest

root = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(root / 'src'), str(root / 'tests/R2')]
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--integration', action='store_true')
    parser.add_argument('--root', type=Path)
    parser.add_argument('--mono')
    args = parser.parse_args()
    result = unittest.TextTestRunner(verbosity=2).run(
        unittest.defaultTestLoader.discover(str(root / 'tests/Step5'), pattern='test_*.py'))
    if result.wasSuccessful() and args.integration:
        from brass_ai.learning.contracts import LAB_ROOT
        sys.path.insert(0, str(root / 'tests/Step5'))
        from integration import run
        print(json.dumps(run(args.root or LAB_ROOT, mono=args.mono), indent=2))
    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    raise SystemExit(main())
