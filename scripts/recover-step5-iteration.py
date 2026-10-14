"""Preserve an uncommitted Step 5 collection before retrying a reviewed fix."""
import argparse
import json
import os
from pathlib import Path
import sys


def main():
    source = Path(__file__).parent.parent
    os.chdir(source)
    sys.path.insert(0, str(source.resolve()/'src'))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--run', required=True)
    parser.add_argument('--expected-iteration', type=int, required=True)
    parser.add_argument('--apply', action='store_true', help='Archive the partial collection; default is validation only')
    args = parser.parse_args()
    from brass_ai.learning.contracts import require
    from brass_ai.recovery.interrupted import archive_uncommitted_iteration
    require(args.root.is_absolute(), 'Pass an absolute --root')
    result = archive_uncommitted_iteration(args.root, args.run,
                                          expected_iteration=args.expected_iteration, apply=args.apply)
    print(json.dumps({k: v for k, v in result.items() if k != 'files'}, indent=2))


if __name__ == '__main__':
    main()
