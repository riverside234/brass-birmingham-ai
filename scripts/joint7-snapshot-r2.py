"""Preview/apply the once-only snapshot R2 baseline hotfix; never launch training."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from brass_ai.recovery.snapshot_r2_hotfix import RUN, preview, run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', default=RUN)
    parser.add_argument('--reports', help='Read-only exported report directory; no checkpoint verification')
    parser.add_argument('--apply', action='store_true')
    parser.add_argument('--defer-until-ready', action='store_true',
                        help='Seed available current-regime evidence; freeze once ten qualify, holding the weight')
    args = parser.parse_args()
    if args.reports:
        parser.error('--reports is read-only') if args.apply or args.defer_until_ready else None
        result = preview(ROOT / args.reports)
    else:
        result = run(ROOT, args.run, apply=args.apply, defer_until_ready=args.defer_until_ready)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
