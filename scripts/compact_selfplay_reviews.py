"""Losslessly compact the self-play resume registry, including legacy v1 files."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from brass_ai.recovery.source_reviews import compact_source_reviews


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('path', nargs='?', type=Path, default=Path(__file__).resolve().parents[1] /
                        'configs/learning/selfplay-resume-sources.json')
    args = parser.parse_args()
    before = args.path.read_bytes()
    receipt = compact_source_reviews(json.loads(before.decode('utf-8-sig')))
    after = (json.dumps(receipt, indent=2, allow_nan=False) + '\n').encode('utf-8')
    if before != after:
        temporary = args.path.with_name(args.path.name + '.tmp')
        temporary.write_bytes(after)
        temporary.replace(args.path)
    print(f'{len(receipt["transitions"])} approvals; {len(before):,} -> {len(after):,} bytes')


if __name__ == '__main__':
    main()
