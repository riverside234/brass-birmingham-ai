"""Check admitted evidence; optionally restore only exactly hash-matching newlines.

No Torch, private game runtime, new qualification or changed bank adapter is needed.
"""
from pathlib import Path
import argparse
import hashlib
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from brass_ai.bank import admission, below, read, require

LAB_ROOT = Path('/data/not_backed_up/yxu209/ai-training/')


def matching_bytes(data, expected):
    """Accept original bytes or an LF/CRLF-only conversion to the recorded hash."""
    lf = data.replace(b'\r\n', b'\n')
    for candidate in (data, lf, lf.replace(b'\n', b'\r\n')):
        if hashlib.sha256(candidate).hexdigest() == expected:
            return candidate
    raise ValueError('Report contents differ beyond LF/CRLF conversion; refusing repair')


def check_reports(root, *, repair=False):
    root = Path(root).resolve()
    pool = read(root / 'training/opponents/pool.json')
    current = read(root / 'training/opponents/current.json')
    expected = {current['qualification']: current['qualification_sha256']}
    admitted = []
    for purpose in ('training', 'evaluation'):
        for identity, entry in pool[purpose].items():
            name, fingerprint = entry['qualification'], entry['sha256']
            require(name not in expected or expected[name] == fingerprint,
                    'Conflicting qualification fingerprints: ' + name)
            expected[name] = fingerprint
            admitted.append((identity, purpose, entry))
    reports, changes, equivalent = {}, [], []
    for name, fingerprint in sorted(expected.items()):
        path = below(root, name)
        original = path.read_bytes()
        try:
            candidate = matching_bytes(original, fingerprint)
        except ValueError as error:
            raise ValueError(f'{name}: {error}; expected {fingerprint}, '
                             f'actual {hashlib.sha256(original).hexdigest()}') from error
        reports[name] = json.loads(candidate.decode('utf-8-sig'))
        if candidate != original:
            equivalent.append(name)
            if repair:
                changes.append((path, original, candidate))
    # Validate all contents and references before touching any report.
    evidence = reports[current['qualification']]
    require(evidence.get('bank', evidence.get('new_bank')) == current['bank'], 'Current bank differs from evidence')
    for identity, purpose, entry in admitted:
        # Current v4 uses a selection receipt, not the original R3 per-profile
        # qualification schema. Exercise the same versioned gates as training.
        admission(root, identity, purpose)
    for path, original, candidate in changes:
        require(path.read_bytes() == original, 'Report changed during validation')
        path.write_bytes(candidate)
    # Recheck after an explicitly requested, byte-exact newline restoration.
    for identity, purpose, _ in admitted:
        admission(root, identity, purpose)
    return dict(passed=True, reports=len(reports), admitted_policies=len(admitted),
                repaired=[p.relative_to(root).as_posix() for p, _, _ in changes],
                line_endings_equivalent=equivalent,
                fingerprints_unchanged=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=LAB_ROOT)
    parser.add_argument('--repair-line-endings', action='store_true')
    args = parser.parse_args()
    require(args.root.is_absolute(), 'Pass an absolute project root')
    print(json.dumps(check_reports(args.root, repair=args.repair_line_endings), indent=2))


if __name__ == '__main__':
    main()
