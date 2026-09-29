"""Four-player pilot/diagnostic runner. Failures are never credited as wins."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from brass_ai.legacy import LAB_ROOT, LegacyWorker


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, default=LAB_ROOT)
    p.add_argument('--profile', default='balanced-v1')
    p.add_argument('--seeds', type=int, nargs='+', default=[9101])
    p.add_argument('--seats', type=int, nargs='+', default=[0, 1, 2, 3])
    p.add_argument('--milliseconds', type=float)
    p.add_argument('--work', type=int)
    p.add_argument('--fixed-work', action='store_true')
    p.add_argument('--diagnosis', action='store_true')
    p.add_argument('--plain', action='store_true')
    p.add_argument('--output', default='pilot')
    args = p.parse_args()
    if any(s not in (-1, 0, 1, 2, 3, 4) for s in args.seats):
        p.error('Seats 0..3: one candidate; -1: all stock; 4: all candidate')
    out = args.root / 'artifacts/legacy-ai' / args.output
    out.mkdir(parents=True, exist_ok=False)
    results = []
    with LegacyWorker(args.root, timeout=1200) as w:
        profile = w.profile(args.profile)
        if args.milliseconds is not None: profile['milliseconds'] = args.milliseconds
        if args.work is not None: profile['work_limit'] = args.work
        manifest = dict(format='legacy-pilot/0.1.0', player_count=4, profile=profile,
                        runtime=w.manifest, seeds=args.seeds, seats=args.seats, fixed_work=args.fixed_work,
                        diagnosis=args.diagnosis, status='development pilot; H5 qualification not implied')
        (out / 'manifest.json').write_text(json.dumps(manifest, indent=2))
        for seed in args.seeds:
            for seat in args.seats:
                start = time.perf_counter()
                result = w.call('run', seed=seed, candidate_seat=seat, player_count=4,
                                profile=profile, fixed_work=args.fixed_work, diagnosis=args.diagnosis,
                                record_selectors=not args.plain)
                filename = f'{seed}-{seat}.json'
                content = json.dumps(result, separators=(',', ':')).encode()
                (out / filename).write_bytes(content)
                summary = {k: result.get(k) for k in ('seed', 'candidate_seat', 'completed', 'failure', 'actions', 'turns', 'result')}
                summary['sha256'] = hashlib.sha256(content).hexdigest()
                summary['elapsed_seconds'] = round(time.perf_counter() - start, 2)
                results.append(summary)
                print(json.dumps(summary), flush=True)
                (out / 'summary.json').write_text(json.dumps(results, indent=2))
    return 0 if all(r['completed'] and not r['failure'] for r in results) else 1


if __name__ == '__main__': raise SystemExit(main())
